#!/usr/bin/env python3
""" https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block_causal_varlen.py
SageAttention: Accurate 8-bit Inference Attention https://arxiv.org/html/2410.02367v6
     N_CTX     pytorch   flash_attn  flash_attn_varlen  sageattn_varlen
0     1024  101.688067   118.003940         109.207398        53.948444
1     2048  140.851681   262.960917         142.428452       110.656421
2     4096  155.690084   536.076564         157.530936       172.889696
3     8192  162.372505  1090.679558         164.438866       232.255332
4    16384  164.710178  2193.152006         168.112195       264.587893
"""

import torch, triton, math
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, q_scale, kv_len,
    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
    start_m, H: tl.constexpr,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M

    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += (lo // BLOCK_N) * H
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)   
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += H
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q, K, V, cu_seqlens_q, cu_seqlens_k,
    Q_scale, K_scale, cu_seqlens_q_scale, cu_seqlens_k_scale,
    Out,  
    stride_qh, stride_qn, stride_kh, stride_kn,  
    stride_vh, stride_vn, stride_oh, stride_on,  
    H: tl.constexpr, num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,  
    BLOCK_N: tl.constexpr, STAGE: tl.constexpr
):
    start_m = tl.program_id(0)
    off_z   = tl.program_id(2).to(tl.int64)
    off_h   = tl.program_id(1).to(tl.int64)

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end   = tl.load(cu_seqlens_q + off_z + 1)

    qo_len = cu_seqlens_q_end - cu_seqlens_q_start
    if (start_m * BLOCK_M) >= qo_len: return #####

    cu_seq_lens_q_scale_start = tl.load(cu_seqlens_q_scale + off_z)
    cu_seq_lens_k_scale_start = tl.load(cu_seqlens_k_scale + off_z)    

    q_scale_offset = cu_seq_lens_q_scale_start * H + off_h + start_m * H
    k_scale_offset = cu_seq_lens_k_scale_start * (H // num_kv_groups) + off_h // num_kv_groups

    cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end   = tl.load(cu_seqlens_k + off_z + 1)

    kv_len = cu_seqlens_k_end - cu_seqlens_k_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    off_hkv = (off_h//num_kv_groups)

    Q_ptrs      = Q   + (cu_seqlens_q_start*stride_qn + off_h  *stride_qh) + offs_m[:, None]*stride_qn + offs_k[None, :]
    K_ptrs      = K   + (cu_seqlens_k_start*stride_kn + off_hkv*stride_kh) + offs_n[None, :]*stride_kn + offs_k[: ,None] 
    V_ptrs      = V   + (cu_seqlens_k_start*stride_vn + off_hkv*stride_vh) + offs_n[:, None]*stride_vn + offs_k[None, :]
    O_block_ptr = Out + (cu_seqlens_q_start*stride_on + off_h  *stride_oh) + offs_m[:, None]*stride_on + offs_k[None, :]

    Q_scale_ptr = Q_scale + q_scale_offset
    K_scale_ptr = K_scale + k_scale_offset
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
        start_m, H // num_kv_groups, BLOCK_M, HEAD_DIM, BLOCK_N, 4 - STAGE, offs_m, offs_n 
    )
    acc, l_i, _ = _attn_fwd_inner(
        acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
        start_m, H // num_kv_groups, BLOCK_M, HEAD_DIM, BLOCK_N,  2, offs_m, offs_n 
    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))


def attn_true_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, 
        k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=torch.float16):

    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    b = cu_seqlens_q.shape[0] - 1

    _, h_qo, head_dim = q.shape
    _, h_kv, _ = k.shape

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale,
        o,  
        q.stride(1), q.stride(0), 
        k.stride(1), k.stride(0),  
        v.stride(1), v.stride(0), 
        o.stride(1), o.stride(0),
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage, 
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4)
    return o



@triton.jit
def quant_per_block_int8_kernel(
    Input, Output, Scale,
    cu_seqlens_input, cu_seqlens_scale,
    stride_ih, stride_in,
    stride_oh, stride_on,
    sm_scale, H: tl.constexpr,
    C: tl.constexpr, BLK: tl.constexpr
):
    off_blk = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_b   = tl.program_id(2)

    cu_seqlens_input_start = tl.load(cu_seqlens_input + off_b)
    cu_seqlens_input_end   = tl.load(cu_seqlens_input + off_b + 1)

    L = cu_seqlens_input_end - cu_seqlens_input_start

    if (off_blk * BLK) >= L: return
    
    cu_seqlens_scale_start = tl.load(cu_seqlens_scale + off_b)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs  = Input  + cu_seqlens_input_start*stride_in + off_h*stride_ih + offs_n[:, None]*stride_in + offs_k[None, :]
    output_ptrs = Output + cu_seqlens_input_start*stride_on + off_h*stride_oh + offs_n[:, None]*stride_on + offs_k[None, :]
    scale_ptrs  = Scale + cu_seqlens_scale_start * H + off_h + off_blk * H

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


def per_block_int8_varlen(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, BLKQ=128, BLKK=64, sm_scale=None):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    h_qo = q.shape[1]
    h_kv = k.shape[1]
    head_dim = q.shape[-1]

    b = cu_seqlens_q.shape[0] - 1
    q_batch_len = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_batch_len = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

    q_scale_len = (q_batch_len + BLKQ - 1) // BLKQ
    k_scale_len = (k_batch_len + BLKK - 1) // BLKK

    cu_seqlens_q_scale = torch.nn.functional.pad(torch.cumsum(q_scale_len, dim=0), (1, 0), value=0)
    cu_seqlens_k_scale = torch.nn.functional.pad(torch.cumsum(k_scale_len, dim=0), (1, 0), value=0)

    q_scale = torch.empty((cu_seqlens_q_scale[-1], h_qo), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((cu_seqlens_k_scale[-1], h_kv), device=k.device, dtype=torch.float32)

    if sm_scale is None: sm_scale = head_dim**-0.5

    grid = ((max_seqlen_q + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_int8_kernel[grid](
        q, q_int8, q_scale,
        cu_seqlens_q, cu_seqlens_q_scale,
        q.stride(1), q.stride(0),
        q_int8.stride(1), q_int8.stride(0),
        sm_scale=(sm_scale * 1.44269504), H=h_qo,
        C=head_dim, BLK=BLKQ
    )

    grid = ((max_seqlen_k + BLKK - 1) // BLKK, h_kv, b)

    quant_per_block_int8_kernel[grid](
        k, k_int8, k_scale,
        cu_seqlens_k, cu_seqlens_k_scale,
        k.stride(1), k.stride(0),
        k_int8.stride(1), k_int8.stride(0),
        sm_scale=1.0, H=h_kv,
        C=head_dim, BLK=BLKK
    )

    return q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale


@torch.compiler.disable
def sageattn_varlen(q, k, v, cu_seqlens, max_seqlen, sm_scale:float=None) -> torch.Tensor:
    """
    Parameters
    ----------
    q : torch.Tensor, shape: ``[cu_seqlens[-1], num_qo_heads, head_dim]``.
    k : torch.Tensor, shape: ``[cu_seqlens[-1], num_kv_heads, head_dim]``.
    v : torch.Tensor, shape: ``[cu_seqlens[-1], num_kv_heads, head_dim]``.

    cu_seqlens : torch.Tensor
        The cumulative sequence lengths for the query, key, value sequences in the batch, used to index into `q`, `k`, `v`. 
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    max_seqlen : int, The maximum sequence length for the query, key, value tensors in the batch.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    Returns
    -------
    The output tensor, shape: ``[cu_seqlens[-1], num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16``.
    - The tensors `cu_seqlens` must have the dtype ``torch.int32`` or ``torch.int64``.
    - !!! `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances. !!!
    """
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    ''' FIXME(DefTruth): make sage attention work compatible with distributed env, for example, xDiT
which launch by torchrun. Without this workaround, sage attention will run into illegal memory access
error after first inference step in distributed env for multi gpus inference. This small workaround
also make sage attention work compatible with torch.compile through non-fullgraph compile mode. '''
    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)
    assert head_dim_og in [64, 128], "Only support 64 or 128 head_dim"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous."

    if dtype == torch.bfloat16: v = v.to(torch.float16) # tại sao phải convert sang float16?
    k = k - k.mean(dim=0, keepdim=True) # Always smooth_k
    if sm_scale is None: sm_scale = 1.0 / (head_dim_og ** 0.5)

    q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale = \
        per_block_int8_varlen(q, k, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, sm_scale=sm_scale)

    o = attn_true_varlen(q_int8, k_int8, v, cu_seqlens, cu_seqlens, max_seqlen, q_scale, \
            k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)
    return o


if __name__ == "__main__":
    import torch.nn.functional as F
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    lines = "pytorch flash_attn flash_attn_varlen sageattn_varlen".split()
    BATCH, N_HEADS, HEAD_DIM = 8, 8, 128

    config = triton.testing.Benchmark(
        line_vals=lines, line_names=lines,
        line_arg="provider", x_names=["N_CTX"], ylabel="ms", 
        x_vals=[2**i for i in range(10, 15)], # 1024 2048 4096 8192 16384   
        # styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        plot_name=f"attn-bs{BATCH}-h{N_HEADS}-d{HEAD_DIM}",
        args=dict(H=N_HEADS, BATCH=BATCH, HEAD_DIM=HEAD_DIM),
    )


    @triton.testing.perf_report([config])
    def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, provider, device="cuda"):
        dtype = torch.bfloat16
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)

        max_seqlen, seq_len = N_CTX, BATCH*N_CTX 
        cu_seqlens = torch.tensor([i*N_CTX for i in range(BATCH + 1)], dtype=torch.int32, device="cuda")                
        qq = q.transpose(1, 2).reshape(seq_len, H, HEAD_DIM) # seq_len, H, D
        kk = k.transpose(1, 2).reshape(seq_len, H, HEAD_DIM) # seq_len, H, D
        vv = v.transpose(1, 2).reshape(seq_len, H, HEAD_DIM) # seq_len, H, D

        def attn_fn(provider, q, k, v):
            if "sageattn" in provider:
                return lambda: sageattn_varlen(qq, kk, vv, cu_seqlens, max_seqlen, sm_scale=1.3)

            if provider == "pytorch":
                return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.3)
            
            if provider == "flash_attn_varlen":
                return lambda: flash_attn_varlen_func(qq, kk, vv, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)

            return lambda: flash_attn_func(q=q, k=k, v=v, dropout_p=float(0.0), softmax_scale=1.3, 
                causal=True, window_size=(-1,-1), alibi_slopes=None, deterministic=False)

        ms = triton.testing.do_bench(attn_fn(provider, q, k, v), warmup=15, rep=50)

        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        total_flops *= 0.5
        return total_flops / ms * 1e-9


    def assert_triton_attn_is_same_as_sdpa():
        torch.manual_seed(0)
        q = torch.randn((4, 32, 1024, 64), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((4, 32, 1024, 64), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((4, 32, 1024, 64), dtype=torch.bfloat16, device="cuda")
        o_torch  = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.3) 

        m = 1024
        c = torch.tensor([0, 1024, 2*1024, 3*1024, 4*1024], dtype=torch.int32, device="cuda")
        q = q.transpose(1, 2).reshape(4*1024, 32, 64) # seq_len, H, D
        k = k.transpose(1, 2).reshape(4*1024, 32, 64) # seq_len, H, D
        v = v.transpose(1, 2).reshape(4*1024, 32, 64) # seq_len, H, D
        o_triton = sageattn_varlen(q, k, v, c, m, sm_scale=1.3)
        o_triton = o_triton.reshape(4, 1024, 32, 64).transpose(1, 2)

        o_flash = flash_attn_varlen_func(q, k, v, c, c, m, m, causal=True, softmax_scale=1.3)
        o_flash = o_flash.reshape(4, 1024, 32, 64).transpose(1, 2)

        assert o_triton.shape == (4, 32, 1024, 64)
        assert o_flash.shape  == (4, 32, 1024, 64)

        x, y = o_triton[0][0], o_flash[0][0]
        if not torch.allclose(x, y, rtol=0.25*1e-1, atol=0.3*1e-1):
            print(f"sage ~= flash AssertionError:\n{x}\n{y}")

        x, y = o_torch[0][0], o_flash[0][0]
        if not torch.allclose(x, y, rtol=0.25*1e-2, atol=0.3*1e-1):
            print(f"torch ~= flash AssertionError:\n{x}\n{y}")

    # 测试推理正确性.
    assert_triton_attn_is_same_as_sdpa()
    bench_flash_attention.run(save_path=None, print_data=True)
    print("total_flops, higher is better.")
