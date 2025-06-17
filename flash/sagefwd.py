""" SageAttention: Accurate 8-bit Inference Attention 
https://ar5iv.org/html/2410.02367v8
"""
import torch, triton, math
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, q_scale, kv_len,
    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, start_m, H: tl.constexpr,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,):

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



_cfgs = [ triton.Config(dict(BLOCK_N=n), num_stages=s, num_warps=w) for m, n, s, w in \
    [ ( 64, 3, 8), (64, 4, 8), (128, 3, 8), (128, 4, 8), ( 256, 3, 8), (256, 4, 8), ]]
@triton.autotune(configs=_cfgs, key=['HEAD_DIM', 'H', 'num_kv_groups', 'STAGE', 'BLOCK_M'],)
@triton.jit
def _attn_fwd(
    Q, K, V, cu_seqlens,
    Q_scale, K_scale, cu_seqlens_scale,
    Out, 
    stride_qh, stride_qn, stride_kh, stride_kn,  
    stride_vh, stride_vn, stride_oh, stride_on,  
    H: tl.constexpr, num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,  
    BLOCK_N: tl.constexpr, STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z   = tl.program_id(2).to(tl.int64)
    off_h   = tl.program_id(1).to(tl.int64)

    cu_seqlens_q_start = tl.load(cu_seqlens + off_z)
    cu_seqlens_q_end   = tl.load(cu_seqlens + off_z + 1)

    qo_len = cu_seqlens_q_end - cu_seqlens_q_start
    if (start_m * BLOCK_M) >= qo_len: return #####

    cu_seq_lens_scale_start = tl.load(cu_seqlens_scale + off_z)
    q_scale_offset = cu_seq_lens_scale_start * H + off_h + start_m * H
    k_scale_offset = cu_seq_lens_scale_start * (H // num_kv_groups) + off_h // num_kv_groups

    cu_seqlens_k_start = tl.load(cu_seqlens + off_z)
    cu_seqlens_k_end   = tl.load(cu_seqlens + off_z + 1)

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


def attn_true_varlen(q, k, v, cu_seqlens, max_seqlen, q_scale, k_scale, cu_seqlens_scale, output_dtype=torch.float16):
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    _, h_qo, head_dim = q.shape
    _, h_kv, _ = k.shape

    HEAD_DIM_K      = head_dim
    num_kv_groups   = h_qo // h_kv
    BLOCK_M         = 128
    num_seqs        = cu_seqlens.shape[0] - 1

    _grid = (triton.cdiv(max_seqlen, BLOCK_M), h_qo, num_seqs)
    _attn_fwd[_grid](
        q, k, v, cu_seqlens,
        q_scale, k_scale, cu_seqlens_scale,
        o, 
        q.stride(1), q.stride(0), 
        k.stride(1), k.stride(0),  
        v.stride(1), v.stride(0), 
        o.stride(1), o.stride(0),
        h_qo, num_kv_groups,
        HEAD_DIM    = HEAD_DIM_K,  
        STAGE       = 3,
        BLOCK_M     = BLOCK_M,
        # num_warps = ( 4 if head_dim == 64 else 8 )
    )
    return o


@triton.jit
def quant_per_block_int8_kernel(
    Input, Output, Scale,
    cu_seqlens_input, cu_seqlens_scale,
    stride_ih, stride_in, stride_oh, stride_on,
    sm_scale: tl.constexpr, H: tl.constexpr,
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
    scale_ptrs  = Scale  + cu_seqlens_scale_start*H + off_h + off_blk*H

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32) * sm_scale # softmax scale
    tensor_scale = tl.max(tl.abs(x)) / 127.0
    x /= tensor_scale
    x += 0.5*tl.where(x>=0, 1, -1)  # round-to-nearest
    tl.store(output_ptrs, x.to(tl.int8), mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, tensor_scale)


def per_block_int8_varlen(q, k, cu_seqlens, max_seqlen, BLK_QK=64, sm_scale=None):
    ''' Inputs:
    q: [total_seqlens, num_qo_heads, head_dim]  
    k: [total_seqlens, num_kv_heads, head_dim]
    cu_seqlens: [batch_size + 1] - cumulative sequence lengths
    
    Returns:
    q_int8: [total_seqlens, num_qo_heads, head_dim] - dtype=int8
    k_int8: [total_seqlens, num_kv_heads, head_dim] - dtype=int8  
    q_scale: [total_blocks, num_qo_heads] - dtype=float32
    k_scale: [total_blocks, num_kv_heads] - dtype=float32
    cu_seqlens_scale: [batch_size + 1] - cumulative block counts for seqs in varlen

    Với:
    total_seqlens = cu_seqlens[-1]
    total_blocks  = cu_seqlens_scale[-1]
    '''
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    h_qo = q.shape[1]
    h_kv = k.shape[1]
    head_dim = q.shape[-1]

    b = cu_seqlens.shape[0] - 1
    batch_len = cu_seqlens[1:] - cu_seqlens[:-1]
    scale_len = (batch_len + BLK_QK - 1) // BLK_QK

    cu_seqlens_scale = torch.cumsum(scale_len, dim=0)
    cu_seqlens_scale = torch.nn.functional.pad(cu_seqlens_scale, (1, 0), value=0) # thêm 0 vào bên trái <= (1, 0)

    q_scale = torch.empty((cu_seqlens_scale[-1], h_qo), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((cu_seqlens_scale[-1], h_kv), device=k.device, dtype=torch.float32)

    if sm_scale is None: sm_scale = (2*head_dim)**-0.5
    grid = ((max_seqlen + BLK_QK - 1) // BLK_QK, h_qo, b)

    quant_per_block_int8_kernel[grid](
        q, q_int8, q_scale,
        cu_seqlens, cu_seqlens_scale,
        q.stride(1), q.stride(0),
        q_int8.stride(1), q_int8.stride(0),
        sm_scale=sm_scale, H=h_qo,
        C=head_dim, BLK=BLK_QK
    )
    grid = ((max_seqlen + BLK_QK - 1) // BLK_QK, h_kv, b)
    quant_per_block_int8_kernel[grid](
        k, k_int8, k_scale,
        cu_seqlens, cu_seqlens_scale,
        k.stride(1), k.stride(0),
        k_int8.stride(1), k_int8.stride(0),
        sm_scale=1.0, H=h_kv,
        C=head_dim, BLK=BLK_QK
    )
    return q_int8, q_scale, k_int8, k_scale, cu_seqlens_scale


@torch.compiler.disable
def sageattn_varlen(q, k, v, cu_seqlens, max_seqlen, sm_scale:float=None) -> torch.Tensor:
    """q: torch.Tensor, shape: [cu_seqlens[-1], num_qo_heads, head_dim].
    k, v: torch.Tensor, shape: [cu_seqlens[-1], num_kv_heads, head_dim].
    sm_scale: Optional[float]: softmax scale, if not provided, set to 1.0 / sqrt(head_dim).
    Return Tensor shape: [cu_seqlens[-1], num_qo_heads, head_dim].
    """
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    torch.cuda.set_device(v.device) ### FIXME(DefTruth)

    head_dim_og = q.size(-1)
    assert head_dim_og in [64, 128], "Only support 64 or 128 head_dim"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous."

    if dtype == torch.bfloat16: v = v.to(torch.float16) # tại sao phải convert v bf16 to fl16? <= vì acc là fp16
    k = k - k.mean(dim=0, keepdim=True)                 # Always smooth_k (zero centering)
    if sm_scale is None: sm_scale = 1.0 / (head_dim_og ** 0.5)

    q_int8, q_scale, k_int8, k_scale, cu_seqlens_scale = per_block_int8_varlen(q, k, cu_seqlens, max_seqlen, sm_scale=sm_scale)
    o = attn_true_varlen(q_int8, k_int8, v, cu_seqlens, max_seqlen, q_scale, k_scale, cu_seqlens_scale, output_dtype=dtype)
    return o

