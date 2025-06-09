# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings, torch, triton, triton.language as tl
from einops import rearrange, reduce
from typing import Optional

from .cumsum import chunk_global_cumsum
from .utils import prepare_chunk_indices, exp, log, safe_exp
from .utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous
tlc = tl.constexpr

@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'IS_VARLEN'],
)
@triton.jit
def parallel_attn_fwd_kernel(
    q, k, v, o,             # query, key, value, output tensors
    g_cumsum,               # gradient cumulative sum (cho gating mechanism)
    lse,                    # LogSumExp (lưu cho backward pass)
    scale,                  # attention scale factor (thường là 1/sqrt(d_k))
    cu_seqlens,             # cumulative sequence lengths (cho variable-length sequences)
    chunk_indices,          # chỉ số chunks cho varlen processing
    T, B: tlc, H: tlc,      # T=seq_len, B=batch_size, H=num_kv_heads
    HQ: tl.constexpr,       # số query heads (có thể khác H cho GQA)
    G: tlc, K: tlc, V: tlc, # G=group_size (HQ//H), K=key_dim, V=value_dim
    BT: tl.constexpr,       # block size cho sequence dimension
    BS: tl.constexpr,       # block size cho source sequence trong inner loop
    BK: tl.constexpr,       # block size cho key dimension
    BV: tl.constexpr,       # block size cho value dimension
    USE_G: tl.constexpr,    # có sử dụng gating mechanism không
    IS_VARLEN: tl.constexpr,# có phải variable-length sequences không
):
    # === 1. GIẢI MÃ THREAD/BLOCK INDICES ===
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # i_v: block index cho value dimension (để xử lý V theo chunks)
    # i_t: block index cho time/sequence dimension (BT tokens mỗi block)
    # i_bh: block index cho batch * head dimension
    
    i_b, i_hq = i_bh // HQ, i_bh % HQ  # batch index và query head index
    i_h = i_hq // G                    # key/value head index (cho GQA: nhiều Q heads dùng chung 1 KV head)

    # === 2. XÁC ĐỊNH SEQUENCE BOUNDARIES ===
    if IS_VARLEN:
        # Variable-length: mỗi sequence có độ dài khác nhau
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos  # chiều dài thực tế của sequence này
    else:
        # Fixed-length: tất cả sequences đều có độ dài T
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    # === 3. TẠO BLOCK POINTERS CHO MEMORY ACCESS ===
    # Pointer cho Q block hiện tại: [BT, BK]
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    # Pointer cho output block: [BT, BV] 
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # Pointer cho LogSumExp (để numerical stability): [BT]
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # === 4. LOAD Q BLOCK VÀO SHARED MEMORY ===
    # Q block được giữ nguyên trong shared memory suốt quá trình tính toán
    # [BT, BK] - BT tokens, BK key dimensions
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)  # apply attention scale (1/sqrt(d_k))
    
    # === 5. KHỞI TẠO ACCUMULATORS CHO ONLINE SOFTMAX ===
    # Output accumulator: [BT, BV] - tích lũy weighted values 
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    
    # Online softmax state - giữ running max và sum cho numerical stability
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)  # running max values
    b_acc = tl.zeros([BT], dtype=tl.float32)              # running sum of exp(x - max)

    # === 6. XỬ LÝ GATING MECHANISM (nếu có) ===
    if USE_G:
        # Load gating values cho Q tokens
        p_g = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None

    # === 7. LOOP QUA CÁC K,V BLOCKS TRƯỚC CURRENT POSITION (FULL ATTENTION) ===
    for i_s in range(0, i_t * BT, BS):
        # Tạo pointers cho K,V blocks hiện tại
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        
        # Load K,V blocks từ global memory
        b_k = tl.load(p_k, boundary_check=(0, 1))  # [BK, BS] - transposed cho efficient matmul
        b_v = tl.load(p_v, boundary_check=(0, 1))  # [BS, BV]
        
        # Tính attention scores: Q @ K^T 
        b_s = tl.dot(b_q, b_k)  # [BT, BS] - attention scores

        # Apply gating nếu có (linear attention variant)
        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]  # add gating bias

        # === ONLINE SOFTMAX UPDATE ===
        # Cập nhật running maximum cho numerical stability
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m  # new_max, old_max
        b_r = exp(b_mp - b_m)  # renormalization factor = exp(old_max - new_max)
        
        # Tính attention weights với new maximum
        b_p = safe_exp(b_s - b_m[:, None])  # [BT, BS] - attention weights
        
        # Cập nhật running sum và output
        b_acc = b_acc * b_r + tl.sum(b_p, 1)              # update normalizer
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)  # update weighted sum

        b_mp = b_m  # save for next iteration

    # === 8. LOOP QUA CURRENT BLOCK VỚI CAUSAL MASKING ===
    # Chỉ xử lý positions trong current block, với causal constraint
    o_q = i_t * BT + tl.arange(0, BT)  # absolute positions của Q tokens
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # Tạo pointers cho K,V trong current block
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        # Absolute positions của K tokens
        o_k = i_s + tl.arange(0, BS)
        
        # Load K,V blocks
        b_k = tl.load(p_k, boundary_check=(0, 1))  # [BK, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))  # [BS, BV]
        
        # Tính attention scores
        b_s = tl.dot(b_q, b_k)  # [BT, BS]
        
        # === CAUSAL MASKING ===
        # Chỉ cho phép attend đến positions <= current position
        b_s = tl.where(o_q[:, None] >= o_k[None, :], b_s, float('-inf'))

        # Apply gating
        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        # Online softmax update (tương tự như loop trước)
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp(b_mp - b_m)
        b_p = safe_exp(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_mp = b_m

    # === 9. FINALIZATION ===
    # Normalize output bằng accumulated softmax denominator
    b_o = b_o / b_acc[:, None]
    # Lưu log(sum_exp) cho backward pass
    b_m += log(b_acc)
    
    # Store results về global memory
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


@triton.jit
def parallel_attn_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_attn_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    dg_cumsum,
    g_cumsum,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT]
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    if USE_G:
        b_dg = tl.zeros([BT, ], dtype=tl.float32)
        p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None
        b_dg = None

    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        b_p = safe_exp(b_s - b_lse[:, None])
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        # [BS]
        o_k = i_s + tl.arange(0, BS)
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k)

        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
            b_s = tl.where(o_q[:, None] >= o_k[None, :], b_s, -float('inf'))

        b_p = safe_exp(b_s - b_lse[:, None])  # SY: important to use safe_exp here to avoid NaN.
        b_p = tl.where(o_q[:, None] >= o_k[None, :], b_p, 0)

        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    g_cumsum,
    lse,
    delta,
    do,
    dk,
    dv,
    dg_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    o_k = i_t * BT + tl.arange(0, BT)

    if USE_G:
        p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
        b_dg = tl.zeros([BT,], dtype=tl.float32)
    else:
        b_gk = None
        b_dg = None

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BS]
        o_q = i_s + tl.arange(0, BS)
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q))
        if USE_G:
            p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
            b_s = tl.where(o_k[:, None] <= o_q[None, :], b_s, -float('inf'))
        b_p = safe_exp(b_s - b_lse[None, :])
        b_p = tl.where(o_k[:, None] <= o_q[None, :], b_p, 0)
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)

    for i_s in range((i_t + 1) * BT, tl.cdiv(T, BS) * BS, BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BS]
        o_q = i_s + tl.arange(0, BS)
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q))
        if USE_G:
            p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        b_p = safe_exp(b_s - b_lse[None, :])
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def parallel_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    scale: float,
    chunk_size: int = 128,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """
    Forward pass của parallel attention với memory-efficient block processing
    
    Args:
        q: Query tensor [B, T, HQ, K] 
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        g_cumsum: Cumulative gating (None nếu không dùng)
        scale: Attention scale factor
        chunk_size: Kích thước block cho sequence dimension
        cu_seqlens: Cumulative sequence lengths cho variable-length
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]  # số query heads (có thể > H cho GQA)
    G = HQ // H      # group size cho Grouped Query Attention
    BT = chunk_size
    
    # === DYNAMIC BLOCK SIZING DỰA TRÊN GPU ARCHITECTURE ===
    if check_shared_mem('hopper', q.device.index):  # H100
        BS = min(64, max(16, triton.next_power_of_2(T)))    # sequence block size
        BK = min(256, max(16, triton.next_power_of_2(K)))   # key dim block size  
        BV = min(256, max(16, triton.next_power_of_2(V)))   # value dim block size
    elif check_shared_mem('ampere', q.device.index):  # A100/3090/4090
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
    else:  # Older GPUs
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
    
    NK = triton.cdiv(K, BK)  # số blocks cho key dimension
    NV = triton.cdiv(V, BV)  # số blocks cho value dimension

    # Chuẩn bị chunk indices cho variable-length sequences
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert NK == 1, "Key dimension không thể lớn hơn 256 (giới hạn shared memory)"

    # Allocate output tensors
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    
    # Launch kernel với grid dimensions
    grid = (NV, NT, B * HQ)  # (value_blocks, time_blocks, batch_head_blocks)
    parallel_attn_fwd_kernel[grid](
        q=q, k=k, v=v, o=o,
        g_cumsum=g_cumsum, lse=lse, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        B=B, T=T, H=H, HQ=HQ, G=G, K=K, V=V,
        BT=BT, BS=BS, BK=BK, BV=BV,
    )
    return o, lse


def parallel_attn_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_attn_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    g_cumsum: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    scale: float = None,
    chunk_size: int = 128,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = chunk_size
    BS = max(16, triton.next_power_of_2(T))
    BS = min(32, BS) if check_shared_mem('ampere') else min(16, BS)  # SY:H100 should at least use BS=64 to use WGMMA
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)

    delta = parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dk = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device)
    grid = (NV, NT, B * HQ)

    dg_cumsum, dg_cumsum_k = None, None
    if g_cumsum is not None:
        dg_cumsum = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        dg_cumsum_k = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    parallel_attn_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        dg_cumsum=dg_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV
    )
    parallel_attn_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        dg_cumsum=dg_cumsum_k,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV
    )
    dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    dv = reduce(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
    if g_cumsum is not None:
        dg_cumsum.add_(dg_cumsum_k)
    return dq, dk, dv, dg_cumsum


@torch.compile
class ParallelAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, cu_seqlens):
        ctx.dtype = q.dtype

        chunk_size = min(128, max(16, triton.next_power_of_2(q.shape[1])))

        g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens) if g is not None else None
        o, lse = parallel_attn_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            scale=scale,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, o, g_cumsum, lse)
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, g_cumsum, lse = ctx.saved_tensors
        dq, dk, dv, dg = parallel_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=g_cumsum,
            lse=lse,
            do=do,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
            cu_seqlens=ctx.cu_seqlens,
        )
        if dg is not None:
            dg = chunk_global_cumsum(dg, cu_seqlens=ctx.cu_seqlens, reverse=True)

        return dq.to(q), dk.to(k), dv.to(v), dg, None, None


def parallel_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA will be applied if HQ is divisible by H.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (Optional[torch.Tensor]):
            log decay factors of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v))
        if g is not None:
            g = rearrange(g, 'b h t ... -> b t h ...')
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"

    o = ParallelAttentionFunction.apply(q, k, v, g, scale, cu_seqlens)
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o
