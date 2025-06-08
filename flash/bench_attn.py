#!/usr/bin/env python3
import triton, torch, torch.nn.functional as F

from sageattn_triton import sageattn_varlen
from linear_attn.parallel_nsa import parallel_nsa

from infllmv2 import infllmv2_sparse_attn_func, generate_topk_indices

try: from flash_attn_interface import flash_attn_varlen_func; FA_ENABLED = 3
except: from attn import flash_attn_varlen_func; FA_ENABLED = 2

if __name__ == "__main__":

    lines = "flash_attn_varlen parallel_nsa infllmv2_varlen sageattn_varlen".split()
    BATCH, N_HEADS, HQ, HEAD_DIM = 4, 64, 4, 128
    assert N_HEADS // HQ == 16 # cần để infllmv2_sparse_attn chạy

    config = triton.testing.Benchmark(
        line_vals=lines, line_names=lines,
        line_arg="provider", x_names=["N_CTX"], ylabel="ms", 
        x_vals=[2**i for i in range(13, 16)], # 8192 16384 32k   
        plot_name=f"attn-bs{BATCH}-h{N_HEADS}-d{HEAD_DIM}",
        args=dict(H=N_HEADS, HQ=HQ, BATCH=BATCH, HEAD_DIM=HEAD_DIM),
    )

    @triton.testing.perf_report([config])
    def bench_flash_attention(BATCH, H, HQ, N_CTX, HEAD_DIM, provider, device="cuda"):
        dtype = torch.bfloat16
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
        k = torch.randn((BATCH, HQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
        v = torch.randn((BATCH, HQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)

        block_size, S = 64, 16
        indices = torch.full((BATCH, HQ, N_CTX, S), N_CTX, dtype=torch.long, device=device)
        for b in range(BATCH):
            for h in range(HQ):
                for t in range(N_CTX):
                    i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                    indices[b, h, t, :len(i_i)] = i_i
        indices = indices.sort(-1)[0]

        max_seqlen, seq_len = N_CTX//2, BATCH*N_CTX
        cu_seqlens = [i for i in range(0, BATCH*N_CTX + max_seqlen, max_seqlen)]
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
        qq = q.transpose(1, 2).reshape(seq_len, H, HEAD_DIM) # seq_len, H, D
        kk = k.transpose(1, 2).reshape(seq_len, HQ, HEAD_DIM) # seq_len, H, D
        vv = v.transpose(1, 2).reshape(seq_len, HQ, HEAD_DIM) # seq_len, H, D

        def attn_fn(provider, q, k, v):
            if provider == "parallel_nsa":
                return lambda: parallel_nsa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_indices=indices, block_size=block_size)

            if provider == "flash_attn_varlen":
                return lambda: flash_attn_varlen_func(qq, kk, vv, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)

            if provider == "infllmv2_varlen":
                from einops import rearrange, repeat
                sparsity=0.8; block_size=64; block_window_size=3
                topk_idx = generate_topk_indices(HQ, qq.shape[0], max_seqlen, sparsity, block_size, "cuda")
                return lambda: infllmv2_sparse_attn_func(qq, kk, vv, cu_seqlens, cu_seqlens, topk_idx, max_seqlen, max_seqlen, block_window_size)

            if provider == "sageattn_varlen":
                return lambda: sageattn_varlen(qq, kk, vv, cu_seqlens, max_seqlen, sm_scale=1.3)

            if provider == "pytorch":
                return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.3)

        ms = triton.testing.do_bench(attn_fn(provider, q, k, v), warmup=15, rep=50)

        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        total_flops *= 0.5
        return total_flops / ms * 1e-9


    def assert_sage_attn_is_same_as_sdpa():
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
        o_sage = sageattn_varlen(q, k, v, c, m, sm_scale=1.3)
        o_sage = o_sage.reshape(4, 1024, 32, 64).transpose(1, 2)
        assert o_sage.shape == (4, 32, 1024, 64)

        if flash_attn_varlen_func:
            o_flash = flash_attn_varlen_func(q, k, v, c, c, m, m, causal=True, softmax_scale=1.3)
            o_flash = o_flash.reshape(4, 1024, 32, 64).transpose(1, 2)
            assert o_flash.shape  == (4, 32, 1024, 64)
            x, y = o_torch[0][0], o_flash[0][0]
            if torch.allclose(x, y, rtol=0.25*1e-2, atol=0.3*1e-1): print("torch ~= flash")
            else: print(f"torch ~= flash AssertionError")

        x, y = o_sage[0][0], o_torch[0][0]
        if torch.allclose(x, y, rtol=0.25*1e-1, atol=0.3*1e-1): print("torch ~= sage")
        else: print(f"torch ~= sage AssertionError")

    assert_sage_attn_is_same_as_sdpa()
    bench_flash_attention.run(save_path=None, print_data=True)
    print("total_flops, higher is better.")
