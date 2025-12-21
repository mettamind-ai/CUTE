#!/usr/bin/env python3
"""
Performance benchmark: Original kernel (with padding) vs Varlen kernel (packed)
"""
import torch
import time

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(42)

HEAD_SIZE = 64
CHUNK_LEN = 16

from torch.utils.cpp_extension import load

FLAGS = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}",
    "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]

print("Compiling kernels...")
load(name="wind_backstepping", 
     sources=["wkv7.cu"], 
     is_python_module=False, verbose=False, extra_cuda_cflags=FLAGS)
load(name="wind_backstepping_varlen", 
     sources=["wkv7_varlen.cu"], 
     is_python_module=False, verbose=False, extra_cuda_cflags=FLAGS)

print()
print("=" * 70)
print("PERFORMANCE BENCHMARK: Original (padded) vs Varlen (packed)")
print("=" * 70)
print()

H = 4  # num heads
C = HEAD_SIZE
scale = 0.1
warmup = 3
iterations = 10


def benchmark_original_padded(seq_lengths, H, C):
    """Run original kernel with padding to max_len (simulates batch processing)"""
    max_len = max(seq_lengths)
    padded_len = ((max_len + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
    B = len(seq_lengths)
    
    w = (torch.randn(B, padded_len, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    q = (torch.randn(B, padded_len, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    k = (torch.randn(B, padded_len, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    v = (torch.randn(B, padded_len, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    a = (torch.randn(B, padded_len, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    b = (torch.randn(B, padded_len, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    
    y = torch.empty_like(v)
    s = torch.empty(B, H, padded_len // CHUNK_LEN, C, C, dtype=torch.float32, device="cuda")
    sa = torch.empty(B, padded_len, H, C, dtype=torch.float32, device="cuda")
    
    dy = torch.ones_like(y)
    dw, dq, dk, dv, da, db = [torch.empty_like(x) for x in [w, q, k, v, a, b]]
    
    for _ in range(warmup):
        torch.ops.wind_backstepping.forward(w, q, k, v, a, b, y, s, sa)
        torch.ops.wind_backstepping.backward(w, q, k, v, a, b, dy, s, sa, dw, dq, dk, dv, da, db)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        torch.ops.wind_backstepping.forward(w, q, k, v, a, b, y, s, sa)
        torch.ops.wind_backstepping.backward(w, q, k, v, a, b, dy, s, sa, dw, dq, dk, dv, da, db)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000
    
    return elapsed, B * padded_len


def benchmark_varlen(seq_lengths, H, C):
    """Run varlen kernel with packed sequences"""
    total_tokens = sum(seq_lengths)
    num_seqs = len(seq_lengths)
    num_chunks = (total_tokens + CHUNK_LEN - 1) // CHUNK_LEN
    
    w = (torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    q = (torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    k = (torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    v = (torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    a = (torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    b = (torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device="cuda") * scale).detach()
    
    cu_seqlens = [0]
    for length in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    
    y = torch.empty_like(v)
    s_chunk = torch.empty(H, num_chunks, C, C, dtype=torch.float32, device="cuda")
    sa = torch.empty(total_tokens, H, C, dtype=torch.float32, device="cuda")
    
    dy = torch.ones_like(y)
    dw, dq, dk, dv, da, db = [torch.empty_like(x) for x in [w, q, k, v, a, b]]
    
    for _ in range(warmup):
        torch.ops.wind_backstepping_varlen.forward_varlen(w, q, k, v, a, b, cu_seqlens, y, s_chunk, sa)
        torch.ops.wind_backstepping_varlen.backward_varlen(w, q, k, v, a, b, dy, cu_seqlens, s_chunk, sa, dw, dq, dk, dv, da, db)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        torch.ops.wind_backstepping_varlen.forward_varlen(w, q, k, v, a, b, cu_seqlens, y, s_chunk, sa)
        torch.ops.wind_backstepping_varlen.backward_varlen(w, q, k, v, a, b, dy, cu_seqlens, s_chunk, sa, dw, dq, dk, dv, da, db)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000
    
    return elapsed, total_tokens


test_cases = [
    ([512] * 8, "8 x 512 (uniform)"),
    ([256, 512, 128, 640, 384, 256, 512, 384], "8 seqs (varied 128-640)"),
    ([1024, 256, 64, 512, 128, 1024, 64, 256], "8 seqs (varied 64-1024)"),
    ([64] * 64, "64 x 64 (many short)"),
    ([32] * 128, "128 x 32 (many very short)"),
    ([2048, 512, 256, 128], "4 seqs (2048, 512, 256, 128)"),
    ([4096], "1 x 4096 (single long)"),
]

print(f"Config: H={H}, C={C}, warmup={warmup}, iterations={iterations}")
print()
print(f"{'Test Case':<35} {'Orig ms':>10} {'Varlen ms':>10} {'Speedup':>10} {'Tokens':>10} {'Padded':>10}")
print("-" * 90)

for seq_lengths, description in test_cases:
    orig_ms, padded_tokens = benchmark_original_padded(seq_lengths, H, C)
    varlen_ms, real_tokens = benchmark_varlen(seq_lengths, H, C)
    speedup = orig_ms / varlen_ms
    
    print(f"{description:<35} {orig_ms:>10.2f} {varlen_ms:>10.2f} {speedup:>9.2f}x {real_tokens:>10} {padded_tokens:>10}")

print()
print("Speedup > 1 means varlen is faster")
print("Padded = tokens processed by original (with padding waste)")
