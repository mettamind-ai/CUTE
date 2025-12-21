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


import random

def generate_varlen_seqs(total_tokens, num_seqs):
    """Generate random sequence lengths that sum to total_tokens"""
    if num_seqs == 1:
        return [total_tokens]
    
    # Generate random split points
    splits = sorted(random.sample(range(1, total_tokens), num_seqs - 1))
    splits = [0] + splits + [total_tokens]
    lengths = [splits[i+1] - splits[i] for i in range(num_seqs)]
    
    # Ensure all lengths are multiples of 16 (CHUNK_LEN) for fair comparison
    # Adjust lengths to be at least 16 and multiples of 16
    adjusted = []
    remaining = total_tokens
    for i, l in enumerate(lengths[:-1]):
        adj_l = max(16, (l // 16) * 16)
        adjusted.append(adj_l)
        remaining -= adj_l
    # Last sequence gets the remainder, adjusted to multiple of 16
    last = max(16, (remaining // 16) * 16)
    adjusted.append(last)
    
    return adjusted


print(f"Config: H={H}, C={C}, warmup={warmup}, iterations={iterations}")

for total_ctx in [4096, 8192]:
    print()
    print("=" * 90)
    print(f"CONTEXT LENGTH: {total_ctx}")
    print("=" * 90)
    print()
    print(f"{'Num Seqs':<12} {'Orig ms':>10} {'Varlen ms':>10} {'Speedup':>10} {'Tokens':>10} {'Padded':>10} {'Waste%':>10}")
    print("-" * 75)
    
    for num_seqs in [5, 10, 20, 30, 40, 50]:
        random.seed(42)  # Reproducible
        seq_lengths = generate_varlen_seqs(total_ctx, num_seqs)
        
        # Ensure total matches
        actual_total = sum(seq_lengths)
        
        orig_ms, padded_tokens = benchmark_original_padded(seq_lengths, H, C)
        varlen_ms, real_tokens = benchmark_varlen(seq_lengths, H, C)
        speedup = orig_ms / varlen_ms
        waste_pct = (padded_tokens - real_tokens) / padded_tokens * 100
        
        print(f"{num_seqs:<12} {orig_ms:>10.2f} {varlen_ms:>10.2f} {speedup:>9.2f}x {real_tokens:>10} {padded_tokens:>10} {waste_pct:>9.1f}%")

print()
print("Speedup > 1 means varlen is faster")
print("Waste% = padding overhead in original kernel")
