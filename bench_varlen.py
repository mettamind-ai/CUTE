#!/usr/bin/env python3
"""
End-to-end benchmark: WinRWKV (original) vs WinRWKV (varlen) on SAME data.

Both models train on SAME total tokens (ctxlen):
- Original: batch=1, seq_len=ctxlen (single long sequence)
- Varlen: packed multiple sequences, total=ctxlen

Fair comparison - same compute, different data organization.
"""
import os, sys, time, random
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.bfloat16)

# Import models
from winrwkv import WinRWKV, fused_loss_fn, HEAD_SIZE, CHUNK_LEN
from winrwkv_varlen import WinRWKVVarlen, fused_loss_fn_varlen


def generate_packed_seqs(total_tokens, avg_seq_len=256):
    """Generate random sequence lengths that sum to total_tokens."""
    lengths = []
    remaining = total_tokens
    while remaining > 0:
        seq_len = min(remaining, max(16, int(random.gauss(avg_seq_len, avg_seq_len // 2))))
        seq_len = ((seq_len + 15) // 16) * 16
        seq_len = min(seq_len, remaining)
        if seq_len < 16:
            seq_len = remaining
        lengths.append(seq_len)
        remaining -= seq_len
    return lengths


def create_data(ctxlen, seq_lengths, vocab_size, device='cuda'):
    """Create data for both models - SAME total tokens."""
    total_tokens = sum(seq_lengths)
    assert total_tokens == ctxlen, f"Total tokens {total_tokens} != ctxlen {ctxlen}"
    
    # For original: batch=1, seq_len=ctxlen
    input_orig = torch.randint(5, vocab_size // 4, (1, ctxlen), dtype=torch.long, device=device)
    target_orig = torch.full((1, ctxlen), -100, dtype=torch.long, device=device)
    target_orig[0, :-1] = input_orig[0, 1:]
    
    # For varlen: packed sequences, total=ctxlen
    input_packed = input_orig.view(-1)  # Same tokens, just flattened
    
    # Target for varlen (shift by 1 within each sequence)
    target_packed = torch.full((ctxlen,), -100, dtype=torch.long, device=device)
    offset = 0
    for seq_len in seq_lengths:
        if seq_len > 1:
            target_packed[offset:offset + seq_len - 1] = input_packed[offset + 1:offset + seq_len]
        offset += seq_len
    
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lengths), 0)), dtype=torch.int32, device=device)
    
    return input_orig, target_orig, input_packed, target_packed, cu_seqlens


def benchmark_original(model, input_ids, target, optimizer, warmup=2, iterations=5):
    """Benchmark original WinRWKV training step."""
    model.train()
    
    for _ in range(warmup):
        optimizer.zero_grad()
        loss = fused_loss_fn(model, input_ids, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = fused_loss_fn(model, input_ids, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iterations * 1000


def benchmark_varlen(model, input_packed, target_packed, cu_seqlens, optimizer, warmup=2, iterations=5):
    """Benchmark varlen WinRWKV training step."""
    model.train()
    
    for _ in range(warmup):
        optimizer.zero_grad()
        loss = fused_loss_fn_varlen(model, input_packed, target_packed, cu_seqlens)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = fused_loss_fn_varlen(model, input_packed, target_packed, cu_seqlens)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iterations * 1000


if __name__ == "__main__":
    print("=" * 80)
    print("END-TO-END BENCHMARK: WinRWKV (original) vs WinRWKV (varlen)")
    print("Same total tokens, different data organization")
    print("=" * 80)
    print()
    
    vocab_size = 4096
    dim, n_layers = 256, 6  # Size M
    warmup, iterations = 2, 5
    
    print(f"Model: dim={dim}, layers={n_layers}, vocab={vocab_size}")
    print(f"Benchmark: warmup={warmup}, iterations={iterations}")
    print()
    
    for ctxlen in [4096, 8192]:
        print("=" * 80)
        print(f"CONTEXT LENGTH: {ctxlen} tokens")
        print("=" * 80)
        print()
        print(f"{'AvgSeqLen':<10} {'NumSeqs':>8} {'Orig ms':>10} {'Varlen ms':>11} {'Speedup':>9}")
        print("-" * 80)
        
        for avg_seq_len in [128, 256, 512, 1024, ctxlen]:
            random.seed(42)
            torch.manual_seed(42)
            
            if avg_seq_len >= ctxlen:
                seq_lengths = [ctxlen]  # Single sequence = same as original
            else:
                seq_lengths = generate_packed_seqs(ctxlen, avg_seq_len)
            num_seqs = len(seq_lengths)
            
            # Create data - same total tokens
            data = create_data(ctxlen, seq_lengths, vocab_size)
            input_orig, target_orig, input_packed, target_packed, cu_seqlens = data
            
            # Create models with same init
            torch.manual_seed(1981)
            model_orig = WinRWKV(vocab_size, n_layers, dim, ctxlen).cuda()
            opt_orig = torch.optim.AdamW(model_orig.parameters(), lr=1e-4)
            
            torch.manual_seed(1981)
            model_var = WinRWKVVarlen(vocab_size, n_layers, dim, ctxlen).cuda()
            opt_var = torch.optim.AdamW(model_var.parameters(), lr=1e-4)
            
            # Benchmark
            orig_ms = benchmark_original(model_orig, input_orig, target_orig, opt_orig, warmup, iterations)
            varlen_ms = benchmark_varlen(model_var, input_packed, target_packed, cu_seqlens, opt_var, warmup, iterations)
            speedup = orig_ms / varlen_ms
            
            label = f"{avg_seq_len}" if avg_seq_len < ctxlen else "full"
            print(f"{label:<10} {num_seqs:>8} {orig_ms:>10.1f} {varlen_ms:>11.1f} {speedup:>8.2f}x")
            
            # Cleanup
            del model_orig, model_var, opt_orig, opt_var
            torch.cuda.empty_cache()
        
        print()
    
    print("Speedup > 1 = varlen faster")
    print("'full' = single sequence (same as original, baseline)")
