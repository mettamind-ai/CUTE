#!/usr/bin/env python3
"""Profile WinRWKV Varlen to see where time is spent."""
import os, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.bfloat16)

from winrwkv_varlen import WinRWKVVarlen, fused_loss_fn_varlen

def profile_forward_backward():
    vocab_size = 4096
    dim, n_layers = 256, 6
    ctxlen = 4096
    
    torch.manual_seed(1981)
    model = WinRWKVVarlen(vocab_size, n_layers, dim, ctxlen).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    seq_lengths = [256] * 16
    total_tokens = sum(seq_lengths)
    
    input_ids = torch.randint(5, vocab_size // 4, (total_tokens,), dtype=torch.long).cuda()
    target = torch.full((total_tokens,), -100, dtype=torch.long).cuda()
    offset = 0
    for seq_len in seq_lengths:
        if seq_len > 1:
            target[offset:offset + seq_len - 1] = input_ids[offset + 1:offset + seq_len]
        offset += seq_len
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lengths), 0)), dtype=torch.int32).cuda()
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    print("=" * 70)
    print("PROFILING WinRWKV Varlen")
    print(f"Config: dim={dim}, layers={n_layers}, ctxlen={ctxlen}, num_seqs={len(seq_lengths)}")
    print("=" * 70)
    
    # Use CUDA events for manual timing
    import torch.cuda as cuda
    
    def time_fn(fn, name, iterations=10):
        # Warmup
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        
        start = cuda.Event(enable_timing=True)
        end = cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        
        return start.elapsed_time(end) / iterations
    
    # Profile individual components
    from winrwkv_varlen import varlen_timeshift, precompute_starts
    
    x = torch.randn(total_tokens, dim, device='cuda', dtype=torch.bfloat16)
    starts = precompute_starts(cu_seqlens, total_tokens)
    
    # Time-shift
    ts_time = time_fn(lambda: varlen_timeshift(x, starts), "time_shift")
    print(f"varlen_timeshift:     {ts_time:.3f} ms")
    
    # Linear projection (simulate receptance/key/value)
    weight = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
    lin_time = time_fn(lambda: x @ weight, "linear")
    print(f"linear (dim->dim):    {lin_time:.3f} ms")
    
    # 6 mixes (xr, xw, xk, xv, xa, xg)
    x_r = torch.randn(1, dim, device='cuda', dtype=torch.bfloat16)
    def six_mixes():
        xx = varlen_timeshift(x, starts)
        xr = x + xx * x_r
        xw = x + xx * x_r
        xk = x + xx * x_r
        xv = x + xx * x_r
        xa = x + xx * x_r
        xg = x + xx * x_r
        return xr, xw, xk, xv, xa, xg
    mix_time = time_fn(six_mixes, "6_mixes")
    print(f"time_shift + 6 mixes: {mix_time:.3f} ms")
    
    # Full forward pass
    def forward_only():
        with torch.no_grad():
            model(input_ids, cu_seqlens, return_logits=True)
    fwd_time = time_fn(forward_only, "forward")
    print(f"full forward:         {fwd_time:.3f} ms")
    
    # Full training step
    def train_step():
        optimizer.zero_grad()
        loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
        loss.backward()
        optimizer.step()
    train_time = time_fn(train_step, "train_step", iterations=5)
    print(f"full train step:      {train_time:.3f} ms")
    
    print()
    print("=" * 70)
    print("BREAKDOWN")
    print("=" * 70)
    
    # Estimate percentages
    # Per layer: 2x time_shift (Tmix + Cmix) + 6 mixes + projections
    ts_per_layer = ts_time * 2  # Tmix + Cmix
    mix_per_layer = mix_time - ts_time  # just the 6 mixes without time_shift
    proj_per_layer = lin_time * 4  # receptance, key, value, output (rough estimate)
    
    total_ts = ts_per_layer * n_layers
    total_mix = mix_per_layer * n_layers
    total_proj = proj_per_layer * n_layers
    
    print(f"Estimated time_shift total:  {total_ts:.2f} ms ({total_ts/train_time*100:.1f}% of train step)")
    print(f"Estimated 6-mixes total:     {total_mix:.2f} ms ({total_mix/train_time*100:.1f}% of train step)")
    print(f"Estimated projections total: {total_proj:.2f} ms ({total_proj/train_time*100:.1f}% of train step)")

if __name__ == "__main__":
    profile_forward_backward()
