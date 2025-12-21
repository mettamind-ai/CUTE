#!/usr/bin/env python3
'''Benchmark WinGPT vs WinRWKV speed'''
import torch, torch.nn.functional as F, time, sys

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(1981)

CTXLEN = 1024
VOCAB_SIZE = 16 * 1024
WARMUP = 2
STEPS = 5

# Model size configs: (dim, n_layers)
# dim must be divisible by 64 (HEAD_SIZE)
CONFIGS = {
    "S":   (128, 6),   # ~18-20M
    "M":   (256, 6),   # ~40M
    "L":   (384, 12),  # ~75M  
    "XL":  (512, 12),  # ~110M
    "XXL": (640, 12),  # ~150M
}

def count_params(model):
    """Count params: (emb_params, other_params, total)"""
    emb = sum(p.numel() for n, p in model.named_parameters() if 'emb' in n or (n == 'head.weight'))
    total = sum(p.numel() for p in model.parameters())
    return emb, total - emb, total

def benchmark_winrwkv(dim, n_layers):
    from winrwkv import WinRWKV, fused_loss_fn as rwkv_loss_fn
    
    model = WinRWKV(VOCAB_SIZE, n_layers, dim, CTXLEN).cuda()
    emb_params, other_params, n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Warmup
    for _ in range(WARMUP):
        input_seq = torch.randint(5, VOCAB_SIZE//4, (1, CTXLEN), dtype=torch.long).cuda()
        target = F.pad(input_seq[:, 1:], (0, 1), mode='constant', value=-100)
        optimizer.zero_grad()
        loss = rwkv_loss_fn(model, input_seq, target)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    start = time.perf_counter()
    for step in range(STEPS):
        input_seq = torch.randint(5, VOCAB_SIZE//4, (1, CTXLEN), dtype=torch.long).cuda()
        target = F.pad(input_seq[:, 1:], (0, 1), mode='constant', value=-100)
        optimizer.zero_grad()
        loss = rwkv_loss_fn(model, input_seq, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    tokens_per_sec = (STEPS * CTXLEN) / elapsed
    ms_per_step = (elapsed / STEPS) * 1000
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return emb_params, other_params, n_params, ms_per_step, tokens_per_sec, peak_mem


def benchmark_wingpt(dim, n_layers):
    from wingpt import WinGPT, fused_loss_fn as gpt_loss_fn, get_cu_max_seqlens_from
    from optimus import Muon1GPU as Muon, convert_int8_mixed_precision
    
    model = WinGPT(VOCAB_SIZE, n_layers, dim, CTXLEN, head_dim=64).cuda()
    emb_params, other_params, n_params = count_params(model)
    convert_int8_mixed_precision(model)
    
    apara = {n: p for n, p in model.named_parameters() if "proj" not in n}
    mpara = [p for n, p in model.named_parameters() if "proj" in n]
    aptim = torch.optim.AdamW(apara.values(), lr=1e-4)
    optim = Muon(mpara)
    model.train()
    
    # Warmup
    for _ in range(WARMUP):
        input_seq = torch.randint(5, VOCAB_SIZE//4, (CTXLEN,), dtype=torch.long).cuda()
        target = F.pad(input_seq[1:], (1, 0), mode='constant', value=-100)
        cu_seqlens, max_seqlen = get_cu_max_seqlens_from(input_seq, eot=0)
        optim.zero_grad(); aptim.zero_grad()
        loss = gpt_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen)
        loss.backward()
        optim.step(); aptim.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    start = time.perf_counter()
    for step in range(STEPS):
        input_seq = torch.randint(5, VOCAB_SIZE//4, (CTXLEN,), dtype=torch.long).cuda()
        target = F.pad(input_seq[1:], (1, 0), mode='constant', value=-100)
        cu_seqlens, max_seqlen = get_cu_max_seqlens_from(input_seq, eot=0)
        optim.zero_grad(); aptim.zero_grad()
        loss = gpt_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen)
        loss.backward()
        optim.step(); aptim.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    tokens_per_sec = (STEPS * CTXLEN) / elapsed
    ms_per_step = (elapsed / STEPS) * 1000
    
    del model, optim, aptim
    torch.cuda.empty_cache()
    
    return emb_params, other_params, n_params, ms_per_step, tokens_per_sec, peak_mem


if __name__ == "__main__":
    print(f"Benchmark: {STEPS} steps, {WARMUP} warmup, {CTXLEN} ctxlen, vocab {VOCAB_SIZE}\n")
    
    results = []
    
    for name, (dim, n_layers) in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {name} (dim={dim}, layers={n_layers})")
        print(f"{'='*60}")
        
        # RWKV
        print(f"\nBenchmarking WinRWKV...")
        rwkv_emb, rwkv_other, rwkv_params, rwkv_ms, rwkv_tps, rwkv_mem = benchmark_winrwkv(dim, n_layers)
        print(f"  Params: {rwkv_params/1e6:.1f}M (emb={rwkv_emb/1e6:.1f}M, other={rwkv_other/1e6:.1f}M)")
        print(f"  Speed: {rwkv_ms:.1f}ms/step, {rwkv_tps:.0f} tok/s, {rwkv_mem:.0f}MB")
        
        # GPT
        print(f"Benchmarking WinGPT...")
        gpt_emb, gpt_other, gpt_params, gpt_ms, gpt_tps, gpt_mem = benchmark_wingpt(dim, n_layers)
        print(f"  Params: {gpt_params/1e6:.1f}M (emb={gpt_emb/1e6:.1f}M, other={gpt_other/1e6:.1f}M)")
        print(f"  Speed: {gpt_ms:.1f}ms/step, {gpt_tps:.0f} tok/s, {gpt_mem:.0f}MB")
        
        speedup = gpt_ms / rwkv_ms
        print(f"  RWKV speedup: {speedup:.2f}x")
        
        results.append((name, dim, n_layers, rwkv_emb, rwkv_other, rwkv_params, rwkv_ms, rwkv_tps, 
                       gpt_emb, gpt_other, gpt_params, gpt_ms, gpt_tps, speedup))
    
    # Summary table
    print(f"\n{'='*90}")
    print("MODEL CONFIGS")
    print(f"{'='*90}")
    print(f"{'Size':<5} {'dim':>5} {'L':>3} | {'RWKV emb':>10} {'RWKV other':>12} {'RWKV total':>12} | {'GPT emb':>10} {'GPT other':>12} {'GPT total':>12}")
    print(f"{'-'*90}")
    for name, dim, layers, re, ro, rt, _, _, ge, go, gt, _, _, _ in results:
        print(f"{name:<5} {dim:>5} {layers:>3} | {re/1e6:>9.1f}M {ro/1e6:>11.1f}M {rt/1e6:>11.1f}M | {ge/1e6:>9.1f}M {go/1e6:>11.1f}M {gt/1e6:>11.1f}M")
    
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Size':<6} {'RWKV ms':>10} {'RWKV tok/s':>12} | {'GPT ms':>10} {'GPT tok/s':>12} | {'Speedup':>8}")
    print(f"{'-'*80}")
    for name, _, _, _, _, _, rms, rtps, _, _, _, gms, gtps, sp in results:
        print(f"{name:<6} {rms:>10.1f} {rtps:>12.0f} | {gms:>10.1f} {gtps:>12.0f} | {sp:>7.2f}x")
