#!/usr/bin/env python3
'''Benchmark WinGPT vs WinRWKV speed'''
import torch, torch.nn.functional as F, time, sys, subprocess, threading

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(1981)

CTXLEN = 8192
VOCAB_SIZE = 16 * 1024
WARMUP = 2
STEPS = 5
NUM_SEQS = 16  # Number of packed sequences per batch

# Model size configs: (dim, n_layers)
# dim must be divisible by 64 (HEAD_SIZE)
CONFIGS = {
    "S":   (128, 6),   # ~18-20M
    "M":   (256, 6),   # ~40M
    "L":   (384, 12),  # ~75M  
    "XL":  (512, 12),  # ~110M
    "XXL": (640, 12),  # ~150M
}


def generate_packed_data(total_len, num_seqs, vocab_size, device='cuda'):
    """Generate packed sequences with random lengths (aligned to 16).
    
    Returns:
        input_ids: (total_len,) packed tokens
        target: (total_len,) targets with -100 at sequence boundaries
        cu_seqlens: (num_seqs+1,) cumulative sequence lengths
        max_seqlen: maximum sequence length
    """
    # Generate random lengths (aligned to 16, min 16)
    min_len = 16
    avg_len = total_len // num_seqs
    lengths = []
    remaining = total_len
    
    for i in range(num_seqs - 1):
        # Random length around average, aligned to 16
        max_possible = remaining - (num_seqs - i - 1) * min_len
        length = min(max_possible, max(min_len, ((avg_len + torch.randint(-avg_len//2, avg_len//2, (1,)).item()) // 16) * 16))
        lengths.append(length)
        remaining -= length
    lengths.append(remaining)  # Last sequence gets the rest
    
    # Ensure all lengths are valid
    lengths = [max(min_len, (l // 16) * 16) for l in lengths]
    
    # Adjust to match total_len exactly
    diff = total_len - sum(lengths)
    if diff != 0:
        lengths[-1] += diff
    
    # Generate cu_seqlens
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    max_seqlen = max(lengths)
    
    # Generate tokens (avoid 0 which is EOT)
    input_ids = torch.randint(5, vocab_size // 4, (total_len,), dtype=torch.long, device=device)
    
    # Generate targets: shift by 1, -100 at sequence boundaries
    target = torch.full((total_len,), -100, dtype=torch.long, device=device)
    for i in range(num_seqs):
        start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
        if end - start > 1:
            target[start:end-1] = input_ids[start+1:end]
    
    return input_ids, target, cu_seqlens, max_seqlen

class GPUMonitor:
    def __init__(self):
        self.samples = []
        self.running = False
        self.thread = None
    
    def start(self):
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._sample)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        return max(self.samples) if self.samples else 0
    
    def _sample(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                self.samples.append(int(out))
            except:
                pass
            time.sleep(0.1)

def count_params(model):
    """Count params: (emb_params, other_params, total)"""
    emb = sum(p.numel() for n, p in model.named_parameters() if 'emb' in n or (n == 'head.weight'))
    total = sum(p.numel() for p in model.parameters())
    return emb, total - emb, total

def benchmark_winrwkv(dim, n_layers, gpu_mon):
    from winrwkv_varlen import WinRWKVVarlen, fused_loss_fn_varlen
    from optimus import convert_int8_ffn_only
    
    model = WinRWKVVarlen(VOCAB_SIZE, n_layers, dim, CTXLEN).cuda()
    convert_int8_ffn_only(model)
    emb_params, other_params, n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Warmup - packed sequences
    for _ in range(WARMUP):
        input_ids, target, cu_seqlens, _ = generate_packed_data(CTXLEN, NUM_SEQS, VOCAB_SIZE)
        optimizer.zero_grad()
        loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_mon.start()
    
    # Benchmark
    start = time.perf_counter()
    for step in range(STEPS):
        input_ids, target, cu_seqlens, _ = generate_packed_data(CTXLEN, NUM_SEQS, VOCAB_SIZE)
        optimizer.zero_grad()
        loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gpu_util = gpu_mon.stop()
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    tokens_per_sec = (STEPS * CTXLEN) / elapsed
    ms_per_step = (elapsed / STEPS) * 1000
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return emb_params, other_params, n_params, ms_per_step, tokens_per_sec, peak_mem, gpu_util


def benchmark_wingpt(dim, n_layers, gpu_mon):
    from wingpt import WinGPT, fused_loss_fn as gpt_loss_fn
    from optimus import convert_int8_mixed_precision
    
    model = WinGPT(VOCAB_SIZE, n_layers, dim, CTXLEN, head_dim=64).cuda()
    emb_params, other_params, n_params = count_params(model)
    convert_int8_mixed_precision(model)
    
    apara = {n: p for n, p in model.named_parameters() if "proj" not in n}
    mpara = [p for n, p in model.named_parameters() if "proj" in n]
    aptim = torch.optim.AdamW(apara.values(), lr=1e-4)
    optim = torch.optim.Muon(mpara, nesterov=False)
    model.train()
    
    # Warmup - packed sequences (same data format as RWKV)
    for _ in range(WARMUP):
        input_ids, target, cu_seqlens, max_seqlen = generate_packed_data(CTXLEN, NUM_SEQS, VOCAB_SIZE)
        optim.zero_grad(); aptim.zero_grad()
        loss = gpt_loss_fn(model, input_ids, target, cu_seqlens, max_seqlen)
        loss.backward()
        optim.step(); aptim.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_mon.start()
    
    # Benchmark
    start = time.perf_counter()
    for step in range(STEPS):
        input_ids, target, cu_seqlens, max_seqlen = generate_packed_data(CTXLEN, NUM_SEQS, VOCAB_SIZE)
        optim.zero_grad(); aptim.zero_grad()
        loss = gpt_loss_fn(model, input_ids, target, cu_seqlens, max_seqlen)
        loss.backward()
        optim.step(); aptim.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gpu_util = gpu_mon.stop()
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    tokens_per_sec = (STEPS * CTXLEN) / elapsed
    ms_per_step = (elapsed / STEPS) * 1000
    
    del model, optim, aptim
    torch.cuda.empty_cache()
    
    return emb_params, other_params, n_params, ms_per_step, tokens_per_sec, peak_mem, gpu_util


def benchmark_rwkv_original(dim, n_layers, gpu_mon):
    """Benchmark original WinRWKV (single sequence, no varlen)"""
    from winrwkv import WinRWKV, fused_loss_fn
    from optimus import convert_int8_ffn_only
    
    model = WinRWKV(VOCAB_SIZE, n_layers, dim, CTXLEN).cuda()
    convert_int8_ffn_only(model)
    emb_params, other_params, n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Warmup - single sequence (batch=1)
    for _ in range(WARMUP):
        input_ids = torch.randint(5, VOCAB_SIZE//4, (1, CTXLEN), dtype=torch.long).cuda()
        target = torch.full((1, CTXLEN), -100, dtype=torch.long).cuda()
        target[:, :-1] = input_ids[:, 1:]
        optimizer.zero_grad()
        loss = fused_loss_fn(model, input_ids, target)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_mon.start()
    
    # Benchmark
    start = time.perf_counter()
    for step in range(STEPS):
        input_ids = torch.randint(5, VOCAB_SIZE//4, (1, CTXLEN), dtype=torch.long).cuda()
        target = torch.full((1, CTXLEN), -100, dtype=torch.long).cuda()
        target[:, :-1] = input_ids[:, 1:]
        optimizer.zero_grad()
        loss = fused_loss_fn(model, input_ids, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gpu_util = gpu_mon.stop()
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    tokens_per_sec = (STEPS * CTXLEN) / elapsed
    ms_per_step = (elapsed / STEPS) * 1000
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return emb_params, other_params, n_params, ms_per_step, tokens_per_sec, peak_mem, gpu_util


def benchmark_rwkv_varlen_single(dim, n_layers, gpu_mon):
    """Benchmark WinRWKV varlen with single sequence (fair comparison)"""
    from winrwkv_varlen import WinRWKVVarlen, fused_loss_fn_varlen
    from optimus import convert_int8_ffn_only
    
    model = WinRWKVVarlen(VOCAB_SIZE, n_layers, dim, CTXLEN).cuda()
    convert_int8_ffn_only(model)
    emb_params, other_params, n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Warmup - single sequence (varlen format)
    for _ in range(WARMUP):
        input_ids = torch.randint(5, VOCAB_SIZE//4, (CTXLEN,), dtype=torch.long).cuda()
        target = torch.full((CTXLEN,), -100, dtype=torch.long).cuda()
        target[:-1] = input_ids[1:]
        cu_seqlens = torch.tensor([0, CTXLEN], dtype=torch.int32).cuda()
        optimizer.zero_grad()
        loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_mon.start()
    
    # Benchmark
    start = time.perf_counter()
    for step in range(STEPS):
        input_ids = torch.randint(5, VOCAB_SIZE//4, (CTXLEN,), dtype=torch.long).cuda()
        target = torch.full((CTXLEN,), -100, dtype=torch.long).cuda()
        target[:-1] = input_ids[1:]
        cu_seqlens = torch.tensor([0, CTXLEN], dtype=torch.int32).cuda()
        optimizer.zero_grad()
        loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gpu_util = gpu_mon.stop()
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    tokens_per_sec = (STEPS * CTXLEN) / elapsed
    ms_per_step = (elapsed / STEPS) * 1000
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return emb_params, other_params, n_params, ms_per_step, tokens_per_sec, peak_mem, gpu_util


def compare_rwkv_kernels():
    """Compare RWKV original vs varlen kernel performance"""
    print(f"\n{'='*70}")
    print("RWKV KERNEL COMPARISON: Original vs Varlen (single sequence)")
    print(f"Config: {STEPS} steps, {WARMUP} warmup, {CTXLEN} ctxlen, vocab {VOCAB_SIZE}")
    print(f"{'='*70}\n")
    
    gpu_mon = GPUMonitor()
    results = []
    
    for name, (dim, n_layers) in CONFIGS.items():
        print(f"Config {name} (dim={dim}, layers={n_layers})...")
        
        # Original kernel
        _, _, _, orig_ms, orig_tps, orig_mem, orig_gpu = benchmark_rwkv_original(dim, n_layers, gpu_mon)
        
        # Varlen kernel (single seq)
        _, _, _, var_ms, var_tps, var_mem, var_gpu = benchmark_rwkv_varlen_single(dim, n_layers, gpu_mon)
        
        speedup = orig_ms / var_ms
        print(f"  Original: {orig_ms:.1f}ms, Varlen: {var_ms:.1f}ms, Speedup: {speedup:.2f}x")
        
        results.append((name, dim, n_layers, orig_ms, orig_tps, orig_gpu, var_ms, var_tps, var_gpu, speedup))
    
    # Summary
    print(f"\n{'='*90}")
    print("RWKV KERNEL COMPARISON RESULTS")
    print(f"{'='*90}")
    print(f"{'Size':<6} {'Orig ms':>10} {'Orig tok/s':>12} {'GPU%':>6} | {'Varlen ms':>10} {'Varlen tok/s':>12} {'GPU%':>6} | {'Speedup':>8}")
    print(f"{'-'*90}")
    for name, _, _, oms, otps, ogpu, vms, vtps, vgpu, sp in results:
        print(f"{name:<6} {oms:>10.1f} {otps:>12.0f} {ogpu:>5}% | {vms:>10.1f} {vtps:>12.0f} {vgpu:>5}% | {sp:>7.2f}x")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Compare RWKV original vs varlen kernel
        compare_rwkv_kernels()
    else:
        # Default: benchmark WinGPT vs WinRWKV varlen
        print(f"Benchmark: {STEPS} steps, {WARMUP} warmup, {CTXLEN} ctxlen, vocab {VOCAB_SIZE}, {NUM_SEQS} seqs/batch\n")
    
    gpu_mon = GPUMonitor()
    results = []
    
    for name, (dim, n_layers) in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {name} (dim={dim}, layers={n_layers})")
        print(f"{'='*60}")
        
        # RWKV
        print(f"\nBenchmarking WinRWKV...")
        rwkv_emb, rwkv_other, rwkv_params, rwkv_ms, rwkv_tps, rwkv_mem, rwkv_gpu = benchmark_winrwkv(dim, n_layers, gpu_mon)
        print(f"  Params: {rwkv_params/1e6:.1f}M (emb={rwkv_emb/1e6:.1f}M, other={rwkv_other/1e6:.1f}M)")
        print(f"  Speed: {rwkv_ms:.1f}ms/step, {rwkv_tps:.0f} tok/s, {rwkv_mem:.0f}MB, GPU {rwkv_gpu}%")
        
        # GPT
        print(f"Benchmarking WinGPT...")
        gpt_emb, gpt_other, gpt_params, gpt_ms, gpt_tps, gpt_mem, gpt_gpu = benchmark_wingpt(dim, n_layers, gpu_mon)
        print(f"  Params: {gpt_params/1e6:.1f}M (emb={gpt_emb/1e6:.1f}M, other={gpt_other/1e6:.1f}M)")
        print(f"  Speed: {gpt_ms:.1f}ms/step, {gpt_tps:.0f} tok/s, {gpt_mem:.0f}MB, GPU {gpt_gpu}%")
        
        speedup = gpt_ms / rwkv_ms
        print(f"  RWKV speedup: {speedup:.2f}x")
        
        results.append((name, dim, n_layers, rwkv_emb, rwkv_other, rwkv_params, rwkv_ms, rwkv_tps, rwkv_gpu,
                       gpt_emb, gpt_other, gpt_params, gpt_ms, gpt_tps, gpt_gpu, speedup))
    
    # Summary table
    print(f"\n{'='*90}")
    print("MODEL CONFIGS")
    print(f"{'='*90}")
    print(f"{'Size':<5} {'dim':>5} {'L':>3} | {'RWKV emb':>10} {'RWKV other':>12} {'RWKV total':>12} | {'GPT emb':>10} {'GPT other':>12} {'GPT total':>12}")
    print(f"{'-'*90}")
    for name, dim, layers, re, ro, rt, _, _, _, ge, go, gt, _, _, _, _ in results:
        print(f"{name:<5} {dim:>5} {layers:>3} | {re/1e6:>9.1f}M {ro/1e6:>11.1f}M {rt/1e6:>11.1f}M | {ge/1e6:>9.1f}M {go/1e6:>11.1f}M {gt/1e6:>11.1f}M")
    
    print(f"\n{'='*90}")
    print("BENCHMARK RESULTS")
    print(f"{'='*90}")
    print(f"{'Size':<6} {'RWKV ms':>10} {'RWKV tok/s':>12} {'GPU%':>6} | {'GPT ms':>10} {'GPT tok/s':>12} {'GPU%':>6} | {'Speedup':>8}")
    print(f"{'-'*90}")
    for name, _, _, _, _, _, rms, rtps, rgpu, _, _, _, gms, gtps, ggpu, sp in results:
        print(f"{name:<6} {rms:>10.1f} {rtps:>12.0f} {rgpu:>5}% | {gms:>10.1f} {gtps:>12.0f} {ggpu:>5}% | {sp:>7.2f}x")
