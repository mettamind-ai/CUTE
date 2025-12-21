#!/usr/bin/env python3
'''Benchmark WinGPT vs WinRWKV speed'''
import torch, torch.nn.functional as F, time, sys

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(1981)

CTXLEN = 1024
VOCAB_SIZE = 64 * 1024
DIM = 256
N_LAYERS = 8
WARMUP = 2
STEPS = 5

def benchmark_winrwkv():
    from winrwkv import WinRWKV, fused_loss_fn as rwkv_loss_fn
    
    print(f"\n{'='*60}")
    print(f"WinRWKV: layers={N_LAYERS}, dim={DIM}, ctxlen={CTXLEN}")
    print(f"{'='*60}")
    
    model = WinRWKV(VOCAB_SIZE, N_LAYERS, DIM, CTXLEN).cuda()
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
    
    print(f"Time: {elapsed:.2f}s ({ms_per_step:.1f}ms/step)")
    print(f"Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"Peak VRAM: {peak_mem:.0f} MB")
    print(f"Final loss: {loss.item():.4f}")
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return ms_per_step, tokens_per_sec, peak_mem


def benchmark_wingpt():
    from wingpt import WinGPT, fused_loss_fn as gpt_loss_fn, get_cu_max_seqlens_from
    from optimus import Muon1GPU as Muon
    
    print(f"\n{'='*60}")
    print(f"WinGPT: layers={N_LAYERS}, dim={DIM}, ctxlen={CTXLEN}")
    print(f"{'='*60}")
    
    model = WinGPT(VOCAB_SIZE, N_LAYERS, DIM, CTXLEN, head_dim=64).cuda()
    
    # Muon for proj layers, Adam for others
    from optimus import convert_int8_mixed_precision
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
    
    print(f"Time: {elapsed:.2f}s ({ms_per_step:.1f}ms/step)")
    print(f"Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"Peak VRAM: {peak_mem:.0f} MB")
    print(f"Final loss: {loss.item():.4f}")
    
    del model, optim, aptim
    torch.cuda.empty_cache()
    
    return ms_per_step, tokens_per_sec, peak_mem


if __name__ == "__main__":
    print(f"Benchmark config: {N_LAYERS} layers, {DIM} dim, {CTXLEN} ctxlen, {STEPS} steps")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "rwkv":
            benchmark_winrwkv()
        elif sys.argv[1] == "gpt":
            benchmark_wingpt()
    else:
        # Run both sequentially
        rwkv_ms, rwkv_tps, rwkv_mem = benchmark_winrwkv()
        gpt_ms, gpt_tps, gpt_mem = benchmark_wingpt()
        
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'ms/step':>10} {'tok/sec':>12} {'VRAM MB':>10}")
        print(f"{'-'*44}")
        print(f"{'WinRWKV':<12} {rwkv_ms:>10.1f} {rwkv_tps:>12.0f} {rwkv_mem:>10.0f}")
        print(f"{'WinGPT':<12} {gpt_ms:>10.1f} {gpt_tps:>12.0f} {gpt_mem:>10.0f}")
        print(f"{'-'*44}")
        speedup = gpt_ms / rwkv_ms if rwkv_ms > 0 else 0
        print(f"RWKV vs GPT speedup: {speedup:.2f}x")
