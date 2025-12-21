# CUTE - CUDA Transformer Experiments

Dự án thử nghiệm các kiến trúc transformer với CUDA kernels tối ưu.

## Models

### WinGPT (`wingpt.py`)
GPT-style transformer với:
- Flash Attention (varlen)
- GTA (Group Tied Attention)
- RoPE/NoPE hybrid (3 short + 1 long pattern)
- MTP (Multi-Token Prediction)
- FusedCE loss

### WinRWKV (`winrwkv.py`)
RWKV7 implementation với:
- Custom CUDA kernel (`wkv7.cu`)
- Sequential block structure (compatible với tools/rwkv7 checkpoint)
- MTP (Multi-Token Prediction)
- FusedCE loss

## Benchmark

**Note**: Chạy trong WSL/Linux, không dùng PowerShell.

```bash
python3 benchmark.py
```

**Training config**: batch=1, seq_len=4096, random tokens, 5 steps (2 warmup), AdamW + Muon optimizer

**Constraints**: `dim % 64 == 0` (HEAD_SIZE), `seq_len % 16 == 0` (CHUNK_LEN), `n_layers >= 2`, `dtype = bfloat16`

### Model Configs (vocab=4k)

| Size | dim | L  | RWKV (emb+other) | GPT (emb+other)  |
|------|-----|----|------------------|------------------|
| S    | 128 |  6 |  1M +  2M =  3M  |  1M +  3M =  4M  |
| M    | 256 |  6 |  2M +  6M =  8M  |  2M +  8M = 10M  |
| L    | 384 | 12 |  3M + 25M = 28M  |  3M + 26M = 29M  |
| XL   | 512 | 12 |  4M + 44M = 48M  |  4M + 41M = 46M  |
| XXL  | 640 | 12 |  5M + 68M = 73M  |  5M + 60M = 66M  |

### Benchmark (vocab=4k, seq=4k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | RWKV GPU% | GPT ms | GPT tok/s | GPT GPU% | Speedup |
|------|---------|------------|-----------|--------|-----------|----------|---------|
| S    | 151     | 27,133     | 94%       | 249    | 16,485    | 100%     | 1.65x   |
| M    | 198     | 20,661     | 99%       | 326    | 12,565    | 99%      | 1.64x   |
| L    | 496     | 8,264      | 100%      | 732    | 5,598     | 100%     | 1.48x   |
| XL   | 646     | 6,344      | 100%      | 918    | 4,460     | 100%     | 1.42x   |
| XXL  | 807     | 5,074      | 100%      | 1205   | 3,398     | 100%     | 1.49x   |

**RWKV nhanh hơn 1.42-1.65x** với vocab=4k. GPU utilization 94-100%.

### Benchmark (vocab=4k, seq=8k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | RWKV GPU% | GPT ms | GPT tok/s | GPT GPU% | Speedup |
|------|---------|------------|-----------|--------|-----------|----------|---------|
| S    | 286     | 28,604     | 100%      | 460    | 17,806    | 100%     | 1.61x   |
| M    | 356     | 23,012     | 100%      | 600    | 13,661    | 100%     | 1.68x   |
| L    | 880     | 9,314      | 100%      | 1421   | 5,765     | 100%     | 1.62x   |
| XL   | 1161    | 7,054      | 100%      | 1821   | 4,499     | 100%     | 1.57x   |
| XXL  | 1526    | 5,368      | 100%      | 2313   | 3,541     | 100%     | 1.52x   |

**RWKV nhanh hơn 1.52-1.68x** với vocab=4k, seq=8k. GPU utilization 100%.

### Model Configs (vocab=16k)

| Size | dim | L  | RWKV (emb+other) | GPT (emb+other)  |
|------|-----|----|------------------|------------------|
| S    | 128 |  6 |  4M +  2M =  6M  |  4M +  3M =  7M  |
| M    | 256 |  6 |  8M +  6M = 15M  |  8M +  8M = 16M  |
| L    | 384 | 12 | 13M + 25M = 38M  | 13M + 26M = 38M  |
| XL   | 512 | 12 | 17M + 44M = 61M  | 17M + 41M = 58M  |
| XXL  | 640 | 12 | 21M + 68M = 89M  | 21M + 60M = 81M  |

### Benchmark (vocab=16k, seq=4k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | RWKV GPU% | GPT ms | GPT tok/s | GPT GPU% | Speedup |
|------|---------|------------|-----------|--------|-----------|----------|---------|
| S    | 158     | 25,867     | 96%       | 293    | 14,000    | 100%     | 1.85x   |
| M    | 258     | 15,895     | 100%      | 351    | 11,666    | 100%     | 1.36x   |
| L    | 558     | 7,344      | 100%      | 808    | 5,071     | 100%     | 1.45x   |
| XL   | 758     | 5,404      | 100%      | 993    | 4,125     | 100%     | 1.31x   |
| XXL  | 896     | 4,569      | 100%      | 1221   | 3,354     | 100%     | 1.36x   |

**RWKV nhanh hơn 1.31-1.85x** với vocab=16k. GPU utilization 96-100%.

### Benchmark (vocab=16k, seq=8k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | RWKV GPU% | GPT ms | GPT tok/s | GPT GPU% | Speedup |
|------|---------|------------|-----------|--------|-----------|----------|---------|
| S    | 306     | 26,732     | 100%      | 657    | 12,466    | 100%     | 2.14x   |
| M    | 471     | 17,388     | 100%      | 765    | 10,715    | 100%     | 1.62x   |
| L    | 1141    | 7,182      | 100%      | 1812   | 4,521     | 100%     | 1.59x   |
| XL   | 1466    | 5,586      | 100%      | 2142   | 3,825     | 100%     | 1.46x   |
| XXL  | 1689    | 4,850      | 100%      | 2416   | 3,391     | 100%     | 1.43x   |

**RWKV nhanh hơn 1.43-2.14x** với vocab=16k, seq=8k. GPU utilization 100%.

### Params difference

RWKV < GPT khi dim nhỏ (S, M, L), RWKV > GPT khi dim lớn (XL, XXL).

**Nguyên nhân**: GPT dùng GTA (Group Tied Attention) với `q_proj` expand 4x dim, trong khi RWKV có nhiều learnable mixing params (`g1, g2, w1, w2, a1, a2, v1, v2`...) scale theo dim. Crossover point ~dim=450.

## Dependencies

- PyTorch 2.x với CUDA
- flash-attn (cho WinGPT)
- einops
- ninja (để compile CUDA kernels)

## Files

- `optimus.py` - Utilities: FusedCE, Muon optimizer, int8 mixed precision
- `wkv7.cu` - RWKV7 CUDA kernel
- `wkv7_varlen.cu` - RWKV7 varlen kernel (packed sequences, xem [bench_varlen.md](bench_varlen.md))
- `winrwkv_varlen.py` - WinRWKV với varlen kernel, **1.6-2x faster** end-to-end
- `tools/rwkv7/` - Reference RWKV7 implementation
- `tools/racoon/` - Alternative RWKV7 implementation
- `flash/` - Flash attention build

## Varlen Time-shift Optimization

**CRITICAL**: Với varlen (packed sequences), time-shift PHẢI vectorize:

```python
# CHẬM (~768 GPU→CPU syncs/step với 16 seqs × 6 layers):
for i in range(num_seqs):
    start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()  # .item() = SYNC!
    xx[start:end] = ...

# NHANH (0 syncs, ~0.13ms compute):
xx[1:] = x[:-1] - x[1:]  # global shift
xx[starts] = 0           # fix sequence boundaries
```

Mỗi `.item()` gây GPU→CPU sync (~10-50μs). 768 syncs = ~20-40ms overhead, gần bằng cả forward pass. Vectorize giúp tăng tốc **1.6-2x** end-to-end.
