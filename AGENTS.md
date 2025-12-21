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

**Training config**: packed sequences (16 seqs/batch), 5 steps (2 warmup), AdamW + Muon optimizer

**Constraints**: `dim % 64 == 0` (HEAD_SIZE), `seq_len % 16 == 0` (CHUNK_LEN), `n_layers >= 2`, `dtype = bfloat16`

### Model Configs (vocab=4k)

| Size | dim | L  | RWKV (emb+other) | GPT (emb+other)  |
|------|-----|----|------------------|------------------|
| S    | 128 |  6 |  1M +  1M =  2M  |  1M +  3M =  4M  |
| M    | 256 |  6 |  2M +  5M =  7M  |  2M +  8M = 10M  |
| L    | 384 | 12 |  3M + 23M = 26M  |  3M + 26M = 29M  |
| XL   | 512 | 12 |  4M + 40M = 44M  |  4M + 41M = 46M  |
| XXL  | 640 | 12 |  5M + 62M = 67M  |  5M + 61M = 66M  |

### Benchmark (vocab=4k, seq=4k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | RWKV GPU% | GPT ms | GPT tok/s | GPT GPU% | Speedup |
|------|---------|------------|-----------|--------|-----------|----------|---------|
| S    | 102     | 40,256     | 59%       | 181    | 22,635    | 99%      | 1.78x   |
| M    | 116     | 35,385     | 85%       | 242    | 16,921    | 98%      | 2.09x   |
| L    | 357     | 11,490     | 96%       | 584    | 7,018     | 100%     | 1.64x   |
| XL   | 466     | 8,799      | 97%       | 787    | 5,207     | 100%     | 1.69x   |
| XXL  | 636     | 6,436      | 98%       | 957    | 4,281     | 100%     | 1.50x   |

**RWKV nhanh hơn 1.50-2.09x** với vocab=4k. GPU utilization 59-100%.

### Benchmark (vocab=4k, seq=8k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | RWKV GPU% | GPT ms | GPT tok/s | GPT GPU% | Speedup |
|------|---------|------------|-----------|--------|-----------|----------|---------|
| S    | 157     | 52,049     | 64%       | 327    | 25,040    | 99%      | 2.08x   |
| M    | 193     | 42,489     | 90%       | 452    | 18,140    | 100%     | 2.34x   |
| L    | 585     | 14,005     | 97%       | 985    | 8,320     | 100%     | 1.68x   |
| XL   | 828     | 9,893      | 97%       | 1205   | 6,800     | 100%     | 1.45x   |
| XXL  | 1047    | 7,828      | 98%       | 1457   | 5,622     | 100%     | 1.39x   |

**RWKV nhanh hơn 1.39-2.34x** với vocab=4k, seq=8k. GPU utilization 64-100%.

### Model Configs (vocab=16k)

| Size | dim | L  | RWKV (emb+other) | GPT (emb+other)  |
|------|-----|----|------------------|------------------|
| S    | 128 |  6 |  4M +  1M =  6M  |  4M +  3M =  7M  |
| M    | 256 |  6 |  8M +  5M = 14M  |  8M +  8M = 16M  |
| L    | 384 | 12 | 13M + 23M = 35M  | 13M + 26M = 38M  |
| XL   | 512 | 12 | 17M + 40M = 57M  | 17M + 41M = 58M  |
| XXL  | 640 | 12 | 21M + 62M = 83M  | 21M + 61M = 81M  |

### Benchmark (vocab=16k, seq=4k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | RWKV GPU% | GPT ms | GPT tok/s | GPT GPU% | Speedup |
|------|---------|------------|-----------|--------|-----------|----------|---------|
| S    | 110     | 37,389     | 58%       | 242    | 16,914    | 79%      | 2.21x   |
| M    | 131     | 31,226     | 71%       | 340    | 12,044    | 98%      | 2.59x   |
| L    | 357     | 11,491     | 92%       | 548    | 7,474     | 100%     | 1.54x   |
| XL   | 450     | 9,112      | 95%       | 732    | 5,599     | 100%     | 1.63x   |
| XXL  | 547     | 7,492      | 97%       | 890    | 4,603     | 100%     | 1.63x   |

**RWKV nhanh hơn 1.54-2.59x** với vocab=16k. GPU utilization 58-100%.

### Benchmark (vocab=16k, seq=8k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | RWKV GPU% | GPT ms | GPT tok/s | GPT GPU% | Speedup |
|------|---------|------------|-----------|--------|-----------|----------|---------|
| S    | 375     | 21,868     | 100%      | 868    | 9,435     | 100%     | 2.32x   |
| M    | 259     | 31,669     | 91%       | 503    | 16,299    | 100%     | 1.94x   |
| L    | 622     | 13,162     | 95%       | 1058   | 7,747     | 100%     | 1.70x   |
| XL   | 830     | 9,870      | 97%       | 1288   | 6,361     | 100%     | 1.55x   |
| XXL  | 1068    | 7,673      | 98%       | 1586   | 5,166     | 100%     | 1.49x   |

**RWKV nhanh hơn 1.49-2.32x** với vocab=16k, seq=8k. GPU utilization 91-100%.

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
