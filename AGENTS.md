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

```bash
wsl python3 benchmark.py
```

**Training config**: batch=1, seq_len=1024, random tokens, 5 steps (2 warmup), AdamW + Muon optimizer

**Constraints**: `dim % 64 == 0` (HEAD_SIZE), `seq_len % 16 == 0` (CHUNK_LEN), `n_layers >= 2`, `dtype = bfloat16`

### Model Configs (vocab=16k)

| Size | dim | L  | RWKV (emb+other) | GPT (total)     |
|------|-----|----|------------------|-----------------|
| S    | 128 |  6 |  4M +  2M =  6M  |  4M +  3M = 7M  |
| M    | 256 |  6 |  8M +  6M = 15M  |  8M +  8M = 16M |
| L    | 384 | 12 | 13M + 25M = 38M  | 13M + 26M = 38M |
| XL   | 512 | 12 | 17M + 44M = 61M  | 17M + 41M = 58M |
| XXL  | 640 | 12 | 21M + 68M = 89M  | 21M + 60M = 81M |

### Benchmark (RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | GPT ms | GPT tok/s | Speedup |
|------|---------|------------|--------|-----------|---------|
| S    | 62      | 16,375     | 115    | 8,889     | 1.84x   |
| M    | 65      | 15,693     | 140    | 7,337     | 2.14x   |
| L    | 133     | 7,704      | 220    | 4,657     | 1.65x   |
| XL   | 157     | 6,507      | 310    | 3,298     | 1.97x   |
| XXL  | 222     | 4,614      | 448    | 2,288     | 2.02x   |

**RWKV nhanh hơn 1.65-2.14x**. Speedup tốt nhất khi dim là power of 2 (128, 256, 512).

### Model Configs (vocab=4k)

| Size | dim | L  | RWKV (emb+other) | GPT (total)    |
|------|-----|----|------------------|----------------|
| S    | 128 |  6 | 1M +  2M =  3M   | 1M +  3M =  4M |
| M    | 256 |  6 | 2M +  6M =  8M   | 2M +  8M = 10M |
| L    | 384 | 12 | 3M + 25M = 28M   | 3M + 26M = 29M |
| XL   | 512 | 12 | 4M + 44M = 48M   | 4M + 41M = 46M |
| XXL  | 640 | 12 | 5M + 68M = 73M   | 5M + 60M = 66M |

### Benchmark (vocab=4k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | GPT ms | GPT tok/s | Speedup |
|------|---------|------------|--------|-----------|---------|
| S    | 66      | 15,622     | 136    | 7,527     | 2.08x   |
| M    | 64      | 16,014     | 107    | 9,617     | 1.67x   |
| L    | 127     | 8,073      | 216    | 4,751     | 1.70x   |
| XL   | 159     | 6,449      | 294    | 3,489     | 1.85x   |
| XXL  | 211     | 4,851      | 405    | 2,530     | 1.92x   |

**RWKV nhanh hơn 1.67-2.08x** với vocab 4k.

### Model Configs (vocab=8k)

| Size | dim | L  | RWKV (emb+other) | GPT (emb+other)  |
|------|-----|----|------------------|------------------|
| S    | 128 |  6 |  2M +  2M =  4M  |  2M +  3M =  5M  |
| M    | 256 |  6 |  4M +  6M = 10M  |  4M +  8M = 12M  |
| L    | 384 | 12 |  6M + 25M = 31M  |  6M + 26M = 32M  |
| XL   | 512 | 12 |  8M + 44M = 52M  |  8M + 41M = 50M  |
| XXL  | 640 | 12 | 10M + 68M = 79M  | 10M + 60M = 71M  |

### Benchmark (vocab=8k, RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | GPT ms | GPT tok/s | Speedup |
|------|---------|------------|--------|-----------|---------|
| S    | 63      | 16,346     | 118    | 8,690     | 1.88x   |
| M    | 70      | 14,540     | 125    | 8,215     | 1.77x   |
| L    | 127     | 8,054      | 222    | 4,609     | 1.75x   |
| XL   | 160     | 6,381      | 302    | 3,386     | 1.88x   |
| XXL  | 211     | 4,862      | 402    | 2,550     | 1.91x   |

**RWKV nhanh hơn 1.75-1.91x** với vocab 8k.

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
- `tools/rwkv7/` - Reference RWKV7 implementation
- `tools/racoon/` - Alternative RWKV7 implementation
- `flash/` - Flash attention build
