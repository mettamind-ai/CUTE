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
# Chạy cả hai
wsl python3 benchmark.py

# Chỉ RWKV
wsl python3 benchmark.py rwkv

# Chỉ GPT
wsl python3 benchmark.py gpt
```

**Constraints**: `dim % 64 == 0` (HEAD_SIZE), `seq_len % 16 == 0` (CHUNK_LEN), `n_layers >= 2`, `dtype = bfloat16`, `vocab = 16k`

### Model Configs

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
