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

Config mặc định trong `benchmark.py`:
- 8 layers, 256 dim, 1024 ctxlen
- 5 steps benchmark, 2 warmup

### Kết quả (RTX 3080)

| Size | RWKV ms | RWKV tok/s | GPT ms | GPT tok/s | Speedup |
|------|---------|------------|--------|-----------|---------|
| 60M  | 239     | 4,278      | 918    | 1,116     | 3.83x   |
| 110M | 301     | 3,405      | 481    | 2,131     | 1.60x   |
| 160M | 324     | 3,163      | 633    | 1,617     | 1.96x   |

**RWKV nhanh hơn 1.6-3.8x** tùy model size.

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
