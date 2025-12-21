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

**Constraints**: `dim % 64 == 0` (HEAD_SIZE), `seq_len % 16 == 0` (CHUNK_LEN), `n_layers >= 2`, `dtype = bfloat16`

### Model Configs

| Name | dim | layers | RWKV params | GPT params |
|------|-----|--------|-------------|------------|
| S    | 128 | 6      | 18M         | 20M        |
| M    | 256 | 6      | 40M         | 41M        |
| L    | 384 | 12     | 75M         | 76M        |
| XL   | 512 | 12     | 111M        | 109M       |
| XXL  | 640 | 12     | 152M        | 144M       |

### Benchmark (RTX 3050 Ti)

| Size | RWKV ms | RWKV tok/s | GPT ms | GPT tok/s | Speedup |
|------|---------|------------|--------|-----------|---------|
| S    | 105     | 9,739      | 150    | 6,811     | 1.43x   |
| M    | 114     | 8,957      | 151    | 6,792     | 1.32x   |
| L    | 194     | 5,286      | 298    | 3,440     | 1.54x   |
| XL   | 242     | 4,228      | 409    | 2,502     | 1.69x   |
| XXL  | 315     | 3,251      | 577    | 1,773     | 1.83x   |

**RWKV nhanh hơn 1.32-1.83x**. Speedup tốt nhất khi dim là power of 2 (128, 256, 512).

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
