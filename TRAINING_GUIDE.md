# WinRWKV Training Guide 🚀

## Quick Start

### 1. Prerequisites

Make sure you have:
- CUDA-capable GPU (RTX 3050 Ti or better recommended)
- Python 3.13+ with PyTorch 2.8.0+
- CUDA Toolkit 12.9+
- All dependencies installed (see `run.sh`)

### 2. Simple Test Run (No Data Required)

Test the model with random data:

```bash
python train_winrwkv.py --bs 1 --ctxlen 1024 --steps 10 --vocab 4096 --dim 256 --layers 6
```

This will:
- Create a small model (256 dim, 6 layers)
- Train for 10 steps with random data
- Show loss, tokens/sec, and memory usage

### 3. Training with Real Data

#### Step 1: Download Data (Optional)

If you want to use FineWeb10B data:

```bash
python data/cached_fineweb10B.py 1  # Download 1 chunk (100M tokens) for testing
# Or download full dataset:
python data/cached_fineweb10B.py 103  # Full FineWeb10B (10.3B tokens)
```

#### Step 2: Start Training

```bash
python train_winrwkv.py \
    --bs 1 \
    --ctxlen 4096 \
    --steps 1000 \
    --vocab 50256 \
    --dim 256 \
    --layers 6 \
    --lr 1e-4 \
    --data_pattern "data/fineweb-tokmon-10B/english-50256-balanced-v2/*train*.bin" \
    --save_every 100
```

---

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--bs` | 1 | Batch size (number of sequences per batch) |
| `--ctxlen` | 4096 | Context length (sequence length, must be divisible by 16) |
| `--steps` | 1000 | Number of training steps |
| `--vocab` | 4096 | Vocabulary size |
| `--dim` | 256 | Model dimension (must be divisible by 64) |
| `--layers` | 6 | Number of transformer layers (must be >= 2) |
| `--lr` | 1e-4 | Learning rate |
| `--data_pattern` | None | Data file pattern (uses random data if not provided) |
| `--use_muon` | False | Use Muon optimizer for projection layers |
| `--use_int8` | False | Use INT8 mixed precision (faster, less VRAM) |
| `--save_dir` | runs/winrwkv | Directory to save checkpoints |
| `--save_every` | 100 | Save checkpoint every N steps |

---

## Model Size Configurations

Based on the benchmark results, here are recommended configs:

### Small Model (S)
```bash
python train_winrwkv.py \
    --dim 128 --layers 6 --vocab 4096 \
    --ctxlen 4096 --bs 1
```
- **Params**: ~2M (RWKV) or ~4M (GPT)
- **VRAM**: ~500-800 MB
- **Speed**: ~40k tok/s (RTX 3050 Ti)

### Medium Model (M)
```bash
python train_winrwkv.py \
    --dim 256 --layers 6 --vocab 4096 \
    --ctxlen 4096 --bs 1
```
- **Params**: ~7M (RWKV) or ~10M (GPT)
- **VRAM**: ~1-2 GB
- **Speed**: ~35k tok/s (RTX 3050 Ti)

### Large Model (L)
```bash
python train_winrwkv.py \
    --dim 384 --layers 12 --vocab 4096 \
    --ctxlen 4096 --bs 1
```
- **Params**: ~26M (RWKV) or ~29M (GPT)
- **VRAM**: ~3-5 GB
- **Speed**: ~11k tok/s (RTX 3050 Ti)

### Extra Large Model (XL)
```bash
python train_winrwkv.py \
    --dim 512 --layers 12 --vocab 4096 \
    --ctxlen 4096 --bs 1
```
- **Params**: ~44M (RWKV) or ~46M (GPT)
- **VRAM**: ~6-10 GB
- **Speed**: ~9k tok/s (RTX 3050 Ti)

---

## Advanced Options

### Use Muon Optimizer (Faster Convergence)

```bash
python train_winrwkv.py \
    --use_muon \
    --dim 256 --layers 6
```

Muon optimizer:
- ✅ ~1.5x faster convergence
- ✅ Uses 1/4 less VRAM
- ✅ Better for projection layers

### Use INT8 Mixed Precision (Less VRAM)

```bash
python train_winrwkv.py \
    --use_int8 \
    --dim 512 --layers 12
```

INT8 mixed precision:
- ✅ ~1.5x speedup
- ✅ ~50% less VRAM
- ✅ Minimal accuracy loss

### Combine Both Optimizations

```bash
python train_winrwkv.py \
    --use_muon --use_int8 \
    --dim 512 --layers 12 \
    --ctxlen 8192
```

---

## Data Format

The training script expects data in the FineWeb format:
- Binary files with `.bin` extension
- Header: 256 int32 values
  - `header[0]` = magic number (20240520)
  - `header[1]` = version (1)
  - `header[2]` = number of tokens
- Body: uint16 token IDs

If no data pattern is provided, the script uses random data for testing.

---

## Training Tips

### 1. Start Small
Begin with a small model to verify everything works:
```bash
python train_winrwkv.py --dim 128 --layers 4 --steps 10
```

### 2. Monitor GPU Usage
Watch GPU utilization:
```bash
watch -n 1 nvidia-smi
```

### 3. Adjust Batch Size
- **Small GPU (8GB)**: `--bs 1`
- **Medium GPU (16GB)**: `--bs 2-4`
- **Large GPU (24GB+)**: `--bs 4-8`

### 4. Sequence Length
- **Shorter sequences**: Faster training, less memory
- **Longer sequences**: Better context, more memory
- Must be divisible by 16 (CHUNK_LEN)

### 5. Learning Rate
- Default: `1e-4` works well for most cases
- If loss doesn't decrease: try `5e-5` or `2e-4`
- With Muon: projection layers use `lr * 5` automatically

---

## Checkpoints

Checkpoints are saved to `runs/winrwkv/` by default (or `--save_dir`).

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Training step number
- Loss value
- Training arguments

To resume training, you'll need to modify the script to load checkpoints (feature coming soon).

---

## Troubleshooting

### Error: "ctxlen must be divisible by 16"
**Solution**: Use a sequence length that's a multiple of 16 (e.g., 1024, 2048, 4096, 8192)

### Error: "dim must be divisible by 64"
**Solution**: Use a model dimension that's a multiple of 64 (e.g., 128, 192, 256, 320, 384, 448, 512)

### Out of Memory (OOM)
**Solutions**:
1. Reduce batch size: `--bs 1`
2. Reduce context length: `--ctxlen 2048`
3. Use INT8: `--use_int8`
4. Use smaller model: `--dim 128 --layers 4`

### CUDA Kernel Compilation Takes Long Time
**Normal**: First run compiles the CUDA kernel (takes 1-5 minutes). Subsequent runs are instant.

### Loss Not Decreasing
**Solutions**:
1. Check learning rate (try `--lr 5e-5` or `--lr 2e-4`)
2. Verify data is loading correctly
3. Use `--use_muon` for better convergence
4. Train for more steps

---

## Example Training Sessions

### Quick Test (5 minutes)
```bash
python train_winrwkv.py \
    --dim 128 --layers 4 --ctxlen 1024 \
    --steps 50 --vocab 4096
```

### Small Model Training (30 minutes)
```bash
python train_winrwkv.py \
    --dim 256 --layers 6 --ctxlen 4096 \
    --steps 500 --vocab 4096 \
    --use_muon --save_every 100
```

### Production Training (Hours/Days)
```bash
python train_winrwkv.py \
    --dim 512 --layers 12 --ctxlen 8192 \
    --steps 10000 --vocab 50256 \
    --use_muon --use_int8 \
    --data_pattern "data/fineweb-tokmon-10B/english-50256-balanced-v2/*train*.bin" \
    --save_every 500
```

---

## What's Different from `pretrain.py`?

- `pretrain.py`: Trains **WinGPT** (transformer with Flash Attention)
- `train_winrwkv.py`: Trains **WinRWKV** (recurrent architecture)

Both use similar data format and optimizers, but:
- WinRWKV is **1.5-2x faster** for inference
- WinRWKV uses less memory
- WinGPT may have better long-context performance

---

## Next Steps

1. **Run a test**: `python train_winrwkv.py --steps 10`
2. **Try with data**: Download data and use `--data_pattern`
3. **Experiment**: Try different model sizes and optimizations
4. **Monitor**: Watch loss decrease and GPU utilization
5. **Scale up**: Increase model size and training steps

Happy training! 🎉
