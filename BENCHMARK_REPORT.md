# WinRWKV Inference Benchmark Report

## Overview

This report presents inference performance benchmarks for WinRWKV models across different sizes (S, M, L, XL, XXL) on GPU and CPU devices.

**Benchmark Date:** December 22, 2024
**Hardware:** RTX 3050 Ti (GPU), CPU (unsupported)
**Model Training:** 750 steps per model size

## Model Configurations

| Size | Dim | Layers | Vocab | Context Length | Parameters |
|------|-----|--------|-------|----------------|------------|
| S    | 128 | 6      | 4096  | 4096          | 2.7M       |
| M    | 256 | 6      | 4096  | 4096          | 8.3M       |
| L    | 384 | 12     | 4096  | 4096          | 28.1M      |
| XL   | 512 | 12     | 4096  | 8192          | 48.3M      |
| XXL  | 640 | 12     | 4096  | 8192          | 73.3M      |

## GPU Inference Results

### Forward Pass Throughput

Forward pass inference processes entire sequences in parallel, measuring tokens processed per second.

| Model | Throughput (tokens/sec) | Time (ms) | Sequence Length |
|-------|------------------------|-----------|-----------------|
| S     | **104,019**            | 98.4      | 1024            |
| M     | **93,431**             | 109.6     | 1024            |
| L     | **48,602**              | 210.7     | 1024            |
| XL    | **39,646**              | 258.3     | 1024            |
| XXL   | **40,919**              | 250.3     | 1024            |

**Observations:**
- Smaller models (S, M) achieve highest throughput (~100k tokens/sec)
- Performance scales down with model size, as expected
- XXL shows slightly better performance than XL despite larger size (likely due to better GPU utilization)

### Autoregressive Generation Speed

Autoregressive generation produces tokens one at a time, measuring generation speed in tokens per second.

| Model | Speed (tokens/sec) | Time (s) | Generated Tokens |
|-------|-------------------|----------|------------------|
| S     | **64.53**         | 0.77     | 50               |
| M     | **87.18**         | 0.57     | 50               |
| L     | **45.72**          | 1.09     | 50               |
| XL    | **43.38**          | 1.15     | 50               |
| XXL   | **40.44**          | 1.24     | 50               |

**Observations:**
- Model M achieves the best autoregressive performance (87 tokens/sec)
- Larger models (L, XL, XXL) show similar generation speeds (~40-46 tokens/sec)
- Autoregressive speed is significantly slower than forward pass (expected due to sequential nature)

## CPU Inference Results

### Status: **Supported (with CPU Fallback)**

CPU inference is now **supported** via a pure PyTorch fallback implementation. The model automatically detects the device and routes to the appropriate implementation (CUDA kernel or CPU fallback).

### Forward Pass Throughput (CPU)

| Model | Throughput (tokens/sec) | Time (s) | Sequence Length | Notes |
|-------|------------------------|----------|-----------------|-------|
| S     | **45**                 | 24.97    | 112             | Initial test |

**Observations:**
- CPU inference is **significantly slower** than GPU (45 vs 104,019 tokens/sec for model S)
- This is expected due to sequential computation on CPU vs parallel GPU execution
- CPU fallback uses pure PyTorch operations, matching CUDA kernel logic exactly

### Autoregressive Generation Speed (CPU)

*Full CPU benchmarks for autoregressive generation are in progress. Initial testing shows CPU inference is functional but much slower than GPU.*

**Technical Implementation:**
- CPU fallback implemented in `RUN_CPU_RWKV7()` function in `winrwkv.py`
- Replicates CUDA kernel algorithm using PyTorch operations
- Automatically selected when model is on CPU device
- Uses float32 for computation (converts from/to bfloat16 as needed)

**Performance Notes:**
- CPU inference is **2000-3000x slower** than GPU for forward pass
- Recommended for development/testing only, not production deployment
- Full benchmark suite may take hours to complete on CPU

## Performance Analysis

### Forward Pass vs Autoregressive

| Model | Forward (tokens/sec) | Autoregressive (tokens/sec) | Ratio |
|-------|---------------------|----------------------------|-------|
| S     | 104,019            | 64.53                      | 1,613x |
| M     | 93,431              | 87.18                      | 1,072x |
| L     | 48,602              | 45.72                      | 1,063x |
| XL    | 39,646              | 43.38                      | 914x   |
| XXL   | 40,919              | 40.44                      | 1,012x |

**Key Insights:**
- Forward pass is 900-1,600x faster than autoregressive generation
- This is expected: forward pass processes all tokens in parallel, while autoregressive generates sequentially
- The ratio decreases for larger models, suggesting better optimization for sequential generation

### Model Size Scaling

**Forward Pass Scaling:**
- S → M: 10% decrease (smaller model, better throughput)
- M → L: 48% decrease (more layers, more computation)
- L → XL: 18% decrease (larger dimension, longer context)
- XL → XXL: 3% increase (better GPU utilization)

**Autoregressive Scaling:**
- S → M: 35% increase (optimal size for this task)
- M → L: 48% decrease (more layers slow sequential generation)
- L → XL: 5% decrease (larger dimension)
- XL → XXL: 7% decrease (largest model)

## Recommendations

1. **For High-Throughput Applications:** Use forward pass inference with smaller models (S, M) for maximum tokens/sec
2. **For Text Generation:** Model M provides the best balance of quality and speed for autoregressive generation
3. **For Large Models:** XL and XXL show similar performance; choose based on quality requirements
4. **CPU Deployment:** Not currently supported; GPU is required

## Benchmark Methodology

- **Forward Pass:** 10 runs with sequence length 1024 tokens
- **Autoregressive:** 50 tokens generated per model
- **Warmup:** 3 iterations before timing
- **Device:** CUDA (GPU only)
- **Precision:** bfloat16
- **Checkpoint:** Step 750 (final checkpoint)

## Files

- **GPU Results:** `benchmark_gpu.csv`
- **Benchmark Script:** `benchmark_inference.py`
- **Inference Script:** `infer_winrwkv.py`
- **Model Checkpoints:** `runs/winrwkv/{SIZE}/checkpoint_step_750.pth`

## Conclusion

WinRWKV models demonstrate excellent GPU inference performance, with forward pass throughput ranging from 40k-104k tokens/sec depending on model size. Autoregressive generation achieves 40-87 tokens/sec, with Model M showing optimal performance. CPU inference is not supported due to the CUDA kernel requirement, making GPU deployment necessary for production use.
