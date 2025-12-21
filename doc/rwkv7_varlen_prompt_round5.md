# RWKV7 Varlen Time-Shift Optimization

## Context

I have implemented a varlen (packed sequences) version of RWKV7 for training. The WKV7 CUDA kernel itself is optimized and shows 1.1x-1.7x speedup over the original padded kernel. However, end-to-end model training is **slower** because the time-shift operation uses a Python loop.

**Evidence Standard**: Only conclude when you have reliable evidence from the provided context, or you can reason it out clearly and defensibly. If evidence is weak, state uncertainty.

---

## Problem: Slow Time-Shift for Varlen

### Original Time-Shift (Batched, Fast)
```python
# x shape: (B, T, C) - batched sequences, all same length
xx = F.pad(x, (0, 0, 1, -1)) - x  # shift right by 1, vectorized
xr = x + xx * self.x_r  # apply mixing
```

### Current Varlen Time-Shift (Slow)
```python
# x shape: (total_tokens, C) - packed sequences, variable lengths
# cu_seqlens: [0, len1, len1+len2, ...] cumulative lengths

xx = torch.zeros_like(x)
num_seqs = cu_seqlens.size(0) - 1
for i in range(num_seqs):
    start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
    if end > start:
        # For each sequence: xx[1:] = x[:-1] - x[1:]
        xx[start + 1:end] = x[start:end - 1] - x[start + 1:end]
        # First token of each seq: xx = 0 (no previous token)

xr = x + xx * self.x_r.squeeze(0)
```

### Benchmark Results

**WKV7 Kernel Only** (forward + backward):
| AvgSeqLen | NumSeqs | Orig ms | Varlen ms | Speedup |
|-----------|---------|---------|-----------|---------|
| 64        | 113     | 4.30    | 2.67      | 1.61x   |
| 128       | 60      | 4.68    | 2.96      | 1.58x   |
| 256       | 31      | 4.16    | 2.91      | 1.43x   |

**End-to-End Model Training** (full forward + backward + optimizer):
| AvgSeqLen | NumSeqs | Orig ms | Varlen ms | Speedup |
|-----------|---------|---------|-----------|---------|
| 128       | 31      | 138.2   | 244.9     | 0.56x   |
| 256       | 17      | 142.1   | 171.4     | 0.83x   |
| 512       | 9       | 135.5   | 137.4     | 0.99x   |

The Python loop kills performance. With 31 sequences, the loop runs 31 times per layer, and there are 6 layers × 2 modules (Tmix + Cmix) = 12 time-shift calls per forward pass.

---

## Constraints

1. **cu_seqlens format**: Must use `cu_seqlens` (int32 tensor) like Flash Attention varlen
2. **Gradient flow**: Time-shift must support autograd (backward pass)
3. **No sequence reordering**: Sequences are packed in order
4. **GPU only**: CUDA tensors, bfloat16 dtype
5. **Typical case**: 10-60 sequences per batch, 64-512 avg seq length, 4096-8192 total tokens

---

## Analysis Needed

### 1. Vectorized PyTorch Solution
Can we eliminate the Python loop using pure PyTorch ops?

Ideas I considered:
- `torch.scatter` / `torch.gather` with index tensors
- Create a "shift mask" tensor once, reuse
- Use `torch.segment_reduce` or similar

### 2. Custom CUDA Kernel
If vectorized PyTorch isn't fast enough, what's the minimal CUDA kernel?

Requirements:
- Input: `x (total_tokens, C)`, `cu_seqlens (num_seqs+1,)`
- Output: `xx (total_tokens, C)` where `xx[i] = x[i-1] - x[i]` if same sequence, else 0
- Must support backward (gradient w.r.t. x)

### 3. Fuse into WKV7 Kernel
Could time-shift be fused into the WKV7 kernel itself? The kernel already processes sequences with cu_seqlens.

### 4. Alternative: Precompute Indices
```python
# Precompute once per batch
prev_indices = compute_prev_indices(cu_seqlens, total_tokens)
# Then vectorized:
xx = x[prev_indices] - x  # where prev_indices[first_of_seq] = self
```

---

## Deliverables

1. **Recommended approach** with reasoning
2. **Code implementation** (PyTorch or CUDA)
3. **Backward pass** implementation if needed
4. **Expected speedup** estimate
5. **Edge cases** to handle (empty sequences, single-token sequences)

---

## Additional Context

### Model Architecture
- RWKV7 with 6-12 layers
- Each layer has Tmix (attention) and Cmix (FFN)
- Both use time-shift: `xx = shift(x) - x`
- Time-shift is applied to compute `xr, xw, xk, xv, xa, xg` (6 variants with different mixing weights)

### Current Varlen Model Code (relevant parts)
```python
class RWKV_Tmix_Varlen(nn.Module):
    def forward(self, x, v_first, cu_seqlens):
        T, C = x.size()
        num_seqs = cu_seqlens.size(0) - 1
        
        # SLOW: Python loop for time-shift
        xx = torch.zeros_like(x)
        for i in range(num_seqs):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            if end > start:
                xx[start + 1:end] = x[start:end - 1] - x[start + 1:end]
        
        # Apply 6 different mixing weights
        xr = x + xx * self.x_r.squeeze(0)
        xw = x + xx * self.x_w.squeeze(0)
        xk = x + xx * self.x_k.squeeze(0)
        xv = x + xx * self.x_v.squeeze(0)
        xa = x + xx * self.x_a.squeeze(0)
        xg = x + xx * self.x_g.squeeze(0)
        
        # ... rest of forward pass uses xr, xw, xk, xv, xa, xg
```

### WKV7 Varlen Kernel Interface
```python
# Already have this working:
torch.ops.wind_backstepping_varlen.forward_varlen(
    w, q, k, v, a, b,  # (total_tokens, H, C) packed tensors
    cu_seqlens,        # (num_seqs + 1,) int32
    y, s_chunk, sa     # outputs
)
```
