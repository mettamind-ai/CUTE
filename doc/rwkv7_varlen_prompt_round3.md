# RWKV7 Varlen Kernel Implementation Review

## Context

I've implemented a variable-length sequence kernel for RWKV7 (`wkv7_varlen.cu`) that supports packed sequences with `cu_seqlens` (like Flash Attention varlen). This is an extension of the original fixed-batch kernel (`wkv7.cu`).

**Goal**: Review the implementation for correctness, edge cases, and potential bugs. Use your full reasoning capabilities - this is a complex CUDA kernel with subtle indexing and numerical issues.

**Evidence Standard**: Only conclude when you have **reliable evidence** from the provided code/tests, or you can **reason it out clearly and defensibly**. If evidence is weak, state uncertainty and propose a conservative fallback.

---

## Original Kernel Math (from wkv7.cu)

The RWKV7 kernel maintains a `C×C` state matrix `S_t` per (sequence, head). Each thread `i` owns row `i` of the state matrix.

**Forward update** (per token `t`):
```
sa_t[r] = Σ_c a_t[c] * S_{t-1}[r,c]           # state-attention
S_t[r,c] = S_{t-1}[r,c] * w_t[c] + sa_t[r] * b_t[c] + v_t[r] * k_t[c]
y_t[r] = Σ_c S_t[r,c] * q_t[c]
```

Where `w_t[c] = exp(-exp(w_input[c]))` (double exp for numerical stability).

**Checkpointing**: Original kernel stores `S_t` at every `(t+1) % 16 == 0` for backward reconstruction.

---

## Varlen Design Decisions (from Pro consultation round 1)

Key differences from original:
1. **Input shape**: `(total_tokens, H, C)` instead of `(B, T, H, C)`
2. **Grid mapping**: `blockIdx.x = seq * H + head` (1D grid, avoids 65535 limit)
3. **Global checkpoints**: Store `s_chunk` at global positions where `(p+1) % 16 == 0`
4. **No s_end storage**: Use "tail forward replay" (max 15 steps) to reconstruct S_end
5. **Boundary handling**: Each block processes one sequence independently

---

## Implementation Code

### wkv7_varlen.cu (complete)

```cuda
/**
 * RWKV7 Variable-Length Sequence Kernel
 */

#include <assert.h>
#include <torch/extension.h>
#include <cuda_bf16.h>

using bf = __nv_bfloat16;

__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return __float2bfloat16_rn(u); }

typedef bf * __restrict__ F_;
typedef float * __restrict__ FF_;
typedef const int * __restrict__ I_;

__global__ void forward_kernel_varlen(
    int total_tokens,
    int H,
    I_ cu_seqlens,      // (num_seqs + 1,)
    int num_seqs,
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
    bf* __restrict__ y_,
    FF_ s_chunk_,       // (H, num_chunks, C, C) transposed
    FF_ sa_
) {
    constexpr int C = _C_;
    constexpr int CHUNK = _CHUNK_LEN_;
    
    int block_id = blockIdx.x;
    int seq = block_id / H;
    int hh = block_id % H;
    int i = threadIdx.x;  // row index (0..C-1)
    
    if (seq >= num_seqs) return;
    
    int start = cu_seqlens[seq];
    int end = cu_seqlens[seq + 1];
    int L = end - start;
    if (L <= 0) return;
    
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    
    int num_chunks = (total_tokens + CHUNK - 1) / CHUNK;
    
    for (int tl = 0; tl < L; ++tl) {
        int p = start + tl;  // global packed token index
        int ind = (p * H + hh) * C + i;
        
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();
        
        // State-attention: sa = sum_j(a[j] * state[j])
        float sa = 0;
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;  // Save for backward
        
        // State update and output
        float v = to_float(v_[ind]);
        float y = 0;
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);
        
        // Checkpoint at global chunk boundaries: (p + 1) % CHUNK == 0
        if (((p + 1) & (CHUNK - 1)) == 0) {
            int chunk = p / CHUNK;
            int base = ((hh * num_chunks + chunk) * C * C) + i;
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                s_chunk_[base + j * C] = state[j];
            }
        }
    }
}


__global__ void backward_kernel_varlen(
    int total_tokens,
    int H,
    I_ cu_seqlens,
    int num_seqs,
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_,
    FF_ s_chunk_,
    FF_ sa_,
    bf* __restrict__ dw_,
    bf* __restrict__ dq_,
    bf* __restrict__ dk_,
    bf* __restrict__ dv_,
    bf* __restrict__ da_,
    bf* __restrict__ db_
) {
    constexpr int C = _C_;
    constexpr int CHUNK = _CHUNK_LEN_;
    
    int block_id = blockIdx.x;
    int seq = block_id / H;
    int hh = block_id % H;
    int i = threadIdx.x;
    
    if (seq >= num_seqs) return;
    
    int start = cu_seqlens[seq];
    int end = cu_seqlens[seq + 1];
    int L = end - start;
    if (L <= 0) return;
    
    int p_end = end - 1;  // last token of this sequence
    int num_chunks = (total_tokens + CHUNK - 1) / CHUNK;
    
    // ========== PRELUDE: Tail Forward Replay to build S_end ==========
    // Find nearest chunk checkpoint <= p_end
    int p0 = (((p_end + 1) >> 4) << 4) - 1;  // bit trick for CHUNK=16
    
    // stateT[r] = S_current[r, i] (column i of state matrix)
    float stateT[C] = {0};
    
    __shared__ float w_sh[C], k_sh[C], b_sh[C], v_sh[C], sa_sh[C];
    __shared__ float q_sh[C], dy_sh[C], a_sh[C];
    __shared__ float dsa_shared[C];
    
    if (p0 >= start) {
        // Load checkpoint S_{p0} from s_chunk
        int chunk = p0 / CHUNK;
        int base = ((hh * num_chunks + chunk) * C * C) + i * C;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            stateT[r] = s_chunk_[base + r];
        }
    } else {
        #pragma unroll
        for (int r = 0; r < C; ++r) stateT[r] = 0.0f;
        p0 = start - 1;
    }
    
    // Replay forward from p0+1 to p_end to build S_{p_end}
    for (int p = p0 + 1; p <= p_end; ++p) {
        int ind = (p * H + hh) * C + i;
        
        __syncthreads();
        w_sh[i] = __expf(-__expf(to_float(w_[ind])));
        k_sh[i] = to_float(k_[ind]);
        b_sh[i] = to_float(b_[ind]);
        v_sh[i] = to_float(v_[ind]);
        sa_sh[i] = sa_[ind];
        __syncthreads();
        
        float wi = w_sh[i], ki = k_sh[i], bi = b_sh[i];
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            stateT[r] = stateT[r] * wi + sa_sh[r] * bi + v_sh[r] * ki;
        }
    }
    // Now stateT = column i of S_{p_end}
    
    // ========== MAIN BACKWARD LOOP ==========
    float dstate[C] = {0};
    float dstateT[C] = {0};
    
    for (int tl = L - 1; tl >= 0; --tl) {
        int p = start + tl;
        int ind = (p * H + hh) * C + i;
        
        __syncthreads();
        q_sh[i] = to_float(q_[ind]);
        float x = to_float(w_[ind]);
        float wi_fac = -__expf(x);
        w_sh[i] = __expf(wi_fac);
        k_sh[i] = to_float(k_[ind]);
        a_sh[i] = to_float(a_[ind]);
        b_sh[i] = to_float(b_[ind]);
        v_sh[i] = to_float(v_[ind]);
        dy_sh[i] = to_float(dy_[ind]);
        sa_sh[i] = sa_[ind];
        __syncthreads();
        
        float wi = w_sh[i];
        float ki = k_sh[i];
        float bi = b_sh[i];
        float ai = a_sh[i];
        float qi = q_sh[i];
        float dyi = dy_sh[i];
        
        // Optional: reload checkpoint at chunk boundaries
        if (tl != L - 1 && ((p + 1) & (CHUNK - 1)) == 0) {
            int chunk = p / CHUNK;
            int base = ((hh * num_chunks + chunk) * C * C) + i * C;
            #pragma unroll
            for (int r = 0; r < C; ++r) {
                stateT[r] = s_chunk_[base + r];
            }
        }
        
        // dq_i = sum_r S_t[r, i] * dy[r]
        float dq_i = 0.0f;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            dq_i += stateT[r] * dy_sh[r];
        }
        dq_[ind] = to_bf(dq_i);
        
        // Reconstruct S_{t-1}[:,i] from S_t[:,i]
        float inv_wi = 1.0f / wi;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            stateT[r] = (stateT[r] - ki * v_sh[r] - bi * sa_sh[r]) * inv_wi;
        }
        
        // Add output gradient contribution to G_t
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            dstate[j] += dyi * q_sh[j];
            dstateT[j] += qi * dy_sh[j];
        }
        
        // Compute gradients for w_i, k_i, b_i
        float dw_i = 0.0f, dk_i = 0.0f, db_i = 0.0f;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            dw_i += dstateT[r] * stateT[r];
            dk_i += dstateT[r] * v_sh[r];
            db_i += dstateT[r] * sa_sh[r];
        }
        
        dw_[ind] = to_bf(dw_i * wi * wi_fac);
        dk_[ind] = to_bf(dk_i);
        db_[ind] = to_bf(db_i);
        
        // Compute dv_i and dsa_i
        float dv_i = 0.0f;
        float dsa_i = 0.0f;
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            dv_i += dstate[c] * k_sh[c];
            dsa_i += dstate[c] * b_sh[c];
        }
        dv_[ind] = to_bf(dv_i);
        
        __syncthreads();
        dsa_shared[i] = dsa_i;
        __syncthreads();
        
        // da_i = sum_r S_{t-1}[r,i] * dsa[r]
        float da_i = 0.0f;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            da_i += stateT[r] * dsa_shared[r];
        }
        da_[ind] = to_bf(da_i);
        
        // Propagate G_{t-1}
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            dstate[c] = dstate[c] * w_sh[c] + dsa_i * a_sh[c];
        }
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            dstateT[r] = dstateT[r] * wi + ai * dsa_shared[r];
        }
    }
}
```

### PyTorch Bindings & Constants

```cuda
// Constants (compile-time defines):
// _C_ = 64 (HEAD_SIZE)
// _CHUNK_LEN_ = 16

void cuda_forward_varlen(
    int total_tokens, int H, int num_seqs,
    int* cu_seqlens,
    bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
    bf* y, float* s_chunk, float* sa
) {
    int num_blocks = num_seqs * H;
    forward_kernel_varlen<<<num_blocks, _C_>>>(
        total_tokens, H, cu_seqlens, num_seqs,
        w, q, k, v, a, b, y, s_chunk, sa
    );
}

void cuda_backward_varlen(
    int total_tokens, int H, int num_seqs,
    int* cu_seqlens,
    bf* w, bf* q, bf* k, bf* v, bf* a, bf* b, bf* dy,
    float* s_chunk, float* sa,
    bf* dw, bf* dq, bf* dk, bf* dv, bf* da, bf* db
) {
    int num_blocks = num_seqs * H;
    backward_kernel_varlen<<<num_blocks, _C_>>>(
        total_tokens, H, cu_seqlens, num_seqs,
        w, q, k, v, a, b, dy, s_chunk, sa,
        dw, dq, dk, dv, da, db
    );
}

// PyTorch interface
void forward_varlen(
    torch::Tensor& w, torch::Tensor& q, torch::Tensor& k, 
    torch::Tensor& v, torch::Tensor& a, torch::Tensor& b,
    torch::Tensor& cu_seqlens,
    torch::Tensor& y, torch::Tensor& s_chunk, torch::Tensor& sa
) {
    int total_tokens = w.size(0);
    int H = w.size(1);
    int num_seqs = cu_seqlens.size(0) - 1;
    
    cuda_forward_varlen(
        total_tokens, H, num_seqs,
        cu_seqlens.data_ptr<int>(),
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(),
        (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(),
        (bf*)y.data_ptr(), s_chunk.data_ptr<float>(), sa.data_ptr<float>()
    );
}

TORCH_LIBRARY(wind_backstepping_varlen, m) {
    m.def("forward_varlen(...) -> ()");
    m.def("backward_varlen(...) -> ()");
}
```

### Test Methodology (Python)

```python
# Ground truth: run original kernel per-sequence with padding
def run_original_per_sequence(seq_lengths, w, q, k, v, a, b, H, C):
    outputs = []
    offset = 0
    for seq_len in seq_lengths:
        # Pad to multiple of CHUNK_LEN=16
        padded_len = ((seq_len + 15) // 16) * 16
        
        # Extract and pad sequence data
        w_seq = F.pad(w[offset:offset+seq_len], (0,0,0,0,0,padded_len-seq_len))
        # ... same for q, k, v, a, b
        
        # Reshape to (1, T, H, C) for original kernel
        w_seq = w_seq.unsqueeze(0).contiguous()
        
        # Run original kernel
        y_seq = OriginalRWKV7.apply(w_seq, q_seq, k_seq, v_seq, a_seq, b_seq)
        
        # Extract only valid (non-padded) output
        outputs.append(y_seq[0, :seq_len])
        offset += seq_len
    
    return torch.cat(outputs, dim=0)

# Varlen kernel wrapper
class VarlenRWKV7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, a, b, cu_seqlens):
        total_tokens, H, C = w.shape
        num_seqs = cu_seqlens.shape[0] - 1
        num_chunks = (total_tokens + 15) // 16
        
        y = torch.empty_like(v)
        s_chunk = torch.empty(H, num_chunks, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(total_tokens, H, C, dtype=torch.float32, device=w.device)
        
        torch.ops.wind_backstepping_varlen.forward_varlen(
            w, q, k, v, a, b, cu_seqlens, y, s_chunk, sa
        )
        ctx.save_for_backward(w, q, k, v, a, b, cu_seqlens, s_chunk, sa)
        return y
```

---

## Test Results

All tests pass with **exact match** (max_diff = 0) for sequences that are multiples of CHUNK_LEN=16.

### Test Suite Summary

```
============================================================
RWKV7 Varlen Kernel Test Suite
============================================================
Compiling original kernel (wkv7.cu)...
Compiling varlen kernel (wkv7_varlen.cu)...
Both kernels compiled successfully!

============================================================
TEST: Forward Correctness
============================================================
Two equal sequences, multiple of CHUNK_LEN: [32, 32]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00
Different lengths, both multiple of CHUNK_LEN: [16, 48]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00
Three sequences, NOT multiples of CHUNK_LEN: [20, 30, 14]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00
Short sequences (< CHUNK_LEN): [5, 10, 3]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00
Single token sequences: [1, 1, 1]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00
Single long sequence: [64]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00
Single sequence, not multiple of CHUNK_LEN: [17]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00
Single sequence, exactly CHUNK_LEN: [16]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00
Single sequence, CHUNK_LEN - 1: [15]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00
Short sequence between long ones: [32, 1, 32]
  ✓ PASS | Max diff: 0.000000e+00, Mean diff: 0.000000e+00

============================================================
TEST: Backward Correctness
============================================================
Single sequence, 2 chunks: seq_len=32
  ✓ dw: max_diff = 0.000000e+00
  ✓ dq: max_diff = 0.000000e+00
  ✓ dk: max_diff = 0.000000e+00
  ✓ dv: max_diff = 0.000000e+00
  ✓ da: max_diff = 0.000000e+00
  ✓ db: max_diff = 0.000000e+00
Single sequence, 3 chunks: seq_len=48
  ✓ dw: max_diff = 0.000000e+00
  ✓ dq: max_diff = 0.000000e+00
  ✓ dk: max_diff = 0.000000e+00
  ✓ dv: max_diff = 0.000000e+00
  ✓ da: max_diff = 0.000000e+00
  ✓ db: max_diff = 0.000000e+00
Single sequence, 4 chunks: seq_len=64
  ✓ dw: max_diff = 0.000000e+00
  ✓ dq: max_diff = 0.000000e+00
  ✓ dk: max_diff = 0.000000e+00
  ✓ dv: max_diff = 0.000000e+00
  ✓ da: max_diff = 0.000000e+00
  ✓ db: max_diff = 0.000000e+00
Single sequence, 1 chunk: seq_len=16
  ✓ dw: max_diff = 0.000000e+00
  ✓ dq: max_diff = 0.000000e+00
  ✓ dk: max_diff = 0.000000e+00
  ✓ dv: max_diff = 0.000000e+00
  ✓ da: max_diff = 0.000000e+00
  ✓ db: max_diff = 0.000000e+00

============================================================
TEST: No Gradient Leakage
============================================================
  ✓ w: A=0.00e+00, B=3.91e-03, C=0.00e+00
  ✓ q: A=0.00e+00, B=4.88e-03, C=0.00e+00
  ✓ k: A=0.00e+00, B=1.95e-03, C=0.00e+00
  ✓ v: A=0.00e+00, B=1.95e-03, C=0.00e+00
  ✓ a: A=0.00e+00, B=9.77e-04, C=0.00e+00
  ✓ b: A=0.00e+00, B=9.77e-04, C=0.00e+00

============================================================
TEST: NaN Prefill (Missing Writes Detection)
============================================================
Single sequence: [16]
  ✓ PASS | y finite: True, sa finite: True
Multiple single tokens: [1, 1, 1]
  ✓ PASS | y finite: True, sa finite: True
Mixed short sequences: [5, 10, 3]
  ✓ PASS | y finite: True, sa finite: True
Multiple chunks: [32, 48]
  ✓ PASS | y finite: True, sa finite: True

============================================================
TEST: Edge Cases
============================================================
Single token: [1]
  ✓ PASS | Max diff: 0.000000e+00
All single tokens: [1, 1, 1, 1]
  ✓ PASS | Max diff: 0.000000e+00
All < CHUNK_LEN: [3, 7, 2, 5]
  ✓ PASS | Max diff: 0.000000e+00
All multiples of CHUNK_LEN: [16, 32, 16]
  ✓ PASS | Max diff: 0.000000e+00
Around CHUNK_LEN boundaries: [15, 17, 31, 33]
  ✓ PASS | Max diff: 0.000000e+00
Long sequence: [128]
  ✓ PASS | Max diff: 0.000000e+00

============================================================
SUMMARY
============================================================
  ✓ PASS: forward
  ✓ PASS: backward
  ✓ PASS: leakage
  ✓ PASS: nan_prefill
  ✓ PASS: edge_cases

ALL TESTS PASSED!
```

---

## Benchmark Results

Varlen kernel vs Original kernel (with padding) on RTX 3050 Ti:

### Context Length 4096

| Num Seqs | Orig ms | Varlen ms | Speedup | Waste% |
|----------|---------|-----------|---------|--------|
| 5        | 9.25    | 9.02      | 1.03x   | 62.1%  |
| 10       | 4.65    | 4.05      | 1.15x   | 72.5%  |
| 20       | 2.68    | 2.19      | 1.22x   | 66.3%  |
| 30       | 4.14    | 2.47      | 1.68x   | 77.5%  |
| 40       | 5.05    | 2.28      | 2.21x   | 82.2%  |
| 50       | 3.78    | 1.56      | 2.42x   | 74.4%  |

### Context Length 8192

| Num Seqs | Orig ms | Varlen ms | Speedup | Waste% |
|----------|---------|-----------|---------|--------|
| 5        | 12.26   | 11.54     | 1.06x   | 62.1%  |
| 10       | 9.48    | 8.13      | 1.17x   | 72.5%  |
| 20       | 5.70    | 4.25      | 1.34x   | 66.8%  |
| 30       | 8.69    | 4.65      | 1.87x   | 77.8%  |
| 40       | 10.27   | 4.28      | 2.40x   | 82.2%  |
| 50       | 7.62    | 3.12      | 2.44x   | 74.4%  |

**Varlen is 1.03x - 2.44x faster** depending on number of sequences.

---

## Analysis Needed

Please analyze the implementation thoroughly:

1. **Correctness Review**
   - Is the forward kernel mathematically correct?
   - Is the backward kernel mathematically correct?
   - Are there any subtle bugs in indexing, especially around sequence boundaries?

2. **Tail Forward Replay Analysis**
   - Is the `p0` calculation correct? `p0 = (((p_end + 1) >> 4) << 4) - 1`
   - Does the replay loop correctly reconstruct S_end?
   - What happens when `p0 < start` (no checkpoint inside sequence)?

3. **Checkpoint Layout**
   - Forward stores: `s_chunk_[base + j * C] = state[j]` where `base = ((hh * num_chunks + chunk) * C * C) + i`
   - Backward loads: `stateT[r] = s_chunk_[base + r]` where `base = ((hh * num_chunks + chunk) * C * C) + i * C`
   - Is this transpose correct? Forward stores row i, backward loads column i?

4. **Edge Cases**
   - Empty sequence (L=0): handled by early return
   - Single token (L=1): no checkpoint, replay from zero
   - Sequence shorter than CHUNK_LEN: no checkpoint inside, replay from zero
   - Sequence boundary exactly at chunk boundary

5. **Potential Bugs**
   - Any race conditions?
   - Any out-of-bounds memory access?
   - Any numerical stability issues?

6. **Missing Tests**
   - What edge cases are NOT covered by the current test suite?
   - What could cause silent failures?

---

## Deliverables

- [ ] Correctness analysis with specific line references
- [ ] List of potential bugs (if any) with severity
- [ ] Missing test cases that should be added
- [ ] Recommendations for improvements
