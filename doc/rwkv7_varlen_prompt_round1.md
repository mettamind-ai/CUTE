# RWKV7 Variable-Length Sequence Support - Deep Research

## Context

I'm implementing RWKV7 (linear attention RNN) for LLM training. The current kernel only supports fixed-length sequences within a batch. I want to add varlen support similar to Flash Attention, allowing multiple sequences of different lengths to be packed into a single batch to avoid wasted compute on padding.

RWKV7 is a linear attention mechanism with O(1) memory per token (vs O(n) for transformers), using recurrent state instead of KV cache.

## Files (Self-Contained)

All necessary code is embedded below. No file uploads needed.

---

## FILE 1: wkv7.cu (RWKV7 CUDA Kernel - COMPLETE)

```cuda
#include <assert.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
using bf = __nv_bfloat16;

__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return __float2bfloat16_rn(u); }

typedef bf * __restrict__ F_;

// Forward: processes B batches, each with T timesteps, H heads, C channels per head
// State recurrence: state[j] = state[j] * w[j] + sa * b[j] + k[j] * v
// Output: y = sum_j(state[j] * q[j])
__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, bf* y_, float* s_, float* sa_) {
    constexpr int C = _C_;  // HEAD_SIZE = 64, compile-time constant
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float state[C] = {0};  // Per-sequence state, initialized to 0
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));  // w in (0, 1), decay factor
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        // State-attention: sa = sum_j(a[j] * state[j])
        float sa = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;  // Save for backward

        float v = to_float(v_[ind]);
        float y = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;  // State update recurrence
            y += s * q[j];  // Output = dot(state, q)
        }
        y_[ind] = to_bf(y);

        // Checkpoint state every CHUNK_LEN tokens for backward pass
        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
        }
    }
}

// Backward: reconstructs state from checkpoints, computes gradients
// Key insight: state_{t-1} can be reconstructed from state_t via:
//   state_{t-1} = (state_t - k_t * v_t - b_t * sa_t) / w_t
__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s_, float * __restrict__ sa_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    // stateT: reconstructed state at time t (going backwards)
    // dstate: gradient w.r.t. state (accumulated)
    // dstateT: transposed gradient for certain computations
    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);  // w = exp(-exp(w_input))
        k[i] = ki = to_float(k_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        v[i] = to_float(v_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        // Load checkpoint at chunk boundaries (end of chunk)
        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
#pragma unroll
            for (int j = 0; j < C; j++) {
                stateT[j] = s_[base + j];
            }
        }

        // dq = sum_j(state_t[j] * dy[j])
        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
        dq_[ind] = to_bf(dq);

        // Reconstruct state_{t-1} from state_t:
        // state_t = state_{t-1} * w + sa * b + k * v
        // => state_{t-1} = (state_t - k * v - b * sa) / w
        float iwi = 1.0f/wi;
#pragma unroll        
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }

        // Compute gradients for w, k, v, b
        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*stateT[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = to_bf(dw * wi * wi_fac);
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);

        // Shared memory for cross-thread communication
        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        // da = sum_j(state_{t-1}[j] * dSb[j])
        float da = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            da += stateT[j]*dSb_shared[j];
        }
        da_[ind] = to_bf(da);

        // Propagate gradients through state recurrence
        // dstate_{t-1} = dstate_t * w + dSb * a
#pragma unroll        
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
        }
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa) {
    forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa);
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s,sa,dw,dq,dk,dv,dz,da);
}

void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr());
}

void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(), 
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr());
}

TORCH_LIBRARY(wind_backstepping, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
```

---

## FILE 2: Python Interface (from winrwkv.py)

```python
HEAD_SIZE = 64
CHUNK_LEN = 16

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape  # B=batch, T=seq_len (fixed), H=heads, C=head_size
        assert T % CHUNK_LEN == 0
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y  = torch.empty_like(v)
        s  = torch.empty(B, H, T//CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype == torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db


def RUN_CUDA_RWKV7(q, w, k, v, a, b):
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC//HEAD_SIZE, HEAD_SIZE) for i in [q, w, k, v, a, b]]
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)
```

---

## FILE 3: Flash Attention Varlen Interface (Reference)

```python
# Flash Attention packs multiple sequences into a single tensor
# cu_seqlens = cumulative sequence lengths, shape (num_seqs + 1,)
# Example: 3 sequences of length [100, 50, 150]
# cu_seqlens = [0, 100, 150, 300]
# total_tokens = 300, packed into shape (total_tokens, num_heads, head_dim)

from flash_attn import flash_attn_varlen_func

# q, k, v shape: (total_tokens, num_heads, head_dim)
# cu_seqlens shape: (batch_size + 1,), dtype=int32
output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal=True
)
# output shape: (total_tokens, num_heads, head_dim)
```

---

## RWKV7 Mathematical Formulation

### Forward Pass (per timestep t)

```
Inputs at time t: q_t, k_t, v_t, w_t, a_t, b_t  (all vectors of size C)

1. State-attention:
   sa_t = sum_j( a_t[j] * state_{t-1}[j] )

2. State update (element-wise for each j):
   state_t[j] = state_{t-1}[j] * w_t[j] + sa_t * b_t[j] + k_t[j] * v_t
   
   where w_t[j] = exp(-exp(w_input_t[j])) in (0, 1) is the decay factor

3. Output:
   y_t = sum_j( state_t[j] * q_t[j] )

Initial condition: state_0 = 0 (zero vector)
```

### Backward Pass

```
Given: dy_t (gradient of loss w.r.t. y_t)

Need to compute: dq_t, dk_t, dv_t, dw_t, da_t, db_t

Key insight for state reconstruction:
  state_t = state_{t-1} * w_t + sa_t * b_t + k_t * v_t
  => state_{t-1} = (state_t - k_t * v_t - b_t * sa_t) / w_t

Gradient propagation through state recurrence:
  dL/d(state_{t-1}) = dL/d(state_t) * w_t + dL/d(sa_t) * a_t
```

---

## Proposed Varlen Design

### Input Changes
- Current: `(B, T, H, C)` where B=batch_size, T=fixed_seq_len
- Proposed: `(1, total_tokens, H, C)` or `(total_tokens, H, C)`
- Add: `cu_seqlens` tensor (int32, length = num_seqs + 1)

### Forward Pseudo-code

```cuda
__global__ void forward_kernel_varlen(int T, int H, int* cu_seqlens, int num_seqs, ...) {
    constexpr int C = _C_;
    int hh = blockIdx.x, i = threadIdx.x;
    
    float state[C] = {0};
    int seq_idx = 0;
    int seq_end = cu_seqlens[1];  // End of first sequence
    
    for (int t = 0; t < T; t++) {
        // Check sequence boundary
        if (t == seq_end) {
            seq_idx++;
            seq_end = cu_seqlens[seq_idx + 1];
            // Reset state for new sequence
            for (int j = 0; j < C; j++) state[j] = 0;
        }
        
        int ind = t*H*C + hh*C + i;
        // ... rest of forward computation unchanged
    }
}
```

### Backward Concerns

1. **State reconstruction across boundaries**: 
   - Backward loop iterates T-1 to 0
   - When crossing boundary backwards (from seq N to seq N-1), stateT should become 0
   - But checkpoint might be from the middle of seq N

2. **Gradient flow**:
   - Gradients must NOT flow across sequence boundaries
   - dstate, dstateT must reset to 0 at boundaries

3. **Checkpoint handling**:
   - Current: checkpoint every CHUNK_LEN tokens globally
   - With varlen: checkpoint might cross sequence boundary
   - Problem: what if checkpoint is at t=50, but seq boundary is at t=48?

---

## Concrete Example for Verification

```
Sequences: [A, B, C] with lengths [3, 2, 4]
cu_seqlens = [0, 3, 5, 9]
total_tokens = 9

Token indices:
- Sequence A: t=0, 1, 2
- Sequence B: t=3, 4
- Sequence C: t=5, 6, 7, 8

Forward execution:
- t=0: state = 0 (init), compute y_0
- t=1: state updated from t=0, compute y_1
- t=2: state updated from t=1, compute y_2
- t=3: state = 0 (RESET - new sequence B), compute y_3
- t=4: state updated from t=3, compute y_4
- t=5: state = 0 (RESET - new sequence C), compute y_5
- t=6,7,8: state evolves within sequence C

Backward execution (t=8 down to 0):
- t=8: dstate starts accumulating for seq C
- t=7,6,5: dstate continues for seq C
- t=4: BOUNDARY CROSSING - entering seq B
  - What should happen to dstate? (should reset to 0)
  - What should happen to stateT? (should be 0, as state_5 started from 0)
- t=4,3: dstate accumulates for seq B
- t=2: BOUNDARY CROSSING - entering seq A
- t=2,1,0: dstate accumulates for seq A
```

---

## Analysis Required

### 1. Mathematical Correctness (CRITICAL)

**Requirement**: Provide step-by-step derivation proving gradient computation remains correct with sequence boundaries.

Specifically:
- Forward: state_0 = 0 for each sequence. What does this imply for backward gradients?
- When t crosses a sequence boundary (going backwards), how should stateT reconstruction be handled?
- Prove that gradients do not leak across sequence boundaries.

**Questions to answer**:
1. At boundary (e.g., t=5 going to t=4), stateT[t=4] should be what value?
2. At boundary, dstate and dstateT should be reset to what?
3. How does the checkpoint at t=chunk_end interact with sequence boundaries?

### 2. Checkpoint Strategy

**Options**:

A. **Global checkpoints** (every CHUNK_LEN tokens)
   - Pro: Simple, predictable memory
   - Con: Checkpoint may cross boundary, backward must handle specially

B. **Per-sequence checkpoints**
   - Pro: Clean boundary handling
   - Con: Variable memory, complex indexing

C. **No checkpoints** (recompute forward in backward)
   - Pro: Simplest, no boundary issues
   - Con: 2x compute cost

D. **Hybrid**: Global checkpoints + boundary markers
   - Checkpoint globally, but also save boundary positions
   - Backward checks if current chunk crosses boundary

**Requirement**: Analyze tradeoffs and recommend with detailed reasoning.

### 3. Edge Cases

- Sequence length < CHUNK_LEN (no checkpoint for that sequence)
- Sequence boundary exactly at chunk boundary (t % CHUNK_LEN == 0)
- Empty sequence (length = 0)
- Single token sequence (length = 1)
- Very long sequence (>> CHUNK_LEN)

### 4. Alternative Approaches

- Padding with mask: wastes compute but no kernel modification needed
- Process sequences separately: simple but loses batching benefits
- Any other approaches?

### 5. Performance Implications

- Branch divergence when checking boundary (all threads in warp check same condition?)
- Memory access pattern with packed sequences
- Shared memory usage changes?

---

## Constraints

- Must be backward-compatible: cu_seqlens = NULL implies old fixed-length behavior
- CHUNK_LEN = 16 (compile-time constant, cannot change)
- HEAD_SIZE = 64 (compile-time constant)
- bfloat16 only
- Target: SM80+ (Ampere and newer GPUs)

---

## Deliverables

- [ ] **Gradient derivation**: Step-by-step mathematical proof for varlen case, verified with the concrete example above
- [ ] **Checkpoint strategy**: Recommendation with detailed reasoning and tradeoff analysis
- [ ] **Edge case handling**: Explicit handling for each edge case listed
- [ ] **Modified backward kernel**: Pseudo-code with detailed comments explaining boundary handling
- [ ] **Performance analysis**: Expected impact and mitigation strategies

---

## CRITICAL INSTRUCTIONS FOR ANALYSIS

> **Rigorous proof requirement**: 
> 
> 1. For every claim about mathematical correctness, provide step-by-step derivation
> 2. For every design decision, provide clear reasoning about tradeoffs
> 3. If uncertain, explicitly state uncertainty and propose a conservative fallback
> 4. Verify using the concrete example (sequences [3, 2, 4] with cu_seqlens = [0, 3, 5, 9])
> 5. Pay special attention to the backward pass - this is where bugs are most likely
>
> **Evidence standard**: Only conclude when you have **reliable evidence** from the provided context, or you can **reason it out clearly and defensibly**. If evidence is weak, state uncertainty and propose a conservative fallback.
>
> **Do NOT**: Make conclusions without derivation. If you need to make assumptions, state them explicitly.
>
> **Verification approach**: Walk through the concrete example step-by-step for both forward and backward passes, showing exact values of state, stateT, dstate at each timestep, especially around boundaries.
