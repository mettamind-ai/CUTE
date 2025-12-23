/**
 * RWKV7 Variable-Length Sequence Kernel
 * 
 * Supports packed sequences with cu_seqlens (like Flash Attention varlen).
 * 
 * Key differences from wkv7.cu:
 * - Input shape: (total_tokens, H, C) instead of (B, T, H, C)
 * - Uses cu_seqlens to identify sequence boundaries
 * - One block per (sequence, head) pair
 * - Backward uses "tail forward replay" instead of s_end checkpoint
 * 
 * NOTE ON OPTIMIZATION:
 * This kernel is intentionally kept in a "good enough" state for correctness
 * and practical performance. Further micro-optimizations are NOT pursued here
 * because they add complexity and risk without guaranteed wins:
 * - Backward is already register-heavy; tuning unroll/launch bounds must be
 *   profiled per GPU, not hard-coded.
 * - The main bandwidth cost is sa/s_chunk; reducing precision or checkpoints
 *   changes numerics and requires careful validation.
 * - For typical workloads in this repo, current speed is sufficient and stable.
 *
 * If you need more speed, profile first and optimize based on bottlenecks.
 */

#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <torch/extension.h>
#include <cuda_bf16.h>

using bf = __nv_bfloat16;

__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return __float2bfloat16_rn(u); }

typedef bf * __restrict__ F_;
typedef float * __restrict__ FF_;
typedef const int * __restrict__ I_;

/**
 * Forward kernel for variable-length sequences.
 * 
 * Grid: (num_seqs * H,) - one block per (seq, head)
 * Block: (C,) - 64 threads, each handles one row of state matrix
 * 
 * Inputs (packed, shape: total_tokens, H, C):
 *   w_, q_, k_, v_, a_, b_
 * 
 * Outputs:
 *   y_: output (total_tokens, H, C)
 *   s_chunk_: checkpoints at global chunk boundaries (H, num_chunks, C, C)
 *   sa_: saved state-attention for backward (total_tokens, H, C)
 */
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
    static_assert((CHUNK & (CHUNK - 1)) == 0, "CHUNK must be power of 2");
    
    // Decode block index to (seq, head)
    int block_id = blockIdx.x;
    int seq = block_id / H;
    int hh = block_id % H;
    int i = threadIdx.x;  // row index (0..C-1)
    
    if (seq >= num_seqs) return;
    
    int start = cu_seqlens[seq];
    int end = cu_seqlens[seq + 1];
    int L = end - start;
    if (L <= 0) return;
    
    // Per-thread state: row i of state matrix S[i, :]
    float state[C] = {0};
    
    // Shared memory for vectors
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    
    // Number of global chunks
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
        
        // Checkpoint at global chunk boundaries
        // Condition: (p + 1) % CHUNK == 0
        if (((p + 1) & (CHUNK - 1)) == 0) {
            int chunk = p / CHUNK;
            // Store like original kernel: s_chunk[hh, chunk, i, j] with stride C
            // base points to start of this chunk's data for head hh
            // Thread i stores its row: state[j] at position [i + j*C]
            int base = ((hh * num_chunks + chunk) * C * C) + i;
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                s_chunk_[base + j * C] = state[j];
            }
        }
    }
}


/**
 * Backward kernel for variable-length sequences.
 * 
 * Uses "tail forward replay" to reconstruct S_end without storing s_end.
 * Max replay steps: CHUNK_LEN - 1 = 15
 */
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
    // p0 = floor((p_end + 1) / CHUNK) * CHUNK - 1
    static_assert((CHUNK & (CHUNK - 1)) == 0, "CHUNK must be power of 2");
    int p0 = ((p_end + 1) & ~(CHUNK - 1)) - 1;  // works for any power-of-2 CHUNK
    
    // stateT[r] = S_current[r, i] (column i of state matrix)
    float stateT[C] = {0};
    
    __shared__ float w_sh[C], k_sh[C], b_sh[C], v_sh[C], sa_sh[C];
    __shared__ float q_sh[C], dy_sh[C], a_sh[C];
    __shared__ float dsa_shared[C];
    
    if (p0 >= start) {
        // Load checkpoint S_{p0} from s_chunk
        // Layout matches original kernel: s_chunk[hh, chunk, i, j] with stride C
        // Thread i loads its column: stateT[r] = S[r, i] from position [r + i*C]
        int chunk = p0 / CHUNK;
        int base = ((hh * num_chunks + chunk) * C * C) + i * C;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            stateT[r] = s_chunk_[base + r];
        }
    } else {
        // No checkpoint inside this sequence => start from zero
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
        sa_sh[i] = sa_[ind];  // Already saved from forward
        __syncthreads();
        
        // Forward update for column i:
        // S_p[r, i] = S_{p-1}[r, i] * w[i] + sa[r] * b[i] + v[r] * k[i]
        float wi = w_sh[i], ki = k_sh[i], bi = b_sh[i];
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            stateT[r] = stateT[r] * wi + sa_sh[r] * bi + v_sh[r] * ki;
        }
    }
    // Now stateT = column i of S_{p_end}
    
    // ========== MAIN BACKWARD LOOP ==========
    // dstate[j] = G_t[row=i, col=j] (row i of gradient matrix)
    // dstateT[j] = G_t[row=j, col=i] (col i of gradient matrix)
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
        
        // Optional: reload checkpoint at chunk boundaries (for numerical stability)
        if (tl != L - 1 && ((p + 1) & (CHUNK - 1)) == 0) {
            int chunk = p / CHUNK;
            // Same layout as initial load: base + r for contiguous column i
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
        
        // Reconstruct S_{t-1}[:, i] from S_t[:, i]
        // S_t[r,i] = S_{t-1}[r,i]*w[i] + sa[r]*b[i] + v[r]*k[i]
        // => S_{t-1}[r,i] = (S_t[r,i] - sa[r]*b[i] - v[r]*k[i]) / w[i]
        float inv_wi = 1.0f / wi;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            stateT[r] = (stateT[r] - ki * v_sh[r] - bi * sa_sh[r]) * inv_wi;
        }
        
        // Add output gradient contribution to G_t
        // G_t[r,c] += dy[r] * q[c]
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            dstate[j] += dyi * q_sh[j];    // row i: G_t[i, j]
            dstateT[j] += qi * dy_sh[j];   // col i: G_t[j, i]
        }
        
        // Compute gradients for w_i, k_i, b_i using column i of G_t and S_{t-1}[:,i]
        float dw_i = 0.0f, dk_i = 0.0f, db_i = 0.0f;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            dw_i += dstateT[r] * stateT[r];  // sum_r G_t[r,i] * S_{t-1}[r,i]
            dk_i += dstateT[r] * v_sh[r];    // sum_r G_t[r,i] * v[r]
            db_i += dstateT[r] * sa_sh[r];   // sum_r G_t[r,i] * sa[r]
        }
        
        // Chain rule through w = exp(-exp(x))
        dw_[ind] = to_bf(dw_i * wi * wi_fac);
        dk_[ind] = to_bf(dk_i);
        db_[ind] = to_bf(db_i);
        
        // Compute dv_i and dsa_i from row i of G_t
        float dv_i = 0.0f;
        float dsa_i = 0.0f;
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            dv_i += dstate[c] * k_sh[c];   // sum_c G_t[i,c] * k[c]
            dsa_i += dstate[c] * b_sh[c];  // sum_c G_t[i,c] * b[c]
        }
        dv_[ind] = to_bf(dv_i);
        
        // Share dsa across threads
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
        
        // Propagate G_{t-1} slices
        // G_{t-1}[r,c] = G_t[r,c] * w[c] + dsa[r] * a[c]
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            dstate[c] = dstate[c] * w_sh[c] + dsa_i * a_sh[c];  // row i
        }
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            dstateT[r] = dstateT[r] * wi + ai * dsa_shared[r];  // col i
        }
    }
}


// ========== C++ Interface ==========

static bool varlen_checks_enabled() {
    const char* v = std::getenv("RWKV7_VARLEN_CHECKS");
    if (!v) return false;
    return (std::strcmp(v, "1") == 0) || (std::strcmp(v, "true") == 0) || (std::strcmp(v, "TRUE") == 0);
}

static void check_cu_seqlens(torch::Tensor& cu_seqlens, int64_t total_tokens) {
    // Heavy checks (sync + CPU copy). Enable via RWKV7_VARLEN_CHECKS=1.
    if (!varlen_checks_enabled()) return;
    auto cu_cpu = cu_seqlens.to(torch::kCPU);
    auto* ptr = cu_cpu.data_ptr<int>();
    const int64_t n = cu_cpu.numel();
    TORCH_CHECK(n >= 2, "cu_seqlens must have at least 2 elements");
    TORCH_CHECK(ptr[0] == 0, "cu_seqlens[0] must be 0");
    int prev = ptr[0];
    for (int64_t i = 1; i < n; ++i) {
        int cur = ptr[i];
        TORCH_CHECK(cur >= prev, "cu_seqlens must be non-decreasing");
        TORCH_CHECK(cur >= 0, "cu_seqlens must be non-negative");
        prev = cur;
    }
    TORCH_CHECK(ptr[n - 1] == total_tokens, "cu_seqlens[-1] must equal total_tokens");
}

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


// ========== PyTorch Bindings ==========

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE_BF16(x) TORCH_CHECK(x.dtype() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_DTYPE_INT32(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be int32")
#define CHECK_DTYPE_FP32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")

void forward_varlen(
    torch::Tensor& w, torch::Tensor& q, torch::Tensor& k, 
    torch::Tensor& v, torch::Tensor& a, torch::Tensor& b,
    torch::Tensor& cu_seqlens,
    torch::Tensor& y, torch::Tensor& s_chunk, torch::Tensor& sa
) {
    // NOTE: to match wkv7.cu, `a` here corresponds to `z`, and `b` corresponds to `a`.
    // Input validation
    CHECK_CUDA(w); CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(v); CHECK_CUDA(a); CHECK_CUDA(b);
    CHECK_CUDA(cu_seqlens); CHECK_CUDA(y); CHECK_CUDA(s_chunk); CHECK_CUDA(sa);
    
    CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v); CHECK_CONTIGUOUS(a); CHECK_CONTIGUOUS(b);
    CHECK_CONTIGUOUS(cu_seqlens); CHECK_CONTIGUOUS(y); CHECK_CONTIGUOUS(s_chunk); CHECK_CONTIGUOUS(sa);
    
    CHECK_DTYPE_BF16(w); CHECK_DTYPE_BF16(q); CHECK_DTYPE_BF16(k);
    CHECK_DTYPE_BF16(v); CHECK_DTYPE_BF16(a); CHECK_DTYPE_BF16(b);
    CHECK_DTYPE_BF16(y);
    CHECK_DTYPE_INT32(cu_seqlens);
    CHECK_DTYPE_FP32(s_chunk); CHECK_DTYPE_FP32(sa);
    
    TORCH_CHECK(w.dim() == 3, "w must be 3D (total_tokens, H, C)");
    TORCH_CHECK(q.sizes() == w.sizes(), "q shape must match w");
    TORCH_CHECK(k.sizes() == w.sizes(), "k shape must match w");
    TORCH_CHECK(v.sizes() == w.sizes(), "v shape must match w");
    TORCH_CHECK(a.sizes() == w.sizes(), "a shape must match w");
    TORCH_CHECK(b.sizes() == w.sizes(), "b shape must match w");
    
    int64_t total_tokens = w.size(0);
    int64_t H = w.size(1);
    int64_t C = w.size(2);
    int64_t num_seqs = cu_seqlens.size(0) - 1;
    
    TORCH_CHECK(num_seqs > 0, "cu_seqlens must have at least 2 elements");
    TORCH_CHECK(total_tokens > 0, "total_tokens must be > 0 (no zero-length sequences allowed)");
    TORCH_CHECK(C == _C_, "C must equal HEAD_SIZE (_C_)");
    TORCH_CHECK(total_tokens <= std::numeric_limits<int>::max(), "total_tokens too large for int indexing");
    TORCH_CHECK(H <= std::numeric_limits<int>::max(), "H too large for int indexing");
    TORCH_CHECK((total_tokens * H * C) <= std::numeric_limits<int>::max(), "index overflow risk: total_tokens*H*C too large");
    
    int64_t num_chunks = (total_tokens + _CHUNK_LEN_ - 1) / _CHUNK_LEN_;
    TORCH_CHECK(s_chunk.dim() == 4, "s_chunk must be 4D (H, num_chunks, C, C)");
    TORCH_CHECK(s_chunk.size(0) == H, "s_chunk dim0 must equal H");
    TORCH_CHECK(s_chunk.size(1) == num_chunks, "s_chunk dim1 must equal num_chunks");
    TORCH_CHECK(s_chunk.size(2) == C && s_chunk.size(3) == C, "s_chunk last dims must be (C, C)");
    TORCH_CHECK(sa.dim() == 3, "sa must be 3D (total_tokens, H, C)");
    TORCH_CHECK(sa.size(0) == total_tokens && sa.size(1) == H && sa.size(2) == C, "sa shape mismatch");
    TORCH_CHECK(y.sizes() == w.sizes(), "y shape must match w");
    
    check_cu_seqlens(cu_seqlens, total_tokens);
    
    cuda_forward_varlen(
        (int)total_tokens, (int)H, (int)num_seqs,
        cu_seqlens.data_ptr<int>(),
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(),
        (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(),
        (bf*)y.data_ptr(), s_chunk.data_ptr<float>(), sa.data_ptr<float>()
    );
}

void backward_varlen(
    torch::Tensor& w, torch::Tensor& q, torch::Tensor& k,
    torch::Tensor& v, torch::Tensor& a, torch::Tensor& b,
    torch::Tensor& dy, torch::Tensor& cu_seqlens,
    torch::Tensor& s_chunk, torch::Tensor& sa,
    torch::Tensor& dw, torch::Tensor& dq, torch::Tensor& dk,
    torch::Tensor& dv, torch::Tensor& da, torch::Tensor& db
) {
    // NOTE: to match wkv7.cu, `a` here corresponds to `z`, and `b` corresponds to `a`.
    // Input validation
    CHECK_CUDA(w); CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(v); CHECK_CUDA(a); CHECK_CUDA(b);
    CHECK_CUDA(dy); CHECK_CUDA(cu_seqlens); CHECK_CUDA(s_chunk); CHECK_CUDA(sa);
    CHECK_CUDA(dw); CHECK_CUDA(dq); CHECK_CUDA(dk); CHECK_CUDA(dv); CHECK_CUDA(da); CHECK_CUDA(db);
    
    CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v); CHECK_CONTIGUOUS(a); CHECK_CONTIGUOUS(b);
    CHECK_CONTIGUOUS(dy); CHECK_CONTIGUOUS(cu_seqlens);
    CHECK_CONTIGUOUS(s_chunk); CHECK_CONTIGUOUS(sa);
    CHECK_CONTIGUOUS(dw); CHECK_CONTIGUOUS(dq); CHECK_CONTIGUOUS(dk);
    CHECK_CONTIGUOUS(dv); CHECK_CONTIGUOUS(da); CHECK_CONTIGUOUS(db);
    
    CHECK_DTYPE_BF16(w); CHECK_DTYPE_BF16(q); CHECK_DTYPE_BF16(k);
    CHECK_DTYPE_BF16(v); CHECK_DTYPE_BF16(a); CHECK_DTYPE_BF16(b);
    CHECK_DTYPE_BF16(dy);
    CHECK_DTYPE_BF16(dw); CHECK_DTYPE_BF16(dq); CHECK_DTYPE_BF16(dk);
    CHECK_DTYPE_BF16(dv); CHECK_DTYPE_BF16(da); CHECK_DTYPE_BF16(db);
    CHECK_DTYPE_INT32(cu_seqlens);
    CHECK_DTYPE_FP32(s_chunk); CHECK_DTYPE_FP32(sa);
    
    TORCH_CHECK(w.dim() == 3, "w must be 3D (total_tokens, H, C)");
    TORCH_CHECK(q.sizes() == w.sizes(), "q shape must match w");
    TORCH_CHECK(k.sizes() == w.sizes(), "k shape must match w");
    TORCH_CHECK(v.sizes() == w.sizes(), "v shape must match w");
    TORCH_CHECK(a.sizes() == w.sizes(), "a shape must match w");
    TORCH_CHECK(b.sizes() == w.sizes(), "b shape must match w");
    TORCH_CHECK(dy.sizes() == w.sizes(), "dy shape must match w");
    
    int64_t total_tokens = w.size(0);
    int64_t H = w.size(1);
    int64_t C = w.size(2);
    int64_t num_seqs = cu_seqlens.size(0) - 1;
    
    TORCH_CHECK(num_seqs > 0, "cu_seqlens must have at least 2 elements");
    TORCH_CHECK(total_tokens > 0, "total_tokens must be > 0 (no zero-length sequences allowed)");
    TORCH_CHECK(C == _C_, "C must equal HEAD_SIZE (_C_)");
    TORCH_CHECK(total_tokens <= std::numeric_limits<int>::max(), "total_tokens too large for int indexing");
    TORCH_CHECK(H <= std::numeric_limits<int>::max(), "H too large for int indexing");
    TORCH_CHECK((total_tokens * H * C) <= std::numeric_limits<int>::max(), "index overflow risk: total_tokens*H*C too large");
    
    int64_t num_chunks = (total_tokens + _CHUNK_LEN_ - 1) / _CHUNK_LEN_;
    TORCH_CHECK(s_chunk.dim() == 4, "s_chunk must be 4D (H, num_chunks, C, C)");
    TORCH_CHECK(s_chunk.size(0) == H, "s_chunk dim0 must equal H");
    TORCH_CHECK(s_chunk.size(1) == num_chunks, "s_chunk dim1 must equal num_chunks");
    TORCH_CHECK(s_chunk.size(2) == C && s_chunk.size(3) == C, "s_chunk last dims must be (C, C)");
    TORCH_CHECK(sa.dim() == 3, "sa must be 3D (total_tokens, H, C)");
    TORCH_CHECK(sa.size(0) == total_tokens && sa.size(1) == H && sa.size(2) == C, "sa shape mismatch");
    TORCH_CHECK(dw.sizes() == w.sizes(), "dw shape must match w");
    TORCH_CHECK(dq.sizes() == w.sizes(), "dq shape must match w");
    TORCH_CHECK(dk.sizes() == w.sizes(), "dk shape must match w");
    TORCH_CHECK(dv.sizes() == w.sizes(), "dv shape must match w");
    TORCH_CHECK(da.sizes() == w.sizes(), "da shape must match w");
    TORCH_CHECK(db.sizes() == w.sizes(), "db shape must match w");
    
    check_cu_seqlens(cu_seqlens, total_tokens);
    
    cuda_backward_varlen(
        (int)total_tokens, (int)H, (int)num_seqs,
        cu_seqlens.data_ptr<int>(),
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(),
        (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(),
        (bf*)dy.data_ptr(),
        s_chunk.data_ptr<float>(), sa.data_ptr<float>(),
        (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(),
        (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr()
    );
}


TORCH_LIBRARY(wind_backstepping_varlen, m) {
    m.def("forward_varlen(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor cu_seqlens, Tensor(a!) y, Tensor(b!) s_chunk, Tensor(c!) sa) -> ()");
    m.def("backward_varlen(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor cu_seqlens, Tensor s_chunk, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) da, Tensor(f!) db) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping_varlen, CUDA, m) {
    m.impl("forward_varlen", &forward_varlen);
    m.impl("backward_varlen", &backward_varlen);
}
