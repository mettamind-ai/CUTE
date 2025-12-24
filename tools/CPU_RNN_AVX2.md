# RNN WKV7 CPU Inference Notes (i9-12900H, AVX2)

Dựa trên i9‑12900H: **không có AVX‑512**, nên **AVX2 + FMA** là đường chính; khi chạy nên **thử P‑cores only** để ổn định hiệu năng. Thiết kế tối ưu RNN‑step cho WKV7 nên:

* **State S lưu column‑major**: `S[j*64 + i] = S(i,j)` để load SIMD theo 8 hàng.
* Tính output **không cần nhân q trong vòng update ma trận**:

  * `SA = S_old * a`
  * `y = S_old*(w⊙q) + SA*(b·q) + v*(k·q)`
  * rồi update `S` bằng `S = S*w + SA*b^T + v*k^T`
* AVX2 xử lý **8 rows/lần** (`__m256`), rất khớp với C=64.

## Code micro‑kernel AVX2 (1 head, 1 token)

```cpp
// g++/clang++: -O3 -mavx2 -mfma
#include <immintrin.h>
#include <cstdint>

static inline void enable_ftz_daz() {
    unsigned int csr = _mm_getcsr();
    csr |= (1u << 15) | (1u << 6); // FTZ + DAZ
    _mm_setcsr(csr);
}

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// S: column-major 64x64, aligned 32 bytes
// vectors: aligned 32 bytes
static inline void wkv7_step_head_avx2_cm64(
    float* __restrict S,           // [4096], S[j*64+i]
    const float* __restrict w,     // [64]  (đã là exp(-exp(x)) nếu muốn match CUDA)
    const float* __restrict q,     // [64]
    const float* __restrict k,     // [64]
    const float* __restrict v,     // [64]
    const float* __restrict a,     // [64]
    const float* __restrict b,     // [64]
    float* __restrict y_out        // [64]
) {
    constexpr int C = 64;
    constexpr int STR = 64;  // stride col-major
    constexpr int VEC = 8;

    alignas(32) float wq[C];
    alignas(32) float SA[C];

    // wq = w ⊙ q (vectorized)
    for (int j = 0; j < C; j += VEC) {
        __m256 wv = _mm256_load_ps(w + j);
        __m256 qv = _mm256_load_ps(q + j);
        _mm256_store_ps(wq + j, _mm256_mul_ps(wv, qv));
    }

    // bq = b·q, kq = k·q
    __m256 acc_bq = _mm256_setzero_ps();
    __m256 acc_kq = _mm256_setzero_ps();
    for (int j = 0; j < C; j += VEC) {
        __m256 qv = _mm256_load_ps(q + j);
        acc_bq = _mm256_fmadd_ps(_mm256_load_ps(b + j), qv, acc_bq);
        acc_kq = _mm256_fmadd_ps(_mm256_load_ps(k + j), qv, acc_kq);
    }
    const float bq = hsum256_ps(acc_bq);
    const float kq = hsum256_ps(acc_kq);

    // Pass1: SA = S_old*a  và y1 = S_old*(wq)
    for (int r = 0; r < C; r += VEC) {
        __m256 sa = _mm256_setzero_ps();
        __m256 y1 = _mm256_setzero_ps();

        float* colp = S + r;           // S_old[r..r+7, 0]
        for (int j = 0; j < C; ++j) {
            __m256 scol = _mm256_load_ps(colp);                 // 8 rows of column j
            sa = _mm256_fmadd_ps(scol, _mm256_set1_ps(a[j]), sa);
            y1 = _mm256_fmadd_ps(scol, _mm256_set1_ps(wq[j]), y1);
            colp += STR;
        }

        _mm256_store_ps(SA + r, sa);

        // y = y1 + SA*bq + v*kq
        __m256 yv = y1;
        yv = _mm256_fmadd_ps(sa, _mm256_set1_ps(bq), yv);
        yv = _mm256_fmadd_ps(_mm256_load_ps(v + r), _mm256_set1_ps(kq), yv);
        _mm256_store_ps(y_out + r, yv);
    }

    // Pass2: update S in-place
    // S_new[:,j] = S_old[:,j]*w[j] + SA*b[j] + v*k[j]
    for (int r = 0; r < C; r += VEC) {
        __m256 sa = _mm256_load_ps(SA + r);
        __m256 vv = _mm256_load_ps(v + r);

        float* colp = S + r;
        for (int j = 0; j < C; ++j) {
            __m256 scol = _mm256_load_ps(colp);
            scol = _mm256_mul_ps(scol, _mm256_set1_ps(w[j]));
            scol = _mm256_fmadd_ps(sa,  _mm256_set1_ps(b[j]), scol);
            scol = _mm256_fmadd_ps(vv,  _mm256_set1_ps(k[j]), scol);
            _mm256_store_ps(colp, scol);
            colp += STR;
        }
    }
}
```

## Wrapper chạy nhiều head (OpenMP, 12900H nên thử ~10 threads trước)

```cpp
#include <omp.h>

void wkv7_step_layer_avx2(
    float* __restrict S_all,          // [H][4096] col-major
    const float* __restrict w_all,    // [H][64]
    const float* __restrict q_all,    // [H][64]
    const float* __restrict k_all,    // [H][64]
    const float* __restrict v_all,    // [H][64]
    const float* __restrict a_all,    // [H][64]
    const float* __restrict b_all,    // [H][64]
    float* __restrict y_all,          // [H][64]
    int H
) {
    // gọi 1 lần/worker thread là đủ (đặt ở init threadpool cũng được)
    enable_ftz_daz();

    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H; ++h) {
        wkv7_step_head_avx2_cm64(
            S_all + (size_t)h * 4096,
            w_all + (size_t)h * 64,
            q_all + (size_t)h * 64,
            k_all + (size_t)h * 64,
            v_all + (size_t)h * 64,
            a_all + (size_t)h * 64,
            b_all + (size_t)h * 64,
            y_all + (size_t)h * 64
        );
    }
}
```

## Build/run nhanh

* Compile:

  * `-O3 -mavx2 -mfma -fopenmp`
* Runtime (gợi ý):

  * thử **P‑cores only**: `OMP_NUM_THREADS=6` hoặc `OMP_NUM_THREADS=12` (HT)
  * nếu dùng Intel OMP: `KMP_HW_SUBSET=6c,1t` hoặc `KMP_HW_SUBSET=6c,2t`
  * `OMP_PROC_BIND=close`
  * `OMP_PLACES=cores` (có thể gồm E‑cores; muốn loại E‑cores thì dùng `KMP_HW_SUBSET`)

## Ghi chú ngắn (đúng trọng tâm)

* **Column‑major** giúp load/store contiguous theo SIMD (8 rows liên tiếp).
* Công thức `y = S*(w⊙q) + (S*a)*(b·q) + v*(k·q)` giúp tính output **không phải nhân q trong vòng update**, thường nhanh hơn trên CPU; công thức này **khớp kernel wkv7.cu trong repo**, nhưng **khác** pseudocode RWKV‑7 paper (decay/removal/replacement + r).
* `_mm256_load_ps` cần **32‑byte aligned**; nếu chưa chắc alignment, dùng `_mm256_loadu_ps` hoặc align allocator.
* `FTZ/DAZ` tránh denormal làm chậm (hay gặp vì `w` nhỏ); nếu dùng OpenMP, nên set **trong mỗi worker thread**.
