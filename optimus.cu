/*******************************************************************
*  optimus.cu  –  scaled INT8 GEMM + row/col scale (RTX 4090)     *
*******************************************************************/
// nvcc -arch=sm_86 -O3 --expt-relaxed-constexpr -std=c++17 -c optimus.cu -o optimus.o
// nvcc -arch=sm_86 -shared -Xcompiler -fPIC optimus.o -o liboptimus.so

#include <cuda.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>

using namespace nvcuda;

// ---------------- parameters you may tune ------------------------
#define TM 128
#define TN 128
#define TK 64
// -----------------------------------------------------------------

// kernel -----------------------------------------------------------
template <int M_PER_TB, int N_PER_TB, int K_PER_TB>
__global__ void s8s8_scales_gemm(const int8_t* __restrict__ A,
                                 const int8_t* __restrict__ B,
                                 float*       __restrict__ C,
                                 const float* __restrict__ row_s,
                                 const float* __restrict__ col_s,
                                 int M, int N, int K,
                                 int lda, int ldb, int ldc)
{
  // block index
  int tb_row = blockIdx.y;
  int tb_col = blockIdx.x;

  // shared memory layout: A‑tile (M_PER_TB × K_PER_TB) | B‑tile (K_PER_TB × N_PER_TB)
  extern __shared__ int8_t smem[];
  int8_t* As = smem;                                   // offset 0
  int8_t* Bs = smem + M_PER_TB * K_PER_TB;             // offset next

  // per‑thread accumulator fragments
  constexpr int FRAG_M = 16, FRAG_N = 16, FRAG_K = 16;
  constexpr int FRAG_PER_WARP_M = M_PER_TB / FRAG_M;
  constexpr int FRAG_PER_WARP_N = N_PER_TB / FRAG_N;
  // INT32 accumulator
  wmma::fragment<wmma::accumulator,
                               FRAG_M, FRAG_N, FRAG_K, int32_t>
      acc_frags[FRAG_PER_WARP_M * FRAG_PER_WARP_N];
  #pragma unroll
  for (auto &f : acc_frags) wmma::fill_fragment(f, 0);

  // pointer to the first element of this TB in global mem
  const int8_t* a_tile_global = A + (tb_row * M_PER_TB) * lda;
  const int8_t* b_tile_global = B + (tb_col * N_PER_TB);
  // main loop over K
  for (int k0 = 0; k0 < K; k0 += K_PER_TB)
  {
    // ----------------------------------------------------------------
    // 1. copy A & B sub‑tiles (INT8) from GMEM → SMEM (coalesced copy)
    // each thread copies multiple elements
    for (int idx = threadIdx.x; idx < M_PER_TB * K_PER_TB + K_PER_TB * N_PER_TB;
         idx += blockDim.x)
    {
      if (idx < M_PER_TB * K_PER_TB) {
        int row = idx / K_PER_TB;
        int col = idx % K_PER_TB;
        // Boundary check to prevent out-of-bounds memory access
        if ((tb_row * M_PER_TB + row) < M && (k0 + col) < K) {
          As[idx] = a_tile_global[row * lda + (k0 + col)];
        } else {
          As[idx] = 0;
        }
      } else {
        int j = idx - M_PER_TB * K_PER_TB;
        int row = j / N_PER_TB;
        int col = j % N_PER_TB;
        // Boundary check to prevent out-of-bounds memory access
        if ((k0 + row) < K && (tb_col * N_PER_TB + col) < N) {
          // transpose B-tile on the fly
          Bs[col * K_PER_TB + row] = b_tile_global[(k0 + row) * ldb + col];
        } else {
          Bs[col * K_PER_TB + row] = 0;
        }
      }
    }
    __syncthreads();

    // ----------------------------------------------------------------
    // 2. tensor‑core MMA on the loaded tile
    for (int kk = 0; kk < K_PER_TB; kk += FRAG_K)
    {
      const int8_t* a_panel = As + kk;
      const int8_t* b_panel = Bs + kk;

      #pragma unroll
      for (int mi = 0; mi < M_PER_TB; mi += FRAG_M)
      #pragma unroll
      for (int nj = 0; nj < N_PER_TB; nj += FRAG_N)
      {
        // load fragments
        wmma::fragment<wmma::matrix_a,
            FRAG_M, FRAG_N, FRAG_K,
            int8_t,
            wmma::row_major> a_frag;

        wmma::fragment<wmma::matrix_b,
            FRAG_M, FRAG_N, FRAG_K,
            int8_t,
            wmma::col_major> b_frag;

        wmma::load_matrix_sync(
            a_frag, a_panel + mi * K_PER_TB, K_PER_TB);
        wmma::load_matrix_sync(
            b_frag, b_panel + nj * K_PER_TB, K_PER_TB);

        int frag_idx = (mi / FRAG_M) * FRAG_PER_WARP_N + (nj / FRAG_N);
        wmma::mma_sync(acc_frags[frag_idx],
                                     a_frag, b_frag, acc_frags[frag_idx]);
      }
    }
    __syncthreads();
  }

  // ------------------------------------------------------------------
  // 3. Epilogue: scale & store to global mem
  int row0 = tb_row * M_PER_TB + threadIdx.y;          // warp‑row offset
  int col0 = tb_col * N_PER_TB + threadIdx.x;          // warp‑col offset

  // we use one thread to write one FP32 element
  for (int mi = threadIdx.y; mi < M_PER_TB; mi += blockDim.y)
  for (int nj = threadIdx.x; nj < N_PER_TB; nj += blockDim.x)
  {
    int frag_idx = (mi / FRAG_M) * FRAG_PER_WARP_N + (nj / FRAG_N);
    auto &frag = acc_frags[frag_idx];

    int local_m = mi % FRAG_M;
    int local_n = nj % FRAG_N;
    int elt_idx = local_m * FRAG_N + local_n;

    int global_m = tb_row * M_PER_TB + mi;
    int global_n = tb_col * N_PER_TB + nj;

    if (global_m < M && global_n < N) {
      float val = static_cast<float>(frag.x[elt_idx]);
      val *= row_s[global_m] * col_s[global_n];
      C[global_m * ldc + global_n] = val;
    }
  }
}

// ------------------------ host wrapper ----------------------------
extern "C"
void launch_scaled_int8(const int8_t* dA, const int8_t* dB,
                        float* dC,
                        const float* dRowS,
                        const float* dColS,
                        int M, int N, int K)
{
  // Initialize output to zeros
  cudaMemset(dC, 0, M * N * sizeof(float));
  
  dim3 block(32, 8, 1);      // 256 threads
  dim3 grid((N + TN - 1)/TN,
            (M + TM - 1)/TM);
  size_t smem = (TM * TK + TK * TN) * sizeof(int8_t);
  
  // Synchronize before launch to ensure previous operations are complete
  cudaDeviceSynchronize();
  
  s8s8_scales_gemm<TM, TN, TK>
      <<<grid, block, smem>>>(
          dA, dB, dC, dRowS, dColS,
          M, N, K,         /*lda*/K, /*ldb*/N, /*ldc*/N);
  
  // Synchronize after launch to ensure kernel completion before checking errors
  cudaDeviceSynchronize();
  
  // Check for CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
}
