/*******************************************************************
*  scaled_gemm_int8.cu  --  NVCC -arch=sm_89 -O3                   *
*******************************************************************/
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

// utilities -------------------------------------------------------
#define CHECK_CUDA(call)  do { cudaError_t err = call; \
  if (err != cudaSuccess) { printf("CUDA error %s:%d: %s\n",     \
     __FILE__, __LINE__, cudaGetErrorString(err)); exit(-1);} } while(0)

// kernel ----------------------------------------------------------
template <int TM, int TN, int TK>
__global__ void s8s8_scales_gemm(const int8_t* __restrict__ A,
                                 const int8_t* __restrict__ B,
                                 float*       __restrict__ C,
                                 const float* __restrict__ row_s,
                                 const float* __restrict__ col_s,
                                 int M, int N, int K, int lda, int ldb, int ldc)
{
  // 1. Threadblock indices
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // 2. Tile pointers in global mem
  const int8_t* A_tile = A + block_row * TM * lda;
  // For row-major B, the starting point is different
  const int8_t* B_tile = B + block_col * TN;

  // 3. Shared memory
  extern __shared__ int8_t smem[];
  int8_t* As = smem;                                  // TM x TK
  int8_t* Bs = smem + TM*TK;                          // TK x TN

  // 4. Accumulator fragment (per‑thread, INT32) - Use 16x8x16 for sm_89
  wmma::fragment<wmma::accumulator, 16, 8, 16, int32_t> c_frag[ (TM/16)*(TN/8) ];
  #pragma unroll
  for (auto &frag : c_frag) wmma::fill_fragment(frag, 0);

  // 5. Main loop over K
  for (int k0=0; k0 < K; k0 += TK) {
    // -- Load A & B tiles to SMEM
    // This part needs to be carefully implemented for performance.
    // A simple implementation is shown for correctness.
    __syncthreads();
    for (int i = threadIdx.x; i < TM * TK; i += blockDim.x) {
        int row = i / TK;
        int col = i % TK;
        As[i] = A_tile[row * lda + col + k0];
    }
    for (int i = threadIdx.x; i < TK * TN; i += blockDim.x) {
        int row = i / TN;
        int col = i % TN;
        // B is row-major, ldb is K
        Bs[i] = B_tile[(row + k0) * ldb + col];
    }
    __syncthreads();

    // -- Iterate subTiles 16x8x16
    for (int kk = 0; kk < TK; kk += 16) {
      const int8_t *tileA = As + kk;
      const int8_t *tileB = Bs + kk*TN;

      #pragma unroll
      for (int i=0; i < TM; i+=16)
      #pragma unroll
      for (int j=0; j < TN; j+=8) {
        // For sm_89, both A and B must be row_major for s8 operands
        wmma::fragment<wmma::matrix_a, 16, 8, 16, int8_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 8, 16, int8_t, wmma::row_major> b_frag; // <-- Changed to row_major
        wmma::load_matrix_sync(a_frag, tileA + i*TK, TK);
        wmma::load_matrix_sync(b_frag, tileB + j, TN); // ldb is TN for row-major tile in smem
        int idx = (i/16)*(TN/8) + (j/8);
        wmma::mma_sync(c_frag[idx], a_frag, b_frag, c_frag[idx]);
      }
    }
    __syncthreads();
  }

  // 6. Epilogue: write C with row/col scale
  int row_base = block_row * TM;
  int col_base = block_col * TN;
  for (int i=0; i < TM; i+=16)
  for (int j=0; j < TN; j+=8) {
    int idx = (i/16)*(TN/8) + (j/8);
    wmma::fragment<wmma::accumulator, 16,8,16,int32_t>& frag = c_frag[idx];
    // convert & scale
    #pragma unroll
    for (int t=0; t < frag.num_elements; ++t) {
      // Mapping from fragment element to matrix coordinates
      int frag_row_in_tile = t / 8;
      int frag_col_in_tile = t % 8;
      int r = row_base + i + frag_row_in_tile;
      int c = col_base + j + frag_col_in_tile;
      if (r < M && c < N) {
        float val = static_cast<float>(frag.x[t]);
        val *= row_s[r] * col_s[c];
        C[r*ldc + c] = val;
      }
    }
  }
}

// host helper -----------------------------------------------------
void launch_scaled_int8(const int8_t* dA, const int8_t* dB,
                        float* dC, const float* row_s, const float* col_s,
                        int M, int N, int K)
{
  const int TM = 64, TN = 64, TK = 64;
  dim3 grid( (N+TN-1)/TN, (M+TM-1)/TM );
  dim3 block(256);
  size_t smem = (TM*TK + TK*TN) * sizeof(int8_t);
  s8s8_scales_gemm<TM,TN,TK><<<grid, block, smem>>>(
      dA, dB, dC, row_s, col_s,
      M, N, K, /*lda*/K, /*ldb*/K, /*ldc*/N); // ldb is now K for row-major B
}
