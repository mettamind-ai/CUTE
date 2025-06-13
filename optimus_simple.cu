/*******************************************************************
*  optimus_simple.cu  –  simplified scaled INT8 GEMM               *
nvcc -arch=sm_86 -O3 --expt-relaxed-constexpr -std=c++17 -c optimus_simple.cu -o optimus.o -Xcompiler -fPIC 
nvcc -arch=sm_86 -shared -Xcompiler -fPIC optimus.o -o liboptimus.so
*******************************************************************/

#include <cuda.h>
#include <stdio.h>

// Simplified kernel that uses standard CUDA operations instead of tensor cores
__global__ void simple_int8_gemm(const int8_t* __restrict__ A,
                                const int8_t* __restrict__ B,
                                float* __restrict__ C,
                                const float* __restrict__ row_scale,
                                const float* __restrict__ col_scale,
                                int M, int N, int K) 
{
    // Calculate global thread coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within matrix bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Calculate dot product for this element
        for (int k = 0; k < K; k++) {
            sum += (float)A[row * K + k] * (float)B[k * N + col];
        }
        
        // Apply scaling factors
        float scaled_result = sum * row_scale[row] * col_scale[col];
        
        // Write result to output matrix
        C[row * N + col] = scaled_result;
    }
}

// Host wrapper function
extern "C"
void launch_scaled_int8(const int8_t* dA, const int8_t* dB,
                       float* dC,
                       const float* dRowS,
                       const float* dColS,
                       int M, int N, int K)
{
    // Clear output first
    cudaMemset(dC, 0, M * N * sizeof(float));
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    simple_int8_gemm<<<gridDim, blockDim>>>(dA, dB, dC, dRowS, dColS, M, N, K);
    
    // Check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}