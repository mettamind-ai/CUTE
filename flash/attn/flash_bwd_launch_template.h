/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_BACKWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_bwd_params params)

DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        flash::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}


template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    flash::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_clear_dkvaccum_kernel(const Flash_bwd_params params) {
    flash::clear_dKVaccum<Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(const Flash_bwd_params params, const int nsplits) {
    flash::convert_dQ<Kernel_traits>(params, nsplits);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(const Flash_bwd_params params) {
    flash::convert_dKV<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    if (params.deterministic) {
        int num_sm = get_num_sm(get_current_device());
        gridDimx = (num_sm + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.b, params.h);

    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    // printf("smem_size_dq_dk_dv = %d\n", smem_size_dq_dk_dv);
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                    // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                    // If Is_local, set Is_causal to false
                    auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap>;
                    if (smem_size_dq_dk_dv >= 48 * 1024)  {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                    }
                    kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });

    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    run_flash_bwd_seqk_parallel<Kernel_traits, Is_causal>(params, stream);
#endif
}

// template<typename T, bool Is_causal>
// void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
//     constexpr static int Headdim = 64;
//     int device;
//     cudaGetDevice(&device);
//     int max_smem_per_block;
//     cudaError status_ = cudaDeviceGetAttribute(
//         &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
//     if (status_ != cudaSuccess) {
//       C10_CUDA_CHECK(status_);
//     }
//     // printf("max_smem_per_block = %d\n", max_smem_per_block);
//         if (max_smem_per_block >= 144 * 1024) {
//             run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_causal>(params, stream);
//         } else {
//             run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_causal>(params, stream);
//         }
// }

template<typename T, bool Is_causal>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    if (max_smem_per_block >= 144 * 1024) {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>, Is_causal>(params, stream);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, T>, Is_causal>(params, stream);
    }
}
