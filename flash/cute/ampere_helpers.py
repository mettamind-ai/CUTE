# Copyright (c) 2025, Tri Dao.
from typing import Type, Callable, Optional

import cutlass
import cutlass.cute as cute


# Tạo layout tối ưu cho shared memory (bộ nhớ chia sẻ) trên GPU để tránh bank conflicts.
def get_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:

    dtype_byte    = dtype.width // 8
    bytes_per_row = k_dim * dtype_byte

    smem_k_block_size = (
              128 if bytes_per_row % 128 == 0
         else (64 if bytes_per_row %  64 == 0
         else (32 if bytes_per_row %  32 == 0
         else  16))) // dtype_byte

    swizzle_bits = (
              4 if smem_k_block_size == 128
        else (3 if smem_k_block_size ==  64 
        else (2 if smem_k_block_size ==  32 
        else  1)))

    swizzle_base = (
              2 if dtype_byte == 4 
        else (3 if dtype_byte == 2 
        else  4))

    smem_m_block_size = ( 8 if k_dim % 32 == 0 else 16 )

    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0, # offset
        cute.make_ordered_layout(
            shape=( smem_m_block_size, smem_k_block_size ), 
            order=( 1, 0 )
        ),
    )



def gemm(                                       # MMA: Matrix Multiply-Accumulate (Nhân Ma trận và Tích lũy)
    tiled_mma: cute.TiledMma,                   # Bộ thực hiện phép nhân ma trận theo tiles
    acc:  cute.Tensor,                          # Tensor tích lũy kết quả
    tCrA: cute.Tensor, tCrB: cute.Tensor,       # Tensor A và B trong thanh ghi (register)
    tCsA: cute.Tensor, tCsB: cute.Tensor,       # Tensor A và B trong bộ nhớ chia sẻ (shared memory)
    smem_thr_copy_A: cute.TiledCopy,            # Đối tượng sao chép dữ liệu từ shared memory
    smem_thr_copy_B: cute.TiledCopy,            # -- như trên --
    hook_fn: Optional[Callable] = None,
    A_in_regs: cutlass.Constexpr[bool] = False, # Cờ xác định dữ liệu đã có sẵn trong thanh ghi chưa
    B_in_regs: cutlass.Constexpr[bool] = False, # -- như trên --
    swap_AB:   cutlass.Constexpr[bool] = False, # Cờ để hoán đổi vai trò của A và B
) -> None:
    if swap_AB:
        return gemm(tiled_mma, acc, tCrB, tCrA, tCsB, tCsA, smem_thr_copy_B, smem_thr_copy_A,
                        hook_fn, A_in_regs=B_in_regs, B_in_regs=A_in_regs, swap_AB=False)

    # `retile()` sắp xếp lại cấu trúc tile của tensor để phù hợp với pattern sao chép
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

    # `None, None, 0` Lấy TẤT CẢ các hàng (M) + Lấy TẤT CẢ các cột (N) + Chỉ lấy tile thứ 0 theo chiều K
    if not A_in_regs: cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    if not B_in_regs: cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])

    for k in range(cute.size(tCsA.shape[2])):
        if k <     cute.size(tCsA.shape[2]) - 1:
            if not A_in_regs: cute.copy(smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_copy_view[None, None, k + 1])
            if not B_in_regs: cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        if cutlass.const_expr(k == 0 and hook_fn is not None): hook_fn() # gọi hook sau first tile computed


def gemm_rs(
    tiled_mma: cute.TiledMma,   # bộ thực thi phép nhân
    acc: cute.Tensor,           # nơi tích luỹ kết quả
    tCrA: cute.Tensor,          # A hoàn toàn in register
    tCrB: cute.Tensor,          # B in register
    tCsB: cute.Tensor,          # B in share memory
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
) -> None:

    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])

    for k in range(cute.size(tCrA.shape[2])):
        if k <     cute.size(tCrA.shape[2]) - 1:
            # Ẩn độ trễ, copy dữ liệu cho bước mma sau ...
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1])

        # ... rồi mới thực hiện mma của bước này
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        if cutlass.const_expr(k == 0 and hook_fn is not None): hook_fn()
