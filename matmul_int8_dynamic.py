''' int8 matmul modded from github.com/gau-nernst/quantized-training
Revert commit quantized-training/commit/d430911a5fcf70ba4d4331933b8d0147927a9d6f
để giữ triton code đơn giản. Commit này có nhiều điểm thú vị:
- `scaled_mm` chấp nhận cả int8 và pf8
- `tile_scaled_mm` kernel mới cho ma trận đã được lượng tử hoá block-wise (32×32)
  => Đọc một block K nhỏ nhiều lần giúp giảm cache miss
  => Độ chính xác cao hơn: Block-wise scale (16, 32 phần tử) sai số lượng tử hoá 
    thấp hơn kiểu per-row/col, đặc biệt với mạng lớn.
- 4090 có thể hỗ trợ (1 phần) pf8, cần tìm hiểu và khai thác!
'''

#################################
##  INT8 Triton Matmul support ##
#################################

import torch, triton
import triton.language as tl
from torch import Tensor

scaled_mm_cfgs = [ # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps) => Prune to speedup autotune ??
    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    (128, 256,  64, 3, 8), ( 64, 256,  32, 4, 4), (128, 128,  32, 4, 4), (128,  64, 32, 4, 4),
    ( 64, 128,  32, 4, 4), (128,  32,  32, 4, 4), ( 64,  32,  32, 5, 2), ( 32,  64, 32, 5, 2),
    # Good config for fp8 inputs
    (128, 256, 128, 3, 8), (256, 128, 128, 3, 8), (256,  64, 128, 4, 4), ( 64, 256, 128, 4, 4),
    (128, 128, 128, 4, 4), (128,  64,  64, 4, 4), ( 64, 128,  64, 4, 4), (128,  32,  64, 4, 4),
    # https://github.com/pytorch/pytorch/blob/7868b65c4d4f34133607b0166f08e9fbf3b257c4/torch/_inductor/kernel/mm_common.py#L172
    ( 64,  64,  32, 2, 4), ( 64, 128,  32, 3, 4), (128,  64,  32, 3, 4),
    ( 64, 128,  32, 4, 8), (128,  64,  32, 4, 8), ( 64,  32,  32, 5, 8),
    ( 32,  64,  32, 5, 8), (128, 128,  32, 2, 8), ( 64,  64,  64, 3, 8),
    # https://github.com/pytorch/ao/blob/main/torchao/prototype/quantized_training/int8_mm.py#L47
    (128, 256, 128, 3, 8), (256, 128, 128, 3, 8),  # no need ??
]
scaled_mm_cfgs = [triton.Config(dict(BLOCK_M=m, BLOCK_N=n, BLOCK_K=k), num_stages=s, num_warps=w) for m, n, k, s, w in scaled_mm_cfgs]
@triton.autotune(configs=scaled_mm_cfgs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _scaled_mm_kernel(
    A_ptr, B_ptr, C_ptr,
    row_scale_ptr, col_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32) # ACC_DTYPE = tl.int32
    for k in range(K, 0, -BLOCK_K):
        a, b = tl.load(A), tl.load(B)  # EVEN_K = True
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    row_scale = tl.load(row_scale_ptr + idx_m, mask=idx_m < M).to(tl.float32)
    col_scale = tl.load(col_scale_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    acc = acc.to(tl.float32) * row_scale * col_scale

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


_grid = lambda meta: ( triton.cdiv(meta["M"], meta["BLOCK_M"])*triton.cdiv(meta["N"], meta["BLOCK_N"]), )
def scaled_mm(A: Tensor, B: Tensor, row_scale_A: Tensor, col_scale_B: Tensor) -> Tensor:
    """Matmul for tile-wise quantized A and B. `A` and `B` are both INT8 to utilize
    INT8 tensor cores. `row_scale_A` and `col_scale_B` are quantization scales for A and B. E.g.
    - if `A` is quantized with tile shape (128, 64), `row_scale_A`'s shape will be `(A.shape[0] / 128, A.shape[1] / 64)`.
    - if `A` is row-wise quantized, `row_scale_A`'s shape will be `(A.shape[0], 1)`.
    """
    assert A.dtype == B.dtype == torch.int8
    assert row_scale_A.dtype == col_scale_B.dtype
    assert A.ndim == B.ndim == row_scale_A.ndim == col_scale_B.ndim == 2
    assert A.shape[1] == B.shape[0]

    # row-scale + col-scale or row-scale + tensor-scale
    assert row_scale_A.shape == (A.shape[0], 1)
    assert col_scale_B.shape in ((1, B.shape[1]), (1, 1))

    assert row_scale_A.is_contiguous()
    assert col_scale_B.is_contiguous()

    M, K = A.shape
    _, N = B.shape

    assert K % 2 == 0             # => EVEN_K = True
    assert A.dtype == torch.int8  # => ACC_DTYPE = tl.int32

    C = torch.empty(M, N, device=A.device, dtype=row_scale_A.dtype)
    _scaled_mm_kernel[_grid](
        A, B, C,
        row_scale_A,
        col_scale_B,
        M, N, K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
    )
    return C


@torch.no_grad()
def quantize_int8(tensor: Tensor, dim=-1, eps=1e-12, sr=False) -> Tensor:
    ''' absmax symmetric quantization, clip(cận_dưới_eps) tránh chia cho 0 '''
    scale = tensor.abs().amax(dim, keepdim=True) / 127 # same dtype
    inv_scale = 1.0 / scale.float().clip(eps)       # little bit faster than 
    tensor = tensor.float() * inv_scale.view(-1, 1) # tensor / scale.clip(eps)
    if sr: tensor = (tensor + torch.rand_like(tensor)).floor()
    else:  tensor = tensor.round()# ^^^stochastic rounding^^^^
    return ( tensor.clip(-128, 127).to(torch.int8), scale )


def _dynamic_int8_mm(A: Tensor, B: Tensor, sr=False) -> Tensor:
    A_i8, row_scale = quantize_int8(A, dim=1, sr=sr)
    B_t_i8, col_scale = quantize_int8(B.T, dim=1, sr=sr)
    return scaled_mm(
        A_i8.contiguous(),
        B_t_i8.contiguous().T,
        row_scale.contiguous(),
        col_scale.T.contiguous(),
    )
