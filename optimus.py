#!/usr/bin/env python3
# INT8 Mixed Precision modified from github.com/gau-nernst/quantized-training
# Muon optimizer modified from https://github.com/KellerJordan/Muon

############################################
##  Modded Triton Matmul to support INT8  ##
############################################

import torch
import triton
import triton.language as tl
from torch import Tensor

lib = torch.library.Library("qtrain", "DEF")
lib_ops = torch.ops.qtrain

# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
configs = [
    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    (128, 256, 64, 3, 8),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 5, 2),
    (32, 64, 32, 5, 2),
    # Good config for fp8 inputs
    (128, 256, 128, 3, 8),
    (256, 128, 128, 3, 8),
    (256, 64, 128, 4, 4),
    (64, 256, 128, 4, 4),
    (128, 128, 128, 4, 4),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4),
    (128, 32, 64, 4, 4),
    # https://github.com/pytorch/pytorch/blob/7868b65c4d4f34133607b0166f08e9fbf3b257c4/torch/_inductor/kernel/mm_common.py#L172
    (64, 64, 32, 2, 4),
    (64, 128, 32, 3, 4),
    (128, 64, 32, 3, 4),
    (64, 128, 32, 4, 8),
    (128, 64, 32, 4, 8),
    (64, 32, 32, 5, 8),
    (32, 64, 32, 5, 8),
    (128, 128, 32, 2, 8),
    (64, 64, 64, 3, 8),
]

configs = [
    triton.Config(dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K), num_stages=num_stages, num_warps=num_warps)
    for BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps in configs
]


def _grid(meta):
    return (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)


# templated matmul from pytorch
# https://github.com/pytorch/pytorch/blob/c2e2602ecdc2ec1f120e19198dfc18fc39f7bd09/torch/_inductor/kernel/mm.py
# also re-tune when stride changes i.e. transpose configuration
@triton.autotune(configs=configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _matmul_kernel(
    # fmt: off
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACC_DTYPE: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    # fmt: on
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b, out_dtype=ACC_DTYPE)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


lib.define("int8_mm(Tensor A, Tensor B) -> Tensor")


def int8_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert A.shape[1] == B.shape[0]
    return lib_ops.int8_mm(A, B)


@torch.library.impl(lib, "int8_mm", "Meta")
def _(a: Tensor, b: Tensor):
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.int32)


@torch.library.impl(lib, "int8_mm", "CUDA")
def _(A: Tensor, B: Tensor):
    return _triton_mm(A, B, torch.int32, torch.int32)


def _triton_mm(A: Tensor, B: Tensor, out_dtype: torch.dtype, acc_dtype: torch.dtype):
    ACC_DTYPE_TRITON = {torch.float32: tl.float32, torch.float16: tl.float16, torch.int32: tl.int32}[acc_dtype]
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape
    EVEN_K = K % 2 == 0
    C = torch.empty(M, N, dtype=out_dtype, device=A.device)
    _matmul_kernel[_grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(), ACC_DTYPE_TRITON, EVEN_K)
    return C


@triton.autotune(configs=configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _scaled_mm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    row_scale_ptr,
    col_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
    COL_SCALE_SCALAR: tl.constexpr = False,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
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
    if COL_SCALE_SCALAR:
        # hack to support BitNet. col_scale is now a scalar
        col_scale = tl.load(col_scale_ptr).to(tl.float32)
    else:
        col_scale = tl.load(col_scale_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    acc = acc.to(tl.float32) * row_scale * col_scale

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


@triton.autotune(
    # need to find more performant configs...
    configs=[
        triton.Config(dict(BLOCK_M=128, BLOCK_N=128), num_stages=2, num_warps=8),
        triton.Config(dict(BLOCK_M=128, BLOCK_N=128), num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K", "stride_ak", "stride_bk"],
)
@triton.jit
def _tile_scaled_mm_kernel(
    A_ptr,  # (M, K)
    B_ptr,  # (K, N)
    C_ptr,  # (M, N)
    scale_A_ptr,  # (M // QUANT_BLOCK_M, K // QUANT_BLOCK_K)
    scale_B_ptr,  # (K // QUANT_BLOCK_K, N // QUANT_BLOCK_N)
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scale_am,
    stride_scale_ak,
    stride_scale_bk,
    stride_scale_bn,
    QUANT_BLOCK_M: tl.constexpr,
    QUANT_BLOCK_N: tl.constexpr,
    QUANT_BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
):
    # NOTE: most of the time, it's most performant with BLOCK_K == QUANT_BLOCK_K
    tl.static_assert(QUANT_BLOCK_K % BLOCK_K == 0)

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

    # NOTE: it seems like we can afford to have QUANT_BLOCK_M and QUANT_BLOCK_N not be constexpr
    A_scale = scale_A_ptr + ((rm // QUANT_BLOCK_M)[:, None] * stride_scale_am)
    B_scale = scale_B_ptr + ((rn // QUANT_BLOCK_N)[None, :] * stride_scale_bn)

    # we use 2 accumulators. acc is the final result. mma_acc is accumulator for MMA before
    # scaling. for every QUANT_BLOCK_K, we will scale mma_acc and accumulate it to acc.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    """ v1
    mma_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        mma_acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

        if (k - BLOCK_K) % QUANT_BLOCK_K == 0:
            a_scale = tl.load(A_scale).to(tl.float32)
            b_scale = tl.load(B_scale).to(tl.float32)
            acc += mma_acc.to(tl.float32) * a_scale * b_scale
            mma_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
            A_scale += stride_scale_ak
            B_scale += stride_scale_bk
    """
    ### v2
    for k in range(K, 0, -QUANT_BLOCK_K):
        mma_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        for _ in tl.static_range(QUANT_BLOCK_K // BLOCK_K):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                a = tl.load(A, mask=rk[None, :] < k, other=0.0)
                b = tl.load(B, mask=rk[:, None] < k, other=0.0)
            mma_acc += tl.dot(a, b)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        a_scale = tl.load(A_scale).to(tl.float32)
        b_scale = tl.load(B_scale).to(tl.float32)
        acc += mma_acc.to(tl.float32) * a_scale * b_scale
        A_scale += stride_scale_ak
        B_scale += stride_scale_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


lib.define("scaled_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")
lib.define("tile_scaled_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")


def scaled_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    """Matmul for tile-wise quantized A and B. `A` and `B` are both INT8 or FP8 to utilize
    INT8/FP8 tensor cores. `scale_A` and `scaled_B` are quantization scales for A and B
    respectively with appropriate shapes.

    E.g.
      - if `A` is quantized with tile shape (128, 64), `scale_A`'s shape will be
    `(A.shape[0] / 128, A.shape[1] / 64)`.
      - if `A` is row-wise quantized, `scale_A`'s shape will be `(A.shape[0], 1)`.
    """
    _f8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    assert (A.dtype == B.dtype == torch.int8) or (A.dtype in _f8 and B.dtype in _f8)
    assert scale_A.dtype == scale_B.dtype
    assert A.ndim == B.ndim == scale_A.ndim == scale_B.ndim == 2
    assert A.shape[1] == B.shape[0]

    # row-scale + col-scale or row-scale + tensor-scale
    if scale_A.shape == (A.shape[0], 1) and scale_B.shape in ((1, B.shape[1]), (1, 1)):
        assert scale_A.is_contiguous()
        assert scale_B.is_contiguous()
        return lib_ops.scaled_mm(A, B, scale_A, scale_B)

    # generic tile-wise scaling
    assert scale_A.shape[1] == scale_B.shape[0]
    return lib_ops.tile_scaled_mm(A, B, scale_A, scale_B)


@torch.library.impl(lib, "scaled_mm", "Meta")
@torch.library.impl(lib, "tile_scaled_mm", "Meta")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=scale_A.dtype)


@torch.library.impl(lib, "scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=row_scale.dtype)
    grid = lambda meta: (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)
    _scaled_mm_kernel[grid](
        A,
        B,
        C,
        row_scale,
        col_scale,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        ACC_DTYPE=tl.int32 if A.dtype == torch.int8 else tl.float32,
        EVEN_K=K % 2 == 0,
        COL_SCALE_SCALAR=col_scale.numel() == 1,
    )
    return C


@torch.library.impl(lib, "tile_scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=scale_A.dtype)
    grid = lambda meta: (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)
    QUANT_BLOCK_K = A.shape[1] // scale_A.shape[1]
    _tile_scaled_mm_kernel[grid](
        A,
        B,
        C,
        scale_A,
        scale_B,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        *scale_A.stride(),
        *scale_B.stride(),
        QUANT_BLOCK_M=A.shape[0] // scale_A.shape[0],
        QUANT_BLOCK_N=B.shape[1] // scale_B.shape[1],
        QUANT_BLOCK_K=QUANT_BLOCK_K,
        BLOCK_K=QUANT_BLOCK_K,
        ACC_DTYPE=tl.int32 if A.dtype == torch.int8 else tl.float32,
        EVEN_K=K % 2 == 0,
    )
    return C


##############################################
##  INT8 Mixed Precision for Linear Module  ##
##############################################

from typing import NamedTuple

import torch, os
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import Tensor, nn

aten = torch.ops.aten

INT8_MIXED_SR = os.getenv('INT8_MIXED_SR', '0')
print(f"INT8_MIXED_SR => {INT8_MIXED_SR}")

@torch.no_grad()
def quantize_int8(tensor: Tensor, *, dim: int = -1, eps: float = 1e-12, sr=False) -> Tensor:
    # absmax symmetric quantization
    dtype = tensor.dtype
    tensor = tensor.float()
    scale = tensor.abs().amax(dim, keepdim=True) / 127
    tensor = tensor / scale.clip(eps)

    if sr: tensor = (tensor + torch.rand_like(tensor)).floor()
    else:  tensor = tensor.round()# ^^^stochastic rounding^^^^

    tensor = tensor.clip(-128, 127).to(torch.int8)
    return tensor, scale.to(dtype)

class MixedPrecisionLinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor):
        return Tensor._make_wrapper_subclass(cls, data.shape, device=data.device,)

    @torch._dynamo.disable
    def __init__(self, data: Tensor):
        self._data = data

    def __tensor_flatten__(self):
        return ["_data"], []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict,
            tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["_data"])

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self._data})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            return _Int8MixedPrecisionLinear.apply(*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    # adapated from FP8 implementation of WeightWithDynamicFloat8CastTensor
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        def unwrap(x: cls):
            return x._data

        out = func(
            *pytree.tree_map_only(cls, unwrap, args),
            **pytree.tree_map_only(cls, unwrap, kwargs),
        )

        if func is aten.copy_.default:
            # return original object
            return args[0]

        elif func in {
            aten.t.default,
            aten.detach.default,
            aten.empty_like.default,
            aten.new_zeros.default,
            aten.slice.Tensor,
            aten.view.default,
            aten.as_strided.default,
            aten._to_copy.default,
            aten._pin_memory.default,
            aten.split.Tensor,
            aten.clone.default,
        }:
            # return new wrapped object
            return pytree.tree_map_only(Tensor, lambda x: cls(x), out)
        else:
            # return new unwrapped object
            return out


def _dynamic_int8_mm(A: Tensor, B: Tensor, sr=False) -> Tensor:
    A_i8, row_scale = quantize_int8(A, dim=1, sr=sr)
    B_t_i8, col_scale = quantize_int8(B.T, dim=1, sr=sr)
    return scaled_mm(
        A_i8.contiguous(),
        B_t_i8.contiguous().T,
        row_scale.contiguous(),
        col_scale.T.contiguous(),
    )

ABIT = INT8_MIXED_SR == "abit" # 1 phép round 8bit
HACK = INT8_MIXED_SR == "hack" # 2 phép round 8bit + 1 phép bf16
FULL = INT8_MIXED_SR == "full" # 3 phép round 8bit
FWD_SR       = INT8_MIXED_SR in ["half", "full", "hack"]
BWD_INPUT_SR = INT8_MIXED_SR in ["half", "full", "hack", "abit"]

class _Int8MixedPrecisionLinear(torch.autograd.Function):
    @staticmethod
    def forward(input: Tensor, weight: MixedPrecisionLinearWeight, bias: Tensor | None = None):
        batch_dims = input.shape[:-1]
        input = input.view(-1, weight.shape[1])
        # Có thể không sử dụng stochasic rounding ở forward vì 
        # x2 slow do activation checkpoint sẽ tính forward 2 lần
        out = _dynamic_int8_mm(input, weight._data.T, sr=FWD_SR)
        out = out.view(*batch_dims, weight.shape[0])
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight._data)
        ctx.bias = bias is not None

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.view(-1, weight.shape[0])
        input = input.view(-1, weight.shape[1])

        if ctx.needs_input_grad[0]:
            grad_input = _dynamic_int8_mm(grad_output, weight, sr=BWD_INPUT_SR)
            # grad_input = grad_output @ weight # original op
            grad_input = grad_input.view(*batch_dims, weight.shape[1])

        if ctx.needs_input_grad[1]:
            if   HACK:  # chậm nhưng, giúp tránh oom khi rounding cả grad_weight 
                grad_weight = grad_output.T @ input
            elif FULL:  # sr=True rất dễ oom nên chỉ dùng ở full mode
                grad_weight = _dynamic_int8_mm(input.T, grad_output, sr=True).T
            else:       # Tăng tốc và giảm vram hết cỡ
                grad_weight = _dynamic_int8_mm(input.T, grad_output, sr=False).T

        if ctx.needs_input_grad[2] and ctx.bias:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


def convert_int8_mixed_precision(module:nn.Module):
    count = 0
    for n, m in module.named_modules():
        if isinstance(m, nn.Linear) and "head" not in n:
            count += 1
            m.weight = nn.Parameter(
                MixedPrecisionLinearWeight(m.weight.detach()),
                requires_grad=m.weight.requires_grad,
            )
    return count



#################################################################
##  Muon Optimizer - MomentUm Orthogonalized by Newton-schulz  ##
#################################################################

""" Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
the advantage that it can be stably run in bfloat16 on the GPU.
NOTE: use Adam for 0D, 1D, embeddings and lm_head, then use Muon for the rest """

import torch, math
import torch.distributed as dist
from torch import Tensor

def newtonschulz(G: Tensor, steps: int) -> Tensor:
    # G: The gradient or momentum matrix to be orthogonalized.
    # steps: Number of Newton-Schulz iterations.
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):  # NS iterations
        A = X @ X.mT
        B = b * A + c * A@A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1): X = X.mT
    return X


# DON'T CHANGE. mini fix to make it works with 1 GPU
class MuonOrigin(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=None, world_size=None):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                if handle is not None: handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "mm_buffer" not in state:
                        state["mm_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["mm_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4: # for the case of conv filters
                        g = g.view(len(g), -1)
                    g = newtonschulz(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0: update_prev() # mẹo update muộn để dist.all_gather có time gather data
                # async all_gather instead of sync all_reduce by @YouJiacheng
                if self.world_size > 1:
                    handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                else: # world_size == 1 → copy thẳng gradient vào view 0
                    update_buffer_views[0].copy_(g.flatten().to(torch.bfloat16))
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
# END MuonOrigin


class Muon(torch.optim.Optimizer):    
    ''' Viết lại MuonOrigin cho dễ đọc, giữ càng nguyên bản càng tốt '''
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, ns_steps=5, rank=0, world_size=1):        
        self.rank = rank  # Lưu rank của quá trình hiện tại trong môi trường phân tán
        self.world_size = world_size  # Lưu tổng số quá trình trong môi trường phân tán
        self.is_dist = (world_size > 1)  # Cờ đánh dấu có đang chạy phân tán hay không
        
        params: list[Tensor] = [*params]  # Chuyển iterator params thành list
        param_groups = []  # Khởi tạo danh sách nhóm tham số
        
        sizes = {  p.numel() for p in params }
        for size in sizes:  # Lặp qua các kích thước khác nhau của tham số
            # Tạo buffer cho cập nhật với kích thước phù hợp
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(  # Tạo nhóm tham số với cùng kích thước
                params=[ p for p in params if p.numel() == size ],  # Lọc tham số có cùng kích thước
                update_buffer=b,  # Tạo view cho từng phần của buffer tương ứng với từng GPU
                update_buffer_views=[b[i] for i in range(world_size)] 
            )
            param_groups.append(group)  # Thêm nhóm vào danh sách các tham số cần cập nhật
        
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)  
        super().__init__(param_groups, defaults)  # Khởi tạo Optimizer
    
    @torch.no_grad()  # Không tính gradient trong hàm step
    def step(self):
        for group in self.param_groups:  # Lặp qua từng nhóm tham số
            update_buffer:            Tensor  = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params:              list[Tensor] = group["params"]
            
            # VÒNG LẶP CHÍNH này chặt đều params list vào các GPUs (world_size)
            for base_i in range(len(params))[::self.world_size]:  # Lặp qua tham số với bước bằng world_size
            # => với world_size = 1 thì base_i đi qua từng tham số 1

                param_idx = base_i + self.rank
                if param_idx >= len(params): # Không bao giờ xảy ra nếu world_view = 1
                    # Sử dụng view của buffer nếu không có tham số tương ứng
                    g = update_buffer_views[self.rank] # sử dụng luôn vì đã được tính ở GPU khác

                else: # param_id hợp lệ
                    p = params[param_idx]  # Lấy tham số tương ứng với rank hiện tại
                    g = p.grad  # Lấy gradient của tham số
                    assert g is not None  # Đảm bảo gradient tồn tại
                    
                    state = self.state[p]  # Lấy trạng thái của tham số
                    if "momentum_buffer" not in state:  # Kiểm tra buffer momentum đã tồn tại chưa
                        state["momentum_buffer"] = torch.zeros_like(g)  # Tạo buffer momentum mới nếu chưa có
                    
                    buf: Tensor = state["momentum_buffer"]  # Lấy buffer momentum
                    buf.lerp_(g, 1 - group["momentum"])  # Cập nhật buffer momentum
                    g = g.lerp_(buf, group["momentum"])  # Áp dụng momentum vào gradient

                    if g.ndim == 4: g = g.view(len(g), -1) # Handle conv filters
                    g = newtonschulz(g, steps=group["ns_steps"]).flatten()

                # Gom gradient từ các quá trình nếu đang phân tán
                if self.is_dist:
                    handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                    handle.wait() # Đợi hoạt động bất đồng bộ hoàn thành nếu đang phân tán
                else: update_buffer_views[0].copy_(g.flatten().to(torch.bfloat16)) # copy thẳng nếu chỉ có 1 GPU

                # Update các tham số trong world hiện tại, tới đây các update_buffer_views
                # đã được tính toán phân tán rồi cập nhập tới từng GPU đầy đủ (nhờ handle.wait ở trên)
                for pw, gw in zip(params[base_i : base_i + self.world_size], update_buffer_views):
                    pw.mul_(1 - group["lr"] * group["weight_decay"])  # Áp dụng weight decay
                    # Cập nhật tham số với gradient đã xử lý
                    a = -group["lr"] * max(1, pw.size(-2) / pw.size(-1))**0.5
                    pw.add_(gw.view_as(pw), alpha=a)
# Muon = torch.compile(Muon)


class Muon1GPU(torch.optim.Optimizer):
    ''' Viết lại Muon cho 1 GPU, bỏ distributed code cho dễ hiểu '''
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, ns_steps=5, **args):
        super().__init__(list(params), dict(lr=lr, wd=weight_decay, mm=momentum, ns=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:               # với mỗi tham số p trong model
                if p.grad is None: continue         # bỏ qua nếu không có gradient

                g, st = p.grad, self.state[p]       # lấy gradient và optim state
                if 'mm' not in st:                  # khởi tạo momentum nếu chưa có
                    st['mm'] = torch.zeros_like(g)  # (g, dtype=torch.bfloat16)

                st['mm'].lerp_(g, 1 - group['mm'])  # Áp dụng momentum vào gradient
                g = g.lerp_(st['mm'], group['mm'])  # tương đương với 2 phép tính:
                    # 1) momentum_state = momentum_state * 0.95 + gradient * 0.05
                    # 2) final_gradient = gradient * 0.05 + momentum_state * 0.95

                if g.ndim != 2: g = g.view(len(g), -1)  # 2D hoá
                g = newtonschulz(g, steps=group['ns'])  # Trực giao Newton-Schulz
                if g.shape != p.shape: g = g.view_as(p) # Reshape back if needed

                # Cập nhật tham số p, theo gradient, learning rate và weight decay với 2 phép tính:
                p.mul_(1 - group['lr']*group['wd'])  # 1) p *= (1 - lr*wd) <= thu nhỏ p nếu wd > 0
                rows, cols = p.size(-2), p.size(-1)  # 2) p -= g * lr * sqrt(max(1, rows / cols))
                p.add_(g, alpha=-group['lr']*max(1, rows/cols)**0.5)

# test muon
if __name__ == "__main__":
    import math, copy, torch
    torch.manual_seed(0)
    device = "cuda"

    # ------------------------------------------------------------------
    # 1) Mô hình: toàn bộ tham số đều có rank ≥ 2
    # ------------------------------------------------------------------
    class ToyNet2D(torch.nn.Module):
        def __init__(self, hidden=128, mlp_dim=256, n_classes=10):
            super().__init__()
            # 2 block Conv (không bias) → ReLU
            self.conv_stack = torch.nn.Sequential(
                torch.nn.Conv2d(16, hidden, 3, padding=1, bias=False),  # weight: 4-D
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
                torch.nn.ReLU(inplace=True),
            )
            # MLP head (toàn Linear bias=False → weight 2-D)
            self.head = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(hidden * 8 * 8, mlp_dim, bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(mlp_dim, n_classes, bias=False),
            )
        def forward(self, x):
                return self.head(self.conv_stack(x))

    # -------------------------------------------------
    # 2) Khởi tạo hai bản sao mô hình để test song song
    # -------------------------------------------------
    model_a = ToyNet2D().to(device)
    model_b = copy.deepcopy(model_a).to(device)
    model_c = copy.deepcopy(model_a).to(device)

    # -------------------------------------------------
    # 3) Tạo dữ liệu ngẫu nhiên
    # -------------------------------------------------
    bs = 2; steps = 5
    data = [torch.randn(bs, 16, 8, 8, device=device) for _ in range(steps)]
    target = [torch.randint(0, 10, (bs,), device=device) for _ in range(steps)]
    criterion = torch.nn.CrossEntropyLoss()

    # -------------------------------------------------
    # 4) Khởi tạo optimizer (dùng siêu tham số chung)
    # -------------------------------------------------
    common = dict(lr=2e-2, weight_decay=1e-2, momentum=0.95, ns_steps=5, rank=0, world_size=1,)
    opt_a  = MuonOrigin(model_a.parameters(), **common)
    opt_b  = Muon(model_b.parameters(), **common)
    opt_c  = Muon1GPU(model_c.parameters(), **common)

    # -------------------------------------------------
    # 5) train 2 mô hình steps
    # -------------------------------------------------
    def train(opt, model):
        for s in range(steps):
            torch.manual_seed(1234) # reseed so dropout mask identical
            opt.zero_grad()
            loss = criterion(model(data[s]), target[s])
            loss.backward()
            opt.step()
        return loss

    loss_a = train(opt_a, model_a)
    loss_b = train(opt_b, model_b)
    loss_c = train(opt_c, model_c)

    # -------------------------------------------------
    # 6) So sánh sai khác trọng số
    # -------------------------------------------------
    with torch.no_grad():
        _ab = [ (x - y).abs().max().item() for x, y in zip(model_a.parameters(), model_b.parameters()) ]
        _ac = [ (x - y).abs().max().item() for x, y in zip(model_a.parameters(), model_c.parameters()) ]
        _bc = [ (x - y).abs().max().item() for x, y in zip(model_b.parameters(), model_c.parameters()) ]

    print(f"a) Loss {loss_a:.6f} for {opt_a.__class__.__name__}")
    print(f"b) Loss {loss_b:.6f} for {opt_b.__class__.__name__}")
    print(f"c) Loss {loss_c:.6f} for {opt_c.__class__.__name__}")
    print(f"""Max weight difference after {steps} steps:
* ab {max(_ab):.4e}
* ac {max(_ac):.4e}
* bc {max(_bc):.4e}
""")
