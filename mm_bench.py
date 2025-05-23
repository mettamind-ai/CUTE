import argparse
import pandas as pd
from triton.testing import do_bench

import torch
import triton
import triton.language as tl
from torch import Tensor

lib = torch.library.Library("mm_bench", "DEF")
lib_ops = torch.ops.mm_bench

# https://github.com/pytorch/pytorch/blob/c2e2602ecdc2ec1f120e19198dfc18fc39f7bd09/torch/_inductor/kernel/mm.py
from optimus import cfgs, _grid
@triton.autotune(configs=cfgs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, 
    stride_bn, stride_cm, stride_cn,
    ACC_DTYPE: tl.constexpr, EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr = 8,
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


def bench_f(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")

def to_layout(x: torch.Tensor, column_major: bool):
    return x.T.contiguous().T if column_major else x.contiguous()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_column_major", action="store_true")
    parser.add_argument("--b_column_major", action="store_true")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    data = []
    sizes = [1024, 2048, 1024*3, 4096, 1024*6, 1024*8]

    for sz in sizes:
        print(f"M=N=K={sz}")

        A_bf16 = torch.randn(sz, sz).bfloat16()
        B_bf16 = torch.randn(sz, sz).bfloat16()

        A_f16 = torch.randn(sz, sz).half()
        B_f16 = torch.randn(sz, sz).half()

        A_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8)
        B_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8)

        A_f8 = torch.randn(sz, sz).to(torch.float8_e4m3fn)
        B_f8 = torch.randn(sz, sz).to(torch.float8_e4m3fn)

        A_bf16, A_f16, A_i8 = [to_layout(x, args.a_column_major) for x in [A_bf16, A_f16, A_i8]]
        B_bf16, B_f16, B_i8 = [to_layout(x, args.b_column_major) for x in [B_bf16, B_f16, B_i8]]

        bf16_time       = bench_f(torch.mm,      A_bf16, B_bf16)
        i8_pytorch_time = bench_f(torch._int_mm, A_i8,   B_i8)
        i8_triton_time  = bench_f(int8_mm,       A_i8,   B_i8)
        torch.testing.assert_close(torch._int_mm(A_i8, B_i8), int8_mm(A_i8, B_i8))

        if torch.cuda.get_device_capability() >= (8, 9):
            f8_triton_time = bench_f(_triton_mm, A_f8, B_f8, torch.bfloat16, torch.float32)
        else: f8_triton_time = float("inf") 
        f16_acc_f16_triton_time = bench_f(_triton_mm, A_f16, B_f16, torch.float16, torch.float16)

        data.append(
            [
                bf16_time / i8_pytorch_time,
                bf16_time / i8_triton_time,
                bf16_time / f8_triton_time,
                bf16_time / f16_acc_f16_triton_time,
            ]
        )

    df = pd.DataFrame(
        data, index=sizes,
        columns=[
            "CuBLAS INT8",
            "Triton INT8",
            "Triton FP8",
            "Triton FP16",
        ],
    )
    print(df.round(2).T.to_markdown())
