# Copyright (c) 2024, Tri Dao, Albert Gu.

import torch

import triton
import triton.language as tl

@triton.autotune(configs=[ triton.Config({'BLOCK_N': x}) for x in [32, 64, 128, 256, 512, 1024] ], key=['ncols'])
@triton.jit
def _swiglu_fwd_kernel(
    X, Y, OUT,
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_out_row,
    ncols, BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N

    X += row * stride_x_row
    Y += row * stride_y_row
    OUT += row * stride_out_row

    cols = start_col + tl.arange(0, BLOCK_N)
    mask = cols < ncols

    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=mask, other=0.).to(tl.float32)

    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=mask)


def _swiglu_fwd(xy):
    if xy.stride(-1) != 1: xy = xy.contiguous()
    batch_shape = xy.shape[:-1]

    xy   = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)

    out = torch.empty_like(x)
    assert out.stride(-1) == 1

    M, N = x.shape
    grid = lambda META: (M, triton.cdiv(N, META['BLOCK_N']))

    with torch.cuda.device(x.device.index):
        _swiglu_fwd_kernel[grid](
            x, y, out,
            x.stride(0), y.stride(0), out.stride(0),
            ncols = N,
        )
    return out.reshape(*batch_shape, out.shape[-1])


@triton.autotune(configs=[ triton.Config({'BLOCK_N': x}) for x in [32, 64, 128, 256, 512, 1024] ], key=['ncols'])
@triton.jit
def _swiglu_bwd_kernel(
    X, Y,
    DOUT, DX, DY,
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_dout_row,
    stride_dx_row,
    stride_dy_row,
    ncols, BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N

    X += row * stride_x_row
    Y += row * stride_y_row

    DOUT += row * stride_dout_row
    DX += row * stride_dx_row
    DY += row * stride_dy_row

    cols = start_col + tl.arange(0, BLOCK_N)
    mask = cols < ncols

    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=mask, other=0.).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=mask, other=0.).to(tl.float32)

    x_sigmoid = tl.sigmoid(x)
    dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
    dy = x * x_sigmoid * dout

    tl.store(DX + cols, dx, mask=mask)
    tl.store(DY + cols, dy, mask=mask)


def _swiglu_bwd(xy, dout):
    if xy.stride(-1) != 1: xy = xy.contiguous()
    if dout.stride(-1) != 1: dout = dout.contiguous()

    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)

    dout = dout.reshape(-1, dout.shape[-1])
    assert dout.shape == x.shape

    dxy = torch.empty_like(xy)
    dx, dy = dxy.chunk(2, dim=-1)
    assert dx.stride(-1) == 1 and dy.stride(-1) == 1

    M, N = x.shape
    grid = lambda META: (M, triton.cdiv(N, META['BLOCK_N']))

    with torch.cuda.device(x.device.index):
        _swiglu_bwd_kernel[grid](
            x, y, dout, dx, dy,
            x.stride(0), y.stride(0), dout.stride(0),
            dx.stride(0), dy.stride(0), ncols = N,
        )
    return dxy.reshape(*batch_shape, dxy.shape[-1])


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy):
        ctx.save_for_backward(xy)
        return _swiglu_fwd(xy)

    @staticmethod
    def backward(ctx, dout):
        xy, = ctx.saved_tensors
        return _swiglu_bwd(xy, dout)

swiglu = SwiGLU.apply
