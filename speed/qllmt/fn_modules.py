import torch
from qllmt.functional.quantization import quantize_simulate, quantize_fp8
from qllmt.functional.hadamard import left_had, right_had, get_hadK, matmul_hadU_cuda
import qllmt

def _apply_had(x, side, transpose=False):
    if side == 'left':
        return left_had(x, transpose=transpose)
    elif side == 'right':
        return right_had(x, transpose=transpose)
    elif side == 'both':
        return right_had(left_had(x, transpose=transpose), transpose=transpose)
    else:
        assert side == 'none', f"Unsupported hadamard side: {side}"
        return x


def _cancel_matmul_hads(out, arg1_mode, arg2_mode, arg1_transposed=False, arg2_transposed=False):
    # apply left transposed hadamard if needed
    left_had_needed = (not arg1_transposed and arg1_mode in ['left', 'both']) or (
            arg1_transposed and arg1_mode in ['right', 'both'])
    out = left_had(out, transpose=True) if left_had_needed else out

    # apply right transposed hadamard if needed
    right_had_needed = (not arg2_transposed and arg2_mode in ['right', 'both']) or (
            arg2_transposed and arg2_mode in ['left', 'both'])
    out = right_had(out, transpose=True) if right_had_needed else out

    # throw an error if a hadamard is stuck in between the two operands
    arg1_mid_had = (not arg1_transposed and arg1_mode in ['right', 'both']) or (
            arg1_transposed and arg1_mode in ['left', 'both'])
    arg2_mid_had = (not arg2_transposed and arg2_mode in ['left', 'both']) or (
            arg2_transposed and arg2_mode in ['right', 'both'])
    assert arg1_mid_had == arg2_mid_had, 'Hadamards are stuck in between the two operands!'

    # return the clean output
    return out


def _matmul_hq_simulate(mat1, mat2, transpose_mat1, transpose_mat2, had_mat1_mode, had_mat2_mode, quant_mat1_mode, quant_mat2_mode):
    # apply necessary hadamard transforms
    had_mat1 = _apply_had(mat1, had_mat1_mode)
    had_mat2 = _apply_had(mat2, had_mat2_mode)

    # apply simulated quantization
    q_had_mat1 = quantize_simulate(had_mat1, quant_mat1_mode)
    q_had_mat2 = quantize_simulate(had_mat2, quant_mat2_mode)

    # compute hadamarded output
    had_out = torch.mm(
        q_had_mat1.mT if transpose_mat1 else q_had_mat1,
        q_had_mat2.mT if transpose_mat2 else q_had_mat2
    )

    # finalize output by cancelling the hadamards
    out = _cancel_matmul_hads(
        had_out,
        arg1_mode=had_mat1_mode,
        arg2_mode=had_mat2_mode,
        arg1_transposed=transpose_mat1,
        arg2_transposed=transpose_mat2
    )
    return out


@torch.compile(dynamic=True)
def contiguous_transpose(x):
    return x.T.contiguous()


@torch.compile(dynamic=True)
def cq_transpose(x):
    dtype = torch.float8_e4m3fn
    scale = torch.norm(x, float('inf')) / torch.finfo(dtype).max
    x = x / scale
    qx = x.to(dtype)
    qx = qx.T.contiguous()
    return qx, scale


class LinearFn(torch.autograd.Function):
    """
    Y=XWt
    Ex=EyW
    Gw=ExtX
    """

    @staticmethod
    @torch.no_grad()
    def forward(ctx, x, w, hq_config, layer_name=None):
        assert x.dtype == w.dtype == torch.bfloat16, f'Only bfloat16 inputs are supported for now. x.dtype={x.dtype}, w.dtype={w.dtype}'
        ctx.save_for_backward(x, w)
        ctx.layer_name = layer_name
        return x @ w.t()

    @staticmethod
    @torch.no_grad()
    def backward(ctx, ey):
        assert ey.dtype == torch.bfloat16, f'Only bfloat16 inputs are supported for now. ey.dtype={ey.dtype}'
        x, w = ctx.saved_tensors
        layer_name = ctx.layer_name
        ex = gw = None
        if ctx.needs_input_grad[0]:
            ex = ey @ w
        if ctx.needs_input_grad[1]:
            gw = ey.t() @ x
        return ex, gw, None, None


@torch.compile(dynamic=True)
def per_tensor_int8_quant_v4(x):
    scale = ((x.abs().max())) / 127
    return (x / scale).round().to(torch.int8), scale


class QTrainLinearFrozen(torch.autograd.Function):

    @staticmethod
    def quantize(x, ):
        return per_tensor_int8_quant_v4(x, )

    @staticmethod
    @torch.no_grad()
    def forward(ctx, x, W, qh_config):
        ctx.save_for_backward(x, W)

        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        hadK, K = get_hadK(x_shape[-1], device=x.device, dtype=x.dtype)
        xH = matmul_hadU_cuda(x, hadK, K)

        hadK, K = get_hadK(W.shape[-1], device=x.device, dtype=x.dtype)
        WH = matmul_hadU_cuda(W, hadK, K)
        qWH, qWH_scale = QTrainLinearFrozen.quantize(WH)

        qxH, qxH_scale = QTrainLinearFrozen.quantize(xH, )
        xWt = qllmt.linear_a8_w8_bfp32_ofp32(qxH, qWH, qxH_scale * qWH_scale).to(x.dtype)

        return xWt.view(*x_shape[:-1], -1).to(x.dtype)

    @staticmethod
    @torch.no_grad()
    def backward(ctx, Ey):
        x, W = ctx.saved_tensors

        Ey_shape = Ey.shape
        Ey = Ey.view(-1, Ey_shape[-1])

        Ex = None
        grad_qHWH = grad_qHWH_scale = None

        # Gradient for input
        if ctx.needs_input_grad[0]:
            # HtG_y
            # next power of 2 after Ey.shape[0]
            Ey_row_padding = (1 << (Ey.shape[0] - 1).bit_length()) - Ey.shape[0]
            Ey = torch.cat([Ey, torch.zeros(Ey_row_padding, Ey.shape[1], device=Ey.device, dtype=Ey.dtype)],
                           dim=0).contiguous()
            hadK, K = get_hadK(Ey.shape[0], device=Ey.device, dtype=Ey.dtype)
            HtEy = matmul_hadU_cuda(Ey.T.contiguous(), hadK, K).T.contiguous()

            # wandb.log({
            #     "HtEy": wandb.Histogram(HtEy[:Ey.shape[0]-Ey_row_padding, :].cpu().numpy()),
            #     "Ey": wandb.Histogram(Ey[:Ey.shape[0]-Ey_row_padding, :].cpu().numpy()),
            # })

            qW, qW_scale = QTrainLinearFrozen.quantize(W, )
            qHtEy, qHtEy_scale = QTrainLinearFrozen.quantize(HtEy, )
            HtEyW = qllmt.linear_a8_w8_bfp32_ofp32(qHtEy, qW.T.contiguous(), qHtEy_scale * qW_scale)
            hadK, K = get_hadK(HtEyW.shape[0], device=HtEyW.device, dtype=HtEyW.dtype)
            EyW = matmul_hadU_cuda(HtEyW.T.contiguous(), hadK, K, transpose=True).T.contiguous()
            Ex = EyW
            Ex = Ex[:Ey.shape[0] - Ey_row_padding, :]
        Ex = Ex.to(Ey.dtype)
        Ex = Ex.view(*Ey_shape[:-1], -1)

        return Ex, None, None
