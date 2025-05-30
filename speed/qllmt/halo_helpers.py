import os, sys, torch
import gemm_int8, qllmt

def _precision_to_dtype(precision: str):
    ''' Returns the corresponding torch dtype for the given precision (in str). '''
    if   precision.lower() == 'int8': return torch.int8
    elif precision.lower() == 'bf16': return torch.bfloat16
    elif precision.lower() == 'fp32': return torch.float32
    elif precision.lower() == 'fp16': return torch.float16
    else: raise ValueError(f'Unsupported precision: {precision}')


@torch.compile(dynamic=True)
def contiguous_transpose(x):
    return x.T.contiguous()


def _matmul_int8_transposed(mat_a, c_a, mat_b, c_b, out_prec=torch.bfloat16):
    """ Perform matrix multiplication between two 2D matrices in int8 format. """
    assert mat_a.is_contiguous() and mat_b.is_contiguous(), 'Input matrices should be contiguous'
    assert mat_a.dtype == mat_b.dtype == torch.int8, 'Only int8 inputs are supported'

    if isinstance(c_a, torch.Tensor): c_a = c_a.item()
    if isinstance(c_b, torch.Tensor): c_b = c_b.item()
    mat_out = gemm_int8.matmul(mat_a, mat_b, c_a * c_b)

    if out_prec != mat_out.dtype:
        print('WARNING: CASTING THE MATMUL OUTPUT')
        mat_out = mat_out.to(dtype=out_prec)
    return mat_out


def _matmul_kernel_by_precision(precision):
    if precision == 'int8': return _matmul_int8_transposed
    else: raise ValueError(f'Unsupported precision: {precision}')


def _quantize_fn_by_precision(precision, fake_quant=False):
    if fake_quant:
        if precision == 'fp8':
            return fake_quantize_fp8_transposable
        elif precision == 'int8':
            return fake_quantize_int8_transposable
        else:
            raise ValueError(f'Unsupported precision: {precision}')
    else:
        if precision == 'fp8':
            return quantize_fp8_tranposable
        elif precision == 'int8':
            return quantize_int8_transposable
        else:
            raise ValueError(f'Unsupported precision: {precision}')
