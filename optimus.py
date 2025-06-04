#!/usr/bin/env python3
''' TẬP HỢP CODE TỐI ƯU ĐỂ TRAIN LLM TRÊN GAMING GPU (30xx, 40xx, 50xx)
INT8 Mixed Precision modded from github.com/gau-nernst/quantized-training
Muon optimizer modded from github.com/nil0x9/flash-muon
Fused CE modded from https://github.com/linkedin/Liger-Kernel
'''
import functools, torch, triton, os
import triton.language as tl, torch.distributed as dist
import torch.nn.functional as F, torch.utils._pytree as pytree

from typing import NamedTuple
from torch import Tensor, nn

aten = torch.ops.aten
lib = torch.library.Library("qtrain", "DEF")
lib_ops = torch.ops.qtrain

##########################
##  INT8 Triton Matmul  ##
##########################

cfgs, _grid = [ # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
    (128, 128,  32, 4, 4), (128,  64,  32, 4, 4), ( 64, 128,  32, 4, 4), (128,  32,  32, 4, 4),
    (128, 128, 128, 4, 4), (128,  64,  64, 4, 4), ( 64, 128,  64, 4, 4), (128,  32,  64, 4, 4),
    ( 64,  64,  32, 2, 4), (128, 128,  32, 2, 8), ( 64, 128,  32, 4, 8), (128,  64,  32, 4, 8),
    (128, 128,  16, 2, 8), ( 64,  64,  16, 2, 4), (256, 128,  64, 4, 8), (128, 256,  64, 4, 8),
    (128, 128,  64, 3, 8), ( 64,  64,  64, 3, 4), ( 32, 128,  32, 2, 4), (128,  32,  32, 2, 4),
], lambda meta: ( triton.cdiv(meta["M"], meta["BLOCK_M"])*triton.cdiv(meta["N"], meta["BLOCK_N"]), )
cfgs = [triton.Config(dict(BLOCK_M=m, BLOCK_N=n, BLOCK_K=k), num_stages=s, num_warps=w) for m, n, k, s, w in cfgs]

@triton.autotune(configs=cfgs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _scaled_mm_kernel(
    A_ptr, B_ptr, C_ptr, A_scale_ptr, B_scale_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr = 8,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(K, 0, -BLOCK_K):
        a, b = tl.load(A), tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    A_scale = tl.load(A_scale_ptr + idx_m, mask=idx_m < M)
    B_scale = tl.load(B_scale_ptr + idx_n, mask=idx_n < N)
    acc = acc.to(tl.float32) * A_scale * B_scale

    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


lib.define("scaled_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")
def scaled_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    """Low-bit matmul tensor cores. `scale_A` and `scale_B` are quantization scales for A and B. E.g.
    - if `X` is quantized with tile shape (128, 64), `scale_X`'s shape will be `(X.shape[0] / 128, X.shape[1] / 64)`.
    - if `X` is row-wise quantized, `scale_X`'s shape will be `(X.shape[0], 1)`.
    """
    assert A.dtype == B.dtype == torch.int8
    assert scale_A.dtype == scale_B.dtype
    assert A.ndim == B.ndim == scale_A.ndim == scale_B.ndim == 2
    assert A.shape[1] == B.shape[0]

    # row-scale + col-scale or row-scale + tensor-scale
    is_row_scale_A     = scale_A.shape == ( A.shape[0], 1 )
    is_col_scale_B     = scale_B.shape == ( 1, B.shape[1] )
    is_tensor_scale_B  = scale_B.shape == ( 1,          1 )

    assert is_row_scale_A and ( is_col_scale_B or is_tensor_scale_B )
    assert scale_A.is_contiguous() and scale_B.is_contiguous()
    return lib_ops.scaled_mm(A, B, scale_A, scale_B)


@torch.library.impl(lib, "scaled_mm", "Meta")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=scale_A.dtype)

@torch.library.impl(lib, "scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, row_scale_A: Tensor, col_scale_B: Tensor):
    M, K = A.shape
    _, N = B.shape

    C = torch.empty(M, N, device=A.device, dtype=row_scale_A.dtype)
    _scaled_mm_kernel[_grid](A, B, C, row_scale_A, col_scale_B, M, N, K, *A.stride(), *B.stride(), *C.stride(),)
    return C

@torch.no_grad()
def quantize_int8(tensor, dim=1, eps=1e-12, sr=False):
    ''' absmax symmetric quantization, clip(cận_dưới_eps) tránh chia cho 0 '''
    scale  = tensor.abs().amax(dim, keepdim=True) / 127
    tensor = tensor.float() / scale.float().clip(eps)
    if sr:   tensor = (tensor + torch.rand_like(tensor)).floor()
    else:    tensor.round_()    # ^^^ stochastic rounding ^^^^
    tensor = tensor.clip(-128, 127).to(torch.int8)
    return ( tensor, scale )

##############################################
##  INT8 Mixed Precision for Linear Module  ##
##############################################

class Int8MixedLinear(torch.autograd.Function):
    @staticmethod
    def forward(inp:Tensor, weight, bias=None):
        assert bias is None
        # Do dùng sample packing (varlen) nên input luôn là ma trận 2 chiều
        A, B  = inp, weight._data.T
        A, As = quantize_int8(A, dim=1, sr=False)
        B, Bs = quantize_int8(B, dim=0, sr=True)  # rounding ma trận nhỏ có
        return scaled_mm(A, B, As, Bs,)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inp, weight, bias = inputs
        assert bias is None
        ctx.save_for_backward(inp, weight._data)
        ctx.bias = False

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_weight = grad_bias = None 

        ## Grad truyền tiếp về layer sau, nên cần độ chính xác cao
        # grad_input = grad_output @ weight # phép nhân nguyên bản
        A, B  = grad_output, weight
        A, As = quantize_int8(A, dim=1, sr=True) # rounding để đạt độ ...
        B, Bs = quantize_int8(B, dim=0, sr=True) # ... chính xác cao hơn
        grad_input = scaled_mm(A, B, As, Bs,)

        if ctx.needs_input_grad[1]:
            ## grad_weight = grad_output.T @ inp; cả 2 là activation nên rất lớn
            ## Áp dụng INT8 matmul ở đây là lợi nhất; sr=False để tránh OOM và max speed
            A, B  = grad_output.T, inp
            A, As = quantize_int8(A, dim=1, sr=False) # không cần round vì grad ko truyền tiếp
            B, Bs = quantize_int8(B, dim=0, sr=False) # ... nó được update thẳng vào weight
            grad_weight = scaled_mm(A, B, As, Bs,)

        if ctx.needs_input_grad[2] and ctx.bias: grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


## Dùng lớp này để gói Linear weight, giúp torch.compile build graph?
class MixedPrecisionLinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor): return Tensor._make_wrapper_subclass(cls, data.shape, device=data.device,)
    @torch._dynamo.disable
    def __init__(self, data: Tensor): self._data = data
    def __tensor_flatten__(self): return ["_data"], []
    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None): return cls(tensor_data_dict["_data"])
    def __repr__(self): return f"{self.__class__.__name__}(data={self._data})"
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()
        if func is F.linear: return Int8MixedLinear.apply(*args, **kwargs)
        with torch._C.DisableTorchFunctionSubclass(): return func(*args, **kwargs)
    @classmethod # Adapted from FP8 implementation of WeightWithDynamicFloat8CastTensor
    def __torch_dispatch__(cls, func, types, args, kwargs):
        def unwrap(x: cls): return x._data
        out = func(*pytree.tree_map_only(cls, unwrap, args), **pytree.tree_map_only(cls, unwrap, kwargs),)
        others = { aten.t.default, aten.detach.default, aten.empty_like.default, 
                   aten.new_zeros.default, aten.slice.Tensor, aten.view.default, aten.as_strided.default, 
                   aten._to_copy.default, aten._pin_memory.default, aten.split.Tensor, aten.clone.default,}
        if func is aten.copy_.default: return args[0] # original object
        elif func in others: return pytree.tree_map_only(Tensor, lambda x: cls(x), out) # new wrapped object
        else: return out # new unwrapped object

import re
def convert_int8_mixed_precision(module:nn.Module, ignore='head'):
    ignore = re.compile(rf'{ignore}')
    names, params = [], 0
    for n, m in module.named_modules():
        if isinstance(m, nn.Linear) and not ignore.search(n): 
            names.append(n)            
            params  += m.weight.numel()
            m.weight = nn.Parameter(MixedPrecisionLinearWeight(m.weight.detach()), requires_grad=m.weight.requires_grad,)
    return names, params

###################################
##  Fused Chunked Cross Entropy  ##
###################################

@triton.jit
def liger_cross_entropy_kernel(
    X_ptr, X_stride,                # input tensor.
    Y_ptr, Y_stride,                # đang tính LCE cho label Y này.
    loss_ptr, loss_stride,          # để lưu loss của label.
    n_cols,                         # (int):   The number of columns in the input tensor.
    n_non_ignore,                   # (float): The number of non-ignored elements in the batch.
    ignore_index,                   # (int):   The index to ignore in the target (-100.)
    lse_square_scale: tl.constexpr, # (float): The scaler of (logsumexp(_input))^2 adding to the loss for training stability
    label_smoothing:  tl.constexpr, # (float): 0.0 means no smoothing.
    BLOCK_SIZE:       tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr     += program_id * X_stride
    Y_ptr     += program_id * Y_stride
    loss_ptr  += program_id * loss_stride

    y = tl.load(Y_ptr)
    if y == ignore_index: # grad == 0 => set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + offs, 0.0, mask=offs  < n_cols)
        return # loss đã đc set = 0 trước nên không cần gán

    m = float("-inf")                              # m is the max value. use the notation from the paper
    d = 0.0; scaled_x_sum = 0.0                    # d is the sum. use the notation from the paper
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)  # we need to store the original value of X_y for the loss calculation
    eps = label_smoothing / n_cols

    ## First pass: Tìm giá trị lớn nhất m và tính tổng exponential
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        X = tl.load(X_ptr + offs, mask=offs<n_cols, other=float("-inf"),).cast(tl.float32)
        if eps > 0: scaled_x_sum += tl.sum(tl.where(offs<n_cols, -eps*X, 0.0))
        m_new = tl.maximum(m, tl.max(X))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X - m_new))
        m = m_new

    ## Second pass: Tính softmax và gradient
    lse = m + tl.log(d)
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        X = tl.load(X_ptr + offs, mask=offs<n_cols, other=float("-inf"),).cast(tl.float32)
        X = tl.exp(X - m) / d                                       # softmax(x_i)
        X = X + 2*lse_square_scale*lse*X - eps                      # smoothing term
        X = tl.where(offs != y, X, X-1+label_smoothing)             # gradient 
        tl.store(X_ptr + offs, X / n_non_ignore, mask=offs<n_cols)  # mean reduction
    tl.debug_barrier()  # to ensure the new result of X_ptr is written

    loss = lse - ori_X_y
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss
    if lse_square_scale > 0: loss += lse_square_scale*lse*lse # z_loss is an auxiliary loss
    tl.store(loss_ptr, loss/n_non_ignore)                     # mean reductiom


MAX_FUSED_SIZE = 65536 // 2
class FusedLinearCrossEntropy(torch.autograd.Function):
    """ Ref https://github.com/mgmalek/efficient_cross_entropy. Vì Cross Entropy Loss là layer cuối,
    TA CÓ THỂ TÍNH GRADIENT NGAY TRONG FORWARD PASS. Nhờ đó không cần lưu _input và target cho backward pass. """
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, _input, weight, target, n_non_ignore=None, ignore_index=-100, lse_square_scale=0.0, label_smoothing=0.0):
        loss_1d = torch.zeros(_input.shape[0], dtype=torch.float32, device=_input.device)

        A, As = quantize_int8(_input, dim=1, sr=False)
        B, Bs = quantize_int8(weight.t(), dim=0, sr=False)
        logits = scaled_mm(A, B, As, Bs,)
        
        V = weight.shape[0]
        liger_cross_entropy_kernel[(logits.shape[0],)](
            X_ptr=logits, X_stride=logits.stride(-2),
            Y_ptr=target, Y_stride=target.stride(-1),          # always 1
            loss_ptr=loss_1d, loss_stride=loss_1d.stride(-1),  # always 1
            n_cols=V, n_non_ignore=n_non_ignore, ignore_index=ignore_index,
            lse_square_scale=lse_square_scale, label_smoothing=label_smoothing,
            BLOCK_SIZE=min(MAX_FUSED_SIZE, triton.next_power_of_2(V)),
            num_warps=32 if torch.version.hip is None else 16,
        )
        grad_input  = ( logits     @ weight ).detach()
        # grad_weight = ( logits.t() @ _input ).detach()
        if weight.requires_grad:
                A, As = quantize_int8(logits.t(), dim=1, sr=False)
                B, Bs = quantize_int8(_input, dim=0, sr=False)
                grad_weight = scaled_mm(A, B, As, Bs,).detach()
        else:   grad_weight = None

        ctx.save_for_backward(grad_input, grad_weight)
        return torch.sum(loss_1d)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        grad_input = grad_input * grad_output
        if grad_weight is not None:
            grad_weight = grad_weight * grad_output
        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None


#################################################################
##  Muon Optimizer - MomentUm Orthogonalized by Newton-schulz  ##
#################################################################

@torch.compile()
def zeropower_via_newtonschulz5(X:Tensor) -> Tensor:
    need_invert = X.size(-2) > X.size(-1)
    if need_invert: X = X.mT                         # Ensure số cột ≥ số hàng; giúp NS hoạt động tốt
    X /= (X.norm(dim=(-2, -1), keepdim=True) + 1e-7) # Ensure spectral norm ≤ 1, điều kiện bắt buộc để NS hội tụ
    for a,b,c in [(3.4445,-4.7750,2.0315)]*5:
        A = X @ X.mT
        X = a*X + (b*A + c*A@A) @ X
    return X.mT if need_invert else X

class Muon1GPU(torch.optim.Optimizer):
    ''' Viết lại Muon cho 1 GPU, bỏ distributed code cho dễ hiểu '''
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, **args):
        super().__init__(list(params), dict(lr=lr, wd=weight_decay, mm=momentum))

    @torch.no_grad()
    @torch.compiler.disable
    def step(self):
        for group in self.param_groups:
            for p in group['params']:                           # với mỗi tham số p trong model
                if p.grad is None: continue                     # bỏ qua nếu không có gradient

                g, st = p.grad, self.state[p]                   # lấy gradient và optim state và khởi tạo momentum nếu chưa có
                if 'mm' not in st: st['mm'] = torch.zeros_like(g, dtype=torch.bfloat16)

                st['mm'].lerp_(g, 1 - group['mm'])              # momentum = momentum * 0.9 + gradient * 0.1
                g = g.lerp_(st['mm'], group['mm'])              # gradient = gradient * 0.1 + momentum * 0.9

                if g.ndim != 2: g = g.view(len(g), -1)          # 2D hoá
                g = zeropower_via_newtonschulz5(g.bfloat16())   # Trực giao Newton-Schulz
                if g.shape != p.shape: g = g.view_as(p)         # Reshape back if needed

                # Cập nhật tham số p, theo gradient, learning rate và weight decay với 2 phép tính:
                p.mul_(1 - group['lr']*group['wd'])  # 1) p *= (1 - lr*wd) <= thu nhỏ p nếu wd > 0
                rows, cols = p.size(-2), p.size(-1)  # 2) p -= g * lr * sqrt(max(1, rows / cols))
                p.add_(g, alpha=-group['lr']*max(1, rows/cols)**0.5)
