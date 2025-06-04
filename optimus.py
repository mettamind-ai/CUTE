#!/usr/bin/env python3
''' TẬP HỢP CODE TỐI ƯU ĐỂ TRAIN LLM TRÊN GAMING GPU (30xx, 40xx, 50xx)
INT8 Mixed Precision modded from github.com/gau-nernst/quantized-training
Muon optimizer modded from github.com/nil0x9/flash-muon
Fused CE modded from https://github.com/linkedin/Liger-Kernel
'''
import functools, torch, triton, os
import triton.language as tl

from typing import NamedTuple
import torch.nn.functional as F
import torch.utils._pytree as pytree
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
    A_ptr, B_ptr, C_ptr,
    A_scale_ptr, B_scale_ptr,
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
            m.weight = nn.Parameter(
                MixedPrecisionLinearWeight(m.weight.detach()),
                requires_grad=m.weight.requires_grad,
            )
    return names, params


###################################
##  Fused Chunked Cross Entropy  ##
###################################

try: from triton.language.extra.libdevice import tanh
except ModuleNotFoundError: from triton.language.extra.cuda.libdevice import tanh

@triton.jit
def liger_cross_entropy_kernel(
    X_ptr, X_stride,        # input tensor.
    Y_ptr, Y_stride,        # đang tính LCE cho label Y này.
    loss_ptr, loss_stride,  # loss của label.
    n_cols,         #  (int): The number of columns in the input tensor.
    n_non_ignore,   #  (float): The number of non-ignored elements in the batch.
    ignore_index,   #  (int): The index to ignore in the target. (-100)
    lse_square_scale: tl.constexpr,  # (float): The scaler of (logsumexp(_input))^2 adding to the loss for training stability
    label_smoothing:  tl.constexpr,  # (float): 0.0 means no smoothing.
    BLOCK_SIZE:       tl.constexpr,
):
    """ This kernel computes both cross entropy loss and the gradient of the input. We only consider hard label + mean reduction.
        Refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.
        - Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
        Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867
        - Label smoothing is a general case of normal cross entropy
        See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    """
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride
    Y_ptr += program_id * Y_stride

    y = tl.load(Y_ptr)
    if y == ignore_index:
        # Vì ignore nên grad == 0 => set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return # loss đã đc set = 0 trước nên không cần gán

    m = float("-inf")                              # m is the max value. use the notation from the paper
    d = 0.0                                        # d is the sum. use the notation from the paper
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)  # we need to store the original value of X_y for the loss calculation

    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf"),).cast(tl.float32)

        block_max = tl.max(X_block)
        if label_smoothing > 0: # scale X beforehand to avoid overflow
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))

        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)
    for i in range(0, n_cols, BLOCK_SIZE):

        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf"),).cast(tl.float32)

        X_block = tl.exp(X_block - m) / d  ### softmax(x_i)
        X_block += 2 * lse_square_scale * lse * X_block

        X_block += -eps                    ### smoothing term
        X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))

        X_block = X_block / n_non_ignore   ### mean reduction
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)
    tl.debug_barrier()  # to ensure the new result of X_ptr is written

    loss = lse - ori_X_y
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    loss = loss / n_non_ignore                  # Normalize the loss, mean reduction
    if lse_square_scale > 0:
        z_loss = lse_square_scale * lse * lse   # z_loss is an auxiliary loss
        loss += z_loss / n_non_ignore           # Normalize the z_loss, mean reduction

    # Lưu loss value đơn (a float value) của target label
    tl.store(loss_ptr + program_id * loss_stride, loss)


MAX_FUSED_SIZE = 65536 // 2
def fused_linear_cross_entropy_forward(_input, weight, target, ignore_index=-100, lse_square_scale=0.0, label_smoothing=0.0):
    total_n_non_ignore = ( target != ignore_index ).sum().item()  # .item() that affects the speed ???
    loss_1d = torch.zeros(_input.shape[0], dtype=torch.float32, device=_input.device)
    logits = _input @ weight.t()
    V = weight.shape[0]
    ## Tính LCE cho từng label một !!! => vocab càng lớn càng chậm
    liger_cross_entropy_kernel[(logits.shape[0],)](
        X_ptr=logits, X_stride=logits.stride(-2),
        Y_ptr=target, Y_stride=target.stride(-1),          # always 1
        loss_ptr=loss_1d, loss_stride=loss_1d.stride(-1),  # always 1
        n_cols=V, n_non_ignore=total_n_non_ignore, ignore_index=ignore_index,
        lse_square_scale=lse_square_scale, label_smoothing=label_smoothing,
        BLOCK_SIZE=min(MAX_FUSED_SIZE, triton.next_power_of_2(V)),
        num_warps=32 if torch.version.hip is None else 16,
    )
    grad_input  = ( logits     @ weight ).detach()
    grad_weight = ( logits.t() @ _input ).detach() if weight.requires_grad else None
    return torch.sum(loss_1d), grad_input, grad_weight

class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, _input, weight, target, ignore_index=-100, lse_square_scale=0.0, label_smoothing=0.0):
        """ Ref https://github.com/mgmalek/efficient_cross_entropy. Vì Cross Entropy Loss là layer cuối,
        TA CÓ THỂ TÍNH GRADIENT NGAY TRONG FORWARD PASS. Nhờ đó không cần lưu _input và target cho backward pass. """
        loss, grad_in, grad_w = fused_linear_cross_entropy_forward(_input, weight, target, ignore_index, lse_square_scale, label_smoothing, )
        ctx.save_for_backward(grad_in, grad_w)
        return loss

    @staticmethod
    @torch.compile()
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output, z_loss_grad):
        del z_loss_grad  # z_loss is only for logging
        grad_input, grad_weight = ctx.saved_tensors
        grad_input = grad_input * grad_output
        if grad_weight is not None:
            grad_weight = grad_weight * grad_output
        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None


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

@torch.compile()
def newtonschulz(G: Tensor, steps: int) -> Tensor:
    # G: The gradient or momentum matrix to be orthogonalized.
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1): X = X.mT
    return X


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
