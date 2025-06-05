#!/usr/bin/env python3
''' TẬP HỢP CODE TỐI ƯU ĐỂ TRAIN LLM TRÊN GAMING GPUs (30xx, 40xx, 50xx)
- INT8 Mixed Precision modded from github.com/gau-nernst/quantized-training
- Muon optimizer modded from github.com/nil0x9/flash-muon
- Fused CE modded from https://github.com/linkedin/Liger-Kernel
'''
import functools, torch, triton, os, re
import triton.language as tl, torch.distributed as dist
import torch.nn.functional as F, torch.utils._pytree as pytree

from typing import NamedTuple
from torch import Tensor, nn

##############################################
##  INT8 Mixed Precision for Linear Module  ##
##############################################
aten = torch.ops.aten
lib = torch.library.Library("qtrain", "DEF")
lib_ops = torch.ops.qtrain

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
    rk =                   tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    A = A_ptr + (ram[:, None] * stride_am +  rk[None, :] * stride_ak)
    B = B_ptr + ( rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(K, 0, -BLOCK_K):
        acc += tl.dot(tl.load(A), tl.load(B))
        A   += BLOCK_K * stride_ak
        B   += BLOCK_K * stride_bk

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
    return lib_ops.scaled_mm(A, B, scale_A, scale_B)

@torch.library.impl(lib, "scaled_mm", "Meta")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=scale_A.dtype)

@torch.library.impl(lib, "scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, row_scale_A: Tensor, col_scale_B: Tensor):
    M, K = A.shape; _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=row_scale_A.dtype)
    _scaled_mm_kernel[_grid](A, B, C, row_scale_A, col_scale_B, M, N, K, *A.stride(), *B.stride(), *C.stride(),)
    return C

@torch.no_grad()
def quantize_int8(tensor, dim=1, eps=1e-12, sr=False):
    scale  = tensor.abs().amax(dim, keepdim=True) / 127
    tensor = tensor.float() / scale.float().clip(eps) # clip(cận_dưới_eps) tránh chia cho 0
    if sr:   tensor = (tensor + torch.rand_like(tensor)).floor()
    else:    tensor.round_()    # ^^^ stochastic rounding ^^^^
    tensor = tensor.clip(-128, 127).to(torch.int8)
    return ( tensor, scale )


class Int8MixedLinear(torch.autograd.Function):
    @staticmethod
    def forward(inp, weight, bias=None):
        A, As = quantize_int8(inp, dim=1, sr=False)
        B, Bs = quantize_int8(weight._data.T, dim=0, sr=True)
        return scaled_mm(A, B, As, Bs,)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inp, weight, _ = inputs
        ctx.save_for_backward(inp, weight._data)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_weight = grad_bias = None 

        A, As = quantize_int8(grad_output, dim=1, sr=True) # rounding để đạt độ ...
        B, Bs = quantize_int8(weight, dim=0, sr=True)      # ... chính xác cao hơn
        grad_input = scaled_mm(A, B, As, Bs,)

        if ctx.needs_input_grad[1]:
            A, As = quantize_int8(grad_output.T, dim=1, sr=False) # không cần round vì grad ko truyền tiếp
            B, Bs = quantize_int8(inp, dim=0, sr=False)           # ... nó được update thẳng vào weight
            grad_weight = scaled_mm(A, B, As, Bs,)

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


def convert_int8_mixed_precision(module:nn.Module, ignore='head'):
    ignore = re.compile(rf'{ignore}')
    names, params = [], 0
    for n, m in module.named_modules():
        if isinstance(m, nn.Linear) and not ignore.search(n): 
            names.append(n)            
            params  += m.weight.numel()
            m.weight = nn.Parameter(MixedPrecisionLinearWeight(m.weight.detach()), requires_grad=m.weight.requires_grad,)
    return names, params

############################
##  Fused  Cross Entropy  ##
############################

@triton.jit
def per_label_cross_entropy_kernel(X_ptr, X_stride, label_ptr, loss_ptr, n_non_ignore, ignore, vocab, CHUNK: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)  # chạy từ 0 tới num_labels
    X_ptr     += program_id * X_stride
    loss_ptr  += program_id

    true_label = tl.load(label_ptr + program_id)
    true_logit = tl.load(X_ptr + true_label).cast(tl.float32)

    offs = tl.arange(0, CHUNK)
    mask = (offs < vocab)
    X_ptr= X_ptr + offs

    if true_label == ignore: # logits' grad as 0     
        tl.store(X_ptr, 0.0)
    else:
        X = tl.load(X_ptr, mask=mask, other=float("-inf")).cast(tl.float32)

        m = tl.max(X, axis=0)       # the max value `m` and the sum `d` are notations in ... 
        d = tl.sum(tl.exp(X - m))   # ... the paper https://www.alphaxiv.org/abs/1805.02867

        LSE  = m + tl.log(d)        # Log-Sum-Exp, "Mức độ lớn" của tất cả logits (normalization term của softmax)
        loss = LSE - true_logit     # loss là khoảng cách mức độ lớn tổng thể và true label logit

        X = tl.exp(X - m)/d                         # softmax
        X = tl.where(offs != true_label, X, X - 1)  # gradient bị tác động bởi true_label_logit

        tl.store(X_ptr, X/n_non_ignore, mask=mask)  # mean reduction
        tl.store(loss_ptr, loss/n_non_ignore)       # mean reduction


class FusedLinearCrossEntropy(torch.autograd.Function):
    """ TÍNH GRADIENT NGAY TRONG FORWARD. Nhờ đó không cần lưu input và target cho backward """
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, _input, weight, target, n_ignores=0, ignore=-100):
        loss_1d = torch.zeros(_input.shape[0], dtype=torch.float32, device=_input.device)        
        logits  = _input @ weight.t()

        N,  V = ( logits.shape[0], logits.shape[1] ) # N là số labels, V là vocab
        ni, C = ( N-n_ignores, triton.next_power_of_2(V) )

        per_label_cross_entropy_kernel[(N,)]( X_ptr=logits, X_stride=logits.stride(-2), label_ptr=target, 
            loss_ptr=loss_1d, n_non_ignore=ni, ignore=ignore, vocab=V, CHUNK=C, num_warps=32, )

        grad_input  = ( logits     @ weight ).detach()
        grad_weight = ( logits.t() @ _input ).detach()

        ctx.save_for_backward(grad_input, grad_weight)
        return torch.sum(loss_1d)  # final loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        grad_input = grad_input * grad_output
        if grad_weight is not None:
            grad_weight = grad_weight * grad_output
        return grad_input, grad_weight, None, None, None


#################################################################
##  Muon Optimizer - MomentUm Orthogonalized by Newton-schulz  ##
#################################################################

@torch.compile()
def zeropower_via_newtonschulz5(X:Tensor) -> Tensor:
    need_invert = X.size(-2) > X.size(-1)
    if need_invert: X = X.mT                            # Ensure số cột ≥ số hàng; giúp NS hoạt động tốt
    X /= ( X.norm(dim=(-2, -1), keepdim=True) + 1e-7 )  # Ensure spectral norm ≤ 1, điều kiện bắt buộc để NS hội tụ
    a , b, c = ( 3.4445, -4.7750, 2.0315 )
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X           # 1
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X           # 2
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X           # 3
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X           # 4
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X           # 5
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
