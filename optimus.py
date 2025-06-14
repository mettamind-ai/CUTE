#!/usr/bin/env python3
''' TẬP HỢP CODE TỐI ƯU ĐỂ TRAIN LLM TRÊN GAMING GPUs (30xx, 40xx, 50xx)
- INT8 Mixed Precision github.com/gau-nernst/quantized-training
- Muon optimizer github.com/nil0x9/flash-muon
- Chunked / fused LCE https://gist.github.com/Chillee/22cd93e11b887db1f596ab754d60a899
'''
import functools, torch, triton, os, re, time
import triton.language as tl, torch.distributed as dist
import torch.nn.functional as F, torch.utils._pytree as pytree

from typing import NamedTuple
from torch import Tensor, nn

##############################################
##  INT8 Mixed Precision for Linear Module  ##
##############################################
lib = torch.library.Library("qtrain", "DEF")
lib_ops = torch.ops.qtrain

cfgs = [triton.Config(dict(BLOCK_M=m, BLOCK_N=n, BLOCK_K=k), num_stages=s, num_warps=w) for m, n, k, s, w in \
[(128, 128, 32, 4, 4), ( 64, 128, 32, 4, 8), (128,  64, 32, 4, 8), (256, 128, 64, 4, 8), (128, 256, 64, 4, 8)]]
@triton.autotune(configs=cfgs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
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


lib.define("scaled_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")
def scaled_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    return lib_ops.scaled_mm(A, B, scale_A, scale_B)

@torch.library.impl(lib, "scaled_mm", "Meta")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=scale_A.dtype)

@torch.library.impl(lib, "scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, row_scale_A: Tensor, col_scale_B: Tensor):
    M, K = A.shape; _, N = B.shape
    assert scale_A.is_contiguous()
    assert scale_B.is_contiguous()
    C = torch.empty(M, N, device=A.device, dtype=row_scale_A.dtype)
    _grid = lambda meta: ( triton.cdiv(meta["M"], meta["BLOCK_M"])*triton.cdiv(meta["N"], meta["BLOCK_N"]), )
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
        B, Bs = quantize_int8(weight._data.T, dim=0, sr=True) # phép rounding này rẻ
        return scaled_mm(A, B, As, Bs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inp, weight, _ = inputs
        ctx.save_for_backward(inp, weight._data)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_weight = grad_bias = None 

        ## grad_input tiếp tục truyền về phía sau nên cần duy trì độ chính xác cao =>
        A, As = quantize_int8(grad_output, dim=1, sr=True) # rounding both để đạt độ
        B, Bs = quantize_int8(weight, dim=0, sr=True)      # ... chính xác cao hơn
        grad_input = scaled_mm(A, B, As, Bs)

        if ctx.needs_input_grad[1]:
            A, As = quantize_int8(grad_output.T, dim=1, sr=False) # không cần round vì grad ko truyền tiếp
            B, Bs = quantize_int8(inp, dim=0, sr=False)           # ... nó được update thẳng vào weight
            grad_weight = scaled_mm(A, B, As, Bs)

        return grad_input, grad_weight, grad_bias


''' Chuyển tiếp F.linear func call tới kernel tuỳ chỉnh (Int8MixedLinear.apply) và cho phép torch.compile
dựng biểu đồ (graph) trơn tru, không làm gián đoạn quá trình trace-&-compile của PyTorch 2.
'''
aten = torch.ops.aten
class Int8MixedLWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor): return Tensor._make_wrapper_subclass(cls, data.shape, device=data.device,)
    @torch._dynamo.disable
    def __init__(self, data: Tensor): self._data = data
    def __tensor_flatten__(self): return ["_data"], []
    def __repr__(self): return f"{self.__class__.__name__}(data={self._data})"
    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None): return cls(tensor_data_dict["_data"])
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()                           # hook vào torch_function để ...
        if func is F.linear: return Int8MixedLinear.apply(*args, **kwargs)              # 1) xử lý riêng F.linear
        with torch._C.DisableTorchFunctionSubclass(): return func(*args, **kwargs)      # 2) các hàm khác giữ nguyên
    @classmethod # Adapted from FP8 implementation of WeightWithDynamicFloat8CastTensor
    def __torch_dispatch__(cls, func, types, args, kwargs): # đảm bảo các operations khác (transpose, clone, view...) vẫn hoạt động
        def unwrap(x: cls): return x._data                  # Weight vẫn có thể được sử dụng như tensor bình thường
        out = func(*pytree.tree_map_only(cls, unwrap, args), **pytree.tree_map_only(cls, unwrap, kwargs),)
        others = { aten.t.default, aten.detach.default, aten.empty_like.default, aten.new_zeros.default, aten.slice.Tensor, aten.view.default, aten.as_strided.default, aten._to_copy.default, aten._pin_memory.default, aten.split.Tensor, aten.clone.default,}
        if func is aten.copy_.default: return args[0]       # original object
        elif func in others: return pytree.tree_map_only(Tensor, lambda x: cls(x), out) # new wrapped object
        else: return out                                    # new unwrapped object


def convert_int8_mixed_precision(module:nn.Module, ignore='head'):
    ignore = re.compile(rf'{ignore}')
    names, params = [], 0
    for n, m in module.named_modules():
        if isinstance(m, nn.Linear) and not ignore.search(n): 
            names.append(n)            
            params  += m.weight.numel()
            m.weight = nn.Parameter(                    # Tạo đối tượng param mới và làm 2 việc: 
                Int8MixedLWeight(m.weight.detach()),    # 1) đón Tensor gốc sau khi tháo rời khỏi graph
                requires_grad=m.weight.requires_grad,   # 2) gắn lại wrapper vào graph với yêu cầu grad như cũ 
            )
    return names, params

#############################
##  Chunked Cross Entropy  ##
#############################

class ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, compiled=True):
        CHUNK_SIZE = min(1024, _input.shape[0])
        def compute_loss(input_chunk, weight, target):
            logits = input_chunk @ weight.t()
            return F.cross_entropy(logits.float(), target)

        grad_weight = torch.zeros_like(weight)
        grad_inputs = []
        loss_acc = torch.zeros((), device=_input.device)

        chunks = _input.shape[0] // CHUNK_SIZE
        def accumulate_chunk(input_chunk, target_chunk):
            (chunk_grad_input, chunk_grad_weight), chunk_loss = torch.func.grad_and_value(compute_loss, argnums=(0,1))(input_chunk, weight, target_chunk)
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)
        
        input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        target_chunks = torch.chunk(target, chunks=chunks, dim=0)
        for input_chunk, target_chunk in zip(input_chunks, target_chunks):
            grad_inputs.append(accumulate_chunk(input_chunk, target_chunk))
        
        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0)/chunks,
            grad_weight/chunks
        )
        return loss_acc / chunks

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input, grad_weight) = ctx.saved_tensors
        return (grad_input, grad_weight, None, None)


###########################
##  Fused Cross Entropy  ##
###########################

@triton.jit
def per_label_cross_entropy(
        logits_ptr,            # [vocab]  — ghi đè thành ∂L/∂logit
        target_ptr, loss_ptr,  # 1 phần tử
        stride:  tl.constexpr, vocab: tl.constexpr,
        ignore:  tl.constexpr, BLOCK: tl.constexpr,
        reduction: tl.constexpr,
    ):

    pid  = tl.program_id(0).to(tl.int64)  # chạy từ 0 tới num_targets
    row  = logits_ptr + pid * stride
    offs = tl.arange(0, BLOCK)

    tgt = tl.load(target_ptr + pid)
    if tgt == ignore: tl.store(row + offs, 0.0); return

    # softmax(xi) = p(xi) = e^xi / Σ(e^xj) = e^(xi-M) / Σ(e^(xj-M))
    x    = tl.load(row + offs, mask=offs < vocab, other=-float("inf")).to(tl.float32)
    M    = tl.max(x, axis=0)
    e_x  = tl.exp(x - M)        # e^(xi-M)
    d    = tl.sum(e_x, axis=0)  # Σ(e^(xj-M))
    lse  = M + tl.log(d)        # log(Σe^logits) => (L)og-(S)um-(E)xp

    grad  = e_x / d             # p(xi) = exp(xi-M) / Σexp(xj-M)
    grad *= 1 + 2e-5 * lse      # z-loss modification
    grad  = tl.where(offs == tgt, grad - 1, grad)   # Cross-entropy gradient

    tgt_logit = tl.load(row + tgt).to(tl.float32)   # load trước khi ghi đè grad vào logits
    tl.store(row + offs, grad * reduction, mask=offs < vocab)

    loss  = lse - tgt_logit     # LCE = Surprise = -log(p_target) = -(x_target - lse)
    loss += 1e-5*lse*lse        # cộng thêm z_loss penalty giúp ổn định training
    tl.store(loss_ptr + pid, loss * reduction)                  


class FusedCE(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, _input, weight, target, n_ignores=0, ignore=-100):

        grad_weight = torch.zeros_like(weight, device=_input.device) if weight.requires_grad else None
        grad_input  = torch.empty_like(_input, device=_input.device)
        losses      = torch.zeros(_input.shape[0], device=_input.device, dtype=torch.float32)

        n_labels, vocab = _input.shape[0], weight.shape[0]
        step = min(1024*4, n_labels // 2) # để luôn test được chunked CE

        for s in range( 0, n_labels, step ):
            e = min(s + step, n_labels)
            logits = ( _input[s:e] @ weight.t() ).contiguous()
            per_label_cross_entropy[( logits.shape[0], )](
                logits_ptr  = logits,
                target_ptr  = target[s:e],
                loss_ptr    = losses[s:e],
                stride      = logits.stride(-2),
                ignore      = ignore,
                vocab       = vocab,
                BLOCK       = triton.next_power_of_2(vocab), 
                num_warps   = 16 if vocab <= 1024*8 else 32,
                reduction   = 1.0 / step,
            )
            grad_input[s:e] = logits @ weight
            if weight.requires_grad: grad_weight += logits.t() @ _input[s:e]

        # Khi n_labels lớn thì cộng trước rồi chia sau giúp ổn định số học hơn
        reduction = 1.0 * step / (n_labels - n_ignores)
        ctx.save_for_backward(
            grad_input .detach() * reduction, 
            grad_weight.detach() * reduction if weight.requires_grad else None
        )
        return torch.sum(losses) * reduction

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        grad_input = grad_input * grad_output
        if grad_weight is not None: grad_weight = grad_weight * grad_output
        return grad_input, grad_weight, None, None, None, None


#################################################################
##  MUON optimizer - MomentUm Orthogonalized by Newton-schulz  ##
#################################################################

@torch.compile()
def zeropower_via_newtonschulz5(X:Tensor)->Tensor:  # zero(excess)power có nghĩa là spectral norm = 1 => perfect balance
    need_invert = X.size(-2) > X.size(-1)           # Sẽ báo lỗi nếu X.dim < 2
    X = X.bfloat16()                                # Need for Speed
    if need_invert: X = X.mT                        # Ensure số cột ≥ số hàng; giúp NS hoạt động tốt
    X /= X.norm(dim=(-2,-1), keepdim=True)+1e-7     # Ensure spectral norm ≤ 1, điều kiện bắt buộc để NS hội tụ
    a, b, c = ( 3.4445, -4.7750, 2.0315 )           # Hằng số tối ưu hóa cho NS iteration, tối ưu sau 5 iters
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 1: error ≈ ε  (NS có sai số giảm theo lũy thừa)
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 2: error ≈ ε²
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 3: error ≈ ε⁴
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 4  ... có thể xem mỗi NS iter như 1 lần khử nhiễu ? ...
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 5: error ≈ ε¹⁶, flatten singular values to range (0.7, 1.3)
    return X.mT if need_invert else X

class Muon1GPU(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.0069, momentum=0.96, **args):
        super().__init__(list(params), dict(lr=lr, wd=weight_decay, mm=momentum))

    @torch.no_grad()
    @torch.compiler.disable
    def step(self):
        for group in self.param_groups:
            for p in group['params']:                   # với mỗi tham số p trong model
                if p.grad is None: continue             # bỏ qua nếu không có gradient

                g, st = p.grad, self.state[p]           # lấy gradient và optim state và khởi tạo momentum nếu chưa có
                if 'mm' not in st: 
                    st['mm'] = torch.zeros_like(g, dtype=torch.bfloat16)

                st['mm'].lerp_(g, 1 - group['mm'])      # momentum = momentum * 0.96 + gradient * 0.04
                g = g.lerp_(st['mm'], group['mm'])      # gradient = gradient * 0.04 + momentum * 0.96

                if g.ndim != 2: g = g.view(len(g), -1)  # 2D hoá
                g = zeropower_via_newtonschulz5(g)      # Trực giao Newton-Schulz g => g(o)rthogonalized
                if g.shape != p.shape: g=g.view_as(p)   # Reshape back if needed

                # Cập nhật tham số p, theo gradient, learning rate và weight decay với 2 phép tính:
                p.mul_(1 - group['lr']*group['wd'])     # 1) p *= (1 - lr*wd) <= thu nhỏ p nếu wd > 0
                rows, cols = p.size(-2), p.size(-1)     # 2) p -= go * lr * sqrt(max(1, rows / cols))
                x = max(1, rows / cols)**0.5 
                p.add_(g, alpha=-group['lr']*x)
