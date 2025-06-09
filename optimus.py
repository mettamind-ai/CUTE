#!/usr/bin/env python3
''' TẬP HỢP CODE TỐI ƯU ĐỂ TRAIN LLM TRÊN GAMING GPUs (30xx, 40xx, 50xx)
- INT8 Mixed Precision github.com/gau-nernst/quantized-training
- Muon optimizer github.com/nil0x9/flash-muon
- Fused LCE github.com/linkedin/Liger-Kernel
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
    (128, 128,  32, 4, 4), (128,  64,  32, 4, 4), ( 64, 128,  32, 4, 4), 
    (128,  32,  32, 4, 4), (128, 128, 128, 4, 4), (128,  64,  64, 4, 4),
    ( 64, 128,  64, 4, 4), (128,  32,  64, 4, 4), ( 64,  64,  32, 2, 4),
    (128, 128,  32, 2, 8), ( 64, 128,  32, 4, 8), (128,  64,  32, 4, 8), 
    (256, 128,  64, 4, 8), (128, 256,  64, 4, 8), (128, 128,  64, 3, 8), 
    ( 64,  64,  64, 3, 4), ( 32, 128,  32, 2, 4), (128,  32,  32, 2, 4),
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
        B, Bs = quantize_int8(weight._data.T, dim=0, sr=True) # phép rounding này rẻ
        return scaled_mm(A, B, As, Bs,)

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
        grad_input = scaled_mm(A, B, As, Bs,)

        if ctx.needs_input_grad[1]:
            A, As = quantize_int8(grad_output.T, dim=1, sr=False) # không cần round vì grad ko truyền tiếp
            B, Bs = quantize_int8(inp, dim=0, sr=False)           # ... nó được update thẳng vào weight
            grad_weight = scaled_mm(A, B, As, Bs,)

        return grad_input, grad_weight, grad_bias


''' Chuyển tiếp F.linear func call tới kernel tuỳ chỉnh (Int8MixedLinear.apply) và cho phép torch.compile
dựng biểu đồ (graph) trơn tru, không làm gián đoạn quá trình trace-&-compile của PyTorch 2.
'''
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

############################
##  Fused  Cross Entropy  ##
############################

@triton.jit
def per_label_cross_entropy(X_ptr, X_stride, label_ptr, loss_ptr, n_non_ignore, ignore, vocab, CHUNK: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)  # chạy từ 0 tới num_labels
    X_ptr     += program_id * X_stride
    loss_ptr  += program_id

    true_label = tl.load(label_ptr + program_id)
    true_logit = tl.load(X_ptr + true_label).cast(tl.float32)

    offs = tl.arange(0, CHUNK)
    mask = (offs < vocab)
    X_ptr= X_ptr + offs

    if true_label == ignore: # logits' grad is 0
        tl.store(X_ptr, 0.0)
    else:
        X = tl.load(X_ptr, mask=mask, other=float("-inf")).cast(tl.float32)

        m = tl.max(X, axis=0)       # the max value `m` and the sum `d` are notations in ... 
        d = tl.sum(tl.exp(X - m))   # ... the paper https://www.alphaxiv.org/abs/1805.02867

        LSE  = m + tl.log(d)        # Log-Sum-Exp, "Mức độ lớn" của tất cả logits
        loss = LSE - true_logit     # loss là khoảng cách mức độ lớn tổng thể và true label logit

        X  = tl.exp(X - m)/d                        # softmax(x_i)
        lse_square_scale = 1e-4                     # scaler of logsumexp(_input)^2; adding for stability

        z_loss = lse_square_scale*LSE*LSE           # An auxiliary loss, Refer to Page14 Loss function section ...
        loss  += z_loss                             # ... in the paper https://www.jmlr.org/papers/v24/22-1144.html

        X *= 1 + 2*lse_square_scale*LSE             # derivative of z-loss: 2*lse_square_scale*lse*softmax(x_i)
        X  = tl.where(offs != true_label, X, X - 1) # gradient bị tác động bởi true_label_logit

        tl.store(X_ptr, X/n_non_ignore, mask=mask)  # mean reduction
        tl.store(loss_ptr, loss/n_non_ignore)       # mean reduction


class FusedLinearCrossEntropy(torch.autograd.Function):
    """ TÍNH GRADIENT NGAY TRONG FORWARD. Nhờ đó không cần lưu input và target cho backward """
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, _input, weight, target, n_ignores=0, ignore=-100):

        grad_weight = torch.zeros_like(weight, device=_input.device) if weight.requires_grad else None
        grad_input  = torch.empty_like(_input, device=_input.device)
        
        n_labels = _input.shape[0]
        losses   = torch.zeros(n_labels, dtype=torch.float32, device=_input.device)

        for s in range( 0, n_labels, 2048 ):
            e = min(s + 2048, n_labels)
            logits = ( _input[s:e] @ weight.t() ).contiguous()

            N,  V = logits.shape[0]                             # N là số labels, V là vocab
            ni, C = (N - n_ignores, triton.next_power_of_2(V))  # TODO: ni cần tính toán chính xác theo chunk

        per_label_cross_entropy[(N,)](
            X_ptr=logits, X_stride=logits.stride(-2), label_ptr=target[s:e], 
            loss_ptr=losses[s:e], n_non_ignore=ni, ignore=ignore, vocab=V, CHUNK=C, num_warps=32, 
        )
        grad_input[s:e] = logits @ weight
        if weight.requires_grad: grad_weight += logits.t() @ _input[s:e]

        ctx.save_for_backward(
            grad_input.detach(), 
            grad_weight.detach() if weight.requires_grad else None
        )
        return torch.sum(losses)  # final loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        grad_input = grad_input * grad_output
        if grad_weight is not None:
            grad_weight = grad_weight * grad_output
        return grad_input, grad_weight, None, None, None


#################################################################
##  MUON optimizer - MomentUm Orthogonalized by Newton-schulz  ##
#################################################################
'''
Muon is "Matrix-structured steepest descent with spectral norm regularization"

Phương pháp này bắt đầu từ ý tưởng steepest descent cơ bản - đi theo hướng giảm nhanh nhất của hàm loss bằng cách di chuyển ngược hướng gradient. Khác với các optimizer truyền thống cập nhật từng tham số riêng lẻ, Muon áp dụng cách tiếp cận có cấu trúc ma trận (matrix-structured). Điều này có nghĩa là thay vì xem mỗi element của weight matrix như các đơn vị độc lập, Muon xử lý toàn bộ ma trận như một thể thống nhất và tìm cách biến đổi nó một cách có hệ thống.

Để đảm bảo quá trình cập nhật không bị "vượt quá giới hạn", Muon áp dụng spectral norm regularization (điều chuẩn chuẩn phổ) - một ràng buộc giới hạn sức mạnh của ma trận update thông qua điều kiện ||O_t||₂ ≤ 1. Ràng buộc này không chỉ đảm bảo tính ổn định mà còn tự nhiên dẫn đến nghiệm tối ưu có dạng O_t = UV^T từ phép phân tích SVD của gradient. Cách tiếp cận này cho phép Muon tự động cân bằng toàn bộ ma trận thông qua một ràng buộc toàn cục, thay vì phải điều chỉnh learning rate cho từng parameter như AdamW. Kết quả là một phương pháp optimization vừa đơn giản vừa hiệu quả, duy trì cấu trúc ma trận trong khi đảm bảo convergence ổn định.

- Chuẩn phổ (spectral norm): Là giá trị singular value lớn nhất của ma trận - ĐO "SỨC MẠNH KÉO DÀI" TỐI ĐA mà ma trận có thể gây ra cho vector.
- UV^T là ma trận trực giao (orthogonal) với spectral norm = 1. Newton-Schulz là cách tính gần đúng UV^T mà không cần SVD đắt đỏ.

TẠI SAO MUON LẠI SỬ DỤNG ĐIỀU CHUẨN CHUẨN PHỔ? (spectral norm regularization)
- Muon giới hạn chuẩn phổ <= 1 nghĩa là giới hạn Sức mạnh kéo dài mà ma trận gây ra cho vector (control maximum damage) tránh bùng nổ gradient
- Sử dụng spectral norm regularization tự nhiên dẫn tới SVD structure (nghiệm tối ưu TỰ ĐỘNG là O=UV^T từ SVD G = UΣV^T)
- Spectral norm liên kết với inverse Fisher matrix approximation là 1 phương pháp đạo hàm bậc 2 (điều này quan trọng)
  Shampoo dùng: E[GG^T]^{-1/4} G E[G^T G]^{-1/4} => Simplify → UV^T (spectral structure) => Muon là minimal version của class này

=> Muon thực sự nhìn optimization landscape từ góc nhìn geometric hoàn toàn khác với Adam !!! Và vì Muon xấp xỉ tối thiểu inverse Hessian, nó có thể nhìn thấy "valleys and ridges" của loss landscape, trong khi đó Adam chỉ nhìn thấy local slopes, đặc biệt khi BATCH SIZE lớn và global structure quan trọng hơn local adaptivity.
'''
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
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 4  ... có thể xem mỗi NS iter như 1 lần khử nhiễu ...
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 5: error ≈ ε¹⁶, flatten singular values to range (0.7, 1.3)
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 6: thêm 1 lần khử nhiễu đẻ ổn định hơn với int8? (0.9, 1.1) ?!?
    return X.mT if need_invert else X
    '''Khi thực hiện phép tính A = X @ X.mT, chúng ta đang tạo ra một hiệu ứng trung bình hóa. Trong phép nhân ma trận này, mỗi phần tử của ma trận kết quả A được hình thành từ tổng của nhiều phép nhân giữa các phần tử khác nhau trong X. Noise ngẫu nhiên, do bản chất không có cấu trúc, có xu hướng triệt tiêu lẫn nhau trong quá trình cộng tổng này, trong khi tín hiệu thật có cấu trúc rõ ràng được củng cố và tăng cường.

    Phép update X = aX + (bA + c*A@A) @ X làm mịn dữ liệu: Sau khi ma trận A đã được "lọc" với X ban đầu, chúng ta áp dụng một phép biến đổi mà trong đó mỗi phần tử mới được tạo thành từ sự kết hợp của nhiều phần tử cũ. Quá trình này tương tự như việc áp dụng bộ lọc không gian, nơi các giá trị lân cận ảnh hưởng và cân bằng lẫn nhau, từ đó làm mịn những biến động đột ngột và bất thường.

    Cuối cùng, xu hướng convergence hướng về cấu trúc orthogonal tạo ra một cơ chế lọc tự nhiên. Sau nhiều iterations, X dần hội tụ về ma trận orthogonal, và quá trình này tự động "đẩy ra" những thành phần không phù hợp với cấu trúc orthogonal, bao gồm cả noise, trong khi bảo tồn những directions quan trọng và có ý nghĩa nhất.
    '''

## POLAR https://alphaxiv.org/abs/2505.16932 | https://x.com/gowerrobert/status/1930292178456039739
coeffs_list = [  (8.28721201814563   , -23.595886519098837  , 17.300387312530933   ),    # iter 1
                 (4.107059111542203  ,  -2.9478499167379106 ,  0.5448431082926601  ),    # iter 2
                 (3.9486908534822946 ,  -2.908902115962949  ,  0.5518191394370137  ),    # iter 3
                 (3.3184196573706015 ,  -2.488488024314874  ,  0.51004894012372    ),    # iter 4
                 (2.300652019954817  ,  -1.6689039845747493 ,  0.4188073119525673  ),    # iter 5
                 (1.891301407787398  ,  -1.2679958271945868 ,  0.37680408948524835 ),    # iter 6
                 (1.8750014808534479 ,  -1.2500016453999487 ,  0.3750001645474248  ),    # iter 7
              ]#  1.875                 -1.25                  0.375         => subsequent coeffs
coeffs_list = [(a/1.01,b/1.01**3,c/1.01**5) for (a,b,c) in coeffs_list] + [(1.875, -1.25 , 0.375)]*3
#    safety factor for numerical stability
@torch.compile()
def PolarExpress(X:Tensor, steps=6)->Tensor:        # coeffs_list cho 5 tới 10 lần lặp
    need_invert = X.size(-2) > X.size(-1)           # Sẽ báo lỗi nếu X.dim < 2
    X = X.bfloat16()                                # Need for Speed
    if need_invert: X = X.mT                        # Ensure số cột ≥ số hàng; giúp NS hoạt động tốt
    # X /= X.norm(dim=(-2,-1), keepdim=True)*1.01   # Ensure spectral norm ≤ 1, <= cách làm tròn này gây NaN
    X   /= X.norm(dim=(-2,-1), keepdim=True)+1e-7   # Ensure spectral norm ≤ 1, điều kiện bắt buộc để NS hội tụ
    for (a,b,c) in coeffs_list[:steps]:
        A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X   # X <- aX + bXˆ3 + cXˆ5
    return X.mT if need_invert else X


class Muon1GPU(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, **args):
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

                st['mm'].lerp_(g, 1 - group['mm'])          # momentum = momentum * 0.95 + gradient * 0.05
                g = g.lerp_(st['mm'], group['mm'])          # gradient = gradient * 0.05 + momentum * 0.95

                if g.ndim != 2: g = g.view(len(g), -1)      # 2D hoá
                # go = zeropower_via_newtonschulz5(g)       # Trực giao Newton-Schulz gốc
                go = PolarExpress(g)                        # Thuật toán Polar Express tính orthogonal grad
                if go.shape != p.shape: go=go.view_as(p)    # Reshape back if needed

                # Cập nhật tham số p, theo gradient, learning rate và weight decay với 2 phép tính:
                p.mul_(1 - group['lr']*group['wd'])     # 1) p *= (1 - lr*wd) <= thu nhỏ p nếu wd > 0
                rows, cols = p.size(-2), p.size(-1)     # 2) p -= go * lr * sqrt(max(1, rows / cols))
                x = max(1, rows / cols)**0.5 
                p.add_(go, alpha=-group['lr']*x)
