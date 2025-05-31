#!/usr/bin/env python3
''' TẬP HỢP CODE TỐI ƯU ĐỂ TRAIN LLM TRÊN GAMING GPU (30xx, 40xx, 50xx)
INT8 Mixed Precision modded from github.com/gau-nernst/quantized-training
Muon optimizer modded from github.com/nil0x9/flash-muon
'''
import torch, triton, os
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
    _scaled_mm_kernel[_grid](
        A, B, C,
        row_scale_A, col_scale_B,
        M, N, K,
        *A.stride(), *B.stride(), *C.stride(),
    )
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
        # print(A, As)#; input()
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
        
        ''' Tile scale, tốt nhất nhưng ko nhanh hơn bf16 là mấy
        A, B = grad_output, weight
        A, As = tile_quantize_int8(A, tile_shape=tile_shape, sr=True)
        B, Bs = tile_quantize_int8(B, tile_shape=tile_shape, sr=True)
        grad_input = scaled_mm(A, B, As, Bs,)
        if ctx.needs_input_grad[1]:
            IT, ITs = tile_quantize_int8(inp.T, tile_shape=tile_shape, sr=False)
            grad_weight = scaled_mm(IT, A, ITs, As).T
        # '''

        ## Grad truyền tiếp về layer sau, nên cần độ chính xác cao
        # grad_input = grad_output @ weight # phép nhân nguyên bản
        A, B  = grad_output, weight
        A, As = quantize_int8(A, dim=1, sr=True) # rounding để đạt độ ...
        B, Bs = quantize_int8(B, dim=0, sr=True) # ... chính xác cao hơn
        grad_input = scaled_mm(A, B, As, Bs,)

        if ctx.needs_input_grad[1]:
            ## grad_weight = grad_output.T @ inp; cả 2 là activation nên rất lớn
            ## Áp dụng INT8 matmul ở đây là lợi nhất; sr=False để tránh OOM
            A, B  = grad_output.T, inp
            A, As = quantize_int8(A, dim=1, sr=False) # không cần round vì grad ko truyền tiếp
            B, Bs = quantize_int8(B, dim=0, sr=False) # ... nó được update thẳng vào weight
            grad_weight = scaled_mm(A, B, As, Bs,)
            ## Thử INT4 Matmul <= Speed tăng chút + vỡ đường loss
            # A,  row_scale = quantize_int4(A)
            # BT, col_scale = quantize_int4(B.T)
            # grad_weight = scaled_int4_mm(A, BT.T, row_scale, col_scale.T,)

        if ctx.needs_input_grad[2] and ctx.bias: grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


## Dùng lớp này để gói Linear weight, giúp torch.compile tối ưu hoá được graph
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
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["_data"])

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self._data})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()
        if func is F.linear: return Int8MixedLinear.apply(*args, **kwargs)
        with torch._C.DisableTorchFunctionSubclass(): return func(*args, **kwargs)

    # Adapted from FP8 implementation of WeightWithDynamicFloat8CastTensor
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        def unwrap(x: cls): return x._data
        out = func(*pytree.tree_map_only(cls, unwrap, args), **pytree.tree_map_only(cls, unwrap, kwargs),)
        others = { 
            aten.t.default, aten.detach.default, aten.empty_like.default, 
            aten.new_zeros.default, aten.slice.Tensor, aten.view.default, aten.as_strided.default, 
            aten._to_copy.default, aten._pin_memory.default, aten.split.Tensor, aten.clone.default, 
        }
        if func is aten.copy_.default: return args[0] # original object
        elif func in others: return pytree.tree_map_only(Tensor, lambda x: cls(x), out) # new wrapped object
        else: return out # new unwrapped object

import re
def convert_int8_mixed_precision(module:nn.Module, ignore='head|k_proj|v_proj'):
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

def newtonschulz(G: Tensor, steps: int, fast=True) -> Tensor:
    # G: The gradient or momentum matrix to be orthogonalized.
    # steps: Number of Newton-Schulz iterations.
    assert G.ndim >= 2
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
                    g = newtonschulz(g, steps=group["ns_steps"], fast=False).flatten()
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
