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

_grid = lambda meta: ( triton.cdiv(meta["M"], meta["BLOCK_M"])*triton.cdiv(meta["N"], meta["BLOCK_N"]), )
_cfgs = [triton.Config(dict(BLOCK_M=m, BLOCK_N=n, BLOCK_K=k), num_stages=s, num_warps=w) for m, n, k, s, w in \
[(128, 128, 32, 4, 4), ( 64, 128, 32, 4, 8), (128,  64, 32, 4, 8), (256, 128, 64, 4, 8), (128, 256, 64, 4, 8)]]

@triton.autotune(configs=_cfgs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _scaled_mm_kernel(
    A_ptr, B_ptr, C_ptr, A_scale_ptr, B_scale_ptr, M, N, K,
    stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, 
    stride_bn: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M:   tl.constexpr, BLOCK_N:   tl.constexpr, BLOCK_K:   tl.constexpr,
    GROUP_M:   tl.constexpr = 8 ): # số khối theo chiều M được nhóm lại (để tối ưu L2 cache)

    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)

    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # `r` range arrays (rm, rn, rk là các mảng chỉ số)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk =                   tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M) # tl.max_contiguous => tối đa BLOCK_M phần tử liền kề trong memory
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N) # tl.multiple_of => gợi ý alignment, chỉ số là bội số của BLOCK_N

    A = A_ptr + (ram[:, None] * stride_am +  rk[None, :] * stride_ak) # 2D layout để chuẩn bị nhân ma trận
    B = B_ptr + ( rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(K, 0, -BLOCK_K):
        acc += tl.dot(tl.load(A), tl.load(B))
        A   += BLOCK_K * stride_ak
        B   += BLOCK_K * stride_bk

    # Không dùng lại `rm, rn`, mà tính trực tiếp để `rm, rn` được giải phóng ở trước vòng for, tiết kiệm registers
    idx_m = ( pid_m * BLOCK_M + tl.arange(0, BLOCK_M) )[:, None]
    idx_n = ( pid_n * BLOCK_N + tl.arange(0, BLOCK_N) )[None, :]

    A_scale = tl.load(A_scale_ptr + idx_m, mask=idx_m < M)
    B_scale = tl.load(B_scale_ptr + idx_n, mask=idx_n < N)
    acc = acc.to(tl.float32) * A_scale * B_scale

    mask  = (idx_m < M) & (idx_n < N)
    index = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(index, mask.shape), acc, mask)


lib.define("scaled_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B, ScalarType? dtype=None) -> Tensor")
def scaled_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor, dtype=None) -> Tensor:
    return lib_ops.scaled_mm(A, B, scale_A, scale_B, dtype)

@torch.library.impl(lib, "scaled_mm", "Meta")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor, dtype=None):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=dtype)

@torch.library.impl(lib, "scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, row_scale_A: Tensor, col_scale_B: Tensor, dtype=None):
    M, K = A.shape; _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=( row_scale_A.dtype if dtype is None else dtype ))
    _scaled_mm_kernel[_grid](A, B, C, row_scale_A, col_scale_B, M, N, K, *A.stride(), *B.stride(), *C.stride(),)
    return C


@torch.no_grad()
def quantize_int8(tensor, dim=1, eps=1e-12, sr=False):
    tensor = tensor.float()                             # float32
    scale  = tensor.abs().amax(dim, keepdim=True) / 127 # float32
    tensor = tensor / scale.clip(eps)                   # float32: clip(cận_dưới_eps) tránh chia cho 0
    if sr:   tensor = (tensor+torch.rand_like(tensor)).floor()  # float32
    else:    tensor.round_()  # ^^^ stochastic rounding ^^^^    # float32
    tensor = tensor.clip(-128, 127).to(torch.int8)      # int8
    return ( tensor, scale )                            # int8, float32


def _fp32_to_bf16_sr(x_f32: Tensor) -> Tensor:
    ''' https://github.com/pytorch/ao/blob/main/torchao/optim/quant_utils.py
    For an FP32 number      [a31, ..., a16, a15, ..., a0] to be converted to BF16
    - Round towards zero:   [a31, ..., a16,   0, ...,  0]
    - Round away from zero: [a31, ..., a16+1, 0, ...,  0]
    (since the value can be negative, we use round towards/away from zero instead of round up/down)
    For stochastic rounding, we round away from zero with the probability of
    [a15, ..., a0] / 2^16, where the bit pattern [a15, ..., a0] is interpreted as uint16  
    we have to use int32 since most arithmetic ops are not implemented for uint32/int16/uint16
    '''
    rand_16bit = torch.randint(0, 1 << 16, x_f32.shape, device=x_f32.device, dtype=torch.int32)
    x_f32_bits = x_f32.view(torch.int32)
    x_fraction = x_f32_bits & 0xFFFF              # lower 16 bits
    x_bf16_towards_zero = x_f32_bits & 0xFFFF0000 # upper 16 bits
    x_f32_bits = (x_f32_bits + rand_16bit) & 0xFFFF0000
    return x_f32_bits.view(torch.float32).bfloat16()


class Int8MixedLinear(torch.autograd.Function):
    @staticmethod
    def forward(inp, weight, bias=None):
        A, As = quantize_int8(inp, dim=1, sr=False)
        B, Bs = quantize_int8(weight._data.T, dim=0, sr=True) # phép rounding này rẻ
        return scaled_mm(A, B, As, Bs, dtype=torch.bfloat16)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inp, weight, _ = inputs
        ctx.save_for_backward(inp, weight._data)

    @staticmethod
    def backward(ctx, grad_out):
        inp, weight = ctx.saved_tensors
        grad_weight = grad_bias = None

        ## grad_input tiếp tục truyền về phía sau nên cần duy trì độ chính xác cao =>
        A, As = quantize_int8(grad_out, dim=1, sr=True) # rounding both để đạt độ chính xác cao
        B, Bs = quantize_int8(weight  , dim=0, sr=True) # phép rounding này rẻ
        grad_input = scaled_mm(A, B, As, Bs, dtype=torch.bfloat16)

        if ctx.needs_input_grad[1]:
            A, As = quantize_int8(grad_out.T, dim=1, sr=False) 
            B, Bs = quantize_int8(inp       , dim=0, sr=False)
            grad_weight = scaled_mm(A, B, As, Bs, dtype=torch.float32)
            grad_weight = _fp32_to_bf16_sr(grad_weight) # phép rounding này rẻ

        return grad_input, grad_weight, grad_bias


###########################
##  Fused Cross Entropy  ##
###########################

@triton.jit
def per_label_cross_entropy(
        logits_ptr, target_ptr, loss_ptr, reduction: tl.constexpr,
        stride:  tl.constexpr, vocab: tl.constexpr, ignore:  tl.constexpr
    ):
    pid  = tl.program_id(0).to(tl.int64)    # pid chạy từ 0 tới num_targets
    offs = tl.arange(0, vocab)              # khoảng địa chỉ để load logits
    row  = logits_ptr + pid * stride        # row trỏ tới vị trí đầu của target logits

    tgt = tl.load(target_ptr + pid)
    if tgt == ignore: tl.store(row + offs, 0.0); return

    # softmax(xi) = p(xi) = e^xi / Σ(e^xj) = e^(xi-M) / Σ(e^(xj-M))
    x         = tl.load(row + offs).to(tl.float32)  # load toàn bộ vocab logits liên quan tới target
    tgt_logit = tl.load(row + tgt ).to(tl.float32)  # load true target logit

    M    = tl.max(x, axis=0)
    e_x  = tl.exp(x - M)            # e^(xi-M)
    d    = tl.sum(e_x, axis=0)      # Σ(e^(xj-M))
    lse  = M + tl.log(d)            # log(Σe^logits) => (L)og-(S)um-(E)xp

    grad = e_x / d                  # p(xi) = exp(xi-M) / Σexp(xj-M)
    grad = grad * (1 + 2e-5 * lse)  # z-loss modification
    grad = tl.where(offs == tgt, grad - 1, grad)    # điều chỉnh grad cho target
    tl.store(row + offs, grad * reduction)          # lưu grad cho target logits

    loss  = lse - tgt_logit         # LCE = Surprise = -log(p_target) = -(x_target - lse)
    loss += 1e-5 * lse * lse        # cộng thêm z_loss penalty giúp ổn định training
    tl.store(loss_ptr + pid, loss * reduction)      # lưu loss cho target


class FusedCE(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, _input, weight, target, n_ignores=0, ignore=-100, ratio=1.0):

        grad_weight = torch.zeros_like(weight, device=_input.device) if weight.requires_grad else None
        grad_input  = torch.empty_like(_input, device=_input.device)
        losses      = torch.zeros(_input.shape[0], device=_input.device, dtype=torch.float32)

        n_labels, vocab = _input.shape[0], weight.shape[0]
        assert vocab == triton.next_power_of_2(vocab), "vocab must be power of 2"
        step = min(1024*16, n_labels)

        for s in range( 0, n_labels, step ):
            e = min(s + step, n_labels)
            chunk_input     = _input[s:e]
            logits          = chunk_input @ weight.t()
            logits          = logits.contiguous()

            per_label_cross_entropy[( logits.shape[0], )](
                logits_ptr  = logits,
                target_ptr  = target[s:e],
                loss_ptr    = losses[s:e],
                stride      = logits.stride(-2),
                ignore      = ignore,
                vocab       = vocab,
                num_warps   = 16 if vocab <= 1024*8 else 32,
                reduction   = ratio / (n_labels - n_ignores),
            )

            grad_input[s:e] = logits @ weight
            if weight.requires_grad: grad_weight = torch.addmm(grad_weight, logits.t(), chunk_input)

        ctx.save_for_backward(grad_input.detach(), grad_weight.detach() if weight.requires_grad else None)
        return torch.sum(losses)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out, *args): # vì ratio đã được nhân thẳng vào losses nên grad_out == 1
        return *ctx.saved_tensors, None, None, None, None


#################################################################
##  MUON optimizer - MomentUm Orthogonalized by Newton-schulz  ##
#################################################################

@torch.compile()
def zeropower_newtonschulz6(X:Tensor)->Tensor:  # zero(excess)power có nghĩa là spectral norm = 1 => perfect balance
    need_invert = X.size(-2) > X.size(-1)       # Sẽ báo lỗi nếu X.dim < 2
    if need_invert: X = X.mT                    # Ensure số cột ≥ số hàng; giúp NS hoạt động tốt
    X /= X.norm(dim=(-2,-1), keepdim=True)+1e-7 # Ensure spectral norm ≤ 1, điều kiện bắt buộc để NS hội tụ
    a, b, c = ( 3.4445, -4.7750, 2.0315 )       # Hằng số tối ưu hóa cho NS iteration, tối ưu sau 5 iters
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X   # iter 1: error ≈ ε  (NS có sai số giảm theo lũy thừa)
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X   # iter 2: error ≈ ε²
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X   # iter 3: error ≈ ε⁴
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X   # iter 4  ... có thể xem mỗi NS iter như 1 lần khử nhiễu ? ...
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X   # iter 5: error ≈ ε¹⁶, flatten singular values to range (0.7, 1.3)
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X
    return X.mT if need_invert else X

class Muon1GPU(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, **args):
        super().__init__(list(params), dict(lr=lr, wd=weight_decay, mm=momentum))

    @torch.no_grad()
    @torch.compiler.disable
    def step(self):
        for group in self.param_groups:
            for p in group['params']:               # với mỗi tham số p trong model
                if p.grad is None: continue         # bỏ qua nếu không có gradient

                g, st = p.grad, self.state[p]       # lấy gradient và optim state và ...
                if 'mm' not in st:                  # ... khởi tạo momentum nếu chưa có
                    st['mm'] = torch.zeros_like(g)

                st['mm'].lerp_(g, 1 - group['mm'])  # momentum = momentum * 0.95 + gradient * 0.05
                g = g.lerp_(st['mm'], group['mm'])  # gradient = gradient * 0.05 + momentum * 0.95

                assert g.dim() == 2, "Muon only supports 2D weight matrices"
                g = zeropower_newtonschulz6(g)      # Trực giao hoá g qua 6 bước

                # Cập nhật tham số p, theo gradient, learning rate và weight decay với 2 phép tính:
                p.mul_(1 - group['lr']*group['wd']) # 1) p *= (1 - lr*wd) <= thu nhỏ p nếu wd > 0
                rows, cols = p.size(-2), p.size(-1) # 2) p -= g * lr * sqrt(max(1, rows / cols))
                x = max(1, rows / cols)**0.5 
                p.add_(g, alpha=-group['lr']*x)


################################
##  OhMaiHead speedup LCE     ##
################################
MAX_ACTIVE_VOCAB = 1024 * 32  # 32k tối ưu cho speed, và vừa đủ 1:3 -> 1:4 pos/ng
class OhMaiHead(nn.Module):
    def __init__(self, dim, vocab, bias=None):
        super().__init__()
        self.active_vocab = triton.next_power_of_2(vocab // 3)
        print(f"OhMaiHead: active_vocab = {self.active_vocab} / {vocab}")

        if  self.active_vocab > MAX_ACTIVE_VOCAB:
            self.active_vocab = MAX_ACTIVE_VOCAB

        self.weight = torch.zeros(vocab, dim, device="cpu", pin_memory=True, dtype=torch.bfloat16)
        self.weight.requires_grad_(False)

        self.active = torch.arange(self.active_vocab, device='cuda')
        self.active.requires_grad_(False)
        self.alpha = 0.69

        w = torch.empty(self.active_vocab, dim, device="cuda", dtype=self.weight.dtype)
        with torch.no_grad(): w.data = self.weight.data[:self.active_vocab].cuda()
        self.active_weight = nn.Parameter(w)

        self.pretrained_norm = torch.ones(vocab, device="cuda") / vocab
        self.pretrained_norm.requires_grad_(False)

        self.running_freq = torch.zeros(vocab, device="cuda")
        self.running_freq.requires_grad_(False)
        self.total_tokens = torch.tensor(0, dtype=torch.int64, device='cuda')

        self.inverse_map = torch.full((vocab,), -1, dtype=torch.long, device='cuda')
        self.inverse_map.requires_grad_(False)
        self.ohmai_stream = torch.cuda.Stream()


    @torch.no_grad()
    @torch.compiler.disable
    def get_active_tokens(self, indices):
        # Phép lấy unique tokens và counts này size thay đổi qua mỗi phép toán => ko compile
        tokens, counts = torch.unique(indices, return_counts=True)
        self.running_freq[tokens] += counts
        self.total_tokens += counts.sum()

        empirical_freq = self.running_freq / self.total_tokens
        combined_score = self.alpha * empirical_freq + (1-self.alpha) * self.pretrained_norm

        sample_probs = combined_score.pow(0.75) # Smooth với power 0.75; Từ Word2Vec paper
        sample_probs = sample_probs / sample_probs .sum()

        mask = torch.ones_like(sample_probs)
        mask[tokens] = 0

        masked_probs = sample_probs * mask
        masked_probs = masked_probs / masked_probs.sum()

        neg_tokens = torch.multinomial(masked_probs, self.active_vocab - len(tokens), replacement=False)
        return torch.cat([ tokens, neg_tokens ])

    @torch.no_grad()
    @torch.compile()
    def activate(self, indices):
        curr_active   = self.get_active_tokens(indices)
        unuse_mask    = ~torch.isin(self.active, curr_active)
        unuse_tokens  = self.active[unuse_mask].cpu()
        unuse_indices = torch.nonzero(unuse_mask, as_tuple=False).flatten()

        with torch.cuda.stream(self.ohmai_stream):
            self.weight.data[unuse_tokens] = self.active_weight.data[unuse_indices].to('cpu', non_blocking=True)

        self.new_tokens = curr_active[~torch.isin(curr_active, self.active)]
        self.new_token_indices = torch.nonzero(~torch.isin(self.active, curr_active), as_tuple=False).flatten()

        assert len(self.new_token_indices) == len(self.new_tokens)
        self.active[self.new_token_indices] = self.new_tokens
        self.new_tokens = self.new_tokens.cpu()

        # Tạo inverse indices và trả về
        self.inverse_map[self.active] = torch.arange(len(self.active), device=indices.device)
        return self.inverse_map[indices]

    @torch.no_grad()
    @torch.compiler.disable
    def update_new_tokens_weight(self):
        self.active_weight.data[ self.new_token_indices ] = \
        self.weight.data[ self.new_tokens ].cuda(non_blocking=True)

    @torch.no_grad()
    @torch.compiler.disable
    def update_async_weight(self):
        self.weight.data[ self.active.cpu().long() ] = \
        self.active_weight.data[ : len(self.active) ].cpu()


########################
##  Int8 Mixed Utils  ##
########################

''' Chuyển tiếp F.linear func call tới kernel tuỳ chỉnh (Int8MixedLinear.apply) và cho phép torch.compile
dựng biểu đồ (graph) trơn tru, không làm gián đoạn quá trình trace-&-compile của PyTorch. '''
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


def convert_int8_mixed_precision(module:nn.Module, ignore='emb|up_att'):  # bỏ unembedding khỏi int8 mixed
    ignore = re.compile(rf'{ignore}')
    int8_names, int8_params = [], 0
    sparse_names, sparse_params = [], []
    for n, m in module.named_modules():
        if isinstance(m, nn.Linear) and not ignore.search(n): 
            if False: # "down_proj" in n:
                sparse_names.append(n)
                sparse_params.append(m)
                # quantize_(m, Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()))
                # m.weight = nn.Parameter(Int8MixedLWeight(m.weight.detach()), requires_grad=m.weight.requires_grad)
            else:
                int8_names.append(n)
                int8_params += m.weight.numel()
                m.weight = nn.Parameter(Int8MixedLWeight(m.weight.detach()), requires_grad=m.weight.requires_grad)
    return int8_names, int8_params, sparse_names, sparse_params
