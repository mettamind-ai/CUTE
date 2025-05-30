#!/usr/bin/env python3
""" Modded from github.com/linkedin/liger-Kernel/blob/main/src/liger_kernel/ops/experimental/embedding.py
"""
import functools
import torch, triton
import triton.language as tl
from torch import nn, Tensor

_to_contiguous = lambda x: x if not isinstance(x, Tensor) else x.contiguous()
def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        args    = [_to_contiguous(arg) for arg in args]
        kwargs  = {k: _to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)
    return wrapper

@triton.jit
def embedding_forward_kernel(
    embeds_ptr, tokens_ptr,     # trỏ tới token_ids cần lấy embedding values
    output_ptr,                 # x0, hay embeddings của batch hiện tại
    vocab, hidim,               # vocab size x hidden dim = kích thước embedding matrix
    BLOCK_SIZE: tl.constexpr,   # kích thước khối đang xử lý
):
    tok_offsets  = tl.program_id(0)*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    feat_offsets = tl.program_id(1)*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Đảm bảo không load và store embedding vượt ngoài khuôn khổ vocab x hidim
    tok_mask  =  tok_offsets < vocab  # token < vocab size
    feat_mask = feat_offsets < hidim  # feat  < hidden dim
    emb_mask  = tok_mask[:, None] & feat_mask[None, :]

    tok_indexes = tl.load(tokens_ptr + tok_offsets, mask=tok_mask)
    emb_offsets = tok_indexes[:, None]*hidim + feat_offsets[None, :] # vị trí trong embedding matrix
    out_offsets = tok_offsets[:, None]*hidim + feat_offsets[None, :] # vị trí trong x0

    emb = embeds_ptr + emb_offsets   # emb_ là sparse
    out = output_ptr + out_offsets   # out_ là continuous

    v = tl.load(emb, mask=emb_mask)
    tl.store(out, v, mask=emb_mask)


@triton.jit
def embedding_backward_kernel(
    grad_output_ptr, grad_weight_ptr, # grad_weight là embeddings_grad
    tokens_ptr, vocab, hidim,
    BLOCK_SIZE : tl.constexpr,
):
    tok_offsets  = tl.program_id(0)*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    feat_offsets = tl.program_id(1)*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Đảm bảo không load và store embedding vượt ngoài khuôn khổ vocab x hidim
    tok_mask  =  tok_offsets < vocab  # token < vocab size
    feat_mask = feat_offsets < hidim  # feat  < hidden dim
    emb_mask  = tok_mask[:, None] & feat_mask[None, :]

    tok_indexes = tl.load(tokens_ptr + tok_offsets, mask=tok_mask)
    emb_offsets = tok_indexes[:, None]*hidim + feat_offsets[None, :] # vị trí trong embedding matrix
    out_offsets = tok_offsets[:, None]*hidim + feat_offsets[None, :] # vị trí trong x0

    output = grad_output_ptr + out_offsets
    weight = grad_weight_ptr + emb_offsets

    v  = tl.load(output, mask=emb_mask)
    v += tl.load(weight, mask=emb_mask)
    tl.store(weight, v,  mask=emb_mask)
    # tl.atomic_add(grad_weight_ptr + emb_offsets, grad_output, mask=emb_mask)
    # https://github.com/triton-lang/triton/commit/236f6b54ce337db009ea573915022dafdbf61b82
    # atomic_add chỉ hỗ trợ float32, khi triton được update sẽ dùng lại được


class OhMaiEmbFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        vocab, hidim = indices.numel(), embeddings.shape[1]
        output = torch.empty(vocab, hidim, device=indices.device, dtype=embeddings.dtype,)

        blsz = triton.next_power_of_2(min(128, hidim))
        grid = ( triton.cdiv(vocab, blsz), triton.cdiv(hidim, blsz), )

        embedding_forward_kernel[grid](embeddings, indices, output, vocab, hidim, blsz)
        ctx.save_for_backward(indices, embeddings)
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        indices, embeddings = ctx.saved_tensors
        vocab, hidim = indices.numel(), embeddings.shape[1]
        grad_output = grad_output.contiguous()
        grad_weight = torch.zeros_like(embeddings) # tốn ở chỗ này

        blsz = triton.next_power_of_2(min(128, hidim))
        grid = ( triton.cdiv(vocab, blsz), triton.cdiv(hidim, blsz), )

        embedding_backward_kernel[grid](grad_output, grad_weight, indices, vocab, hidim, blsz)
        return grad_weight, None


# NOTE: Disable compile graph để có thể sửa đổi active_weight tuỳ theo data batch
# https://docs.pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html#torch-compiler-disable
@torch.compiler.disable
class OhMaiEmbedding(nn.Module):
    """ Chỉ load tokens có trong current batch vào vram
- Giữ toàn bộ embedding matrix ở CPU
- Chỉ load embedding matrix của active_vocab vào vram => `active_weight`
- Có cơ chế mapping để biết token_id ở `active_weight` index nào
- Có thao tác để đổi active_vocab
    """
    def __init__(self, vocab, hidim, active_vocab=None):
        super().__init__()
        self.vocab = vocab
        self.hidim = hidim

        # Pinned Memory → GPU Memory   vs   CPU Memory → Staging Buffer → GPU Memory
        self.weight = torch.randn(vocab, hidim, device="cpu", pin_memory=True, dtype=torch.bfloat16)
        self.weight.requires_grad_(False)

        self.active_weight = None
        self.active_tokens = None # Cần kích hoạt mỗi lần forward

        if active_vocab is None: active_vocab = vocab // 2  # a safe assumption
        w = torch.empty(active_vocab, self.hidim, device="cuda", dtype=self.weight.dtype)
        self.active_weight = nn.Parameter(w)
        self.active_vocab = active_vocab

        # Khởi tạo CUDA stream cho async transfer
        self.update_stream = torch.cuda.Stream()


    def activate(self, indices, active=None, inverse=None, force=False):
        if not force: assert self.active_tokens is None, "need to call .update_embeddings() after optimizer step"
        if active is None:
                active, inverse = torch.unique(indices, return_inverse=True, sorted=True)
                self.active_tokens = active.cpu().to(torch.long)
        else:   self.active_tokens = active

        n = len(self.active_tokens)
        assert n <= self.active_vocab, f"OhMai found {n} > active_vocab"

        with torch.no_grad():  # load active tokens' embeddings to GPU
            self.update_stream.synchronize() # Finish update_embeddings first
            self.active_weight.data[:n] = self.weight[self.active_tokens]
        return inverse


    def update_embeddings(self):  
        assert self.active_weight.grad is not None # => grad đã chảy tới

        # Sử dụng stream để async transfer
        with torch.cuda.stream(self.update_stream):
            v = self.active_weight[:len(self.active_tokens)]
            self.weight[self.active_tokens] = v.cpu().to(self.weight.dtype)

        self.active_tokens = None # clear inactive data


    def forward(self, indices, active=None, inverse=None, force=False):
        inverse = self.activate(indices, active, inverse, force)
        x = OhMaiEmbFunction.apply(self.active_weight, inverse)
        return x


########################
##  TESTING  TESTING  ##
########################

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from liger_kernel import LigerEmbedding

    vocab, dim, ctx = 6400, 128, 32
 
    torch.manual_seed(1981)
    e0 = OhMaiEmbedding(vocab, dim).cuda()
 
    torch.manual_seed(1981)
    e1 =   nn.Embedding(vocab, dim).cuda()
 
    torch.manual_seed(1981)
    e2 = LigerEmbedding(vocab, dim).cuda()

    params = \
        list(e0.parameters()) + \
        list(e1.parameters()) + \
        list(e2.parameters())

    optimizer = torch.optim.AdamW(params, lr=0.1)

    for i in range(3):
        optimizer.zero_grad()
    
        x = torch.randint(0, ctx//2, (ctx,), dtype=torch.int16).cuda()
        
        losses = []
        for e in [e0, e1, e2]:
            y = e(x.long())
            # Tạo loss giả để có gradient
            target = torch.randn_like(y)
            loss = torch.nn.functional.mse_loss(y, target)
            losses.append(loss.item())
            loss.backward()  # Tính gradient

        # Kiểm tra gradient
        active_weight_clone = e0.active_weight.clone()
        assert e0.active_weight.grad is not None

        # Apply gradients
        optimizer.step()

        assert not torch.allclose(active_weight_clone, e0.active_weight), "active_weight không đổi"
        losses = [ f"{l:.4f}" for l in losses ]
        print(f"Optimizer step {i}, losses {', '.join(losses)}")

        e0.update_embeddings()
    # END FOR