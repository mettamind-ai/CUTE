""" Modded from github.com/linkedin/liger-Kernel/blob/main/src/liger_kernel/ops/experimental/embedding.py
Với n batches of data thì không cần phải load hết toàn bộ embedding matrix vào vram
- Giữ toàn bộ embedding matrix ở CPU
- Chỉ load embedding matrix của active_vocab vào vram => `active_embbedings`
- Có cơ chế mapping để biết token_id ở `active_embbedings` index nào
- Có thao tác để đổi active_vocab
"""
import functools
import torch, triton
import triton.language as tl
from torch import nn, Tensor

_to_contiguous = lambda x: x if not isinstance(x, Tensor) else x.contiguous()
def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        args = [_to_contiguous(arg) for arg in args]
        kwarg= {k: _to_contiguous(v) for k, v in kwargs.items()}
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
    # hiện tại atomic_add mới chỉ hỗ trợ float32, khi triton được update sẽ dùng lại được

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


class OhMaiEmbedding(nn.Module):
    def __init__(self, vocab, hidim):
        super().__init__()
        self.vocab = vocab
        self.weight = nn.Parameter(torch.randn(vocab, hidim).bfloat16())
        # Cần kích hoạt active_vocab và init_active weight mỗi lần forward
        self.active_vocab = 0
        self.active_weight = None
    
    # def prepare_active_weight():

    def forward(self, indices):
        return OhMaiEmbFunction.apply(self.weight, indices)
