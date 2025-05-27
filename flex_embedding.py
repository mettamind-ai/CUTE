""" Modded from github.com/linkedin/liger-Kernel/blob/main/src/liger_kernel/ops/experimental/embedding.py
Với n batches of data thì không cần phải load hết toàn bộ embedding matrix vào vram
- Giữ toàn bộ embedding matrix ở CPU
- Chỉ load embedding matrix của curr_vocab vào vram => `curr_emb_matrix`
- Có cơ chế mapping để biết token_id ở `curr_emb_matrix` index nào
- Có thao tác để đổi curr_vocab
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
    embeddings_ptr, tokens_ptr, # trỏ tới token_ids cần lấy embedding values
    output_ptr,                 # x0, hay embeddings của batch hiện tại
    vocab, hidim : tl.constexpr,# vocab size x hidden dim = kích thước embedding matrix
    BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_N : tl.constexpr, # kích thước khối
):
    tok_offsets  = tl.program_id(0)*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    feat_offsets = tl.program_id(1)*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Đảm bảo không load và store embedding vượt ngoài khuôn khổ vocab x hidim
    tok_mask  =  tok_offsets < vocab  # token < vocab size
    feat_mask = feat_offsets < hidim  # feat  < hidden dim
    emb_mask  = tok_mask[:, None] & feat_mask[None, :]

    tok_indexes = tl.load(tokens_ptr + tok_offsets, mask=tok_mask)
    emb_offsets = tok_indexes[:, None]*hidim + feat_offsets[None, :] # vị trí trong embedding matrix
    out_offsets = tok_offsets[:, None]*hidim + feat_offsets[None, :] # vị trí trong x0

    embeddings = tl.load(embeddings_ptr + emb_offsets, mask=emb_mask) # emb_ là sparse
    tl.store(output_ptr + out_offsets, embeddings, mask=emb_mask)     # out_ là continuous


@triton.jit
def embedding_backward_kernel(
    grad_output_ptr, grad_weight_ptr, # grad_weight là embeddings_grad
    tokens_ptr,
    vocab, hidim : tl.constexpr,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
):
    tok_offsets  = tl.program_id(0)*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    feat_offsets = tl.program_id(1)*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Đảm bảo không load và store embedding vượt ngoài khuôn khổ vocab x hidim
    tok_mask  =  tok_offsets < vocab  # token < vocab size
    feat_mask = feat_offsets < hidim  # feat  < hidden dim
    emb_mask  = tok_mask[:, None] & feat_mask[None, :]

    tok_indexes = tl.load(tokens_ptr + tok_offsets, mask=tok_mask)
    emb_offsets = tok_indexes[:, None]*hidim + feat_offsets[None, :] # vị trí trong embedding matrix
    out_offsets = tok_offsets[:, None]*hidim + feat_offsets[None, :] # vị trí trong x0

    grad_output = tl.load(grad_output_ptr + out_offsets, mask=emb_mask)
    tl.atomic_add(grad_weight_ptr + emb_offsets, grad_output, mask=emb_mask)


class FlexEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        vocab = indices.numel()
        hidim = embeddings.shape[1]
        output = torch.empty(vocab, hidim, device=indices.device, dtype=embeddings.dtype,)

        _n = min(128, hidim)
        _n = triton.next_power_of_2(_n)
        grid = ( triton.cdiv(vocab, _n), triton.cdiv(hidim, _n), )

        embedding_forward_kernel[grid](
            embeddings, indices, output,
            vocab, hidim, _n, _n,
        )

        ctx.save_for_backward(indices, embeddings)
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        indices, embedding_table = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, embedding_table.shape[1])

        grad_weight = torch.zeros_like(embedding_table)

        vocab = indices.numel()
        hidim = embedding_table.shape[1]

        BLOCK_SIZE_M = triton.next_power_of_2(min(128, hidim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, hidim))
        grid = (
            triton.cdiv(vocab, BLOCK_SIZE_M),
            triton.cdiv(hidim, BLOCK_SIZE_N),
        )

        embedding_backward_kernel[grid](
            grad_output,
            grad_weight,
            indices,
            vocab,
            hidim=hidim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        return grad_weight, None


class FlexEmbedding(nn.Module):
    def __init__(self, num_embeddings, hidim, padding_idx: int = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidim = hidim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.randn(num_embeddings, hidim).float())

        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, indices):
        embedded = FlexEmbeddingFunction.apply(self.weight, indices)
        if self.padding_idx is not None:
            embedded = embedded.clone()
            embedded[indices == self.padding_idx] = 0
        return embedded
