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
    token_offsets = tl.program_id(0)*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    embed_offsets = tl.program_id(1)*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    token_mask = token_offsets < vocab
    embed_mask = embed_offsets < hidim

    tokens = tl.load(tokens_ptr + token_offsets, mask=token_mask)
    mask = token_mask[:, None] & embed_mask[None, :]

    offsets = tokens[:, None]*hidim + embed_offsets[None, :] # M x N
    embeddings = tl.load(embeddings_ptr + offsets, mask=mask)

    offsets = token_offsets[:, None]*hidim + embed_offsets[None, :]
    tl.store(output_ptr + offsets, embeddings, mask=mask)


@triton.jit
def embedding_backward_kernel(
    grad_output_ptr,
    grad_weight_ptr,
    indices_ptr,
    vocab,
    hidim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < vocab
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < hidim

    grad_output = tl.load(
        grad_output_ptr + offsets_m[:, None] * hidim + offsets_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    grad_weight_offsets = indices[:, None] * hidim + offsets_n[None, :]

    tl.atomic_add(
        grad_weight_ptr + grad_weight_offsets,
        grad_output,
        mask=mask_m[:, None] & mask_n[None, :],
    )


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
