#!/usr/bin/env python3
''' bản thuần pytorch
import torch
from torch import nn, Tensor

class OhMaiEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        ctx.save_for_backward(embeddings, indices)
        return embeddings[indices]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        embeddings, indices = ctx.saved_tensors
        grad_weight = torch.zeros_like(embeddings)
        grad_weight.index_add_(0, indices, grad_output)
        return grad_weight, None
'''

import torch, triton
import triton.language as tl
from torch import nn, Tensor

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
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        vocab, hidim = indices.numel(), embeddings.shape[1]
        output = torch.empty(vocab, hidim, device=indices.device, dtype=embeddings.dtype,)

        blsz = triton.next_power_of_2(min(128, hidim))
        grid = ( triton.cdiv(vocab, blsz), triton.cdiv(hidim, blsz), )

        embedding_forward_kernel[grid](embeddings, indices, output, vocab, hidim, blsz)
        ctx.save_for_backward(indices, embeddings)
        return output

    @staticmethod
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

        # Pinned Memory → GPU Memory _vs_ CPU Memory → Staging Buffer → GPU Memory
        self.weight = torch.randn(vocab, hidim, device="cpu", pin_memory=True, dtype=torch.bfloat16)
        self.weight.requires_grad_(False)

        if active_vocab is None: active_vocab = vocab // 2  # a safe assumption
        w = torch.empty(active_vocab, self.hidim, device="cuda", dtype=self.weight.dtype)

        self.active_weight = nn.Parameter(w)
        self.active_vocab = active_vocab

        self.active = torch.tensor([], dtype=torch.long, device='cuda')
        self.active.requires_grad_(False)

        self.inverse_map = torch.full((vocab,), -1, dtype=torch.long, device='cuda')
        self.inverse_map.requires_grad_(False)

        # Khởi tạo CUDA stream cho async transfer
        self.update_stream = torch.cuda.Stream()


    @torch.no_grad()
    def activate(self, indices):
        prev_active = self.active.clone()
        curr_active = torch.unique(indices).long()
        assert len(curr_active) <= self.active_vocab, f"OhMai found {len(curr_active)} > active_vocab"

        # Dùng unuse_tokens và unuse_embeds để update vào weight trên CPU
        unuse_mask    = ~torch.isin(prev_active, curr_active)
        unuse_tokens  = prev_active[unuse_mask]

        unuse_indices = torch.nonzero(unuse_mask, as_tuple=False).flatten()
        unuse_embeds  = self.active_weight.data[unuse_indices].clone()

        # Cập nhật self.active chỉ ở những phần tử mới trong curr_active
        pad_size = len(curr_active) - len(prev_active)
        self.active = torch.nn.functional.pad(self.active, (0, pad_size), value=-1)

        new_mask = ~torch.isin(self.active, curr_active)
        new_token_indices = torch.nonzero(new_mask, as_tuple=False).flatten()

        mask = ~torch.isin(curr_active, prev_active)
        new_token_indices = torch.nonzero(mask, as_tuple=False).flatten()
        new_tokens  = curr_active[new_token_indices]
        self.active[new_token_indices] = new_tokens
        
        # Update new token embeddings
        self.update_stream.synchronize() # đồng bộ hoá lần update trước
        new_embs = self.weight[new_tokens.cpu()].to(device=self.active_weight.device)
        self.active_weight.data[ new_token_indices ] = new_embs

        # Sử dụng stream để async transfer unuse_embeddings from GPU to CPU
        with torch.cuda.stream(self.update_stream):
            self.weight[unuse_tokens.cpu()] = unuse_embeds.cpu()

        # Tạo inverse indices và trả về
        self.inverse_map[self.active] = torch.arange(len(self.active), device=indices.device)
        return self.inverse_map[indices]


    def update_embeddings(self):
        self.update_stream.synchronize()
        self.weight[self.active.cpu().to(torch.long)] = self.active_weight[:len(self.active)].cpu()


    def forward(self, indices):
        inverse = self.activate(indices)
        return OhMaiEmbFunction.apply(self.active_weight, inverse)


########################
##  TESTING  TESTING  ##
########################

if __name__ == "__main__":
    # import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # extend path to real `..`
    vocab, dim, ctx = 6400, 128, 32
 
    torch.manual_seed(1981)
    e0 = OhMaiEmbedding(vocab, dim).cuda()
 
    torch.manual_seed(1981)
    e1 =   nn.Embedding(vocab, dim).cuda()
 
    params = \
        list(e0.parameters()) + \
        list(e1.parameters())

    optimizer = torch.optim.AdamW(params, lr=0.1)

    for i in range(5):
        optimizer.zero_grad()
    
        x = torch.randint(0, ctx//2, (ctx,), dtype=torch.int16).cuda()
        
        losses = []
        for e in [e0, e1]:
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
