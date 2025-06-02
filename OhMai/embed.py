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
    def __init__(self, vocab, dim, active_vocab=None):
        super().__init__()
        self.vocab = vocab

        # Pinned Memory → GPU Memory _vs_ CPU Memory → Staging Buffer → GPU Memory
        self.weight = torch.randn(vocab, dim, device="cpu", pin_memory=True, dtype=torch.bfloat16)
        self.weight.requires_grad_(False)

        if active_vocab is None: active_vocab = vocab // 2  # a safe assumption
        w = torch.empty(active_vocab, dim, device="cuda", dtype=self.weight.dtype)

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
        curr_active = torch.unique(indices).long()
        assert len(curr_active) <= self.active_vocab, f"OhMai found {len(curr_active)} > active_vocab"

        # Dùng unuse_tokens và unuse_weights để update vào weight trên CPU
        unuse_mask    = ~torch.isin(self.active, curr_active)
        unuse_tokens  = self.active[unuse_mask]

        unuse_indices = torch.nonzero(unuse_mask, as_tuple=False).flatten()
        unuse_weights  = self.active_weight.data[unuse_indices].clone()

        # Cập nhật self.active chỉ ở những phần tử mới trong curr_active
        pad_size = len(curr_active) - len(self.active)
        self.active = torch.nn.functional.pad(self.active, (0, pad_size), value=-1)
        assert len(self.active) == len(curr_active)

        new_tokens = curr_active[~torch.isin(curr_active, self.active)]
        new_token_indices = torch.nonzero(~torch.isin(self.active, curr_active), as_tuple=False).flatten()
        assert len(new_token_indices) == len(new_tokens), f"{len(new_token_indices)} != {len(new_tokens)}"
        self.active[new_token_indices] = new_tokens

        # Update new token embeddings
        self.update_stream.synchronize() # đồng bộ hoá lần update trước
        new_weights = self.weight[new_tokens.cpu()].to(device=self.active_weight.device)
        self.active_weight.data[ new_token_indices ] = new_weights

        # Sử dụng stream để async transfer unuse_embeddings from GPU to CPU
        with torch.cuda.stream(self.update_stream):
            self.weight[unuse_tokens.cpu()] = unuse_weights.cpu()

        # Tạo inverse indices và trả về
        self.inverse_map[self.active] = torch.arange(len(self.active), device=indices.device)
        return self.inverse_map[indices]


    def update_async_weight(self):
        self.update_stream.synchronize()
        self.weight[self.active.cpu().to(torch.long)] = self.active_weight[:len(self.active)].cpu()


    def forward(self, indices):
        inverse = self.activate(indices)
        return OhMaiEmbFunction.apply(self.active_weight, inverse)
