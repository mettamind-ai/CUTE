#!/usr/bin/env python3
import torch
from torch import nn, Tensor

class OhMaiEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        ctx.save_for_backward(indices, embeddings)
        return embeddings[indices]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, embeddings = ctx.saved_tensors
        grad_weight = torch.zeros_like(embeddings)
        grad_weight[indices] = grad_output
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
        prev_active = self.active
        self.active = torch.unique(indices).long()
        assert len(self.active) <= self.active_vocab, f"OhMai found {len(self.active)} > active_vocab"

        # Kiểm tra xem có phần tử nào của prev_active nào không có trong self.active không?
        unuse_mask    = ~torch.isin(prev_active, self.active)
        unuse_indices = torch.nonzero(unuse_mask, as_tuple=False).squeeze(-1)

        # Dùng unuse_tokens và unuse_embeds để update vào weight trên CPU
        unuse_tokens  = prev_active[unuse_mask]  # clone đê tạo 1 bản copy
        unuse_embeds  = self.active_weight.data[unuse_indices].clone()
        reuse_indices = torch.nonzero(~unuse_mask, as_tuple=False).squeeze(-1)

        # Kiểm tra xem có phần tử nào của self.active nào có trong prev_active không?
        reuse_mask  = torch.isin(self.active, prev_active)
        reuse, neww = self.active[reuse_mask], self.active[~reuse_mask]
        self.active = torch.cat([reuse, neww])

        reuse = self.active_weight.data[reuse_indices].clone()
        self.update_stream.synchronize() # đồng bộ hoá lần update trước
        newww = self.weight[neww.cpu()].to(device=self.active_weight.device)
        self.active_weight.data[ : len(self.active) ] = torch.cat([reuse, newww])

        # Sử dụng stream để async transfer unuse_embeddings from GPU to CPU
        with torch.cuda.stream(self.update_stream):
            self.weight[unuse_tokens.cpu().to(torch.long)] = unuse_embeds.cpu()

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
