#!/usr/bin/env python3
import torch, triton
import triton.language as tl
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
    # import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # extend path to `..`
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