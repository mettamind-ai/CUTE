import torch
from torch import nn, Tensor

################################################################
""" Ô Mai Nhúng: chỉ load tokens có trong current batch vào vram
- Giữ toàn bộ embedding matrix ở CPU
- Chỉ load embedding matrix của active_vocab vào vram => `active_weight`
- Có cơ chế mapping để biết token_id ở `active_weight` index nào
- Có thao tác để đổi active_vocab
"""

class OhMaiEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        ctx.save_for_backward(embeddings, indices)
        return embeddings[indices]

    @staticmethod
    @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        embeddings, indices = ctx.saved_tensors
        grad_weight = torch.zeros_like(embeddings)
        grad_weight.index_add_(0, indices, grad_output)
        return grad_weight, None


class OhMaiEmbedding(nn.Module):
    def __init__(self, vocab, dim, active_vocab=None):
        super().__init__()
        self.vocab = vocab

        self.weight = torch.randn(vocab, dim, device="cpu", dtype=torch.bfloat16)
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
    @torch.compiler.disable
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
        self.active_weight.data[ new_token_indices ] = \
        self.weight[new_tokens.cpu()].pin_memory().cuda(non_blocking=True)

        # Sử dụng stream để async transfer unuse_embeddings from GPU to CPU
        with torch.cuda.stream(self.update_stream):
            self.weight[unuse_tokens.cpu()] = unuse_weights.cpu()

        # Tạo inverse indices và trả về
        self.inverse_map[self.active] = torch.arange(len(self.active), device=indices.device)
        return self.inverse_map[indices]


    @torch.no_grad()
    @torch.compiler.disable
    def update_async_weight(self):
        self.weight[self.active.cpu().to(torch.long)] = self.active_weight[:len(self.active)].cpu()


    def forward(self, indices):
        inverse = self.activate(indices)
        return OhMaiEmbFunction.apply(self.active_weight, inverse)


############################################################
''' Oh My Head Làm tương tự Ô Mai Nhúng, giữ full head ở CPU
Load active_vocab trong GPU
Do lượng active_vocab x3-x5 lần Emb nên dự đoán IO sẽ bị nặng hơn, 
bù lại tiết kiệm rất nhiều vram và lượng computing save được lúc compute loss là nhiều.
'''
MAX_ACTIVE_VOCAB = 1024 * 32  # 32k tối ưu cho speed, và vừa đủ 1:3 -> 1:4 pos/ng
class OhMaiHead(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()

        self.active_vocab = vocab // 2

        if  self.active_vocab > MAX_ACTIVE_VOCAB:
            self.active_vocab = MAX_ACTIVE_VOCAB

        self.weight = torch.zeros(vocab, dim, device="cpu", dtype=torch.bfloat16)
        self.weight.requires_grad_(False)

        self.active = torch.arange(self.active_vocab, device='cuda')
        self.active.requires_grad_(False)
        self.alpha = torch.tensor(0.69, device='cuda') # :D

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
        self.maistream = torch.cuda.Stream()

    @torch.no_grad()
    @torch.compile()
    def get_active_tokens(self, indices):
        tokens, counts = torch.unique(indices, return_counts=True)
        self.running_freq[tokens] += counts
        self.total_tokens += counts.sum()
        empirical_freq = self.running_freq / self.total_tokens

        combined_score = self.alpha * self.running_freq + (1-self.alpha) * self.pretrained_norm     
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
        with torch.cuda.stream(self.maistream):
            curr_active   = self.get_active_tokens(indices)
            unuse_mask    = ~torch.isin(self.active, curr_active)
            unuse_tokens  = self.active[unuse_mask]

            unuse_indices = torch.nonzero(unuse_mask, as_tuple=False).flatten()
            self.new_tokens = curr_active[~torch.isin(curr_active, self.active)]
            self.new_token_indices = torch.nonzero(~torch.isin(self.active, curr_active), as_tuple=False).flatten()

            assert len(self.new_token_indices) == len(self.new_tokens)
            self.active[self.new_token_indices] = self.new_tokens

            self.new_tokens = self.new_tokens.cpu()
            self.weight.data[unuse_tokens.cpu()] = self.active_weight.data[unuse_indices].cpu()

            # Tạo inverse indices và trả về
            self.inverse_map[self.active] = torch.arange(len(self.active), device=indices.device)
            return self.inverse_map[indices]

    @torch.no_grad()
    @torch.compiler.disable
    def update_new_tokens_weight(self):
        self.active_weight.data[ self.new_token_indices ] = \
        self.weight.data[ self.new_tokens ].pin_memory().cuda(non_blocking=True)

    @torch.no_grad()
    @torch.compiler.disable
    def update_async_weight(self):
        self.weight.data[self.active.cpu().to(torch.long)] = self.active_weight.data[:len(self.active)].cpu()
