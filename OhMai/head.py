''' Oh My Head Làm tương tự Ô Mai Nhúng, giữ full head ở CPU
Load active_vocab trong GPU
Do lượng active_vocab x3-x5 lần Emb nên dự đoán IO sẽ bị nặng hơn, 
bù lại tiết kiệm rất nhiều vram và lượng computing save được lúc compute loss là nhiều.
'''
import torch
from torch import nn

# 32k tối ưu cho speed, và vừa đủ 1:3 -> 1:4 pos/ng
MAX_ACTIVE_VOCAB = 1024 * 32

@torch.compiler.disable
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


    @torch.no_grad
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
            self.weight[unuse_tokens.cpu()] = self.active_weight.data[unuse_indices].cpu()

            # Tạo inverse indices và trả về
            self.inverse_map[self.active] = torch.arange(len(self.active), device=indices.device)
            return self.inverse_map[indices]

    def update_new_tokens_weight(self):
        self.active_weight.data[ self.new_token_indices ] = \
        self.weight[ self.new_tokens ].pin_memory().cuda(non_blocking=True)

    def update_async_weight(self):
        self.weight[self.active.cpu().to(torch.long)] = self.active_weight[:len(self.active)].cpu()
