''' Oh My Head Làm tương tự Ô Mai Nhúng, giữ full head ở CPU
Load active_vocab trong GPU
Do lượng active_vocab x3-x5 lần Emb nên dự đoán IO sẽ bị nặng hơn, 
bù lại tiết kiệm rất nhiều vram và lượng computing save được lúc compute loss là nhiều.
'''
import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # extend path to real `..`
from optimus import Int8MixedLinear

import torch
from torch import nn

MAX_ACTIVE_VOCAB = 32768 # 32768 tối ưu cho speed, 51200 cân bằng speed 1:5 pos/neg ratio

@torch.compiler.disable
class OhMaiHead(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()

        self.active_vocab = vocab // 2

        if  self.active_vocab > MAX_ACTIVE_VOCAB:
            self.active_vocab = MAX_ACTIVE_VOCAB

        # Pinned Memory → GPU Memory _vs_ CPU Memory → Staging Buffer → GPU Memory
        self.weight = torch.zeros(vocab, dim, device="cpu", pin_memory=True, dtype=torch.bfloat16)
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

        # Khởi tạo CUDA stream cho async transfer
        self.update_stream = torch.cuda.Stream()

        ## Hot tokens config playground/683d5a3ef829ed36a76977b1
        self.hot_size = self.active_vocab // 2  # ~16k hot tokens
        self.hot_tokens = torch.arange(self.hot_size, device='cuda')
        self.hot_tokens.requires_grad_(False)  # Không cần gradient
        self.steps_count = 0


    @torch.no_grad
    def get_hot_tokens(self):
        self.steps_count += 1
        if self.steps_count % 20 != 0: return self.hot_tokens

        # Get new hot tokens based on frequency
        self.hot_tokens = torch.topk(self.running_freq, self.hot_size).indices
        
        # Update active_weight
        self.update_async_weight()  # sync with cpu weight first
        self.active[:self.hot_size] = self.hot_tokens
        self.active_weight.data[:self.hot_size] = self.weight[self.hot_tokens.cpu()].cuda()

        return self.hot_tokens


    @torch.no_grad
    def get_active_tokens(self, indices):
        batch_tokens, counts = torch.unique(indices, return_counts=True)
        self.running_freq[tokens] += counts
        self.total_tokens += counts.sum()
        empirical_freq = self.running_freq / self.total_tokens

        combined_score = self.alpha * empirical_freq + (1-self.alpha) * self.pretrained_norm     
        sample_probs = combined_score.pow(0.75) # Smooth với power 0.75; Từ Word2Vec paper
        sample_probs = sample_probs / sample_probs .sum()

        # essential_tokens = combine hot + batch tokens (unique)
        hot_tokens = self.get_hot_tokens()
        batch_cold = batch_tokens[~torch.isin(batch_tokens, hot_tokens)]
        essential = torch.cat([hot_tokens, batch_cold])

        mask = torch.ones_like(sample_probs)
        mask[essential] = 0

        masked_probs = sample_probs * mask
        masked_probs = masked_probs / masked_probs.sum()

        need_neg = self.active_vocab - len(essential)
        neg_tokens = torch.multinomial(masked_probs, need_neg, replacement=False)

        # IMPORTANT: Return hot tokens FIRST
        return torch.cat([hot_tokens, batch_cold, neg_tokens])



    @torch.no_grad()
    def activate(self, indices):
        curr_active = self.get_active_tokens(indices)

        # Hot tokens (0 → hot_size) giữ nguyên, không swap
        # Chỉ xử lý cold zone (hot_size → active_vocab)        
        cold_start, cold_end = self.hot_size, self.active_vocab
        
        # Current cold tokens
        cold_curr = curr_active[cold_start:cold_end]
        cold_prev = self.active[cold_start:cold_end]
        
        # Find cold tokens to swap out  
        unuse_mask = ~torch.isin(cold_prev, cold_curr)
        unuse_indices = torch.nonzero(unuse_mask).flatten() + cold_start
        unuse_tokens = self.active[unuse_indices]
        
        # Update active array (only cold part)
        self.active[cold_start:cold_end] = cold_curr

        unuse_indices = torch.nonzero(unuse_mask, as_tuple=False).flatten()
        unuse_weights = self.active_weight.data[unuse_indices].clone()

        new_tokens = curr_active[~torch.isin(curr_active, self.active)]
        new_token_indices = torch.nonzero(~torch.isin(self.active, curr_active), as_tuple=False).flatten()

        assert len(new_token_indices) == len(new_tokens), f"{len(new_token_indices)} != {len(new_tokens)}"
        self.active[new_token_indices] = new_tokens

        # Update new token weights
        self.update_stream.synchronize() # đồng bộ hoá lần update trước
        new_weights = self.weight[new_tokens.cpu()].to(device=self.active_weight.device)
        self.active_weight.data[ new_token_indices ] = new_weights

        # Sử dụng stream để async transfer unuse_weights from GPU to CPU
        with torch.cuda.stream(self.update_stream):
            self.weight[unuse_tokens.cpu()] = unuse_weights.cpu()

        # Tạo inverse indices và trả về
        self.inverse_map[self.active] = torch.arange(len(self.active), device=indices.device)
        return self.inverse_map[indices]

    def update_async_weight(self):
        self.update_stream.synchronize()
        self.weight[self.active.cpu().to(torch.long)] = self.active_weight[:len(self.active)].cpu()
