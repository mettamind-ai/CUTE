''' Oh My Head Làm tương tự Ô Mai Nhúng, giữ full head ở CPU
Load active_vocab trong GPU
Do lượng active_vocab x3-x5 lần Emb nên dự đoán IO sẽ bị nặng hơn, 
bù lại tiết kiệm rất nhiều vram và lượng computing save được lúc compute loss là nhiều.
'''
import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # extend path to real `..`
from optimus import Int8MixedLinear

import torch
from torch import nn

# 32k tối ưu cho speed, và vừa đủ 1:3 -> 1:4 pos/ng
MAX_ACTIVE_VOCAB = 32*1024 

@torch.compiler.disable
class OhMaiHead(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.vocab_size = vocab
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
        self.hot_size = self.active_vocab//2
        self.hot_tokens = torch.arange(self.hot_size, device='cuda')
        self.hot_tokens.requires_grad_(False)  # Không cần gradient
        self.steps_count = 0


    @torch.no_grad
    def get_hot_tokens(self):
        self.steps_count += 1
        if self.steps_count % 30 != 0: return self.hot_tokens

        # Get new hot tokens based on frequency
        new_hot_tokens = torch.topk(self.running_freq, self.hot_size).indices
        if torch.equal(new_hot_tokens, self.hot_tokens): return self.hot_tokens

        old_hot = self.hot_tokens
        self.hot_tokens = new_hot_tokens

        # Find which positions changed
        position_changed = (new_hot_tokens != old_hot)
        changed_indices = position_changed.nonzero().flatten()

        if len(changed_indices) > 0:
            # Sync old tokens at changed positions
            with torch.cuda.stream(self.update_stream):
                self.weight[old_hot[changed_indices].cpu()] = self.active_weight[changed_indices].cpu()

            # Đồng bộ GPU active_weight trước khi làm việc khác
            new_changed = new_hot_tokens[changed_indices]
            self.active[changed_indices] = new_changed
            self.active_weight.data[changed_indices] = self.weight[new_changed.cpu()].cuda()

        # Trả về sau khi đã đồng bộ active_weight
        return self.hot_tokens

    def update_new_tokens_weight(self):
        new_weights = self.weight[ self.new_tokens.cpu() ].to(device=self.active_weight.device)
        self.active_weight.data[ self.new_token_indices  ] = new_weights

    @torch.no_grad
    def get_active_tokens(self, indices):
        if self.steps_count % 50000 != 0:
            batch_tokens = torch.unique(indices)
        else: # 5 steps update 1 lần
            batch_tokens, counts = torch.unique(indices, return_counts=True)
            self.running_freq[batch_tokens] += counts
            self.total_tokens += counts.sum()
            empirical_freq = self.running_freq / self.total_tokens

            combined_score = self.alpha * empirical_freq + (1-self.alpha) * self.pretrained_norm     
            sample_probs = combined_score.pow(0.75) # Smooth với power 0.75; Từ Word2Vec paper
            self.sample_probs = sample_probs / sample_probs.sum()

        # essential_tokens = combine hot + batch tokens (unique)
        hot_tokens = self.get_hot_tokens()
        batch_cold = batch_tokens[~torch.isin(batch_tokens, hot_tokens)]
        essential = torch.cat([hot_tokens, batch_cold])
        need_neg = self.active_vocab - len(essential)

        ## Lấy mẫu neg_tokens nằm ngoài essential dựa trên sample_probs 
        # noise = torch.rand_like(self.sample_probs)
        # perturbed = self.sample_probs * noise
        # perturbed[essential] = 0
        # neg_tokens = torch.topk(perturbed, need_neg).indices

        # Lấy ngẫu nhiên
        perm = torch.randperm(vocab_size)
        neg_tokens = perm[~torch.isin(perm, essential)][:need_neg]

        # IMPORTANT: Return hot tokens FIRST
        return torch.cat([essential, neg_tokens])



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

        self.new_tokens = curr_active[~torch.isin(curr_active, self.active)]
        self.new_token_indices = torch.nonzero(~torch.isin(self.active, curr_active), as_tuple=False).flatten()

        assert len(self.new_token_indices) == len(self.new_tokens)
        self.active[self.new_token_indices] = self.new_tokens

        # Sử dụng stream để async transfer unuse_weights from GPU to CPU
        with torch.cuda.stream(self.update_stream):
            self.weight[unuse_tokens.cpu()] = unuse_weights.cpu()

        # Tạo inverse indices và trả về
        self.inverse_map[self.active] = torch.arange(len(self.active), device=indices.device)
        return self.inverse_map[indices]


    def update_async_weight(self):
        # self.update_stream.synchronize()  # đồng bộ hoá lần update trước
        self.weight[self.active.cpu().to(torch.long)] = self.active_weight[:len(self.active)].cpu()
