#!/usr/bin/env python3
## GPT for the WIN (cải biên từ modded nanogpt)

import os, torch # Tránh lỗi và tăng tốc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.coordinate_descent_tuning = True
torch.set_default_dtype(torch.bfloat16)

from torch import Tensor, nn
import torch.nn.functional as F

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

@torch.no_grad()
def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5) # 0.5 is a bit better
    bound = (3 ** 0.5) * std         # than default 1/sqrt(3)
    return w.uniform_(-bound, bound)


####################################
##  ReLuSquareMLP Channel Mixing  ##
####################################

class ReLuSquareMLP(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        hdim = int(3 * dim) 

        self.fc = nn.Linear(dim, hdim, bias=False)
        self.proj = nn.Linear(hdim, dim, bias=False)
        
        with torch.no_grad():
            self.fc.weight.copy_(init_linear(torch.empty(hdim, dim)))
            self.proj.weight.zero_()
        
        # Add weight decay multiplier attribute to the weights
        self.fc.weight.wd_mul = 2.0  # điều chỉnh hệ số weight decay
        self.proj.weight.wd_mul = 2.0  # gấp đôi so với mặc định 

    def forward(self, x:Tensor):
        y = self.fc(x)
        y = F.relu(y).square() 
        x = self.proj(y)
        return x


#####################################
## CausalSelfAttention Time Mixing ##
#####################################

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        base, half, dtype = (1/1024), (dim//4), torch.float32
        angular_freq = base  **  torch.linspace(0, 1, steps=half, dtype=dtype)
        angular_freq = torch.cat([angular_freq, torch.zeros(half, dtype=dtype)])
        # Tần số góc, nửa đầu giảm dần từ 1 tới base và nửa còn lại là zeros

        positions = torch.arange(max_seq_len, dtype=dtype)
        theta = torch.einsum("i,j -> ij", positions, angular_freq)
        # theta[i, j] = positions[i] * angular_freq[j]

        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)


    def forward(self, x_BTHD: Tensor):
        seq_len = x_BTHD.size(-3) # batch, T seq_len, head, dim (of head)
        assert self.cos.size(0) >= seq_len, f"{self.cos.size(0)} >= {seq_len}?"

        cos = self.cos[None, :seq_len, None, :] # [1, seq_len, 1, dim]
        sin = self.sin[None, :seq_len, None, :] # [1, seq_len, 1, dim]

        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)

        y1 = x1 * (+cos) + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim:int, num_heads:int, num_kv_heads:int, 
            seq_len:int, head_dim=128, window=None):
        super().__init__() # dim=hidden_size=embedding=feature=representation

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim

        if window: # SWA chậm hơn full attn
                l, w, mask = seq_len, window, torch.zeros(l, l)
                for i in range(l): mask[i, max(0, i-w) : min(l, i+w+1)] = 1
                self.attn_mask = mask
        else:   self.attn_mask = None

        q_inner_dim = num_heads * head_dim
        kv_inner_dim = num_kv_heads * head_dim
                
        # Create separate projection matrices for queries vs key/values
        self.q_proj = nn.Linear(dim, q_inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_inner_dim, bias=False)
        self.out_proj = nn.Linear(q_inner_dim, dim, bias=False)

        # Set the weights directly
        with torch.no_grad():
            self.q_proj.weight.copy_(init_linear(torch.empty(q_inner_dim, dim)))
            self.k_proj.weight.copy_(init_linear(torch.empty(kv_inner_dim, dim)))
            self.v_proj.weight.copy_(init_linear(torch.empty(kv_inner_dim, dim)))
            self.out_proj.weight.zero_() # zero init

        self.rotary = Rotary(head_dim, seq_len)
        self.attn_scale = 0.12

    """ Implement casual_conv1d đơn giản
        self.conv_kernel = 3
        self.kv_conv = torch.ones(
            kv_inner_dim,       # số kênh (C / D / hidden dim)
            1,                  # số kênh đầu vào cho mỗi nhóm
            self.conv_kernel,   # kích thước cửa sổ trượt conv
        ).cuda() / self.conv_kernel # khởi tạo 1 / k => avg

    def causal_moving_avg(self, x):
        B, T, C = x.shape
        x = x.view(B, C, T)
        pad_left = self.conv_kernel - 1
        y = F.pad(x, (pad_left, 0))
        y = F.conv1d(y, self.kv_conv, groups=C)
        return (x + y).view(B, T, C)
    # """

    def forward(self, x:Tensor, v_emb:Tensor|None, sa_lambdas:Tensor):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)      # B, T, C
        # k, v = self.causal_moving_avg(k), self.causal_moving_avg(v) # B, T, C

        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim
        B, T, C   = k.shape; assert C == Hkv * D

        ## Chuyển q, k, v hành x_BTHD
        q = q.view(B, T, H,   D)
        k = k.view(B, T, Hkv, D)
        v = v.view(B, T, Hkv, D)

        q, k, v = norm(q), norm(k), norm(v)
        q, k = self.rotary(q), self.rotary(k)

        if sa_lambdas is not None and v_emb is not None:
            # Trộn value với value embedding (sa = self-attention)
            v = sa_lambdas[0]*v + sa_lambdas[1]*v_emb.view(B, T, Hkv, D)

        # Make tensors contiguous and transpose for attention
        q = q.transpose(1, 2).contiguous()  # BTHD -> BHTD
        k = k.transpose(1, 2).contiguous()  # BTHD -> BHTD
        v = v.transpose(1, 2).contiguous()  # BTHD -> BHTD
        
        # Repeat KV heads to match query head count (GQA)
        if self.num_kv_groups > 1:
            k = torch.repeat_interleave(k, repeats=self.num_kv_groups, dim=1)
            v = torch.repeat_interleave(v, repeats=self.num_kv_groups, dim=1)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True, attn_mask=self.attn_mask,
            dropout_p=0.0, scale=self.attn_scale,
        )
        
        # Transpose back to original shape [B, T, H, D]
        attn_output = attn_output.transpose(1, 2).contiguous()

        y = attn_output.reshape(B, T, H * D)
        y = self.out_proj(y)  # y có shape (B, T, dim)
        return y  # trả về y có shape giống hệt x đầu vào


##############################
## Transformer for the WIN  ##
##############################

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, max_seq_len, head_dim=128):
        super().__init__()
        self.mlp = ReLuSquareMLP(dim)
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, max_seq_len, head_dim=head_dim)

    def forward(self, x:Tensor, te, ve, lambdas, sa_lambdas):
        x = lambdas[0]*x + lambdas[1]*te     # trộn với tok emb
        x = x + self.attn(x, ve, sa_lambdas) # residual connect
        x = x + self.mlp(norm(x))            # residual connect
        return x


class Future(nn.Module):
    """ Dự đoán xa hơn 1 token, ideas from Multi-Token Prediction, DeepSeek và MiMo papers """
    def __init__(self, dim, num_heads, num_kv_heads, max_seq_len, head_dim=128):
        super().__init__()
        self.block = Block(dim, num_heads, num_kv_heads, max_seq_len, head_dim=head_dim)
        self.proj = nn.Linear(2*dim, dim, bias=False)
        with torch.no_grad(): self.proj.weight.copy_(init_linear(torch.empty(dim, 2*dim)))

    def forward(self, x, x0, te, ve, l, s):
        # trộn feat của last layer với token embed gốc (x0)
        x = torch.cat((norm(x), x0), dim=2)
        x = self.proj(x) # mlp mixer
        x = self.block(x, te, ve, l, s)
        return norm(x)


class WinGPT(nn.Module):
    def __init__(self, vocab_size:int, n_layers:int, num_heads:int, num_kv_heads:int,
            dim:int, max_seq_len:int, head_dim=128, ve=4, te=1, exits=2, future_percent=0):
        super().__init__()
        self.n_layers = n_layers

        blocks = [ Block(dim, num_heads, num_kv_heads, max_seq_len, head_dim) for _ in range(n_layers) ]
        self.future_ratio = future_percent / 100.0

        if future_percent > 0: blocks.append(Future(dim, num_heads, num_kv_heads, max_seq_len, head_dim))
        self.blocks = nn.ModuleList(blocks)
        n_blks = len(self.blocks)

        if ve > n_layers: ve = n_layers
        if te > n_layers: te = n_layers

        dd = dim//2
        self.val_embs = nn.ModuleList([ nn.Embedding(vocab_size, num_kv_heads * head_dim) for _ in range(ve) ])     
        self.tok_embs = [ nn.Embedding(vocab_size, dd) for _ in range(te) ]
        self.tok_proj = [ nn.Linear(dd, dim, bias=False) for _ in range(te) ]

        with torch.no_grad():
            for x in self.tok_proj:
                x.weight.copy_(init_linear(torch.empty(dim, dd)))

        self.tok_embs[0] = nn.Embedding(vocab_size, dim) # full for first tok emb
        self.tok_proj[0] = nn.Module() # placeholder, do nothing

        self.tok_embs = nn.ModuleList(self.tok_embs)
        self.tok_proj = nn.ModuleList(self.tok_proj)

        self.scalars = nn.Parameter(torch.cat([
          torch.ones(n_blks),  # skip_weights khởi tạo là 1 cho tất cả layers
          *[torch.tensor([1.0, 0.0]) for _ in range(n_blks)], # token emb mix
          *[torch.tensor([0.5, 0.5]) for _ in range(n_blks)], # value emb mix
        ]))

        self.skip_from = { (n_layers-i): i for i in range(2, (n_layers-1) // 2, 2) }
        print("WinGPT.skip_from", self.skip_from)
         
        ## Future and tied head(s)
        # Vì head dùng để phóng chiếu embedding ra token nên có thể dùng chung cho mọi
        # loại tác vụ predict token bao gồm các điểm exits và next of next token prediction   
        tied_head = nn.Linear(dim, vocab_size, bias=False)
        with torch.no_grad(): tied_head.weight.zero_()
        self.lm_heads = nn.ModuleList([ tied_head for i in range(exits) ])

        # Điểm predict next token và hệ số cho từng điểm
        assert exits in [1, 2, 3]
        if exits == 1: self.exit_ids = [n_layers-1 ]
        if exits == 2: self.exit_ids = [n_layers-1, n_layers//2 ]
        if exits == 3: self.exit_ids = [n_layers-1, n_layers - n_layers//3, n_layers//3 ]
        assert self.exit_ids[0] == n_layers-1, "điểm thoát đầu tiên phải là layer cuối"
        if exits == 1: self.exit_scales = [1. ]
        if exits == 2: self.exit_scales = [0.8, 0.2]
        if exits == 3: self.exit_scales = [0.7, 0.2, 0.1]
        assert sum(self.exit_scales) >= 0.999999999999999


    def forward(self, input_seq:Tensor):
        # Vì embeddings lưu ở float32 nên sẽ cần convert sang bf16
        n_blks = len(self.blocks)
        v_embs = [ norm(emb(input_seq).bfloat16()) for emb in self.val_embs ]
        t_embs = [ norm(emb(input_seq).bfloat16()) for emb in self.tok_embs ]

        for i in range(1, len(t_embs)): # phóng to
            t_embs[i] = self.tok_proj[i](t_embs[i])

        x = x0 = t_embs[0]
        t_embs += [x0]*(n_blks - len(t_embs))
        assert len(t_embs) == n_blks
    
        skips = [None] * (n_blks - 2 * len(v_embs))
        v_embs = (v_embs + skips + v_embs)[:n_blks]
        assert len(v_embs) == n_blks # u-shape

        skip_weights = self.scalars[ :n_blks]
        lambdas      = self.scalars[1*n_blks : 3*n_blks].view(-1, 2)
        sa_lambdas   = self.scalars[3*n_blks : 5*n_blks].view(-1, 2)
        layer_outputs = []

        for i in range(self.n_layers):
            if i in self.skip_from:
                k = self.skip_from[i]
                x += skip_weights[k] * layer_outputs[k]
            
            def fwd(blk, te, ve, l, s): return lambda x: blk(x, te, ve, l, s)
            f = fwd(self.blocks[i], t_embs[i], v_embs[i], lambdas[i], sa_lambdas[i])
            x = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)
            layer_outputs.append(x)

        return layer_outputs, t_embs, v_embs, lambdas, sa_lambdas


###################
## Loss function ##
###################

def _loss_fn(_loss_method, model, input_seq, target, future):
    layer_outputs, t, v, l, s = model(input_seq)
    target = target.flatten()
    loss = 0

    for i, head in enumerate(model.lm_heads):
        layer_id = model.exit_ids[i]
        hidden = norm(layer_outputs[layer_id])
        #########################################
        x, _ = _loss_method(hidden, target, head)
        #########################################
        loss += model.exit_scales[i] * x
    if model.n_layers == len(model.blocks): return loss

    future_loss, _ = _loss_method(
        model.blocks[-1](layer_outputs[-1], t[0], t[-1], v[-1], l[-1], s[-1]),
        future.flatten(),  # lm_head của main task nằm đầu
        model.lm_heads[0], # tied embed với main task head 
    )
    return loss * (1 - model.future_ratio) + future_loss * model.future_ratio


def simple_loss_fn(model, input_seq, target, future):
    def _loss_method(hidden, target, head):
        logits = head(hidden)
        logits = logits.view(-1, logits.size(-1))
        logits = 15*logits*torch.rsqrt(logits.square() + 15*15)
        return F.cross_entropy(logits.float(), target), None
    return _loss_fn(_loss_method, model, input_seq, target, future)


try: # pip install liger_kernel
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    def fused_loss_fn(model, input_seq, target, future):
        def _loss_method(hidden, target, head):
            hidden = hidden.view(-1, hidden.size(-1))
            return LigerFusedLinearCrossEntropyFunction.apply(hidden, head.weight, target)
        return _loss_fn(_loss_method, model, input_seq, target, future)
except: None

## TEST MODEL
if __name__ == "__main__":
    import os
    import numpy as np
    from optimus import Muon, convert_int8_mixed_precision

    loss_fn = simple_loss_fn
    vocab_size = 1981

    # Clear cache and reset peak memory stats
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    
    # Use a medium-sized model to show memory benefits
    dim, n_layers = 128, 8
    num_heads, num_kv_heads = 8, 4
    print(f"Model config: layers={n_layers}, dim={dim}, heads={num_heads}/{num_kv_heads}")
    
    batch_size, seq_len = 2, 256
    model = WinGPT(vocab_size, n_layers, num_heads, num_kv_heads, dim, seq_len, future_percent=20).cuda()

    if os.environ.get('int8', '0') == '1':
        print(convert_int8_mixed_precision(model), "linear converted to int8") # lỗi trên 3050

    ## Generate sequences with batch dimension
    input_seq = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    target    = torch.randint(0, vocab_size, (batch_size, seq_len,)).cuda()
    future    = torch.randint(0, vocab_size, (batch_size, seq_len,)).cuda()

    aptim = torch.optim.Adam([p for n, p in model.named_parameters() if "fc" not in n and "proj" not in n])
    optim = Muon([p for n, p in model.named_parameters() if "fc" in n or "proj" in n])

    # Memory after model initialization
    after_init_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    print(f"Peak VRAM after model initialization: {after_init_memory:.2f} MB")

    for step in range(10):
        loss = loss_fn(model, input_seq, target, future)
        loss.backward()
        optim.step(); optim.zero_grad()
        aptim.step(); aptim.zero_grad()

        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"step {step}, loss {loss.item():.4f}, Peak VRAM: {current_memory:.2f} MB")