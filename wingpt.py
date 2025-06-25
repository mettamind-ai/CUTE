#!/usr/bin/env python3
## GPT for the WIN (cải biên từ modded nanogpt)
import os, math, torch, torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from optimus import Int8MixedLinear, FusedCE
from flash.attn import flash_attn_varlen_func

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.coordinate_descent_tuning = True
torch.set_float32_matmul_precision('high') # better for f32 head
torch.backends.cuda.matmul.allow_tf32  = True
torch.set_default_dtype(torch.bfloat16)

def norm(x: Tensor): # root mean square của các phần tử theo chiều cuối
    return F.rms_norm(x, (x.size(-1),))

@torch.no_grad()
def init_linear(w: Tensor, scale=1):
    std = 0.5 * (w.size(-1) ** -0.5) # 0.5 is a bit better ...
    bound = (3 ** 0.5) * std * scale # ... than default 1/sqrt(3)
    return w.uniform_(-bound, bound)

class Rotary(nn.Module):
    def __init__(self, dim: int, ctxlen: int):
        super().__init__()
        base, half, dtype = (1/10000), (dim//4), torch.float32
        angular_freq = base  **  torch.linspace(0, 1, steps=half, dtype=dtype)
        angular_freq = torch.cat([angular_freq, torch.zeros(half, dtype=dtype)])
        # Tần số góc, nửa đầu giảm dần từ 1 tới base và nửa còn lại là zeros

        positions = torch.arange(ctxlen, dtype=dtype)
        theta = torch.einsum("i,j -> ij", positions, angular_freq)
        # theta[i, j] = positions[i] * angular_freq[j]

        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_THD: Tensor):
        ctxlen = x_THD.size(-3) # T ctxlen, head, dim (of head)
        assert self.cos.size(0) >= ctxlen, f"{self.cos.size(0)} >= {ctxlen}?"

        cos = self.cos[:ctxlen, None, :] # [ctxlen, 1, dim]
        sin = self.sin[:ctxlen, None, :] # [ctxlen, 1, dim]

        x1, x2 = x_THD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1     = x1 * (+cos) + x2 * sin
        y2     = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_THD)


class ReLuSquareMLP(nn.Module):
    def __init__(self, dim:int, hdim=None, odim=None, expansion_factor=2, zero_out=True):
        super().__init__()
        if not hdim: hdim = int(dim*expansion_factor)
        if not odim: odim = dim

        self.fc1_proj = nn.Linear(dim, hdim, bias=False)
        self.fc2_proj = nn.Linear(hdim, odim, bias=False)

        # Add weight decay multiplier attribute to the weights
        self.fc1_proj.weight.wd_mul = 2.0  # điều chỉnh hệ số weight decay
        self.fc2_proj.weight.wd_mul = 2.0  # gấp đôi so với mặc định (follow modded gpt)

        with torch.no_grad():
            self.             fc1_proj.weight.copy_(init_linear(torch.empty(hdim, dim)))
            if zero_out: self.fc2_proj.weight.zero_() # sẽ đc residual connect nên khởi tạo là 0
            else:        self.fc2_proj.weight.copy_(init_linear(torch.empty(odim, hdim)))

    def forward(self, x):
        y = F.relu(self.fc1_proj(x)).square()
        return self.fc2_proj(y)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim:int, ctxlen:int, head_dim=128, long=False, layer_id=-1):
        super().__init__() # dim = hidden_size = embedding = feature = representation

        self.num_heads = dim // head_dim
        self.head_dim  = head_dim
        self.ctxlen   = ctxlen
        self.qk_proj   = nn.Linear(dim, 2*dim, bias=False)
        with torch.no_grad(): self.qk_proj.weight.copy_(init_linear(torch.empty(2*dim, dim)))

        if long: self.rope, self.window  = False, 1024*4
        else:    self.rope, self.window  = True,  1024

        print(f"Layer {layer_id} => {'RoPE' if self.rope else 'Nope'}, win {self.window}")
        self.attn_scale = 0.12


    def forward(self, x, v_emb, input_seq, cu_seqlens, max_seqlen, rotary):
        q, k    = self.qk_proj(x).chunk(2, dim=-1)
        v       = v_emb(input_seq)

        T, H, D = self.ctxlen, self.num_heads, self.head_dim
        q, k, v = q.view(T, H, D), k.view(T, H, D), v.view(T, H, D)

        v = norm(v) # norm head_dim (64 hoặc 128)
        if self.rope: q, k = rotary(q), rotary(k)

        o = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            softmax_scale=self.attn_scale, window_size=(self.window, 0),
        )
        return o.view(T, H * D)

##############################
## Transformer for the WIN  ##
##############################
class Block(nn.Module):
    def __init__(self, dim, expansion, ctxlen, head_dim, layer_id, n_layers):
        super().__init__()
        self.layer_id = layer_id
        self.long = layer_id % 5 == 4 # 4 ngắn + 1 dài
        self.mlp = ReLuSquareMLP(dim, dim*expansion) if 1 <= layer_id and layer_id < n_layers - 1 else None
        self.attn = CausalSelfAttention(dim, ctxlen, head_dim, self.long, layer_id)

    def forward(self, x, v_emb, input_seq, cu_seqlens, max_seqlen, rotary):
        xn = norm(x)
        attn = self.attn(xn, v_emb, input_seq, cu_seqlens, max_seqlen, rotary)
        if self.mlp is None: return x + attn
        else:                return x + attn + self.mlp(xn)

class WinGPT(nn.Module):
    def __init__(self, vocab_size, n_layers, dim, ctxlen, head_dim=128, expansion=2):
        super().__init__()
        self.rotary    = Rotary(head_dim, ctxlen)
        self.dim       = dim
        self.blocks    = nn.ModuleList([Block(dim, expansion, ctxlen, head_dim, i, n_layers) for i in range(n_layers)])
        self.embeds    = nn.ModuleList([nn.Embedding(vocab_size, dim) for _ in range(n_layers + 1)])
        self.head2_mlp = ReLuSquareMLP(2*dim, hdim=3*dim, odim=dim, zero_out=False) # predict next of next token (MTP)
        self.unembeds  = nn.Linear(dim, vocab_size, bias=False)
        with torch.no_grad(): self.unembeds.weight.zero_()

    def forward(self, input_seq, cu_seqlens, max_seqlen):
        x = checkpoint(self.embeds[0], input_seq, use_reentrant=False)
        for i, blk in enumerate(self.blocks):
            f = lambda x, i, blk: blk(x, self.embeds[i + 1], input_seq, cu_seqlens, max_seqlen, self.rotary)
            x = checkpoint(f, x, i, blk, use_reentrant=False)
        return x


def fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen, n_ignore=0, ignore=-100):
    x = model(input_seq, cu_seqlens, max_seqlen) 
    def prepare():

        zeros = torch.zeros_like(x[:1])
        xx    = torch.cat([zeros, x[:-1]], dim=0) # x dịch phải

        x0    = model.embeds[0](input_seq)
        xx_x0 = torch.cat([xx, x0], dim=-1)

        y     = model.head2_mlp(norm(xx_x0))
        ty    = F.pad(target[1:], (1, 0), mode='constant', value=ignore)

        return norm(x), norm(y), ty
    xn, yn, ty = checkpoint(prepare, use_reentrant=False)

    ## Tính loss cho NTP (x) và MTP (y) và cộng lại ưu tiên nhiệm vụ chính NTP
    w     = model.unembeds.weight
    xloss = FusedCE.apply(xn,  w, target, n_ignore, ignore, 0.7)  # NTP: Next token prediction
    yloss = FusedCE.apply(yn,  w, ty,     n_ignore, ignore, 0.3)  # MTP: Next of next token prediction
    return xloss + yloss


def get_cu_max_seqlens_from(input_seq, eot=6399):
        mask = (input_seq == eot)
        mask[-1] = True
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), torch.where(mask)[0].to(torch.int32) + 1,])
        max_seqlen = int(torch.max(torch.diff(cu_seqlens)))
        return cu_seqlens, max_seqlen


########################
##  TESTING  TESTING  ##
########################

if __name__ == "__main__":
    import numpy as np
    from optimus import Muon1GPU as Muon
    from optimus import convert_int8_mixed_precision

    seed = 1981
    ctxlen = 256
    vocab_size = 32*1024
    dim, head_dim, n_layers = 128, 64, 8
    print(f"win config: layers={n_layers}, dim={dim}, heads={dim//head_dim}; ctxlen={ctxlen}")

    torch.manual_seed(seed)
    model = WinGPT(vocab_size, n_layers, dim, ctxlen, head_dim=head_dim).cuda()

    apara = {n: p for n, p in model.named_parameters() if "proj" not in n}
    opara = [p for n, p in model.named_parameters() if "proj" in n]

    print("\nAdam:", apara.keys())
    apara = list(apara.values())

    aptim = torch.optim.Adam(apara)
    optim = Muon(opara)

    after_init_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    print(f"Peak VRAM after model initialization: {after_init_memory:.2f} MB")

    model.train()

    for step in range(10):
        ## Generate sequences with batch dimension
        input_seq = torch.randint(5, vocab_size//4, (ctxlen,), dtype=torch.long).cuda()
        target    = torch.randint(5, vocab_size//4, (ctxlen,), dtype=torch.long).cuda()
        cu_seqlens, max_seqlen = get_cu_max_seqlens_from(input_seq)

        optim.zero_grad()
        aptim.zero_grad()

        loss_model = fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen)
 
        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"step {step}, loss_model {loss_model.item():.4f}, ", end="")

        loss_model.backward()
        optim.step()
        aptim.step()

        print(f"Peak VRAM: {current_memory:.2f} MB")
