#!/usr/bin/env python3
''' GPT for the WIN
- half rope và v_emb idea từ https://github.com/KellerJordan/modded-nanogpt
- bỏ o_proj, inspired by https://www.alphaxiv.org/abs/2311.01906
- Áp dụng GTA from https://arxiv.org/abs/2505.21487v1
- parallel transformer x = x + attn(norm(x)) + mlp(norm(x))
- 1 long NoPE : 4 short RoPE SWA; idea từ Gemma và RNoPE (Command A)
- Không norm q, k để bảo toàn NoPE (Command A)
- MTP dùng concat(last_hidden, next token embedding) DeepSeek v3
'''
import os, math, torch, torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from optimus import FusedCE
from flash.attn import flash_attn_varlen_func
from einops import repeat

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.coordinate_descent_tuning = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_default_dtype(torch.bfloat16)

@torch.no_grad()
def init_linear(w: Tensor, scale=1):
    std = 0.632 / math.sqrt(w.size(-1)) # 0.632 follow https://www.alphaxiv.org/abs/2312.16903 
    bound = math.sqrt(3) * std * scale
    return w.uniform_(-bound, bound)

def norm(x: Tensor): # root mean square của các phần tử theo chiều cuối
    return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim: int, ctxlen: int):
        super().__init__()
        base, half, dtype = 1/10000, dim//4, torch.float32
        angular_freq = base  **  torch.linspace(0, 1, steps=half, dtype=dtype)
        angular_freq = torch.cat([angular_freq, torch.zeros(half, dtype=dtype)])
        # Tần số góc, nửa đầu giảm dần từ 1 tới base và nửa còn lại là zeros

        positions = torch.arange(ctxlen, dtype=dtype)
        theta = torch.einsum("i,j -> ij", positions, angular_freq)

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
    def __init__(self, dim:int, hdim=None, odim=None, expansion=2):
        super().__init__()
        if not hdim: hdim = int(dim*expansion)
        if not odim: odim = dim

        self.fc1_proj = nn.Linear(dim, hdim, bias=False)
        self.fc2_proj = nn.Linear(hdim, odim, bias=False)

        with torch.no_grad():
            self.fc1_proj.weight.copy_(init_linear(torch.empty(hdim, dim)))
            self.fc2_proj.weight.copy_(init_linear(torch.empty(odim, hdim)))

    def forward(self, x):
        y = F.relu(self.fc1_proj(x)).square()
        return self.fc2_proj(y)


##############################
## Transformer for the WIN  ##
##############################
class Block(nn.Module):
    def __init__(self, dim, head_dim, layer_id, n_layers):
        super().__init__()
        self.layer_id = layer_id
        self.long = layer_id % 5 == 4 # 4 ngắn + 1 dài

        self.window = 512*8 if self.long else 512*2
        print(f"Layer {layer_id} => {'Nope' if self.long else 'RoPE'}, win {self.window}")

        self.head_dim  = head_dim
        self.num_heads = dim // head_dim

        self.skip_mlp = layer_id == n_layers - 1 # bỏ MLP ở layer cuối
        if self.skip_mlp:
            self.up_proj = nn.Linear(dim, dim, bias=False)
            with torch.no_grad(): self.up_proj.weight.copy_(init_linear(torch.empty(dim, dim)))
        else:
            self.up_proj = nn.Linear(dim, 4*dim, bias=False)
            self.down_proj = nn.Linear(3*dim, dim, bias=False)

            with torch.no_grad():
                self.up_proj.weight.copy_(init_linear(torch.empty(4*dim, dim)))
                self.down_proj.weight.zero_()

    def forward(self, x, ve, input_seq, cu_seqlens, max_seqlen, rotary):
        T, H, HD = x.shape[0], self.num_heads, self.head_dim
        ID, D    = self.up_proj.weight.shape

        def prepare():
            xn   = norm(x) if self.layer_id > 0 else x
            qy   = self.up_proj(xn)

            q, y = torch.split(qy, [D, ID - D], dim=-1)
            v    = ve(input_seq)

            q = q.view(T, H, HD)
            v = v.view(T, H, HD)

            k = qy[..., -HD//2 : ]
            k = k.view(T, 1, HD//2)
            k = repeat(k, 'T 1 d -> T h d', h=H)
            k = torch.cat([k, v[..., HD//2 : ]], dim=-1)

            if not self.long: q, k = rotary(q), rotary(k)
            return q, k, v, y
        q, k, v, y = checkpoint(prepare, use_reentrant=False)

        # TODO: Vì v hình thành từ embeddings và k hình thành phần lớn từ v nên có thể save nhiều vram khi
        # fuse init v, k vào trong code tính flash attention
        o = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, \
            window_size=(self.window, 0), softcap=50).view(T, D) # https://www.alphaxiv.org/abs/2410.16682

        return x + o if self.skip_mlp else \
               x + o + self.down_proj(F.relu(y).square())


class WinGPT(nn.Module):
    def __init__(self, vocab_size, n_layers, dim, ctxlen, head_dim):
        super().__init__()
        self.rotary   = Rotary(head_dim, ctxlen)
        self.blocks   = nn.ModuleList([Block(dim, head_dim, i, n_layers) for i in range(n_layers)])
        self.embeds   = nn.Embedding(vocab_size, dim, dtype=torch.float32)
        self.v_embeds = nn.ModuleList([nn.Embedding(vocab_size, dim) for _ in range(n_layers)])
        self.mtp_head = ReLuSquareMLP(2*dim, hdim=3*dim, odim=dim) # predict next of next token
        self.unembeds = nn.Linear(dim, vocab_size, bias=False)
        with torch.no_grad(): self.unembeds.weight.zero_()


    def forward(self, input_seq, cu_seqlens, max_seqlen):
        # norm emb để tạo large residuals https://www.alphaxiv.org/abs/2312.16903
        x = x0 = norm(self.embeds(input_seq)).bfloat16()
        B, V, R = self.blocks, self.v_embeds, self.rotary
        for k in range(len(B)):
            x   = B[k](x, V[k], input_seq, cu_seqlens, max_seqlen, R)
        return x, x0


def fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen, n_ignore=0, ignore=-100, cu_steps=1):
    x, x0 = model(input_seq, cu_seqlens, max_seqlen)
    def prepare():
        xn    = norm(x)
        zeros = torch.zeros_like(xn[:1])
        xx    = torch.cat([zeros, xn[:-1]], dim=0)  # x dịch phải
        xx_x0 = torch.cat([xx, x0], dim=-1)
        y     = model.mtp_head(xx_x0)
        return xn, norm(y)
    xn, yn = checkpoint(prepare, use_reentrant=False)

    ## Tính loss cho NTP (x) và MTP (y) và cộng lại ưu tiên nhiệm vụ chính NTP
    target[0] = ignore; n_ignore += 1
    w = model.unembeds.weight

    xloss = FusedCE.apply(xn, w, target, n_ignore, ignore, 0.7 / cu_steps)  # NTP: Next token prediction
    yloss = FusedCE.apply(yn, w, target, n_ignore, ignore, 0.3 / cu_steps)  # MTP: Next of next token prediction
    return xloss + yloss


def get_cu_max_seqlens_from(input_seq, eot):
        mask       = (input_seq == eot)
        mask[-1]   = True
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=input_seq.device), torch.where(mask)[0].to(torch.int32) + 1,])
        max_seqlen = int(torch.max(torch.diff(cu_seqlens)))
        return cu_seqlens, max_seqlen


########################
##  TESTING  TESTING  ##
########################

if __name__ == "__main__":
    from optimus import Muon1GPU as Muon

    seed = 1981
    ctxlen = 1024
    vocab_size = 32*1024
    dim, head_dim, n_layers = 256, 64, 8
    print(f"win config: layers={n_layers}, dim={dim}, heads={dim//head_dim}; ctxlen={ctxlen}")

    torch.manual_seed(seed)
    model = WinGPT(vocab_size, n_layers, dim, ctxlen, head_dim).cuda()

    from optimus import convert_int8_mixed_precision
    convert_int8_mixed_precision(model)

    apara = {n: p for n, p in model.named_parameters() if "proj" not in n}
    opara = [p for n, p in model.named_parameters() if "proj" in n]

    print("\nAdam:", apara.keys())
    apara = list(apara.values())

    aptim = torch.optim.AdamW(apara)
    optim = Muon(opara)

    after_init_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    print(f"Peak VRAM after model initialization: {after_init_memory:.2f} MB")

    model.train()

    for step in range(10):
        ## Generate sequences with batch dimension
        input_seq = torch.randint(5, vocab_size//4, (ctxlen,), dtype=torch.long).cuda()
        target    = F.pad(input_seq[1:], (1, 0), mode='constant', value=-100)
        cu_seqlens, max_seqlen = get_cu_max_seqlens_from(input_seq, eot=0)

        optim.zero_grad()
        aptim.zero_grad()

        loss_model     = fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen)
        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"step {step}, loss_model {loss_model.item():.4f}, ", end="")

        loss_model.backward()
        optim.step()
        aptim.step()

        print(f"Peak VRAM: {current_memory:.2f} MB")
