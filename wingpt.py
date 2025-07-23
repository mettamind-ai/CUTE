#!/usr/bin/env python3
''' GPT for the WIN
- half rope và v_emb idea từ https://github.com/KellerJordan/modded-nanogpt
- bỏ o_proj, inspired by https://www.alphaxiv.org/abs/2311.01906
- Áp dụng GTA from https://arxiv.org/abs/2505.21487v1
- parallel transformer x = x + attn(norm(x)) + mlp(norm(x))
- 1 long NoPE : 3 short RoPE SWA; idea từ Gemma và RNoPE paper
- Không norm q, k để bảo toàn Q @ K (Command A paper)
- MTP dùng concat(last_hidden, next token embedding) from DeepSeek V3
- Các kỹ thuật tối ưu khác trong `optimus.py` (int8 mixed matmul, fused linear LCE, Muon optimizer, OhMaiHead)
'''
import os, math, torch, torch.nn.functional as F, time
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from optimus import FusedCE, convert_int8_mixed_precision, OhMaiHead
from flash.attn import flash_attn_varlen_func
from flash.ops.swiglu import swiglu
from einops import repeat

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.coordinate_descent_tuning = True
torch.set_default_dtype(torch.bfloat16)

@torch.no_grad()
def init_linear(w: Tensor):
    val = 0.632  # change from 0.5 to 0.632 if follow https://www.alphaxiv.org/abs/2312.16903
    std = val * (w.size(-1) ** -0.5)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)

class Rotary(nn.Module):
    def __init__(self, dim: int, ctxlen: int):
        super().__init__()
        self.rotary_dim = dim // 2
        base, half = 1/10_000, self.rotary_dim // 2
        angular_freq = base ** torch.linspace(0, 1, steps=half, dtype=torch.float32)
        positions = torch.arange(ctxlen, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", positions, angular_freq)  # theta[i, j] = p[i] * a[j]
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)


    def forward(self, x: Tensor, half=False):
        ctxlen, head, dim = x.shape
        assert self.cos.shape[0] >= ctxlen

        if half:
            assert self.rotary_dim == dim // 2 and dim % 2 == 0
            x_pass, x_rot = x.chunk(2, dim=-1)
        else:
            assert self.rotary_dim == dim
            x_rot = x

        ## Áp dụng phép quay cho x_rot
        cos    = self.cos[:ctxlen, None, :]
        sin    = self.sin[:ctxlen, None, :]
        x1, x2 = x_rot.to(dtype=torch.float32).chunk(2, dim=-1)
        y1     = x1 * (+cos) + x2 * sin
        y2     = x1 * (-sin) + x2 * cos
        x_rot  = torch.cat((y1, y2), -1).type_as(x)

        if half: return torch.cat((x_pass, x_rot), dim=-1)
        else:    return x_rot


SLIDING_WINDOW = 1024
class Block(nn.Module):
    def __init__(self, dim, head_dim, vocab_size, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.long = layer_id % 4 == 3 # 3 ngắn + 1 dài

        self.group = 4 # query head per group, cân bằng cho cả model nhỡ và lớn
        self.vdim = dim//self.group

        self.window = SLIDING_WINDOW * 8 if self.long else SLIDING_WINDOW
        print(f"Layer {layer_id} => {'Nope' if self.long else 'RoPE'}, win {self.window}")

        self.head_dim  = head_dim
        self.num_heads = dim // head_dim

        inter_dim_ffn = int(4 * dim)
        self.up_proj = nn.Linear(dim, inter_dim_ffn, bias=False)
        self.down_proj = nn.Linear(inter_dim_ffn, dim, bias=False)

        with torch.no_grad():
            self.up_proj.weight.copy_(init_linear(torch.empty(inter_dim_ffn, dim)))
            self.down_proj.weight.zero_()


    def forward(self, x, cu_seqlens, max_seqlen, input_seq, rotary):
        T, D  = x.shape
        H, HD = self.num_heads, self.head_dim
        G, VD = self.group, self.vdim
        k_v_q = [HD//2, VD, D]

        up = self.up_proj(norm(x))
        def prepare():

            k, v, q = torch.split(up[..., : sum(k_v_q)], k_v_q, dim=-1)
            # Group Tied https://github.com/Dao-AILab/grouped-latent-attention/blob/main/modeling_llama_GTA.py#L487
            q = q.view(T, H   , HD   )    # Q       ∈ R^(ctxlen, head_q,  dim)
            v = v.view(T, H//G, HD   )    # KV_half ∈ R^(ctxlen, head_kv, dim/2)
            k = k.view(T, 1   , HD//2)    # K_RoPE  ∈ R^(ctxlen, 1,       dim/2)
            k = repeat(k, 'T 1 d -> T h d', h=H//G)

            if not self.long: q, k = rotary(q, half=True), rotary(k)
            k = torch.cat([v[..., : HD//2], k], dim=-1)

            # TODO: Với non-rotate tạo hàm flash_attn mới nhận k chưa repeat để speedup cho long attention
            att = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, \
                window_size=(self.window, 0), softcap=50).view(T, D)  # softcap https://www.alphaxiv.org/abs/2410.16682

            act = F.relu(up).square()
            return x + att, act

        x_att, act = checkpoint(prepare, use_reentrant=False)
        return x_att + self.down_proj(act)


def norm(x: Tensor): # root mean square của các phần tử theo chiều cuối
    return F.rms_norm(x, (x.size(-1),))

class WinGPT(nn.Module):
    def __init__(self, vocab_size, n_layers, dim, ctxlen, head_dim):
        super().__init__()
        self.rotary   = Rotary(head_dim, ctxlen)
        self.blocks   = nn.ModuleList([Block(dim, head_dim, vocab_size, i) for i in range(n_layers)])
        self.embeds   = nn.Embedding(vocab_size, dim)
        self.mtp_head = Block(dim, head_dim, vocab_size, -2)
        self.mtp_proj = nn.Linear(2*dim, dim, bias=False)
        self.unembeds = OhMaiHead(dim, vocab_size, bias=False)
        with torch.no_grad():
            self.mtp_proj.weight.copy_(init_linear(torch.empty(dim, 2*dim)))

    def forward(self, input_seq, cu_seqlens, max_seqlen):
        x = x0 = norm(self.embeds(input_seq))  # norm emb để tạo large residuals https://www.alphaxiv.org/abs/2312.16903
        for blk in self.blocks: x = blk(x, cu_seqlens, max_seqlen, input_seq, self.rotary)
        return norm(x), x0


def fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen, n_ignore=1, ignore=-100, cu_steps=1):
    target = model.unembeds.activate(target)
    xn, x0 = model(input_seq, cu_seqlens, max_seqlen)
    model.unembeds.update_new_tokens_weight()

    def prepare():
        zeros = torch.zeros_like(xn[:1])
        xx    = torch.cat([zeros, xn[:-1]], dim=0)  # x dịch phải
        y0    = torch.cat([xx, x0], dim=-1)
        y1    = model.mtp_proj(y0)
        return y1

    y1 = checkpoint(prepare, use_reentrant=False)
    y2 = model.mtp_head(y1, cu_seqlens, max_seqlen, input_seq, model.rotary)
    yn = norm(y2)
    target[0] = ignore

    mtp_loss = FusedCE.apply(yn, model.unembeds.active_weight, target, n_ignore, ignore, 0.2 / cu_steps)
    ntp_loss = FusedCE.apply(xn, model.unembeds.active_weight, target, n_ignore, ignore, 0.8 / cu_steps)
    return mtp_loss + ntp_loss


def get_cu_max_seqlens_from(input_seq, eot):
        mask       = (input_seq == eot)
        mask[-1]   = True
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=input_seq.device), torch.where(mask)[0].to(torch.int32) + 1])
        max_seqlen = int(torch.max(torch.diff(cu_seqlens)))
        return cu_seqlens, max_seqlen


########################
##  TESTING  TESTING  ##
########################
if __name__ == "__main__":
    from optimus import Muon1GPU as Muon

    ctxlen = 1024
    vocab_size = 64*1024
    dim, head_dim, n_layers = 256, 64, 8
    print(f"win config: layers={n_layers}, dim={dim}, heads={dim//head_dim}; ctxlen={ctxlen}")

    torch.manual_seed(1981)
    model = WinGPT(vocab_size, n_layers, dim, ctxlen, head_dim).cuda()
    convert_int8_mixed_precision(model)

    apara = {n: p for n, p in model.named_parameters() if "proj" not in n}
    mpara = [p for n, p in model.named_parameters() if "proj" in n]
    print("\nAdam:", apara.keys())

    aptim = torch.optim.AdamW(apara.values())
    optim = Muon(mpara)

    after_init_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    print(f"Peak VRAM after model initialization: {after_init_memory:.2f} MB")
    model.train()

    for step in range(10):
        ## Generate sequences with batch dimension
        input_seq = torch.randint(5, vocab_size//4, (ctxlen,), dtype=torch.long).cuda()
        target    = F.pad(input_seq[1:], (1, 0), mode='constant', value=-100)
        cu_seqlens, max_seqlen = get_cu_max_seqlens_from(input_seq, eot=0)

        optim.zero_grad(); aptim.zero_grad()

        loss_model     = fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen)
        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"step {step}, loss_model {loss_model.item():.4f}, Peak VRAM: {current_memory:.2f} MB")

        loss_model.backward()
        optim.step(); aptim.step()

    model.unembeds.update_async_weight()
