#!/usr/bin/env python3
''' GPT for the WIN
- half rope và v_emb idea từ https://github.com/KellerJordan/modded-nanogpt
- bỏ o_proj, inspired by https://www.alphaxiv.org/abs/2311.01906
- Áp dụng GTA from https://arxiv.org/abs/2505.21487v1
- parallel transformer x = x + attn(norm(x)) + mlp(norm(x))
- 1 long NoPE : 4 short RoPE SWA; idea từ Gemma và RNoPE paper
- Không norm q, k để bảo toàn Q @ K (Command A paper)
- MTP dùng concat(last_hidden, next token embedding) from DeepSeek V3
- Các kỹ thuật tối ưu khác trong `optimus.py` (int8 mixed matmul, fused linear LCE, Muon optimizer)
'''
import os, math, torch, torch.nn.functional as F, time
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from optimus import FusedCE, convert_int8_mixed_precision
from flash.attn import flash_attn_varlen_func
from einops import repeat

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.coordinate_descent_tuning = True
torch.set_default_dtype(torch.bfloat16)


TIMESPENT = {'norm': 0, 'prepare': 0, 'attn': 0, 'up': 0, 'down': 0, 'LCE': 0}
###
@torch.compiler.disable
def time_time(): return time.time()
###
def measure(timer, func):
    global TIMESPENT
    started_at = time_time()
    result = func()
    TIMESPENT[timer] += time_time() - started_at
    return result
###
def timespent():
    total = sum(TIMESPENT.values())
    return {timer: int(spent * 1000 / total) / 10 for timer, spent in TIMESPENT.items()}


@torch.no_grad()
def init_linear(w: Tensor):
    val = 0.632  # change from 0.5 to 0.632 if follow https://www.alphaxiv.org/abs/2312.16903
    std = val * (w.size(-1) ** -0.5)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)

def norm(x: Tensor): # root mean square của các phần tử theo chiều cuối
    func = lambda: F.rms_norm(x, (x.size(-1),))
    return measure("norm", func)

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


    def forward(self, x: Tensor, cu_seqlens, max_seqlen, half=False):
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
        self.vdim = dim//(2 * self.group)
        self.ple = nn.Embedding(vocab_size, self.vdim)

        self.window = SLIDING_WINDOW * 8 if self.long else SLIDING_WINDOW
        print(f"Layer {layer_id} => {'Nope' if self.long else 'RoPE'}, win {self.window}")

        self.head_dim  = head_dim
        self.num_heads = dim // head_dim

        self.  up_proj = nn.Linear(dim, 4*dim, bias=False)
        self.down_proj = nn.Linear(4*dim, dim, bias=False)

        with torch.no_grad():
            self.  up_proj.weight.copy_(init_linear(torch.empty(4*dim, dim)))
            self.down_proj.weight.zero_()


    def forward(self, x, cu_seqlens, max_seqlen, input_seq, rotary):
        T, D  = x.shape
        H, HD = self.num_heads, self.head_dim
        G, VD = self.group, self.vdim

        xn = x if self.layer_id == 0 else norm(x)
        up = measure("up", lambda: self.up_proj(xn))

        def prepare():
            e       = self.ple(input_seq)       # get per-layer embedding
            q, v, k = torch.split(up[..., : D + VD + HD//2], [D, VD, HD//2], dim=-1)

            # Group Tied Attention https://github.com/Dao-AILab/grouped-latent-attention/blob/main/modeling_llama_GTA.py#L487
            q       = q.view(T, H   , HD   )    # Q       ∈ R^(ctxlen, head_q, dim)
            kv_half = v.view(T, H//G, HD//2)    # KV_half ∈ R^(ctxlen, head_kv, dim//2)
            v_half  = e.view(T, H//G, HD//2)    # Nửa còn lại của value lấy từ PLE (per layer embedding)
            k       = k.view(T, 1   , HD//2)    # K_RoPE  ∈ R^(ctxlen, 1, dim/2)
            k_half  = repeat(k, 'T 1 d -> T h d', h=H//G)

            if not self.long:  # Chỉ áp dụng rope cho short layers
                q = rotary(q, cu_seqlens, max_seqlen, half=True) # quay nửa sau dim
                k_half = rotary(k_half, cu_seqlens, max_seqlen)  # quay toàn bộ dim//2

            k_full  = torch.cat([kv_half, k_half], dim=-1)
            v_full  = torch.cat([kv_half, v_half], dim=-1)

            return q, k_full, v_full, F.relu(up).square() # F.sigmoid(up)*up
        q, k, v, act = measure("prepare", lambda: checkpoint(prepare, use_reentrant=False))

        attn = lambda: flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, \
            window_size=(self.window, 0), softcap=50).view(T, D)  # softcap https://www.alphaxiv.org/abs/2410.16682

        return x + measure("attn", attn) + measure("down", lambda: self.down_proj(act))


class WinGPT(nn.Module):
    def __init__(self, vocab_size, n_layers, dim, ctxlen, head_dim):
        super().__init__()
        self.rotary   = Rotary(head_dim, ctxlen)
        self.blocks   = nn.ModuleList([Block(dim, head_dim, vocab_size, i) for i in range(n_layers)])
        self.embeds   = nn.Embedding(vocab_size, dim)
        self.mtp_head = Block(dim, head_dim, vocab_size, -2)
        self.mtp_proj = nn.Linear(2*dim, dim, bias=False)
        self.unembeds = nn.Linear(dim, vocab_size, bias=False)
        with torch.no_grad():
            self.mtp_proj.weight.copy_(init_linear(torch.empty(dim, 2*dim)))
            self.unembeds.weight.zero_()

    def forward(self, input_seq, cu_seqlens, max_seqlen):
        # norm emb để tạo large residuals https://www.alphaxiv.org/abs/2312.16903
        x = norm(self.embeds(input_seq))
        for blk in self.blocks: x = blk(x, cu_seqlens, max_seqlen, input_seq, self.rotary)
        return x


def fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen, n_ignore=0, ignore=-100, cu_steps=1):
    x = model(input_seq, cu_seqlens, max_seqlen)
    def prepare():
        xn    = norm(x)
        x0    = norm(model.embeds(input_seq))
        zeros = torch.zeros_like(xn[:1])
        xx    = torch.cat([zeros, xn[:-1]], dim=0)  # x dịch phải
        xx_x0 = torch.cat([xx, x0], dim=-1)
        y     = model.mtp_proj(xx_x0)
        return xn, y

    xn, y = checkpoint(prepare, use_reentrant=False)
    yn = norm(model.mtp_head(y, cu_seqlens, max_seqlen, input_seq, model.rotary))

    ## Tính loss cho NTP (x) và MTP (y) và cộng lại ưu tiên nhiệm vụ chính NTP
    target[0] = ignore
    n_ignore += 1
    w = model.unembeds.weight

    def final():
        xloss = FusedCE.apply(xn, w, target, n_ignore, ignore, 0.7 / cu_steps)  # NTP: Next token prediction
        yloss = FusedCE.apply(yn, w, target, n_ignore, ignore, 0.3 / cu_steps)  # MTP: Next of next token prediction
        return xloss + yloss
    return measure("LCE", final)


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
    vocab_size = 32*1024
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

    print("timespent", timespent())