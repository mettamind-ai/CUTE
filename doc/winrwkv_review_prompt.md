# Review: WinRWKV Implementation

## Context

WinRWKV là một implementation của RWKV7 theo style của WinGPT - một GPT implementation tối ưu. Mục tiêu là giữ nguyên tinh thần RWKV7 (đặc biệt model init) trong khi áp dụng các kỹ thuật tinh hoa từ WinGPT.

**RWKV7 Reference** (`tools/rwkv7/model.py`): Implementation gốc với PyTorch Lightning, DeepSpeed integration.

**WinGPT** (`wingpt.py`): GPT với GTA attention, MTP, parallel block, INT8 mixed precision, Muon optimizer.

**WinRWKV** (`winrwkv.py`): Kết hợp RWKV7 core với WinGPT optimizations.

## Analysis Needed

1. **Model Init Correctness**: So sánh init weights giữa WinRWKV và RWKV7 reference
2. **Architecture Fidelity**: RWKV7 time/channel mixing có được preserve đúng không?
3. **WinGPT Techniques Applied**: Những gì đã áp dụng, còn thiếu gì?
4. **Potential Issues**: Edge cases, numerical stability, performance concerns

## Constraints
- Phải tương thích với CUDA kernel `wkv7.cu`
- Phải work với `optimus.py` (FusedCE, OhMaiHead, Muon, INT8)
- Model init phải reproducible với RWKV7 reference

## Evidence Standard
> Only conclude when you have **reliable evidence** from the provided context, or you can **reason it out clearly and defensibly**. If evidence is weak, state uncertainty and propose a conservative fallback.

---

## CODE: WinRWKV (`winrwkv.py`)

```python
#!/usr/bin/env python3
''' RWKV7 for the WIN
- Áp dụng RWKV7 time mixing từ tools/rwkv7
- parallel transformer x = x + tmix(norm(x)) + cmix(norm(x))
- Các kỹ thuật tối ưu khác trong `optimus.py`
'''
import os, math, torch, torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch.utils.cpp_extension import load
from optimus import FusedCE, convert_int8_mixed_precision, OhMaiHead

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.coordinate_descent_tuning = True
torch.set_default_dtype(torch.bfloat16)

HEAD_SIZE = 64
CHUNK_LEN = 16

flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}",
    "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
cu = f'{os.path.dirname(os.path.abspath(__file__))}/tools/racoon/wkv7.cu'
load(name="wind_backstepping", sources=[cu], is_python_module=False, verbose=True, extra_cuda_cflags=flags)


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y  = torch.empty_like(v)
        s  = torch.empty(B, H, T//CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype == torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db


def RUN_CUDA_RWKV7(q, w, k, v, a, b):
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC//HEAD_SIZE, HEAD_SIZE) for i in [q, w, k, v, a, b]]
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))
```

```python
@torch.no_grad()
def ortho_init(x, scale):
    shape = x.shape
    orig_dtype = x.dtype
    x = x.float()  # orthogonal init requires float32
    if len(shape) == 2:
        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
        nn.init.orthogonal_(x, gain=gain * scale)
    elif len(shape) == 3:
        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
        for i in range(shape[0]):
            nn.init.orthogonal_(x[i], gain=gain * scale)
    return x.to(orig_dtype)


class RWKV_Tmix(nn.Module):
    def __init__(self, dim, layer_id, n_layers):
        super().__init__()
        self.layer_id = layer_id
        self.head_size = HEAD_SIZE
        self.n_head = dim // self.head_size
        assert dim % self.n_head == 0
        H, N, C = self.n_head, self.head_size, dim

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layers - 1)  # Q: Division by zero if n_layers=1?
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layers)
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            D_DECAY_LORA = max(32, int(round((1.8 * (C ** 0.5)) / 32) * 32))
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, C) + 0.5)

            D_AAA_LORA = max(32, int(round((1.8 * (C ** 0.5)) / 32) * 32))
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))

            D_MV_LORA = max(32, int(round((1.3 * (C ** 0.5)) / 32) * 32))
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

            D_GATE_LORA = max(32, int(round((0.6 * (C ** 0.8)) / 32) * 32))
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.output.weight.data.zero_()
```

```python
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = F.pad(x, (0, 0, 1, -1)) - x  # time_shift

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr.view(B * T, C)).view(B, T, C)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk.view(B * T, C)).view(B, T, C)
        v = self.value(xv.view(B * T, C)).view(B, T, C)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RUN_CUDA_RWKV7(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x.view(B * T, C)).view(B, T, C) * g
        return x, v_first


class RWKV_CMix(nn.Module):
    def __init__(self, dim, layer_id, n_layers):
        super().__init__()
        self.layer_id = layer_id

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layers)
            ddd = torch.ones(1, 1, dim)
            for i in range(dim):
                ddd[0, 0, i] = i / dim
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))

        self.key = nn.Linear(dim, dim * 4, bias=False)
        self.value = nn.Linear(dim * 4, dim, bias=False)

        with torch.no_grad():
            self.key.weight.data.uniform_(-0.5 / (dim ** 0.5), 0.5 / (dim ** 0.5))
            self.value.weight.data.zero_()

    def forward(self, x):
        B, T, C = x.size()
        xx = F.pad(x, (0, 0, 1, -1)) - x  # time_shift
        k = x + xx * self.x_k
        k = self.key(k.view(B * T, C))
        k = torch.relu(k) ** 2
        return self.value(k).view(B, T, C)


class RwkvBlock(nn.Module):
    def __init__(self, dim, layer_id, n_layers):
        super().__init__()
        self.layer_id = layer_id
        self.att = RWKV_Tmix(dim, layer_id, n_layers)
        self.ffn = RWKV_CMix(dim, layer_id, n_layers)

    def forward(self, x, v_first):
        xn = norm(x)
        x_att, v_first = self.att(xn, v_first)
        x_ffn = self.ffn(xn)
        return x + x_att + x_ffn, v_first  # parallel  # Q: RWKV7 uses sequential, is parallel OK?
```

```python
class WinRWKV(nn.Module):
    def __init__(self, vocab_size, n_layers, dim, ctxlen):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.ctxlen = ctxlen
        self.emb_scale = math.sqrt(dim)  # Q: RWKV7 doesn't scale embeddings

        self.embeds = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RwkvBlock(dim, i, n_layers) for i in range(n_layers)])
        self.mtp_head = RwkvBlock(dim, n_layers, n_layers + 1)  # extra block for MTP
        self.mtp_proj = nn.Linear(2 * dim, dim, bias=False)
        self.unembeds = OhMaiHead(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        print("\n" + "#" * 76)
        print("# Init model weight")
        print("#" * 76 + "\n")

        n_params = 0
        for n, p in self.named_parameters():
            shape = p.shape
            n_params += p.numel()

            # ln_x.weight special init: blocks.0.att.ln_x.weight
            if 'ln_x.weight' in n:
                parts = n.split('.')
                layer_id = int(parts[1]) if parts[0] == 'blocks' else self.n_layers
                layer_scale = (1 + layer_id) / self.n_layers
                p.data.fill_(layer_scale ** 0.7)

            # emb init
            elif n == "embeds.weight":
                scale = 1e-4
                nn.init.uniform_(p, -scale, scale)
                print(f"{n}: uniform({-scale}, {scale})")

            # mtp_proj init
            elif n == "mtp_proj.weight":
                std = 0.632 * (p.size(-1) ** -0.5)
                bound = (3 ** 0.5) * std
                nn.init.uniform_(p, -bound, bound)
                print(f"{n}: uniform({-bound:.4f}, {bound:.4f})")

        print(f'\nTotal params: {n_params:,}')

    def forward(self, input_seq):
        B, T = input_seq.shape if input_seq.dim() == 2 else (1, input_seq.shape[0])
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)

        x = self.embeds(input_seq) * self.emb_scale
        v_first = torch.empty_like(x)

        for blk in self.blocks:
            x, v_first = checkpoint(blk, x, v_first, use_reentrant=False)

        return norm(x)
```

```python
def fused_loss_fn(model, input_seq, target, n_ignore=1, ignore=-100):
    target = model.unembeds.activate(target)
    xn = model(input_seq)
    model.unembeds.update_new_tokens_weight()

    B, T, C = xn.shape

    def prepare_mtp():
        zeros = torch.zeros_like(xn[:, :1])
        xx = torch.cat([zeros, xn[:, :-1]], dim=1)  # xn shift right
        x0 = norm(model.embeds(input_seq)) * model.emb_scale
        y0 = torch.cat([xx, x0], dim=-1)
        y1 = model.mtp_proj(y0.view(B * T, 2 * C)).view(B, T, C)
        return y1

    y1 = checkpoint(prepare_mtp, use_reentrant=False)
    v_first_mtp = torch.empty_like(y1)
    y2, _ = model.mtp_head(y1, v_first_mtp)
    yn = norm(y2)

    # Flatten for FusedCE: (B, T, C) -> (B*T, C), target (B, T) -> (B*T,)
    xn_flat = xn.view(B * T, C)
    yn_flat = yn.view(B * T, C)
    target_flat = target.view(B * T)
    target_flat[0] = ignore

    mtp_loss = FusedCE.apply(yn_flat, model.unembeds.active_weight, target_flat, n_ignore, ignore, 0.2)
    ntp_loss = FusedCE.apply(xn_flat, model.unembeds.active_weight, target_flat, n_ignore, ignore, 0.8)
    return mtp_loss + ntp_loss
```

---

## CODE: RWKV7 Reference (`tools/rwkv7/model.py`) - Full Init Code

### RWKV_Tmix_x070.__init__ - FULL CODE (Reference)
```python
class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C): ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 -  torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 -  torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 -  torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 -  torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]): nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32))
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5)  # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32))
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32))
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32))
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()
```

### RWKV_CMix_x070.__init__ - FULL CODE (Reference)
```python
class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()
```

### Block Structure (Reference)
```python
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)  # Q: WinRWKV doesn't have ln0

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x))  # SEQUENTIAL, not parallel
        return x, v_first
```

### RWKV Model Structure (Reference)
```python
class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)  # NOTE: No emb_scale here!

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)
        return x
```

### generate_init_weight - FULL CODE (Reference)
```python
def generate_init_weight(self):
    m = {}
    n_params = 0
    for n in self.state_dict():
        p = self.state_dict()[n]
        shape = p.shape

        scale = 1.0
        if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
            if 'ln_x.weight' in n:
                layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                m[n] = (p * 0.0) + (layer_scale ** 0.7)
            else:
                m[n] = p
        elif n == "emb.weight":
            m[n] = p
            scale = -1e-4
            nn.init.uniform_(m[n], a=scale, b=-scale)
        elif n == "head.weight":
            m[n] = p
            if self.args.vocab_size > self.args.n_embd:
                scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
            else:
                scale = 0.5
            nn.init.orthogonal_(m[n], gain=scale)
        else:
            assert n.endswith('.weight')

            zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
            for kk in zero:
                if kk in n:
                    scale = 0

            for kk in [".att.key."]:
                if kk in n:
                    scale = 0.1
            for kk in [".att.gate."]:
                if kk in n:
                    scale = 0.1

            if self.args.accelerator.upper() == "GPU":
                m[n] = torch.empty((shape[0], shape[1]), device="cuda")
            else:
                m[n] = torch.empty((shape[0], shape[1]))

            if scale == 0:
                nn.init.zeros_(m[n])
            elif scale < 0:
                nn.init.uniform_(m[n], a=scale, b=-scale)
            else:
                nn.init.orthogonal_(m[n], gain=scale)

        m[n] = m[n].cpu()
        if os.environ["RWKV_FLOAT_MODE"] == "fp16":
            m[n] = m[n].half()
        elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
            m[n] = m[n].bfloat16()
        n_params += m[n].numel()

    return m
```

### Key Init Rules from generate_init_weight:
| Weight Pattern | Scale | Init Method |
|----------------|-------|-------------|
| `ln_*`, `.ln*`, `time_*`, `_mask`, `pos_emb`, `_w`, `_w1`, `_w2`, `_bias` | - | Keep default |
| `ln_x.weight` | `(layer+1)/n_layer)^0.7` | Fill constant |
| `emb.weight` | `-1e-4` | uniform(-1e-4, 1e-4) |
| `head.weight` | `0.5*sqrt(vocab/embd)` or `0.5` | orthogonal |
| `.att.output.weight` | `0` | zeros |
| `.ffn.value.weight` | `0` | zeros |
| `.att.key.weight` | `0.1` | orthogonal |
| `.att.gate.weight` | `0.1` | orthogonal |
| Other `.weight` | `1.0` | orthogonal |

---

## CODE: WinGPT Reference (`wingpt.py`) - Key Techniques

### Techniques Applied to WinRWKV
```python
# 1. emb_scale = sqrt(dim) - large residuals
self.emb_scale = math.sqrt(dim)
x = self.embeds(input_seq) * self.emb_scale

# 2. MTP (Multi-Token Prediction)
self.mtp_head = RwkvBlock(...)
self.mtp_proj = nn.Linear(2 * dim, dim, bias=False)
# Loss: 0.8 NTP + 0.2 MTP

# 3. Parallel block (from WinGPT)
return x + x_att + x_ffn, v_first  # parallel

# 4. Gradient checkpointing
x, v_first = checkpoint(blk, x, v_first, use_reentrant=False)

# 5. OhMaiHead for output
self.unembeds = OhMaiHead(dim, vocab_size, bias=False)

# 6. FusedCE loss
FusedCE.apply(xn_flat, model.unembeds.active_weight, target_flat, ...)

# 7. INT8 mixed precision
convert_int8_mixed_precision(model)

# 8. Muon optimizer for proj layers
mpara = [p for n, p in model.named_parameters() if "proj" in n]
optim = Muon(mpara)
```

### Techniques NOT Applied (from WinGPT)
```python
# 1. GTA (Grouped Tied Attention) - N/A, RWKV uses different mechanism
# 2. RoPE/NoPE mixed - N/A, RWKV doesn't use positional encoding
# 3. Flash Attention - N/A, RWKV uses WKV kernel
# 4. Softcap - N/A, RWKV doesn't have attention scores
```

---

## Deliverables

- [ ] **Init Correctness Table**: Compare each weight init between WinRWKV vs RWKV7 reference
- [ ] **Architecture Diff Analysis**: Sequential vs Parallel block impact
- [ ] **Missing Inits**: What RWKV7 inits are missing in WinRWKV?
- [ ] **WinGPT Techniques Checklist**: What's applied, what's missing, what's N/A
- [ ] **Potential Issues**: Division by zero, numerical stability, edge cases
- [ ] **Recommendations**: Priority-ordered fixes

---

## Specific Questions

1. **Parallel vs Sequential Block**: RWKV7 uses `x = x + att; x = x + ffn` (sequential). WinRWKV uses `x + att + ffn` (parallel). Is this a problem for RWKV's state-based mechanism?

2. **Missing ln0**: RWKV7 has `ln0` at layer 0 before attention. WinRWKV doesn't. Impact?

3. **emb_scale**: RWKV7 doesn't scale embeddings. WinRWKV uses `sqrt(dim)`. Conflict with RWKV7's small embedding init?

4. **head.weight init**: RWKV7 uses orthogonal init for head. WinRWKV uses OhMaiHead. Compatible?

5. **key.weight init**: RWKV7 reference uses orthogonal with scale 0.1. WinRWKV uses uniform. Impact on training?

6. **Division by zero**: `ratio_0_to_1 = layer_id / (n_layers - 1)` fails when n_layers=1. Edge case?

7. **MTP v_first**: MTP head creates new `v_first_mtp = torch.empty_like(y1)`. Should it share v_first from main model?


---

## CODE: RWKV7 Optimizer Config (Reference)

### configure_optimizers - Learning Rate Groups
```python
def configure_optimizers(self):
    args = self.args
    
    lr_decay = set()
    lr_1x = set()
    lr_2x = set()
    for n, p in self.named_parameters():
        if ("att.w0" in n):
            lr_2x.add(n)  # NOTE: w0 gets 2x learning rate!
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
            lr_decay.add(n)
        else:
            lr_1x.add(n)

    optim_groups = [
        {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
    ]
    # ... uses FusedAdam or DeepSpeedCPUAdam
```

### Q: WinRWKV uses Muon for "proj" params, Adam for others. Does this conflict with RWKV7's lr_2x for att.w0?

---

## Summary Checklist for Review

### Init Comparison Table (to be filled by reviewer)
| Component | RWKV7 Reference | WinRWKV | Match? |
|-----------|-----------------|---------|--------|
| `x_r, x_w, x_k, x_v, x_a, x_g` | ratio-based pow | ? | ? |
| `w0, w1, w2` (decay LoRA) | zeros, ortho(0.1), decay_speed+0.5 | ? | ? |
| `a0, a1, a2` (AAA LoRA) | zeros, zeros, ortho(0.1) | ? | ? |
| `v0, v1, v2` (MV LoRA) | 1.0, zeros, ortho(0.1) | ? | ? |
| `g1, g2` (gate LoRA) | zeros, ortho(0.1) | ? | ? |
| `k_k, k_a, r_k` | 0.85, 1.0, zeros | ? | ? |
| `receptance.weight` | uniform(-0.5/√C, 0.5/√C) | ? | ? |
| `key.weight` | uniform(-0.05/√C, 0.05/√C) | ? | ? |
| `value.weight` | uniform(-0.5/√C, 0.5/√C) | ? | ? |
| `output.weight` | zeros | ? | ? |
| `ln_x` | GroupNorm(H, C, eps=64e-5) | ? | ? |
| `emb.weight` | uniform(-1e-4, 1e-4) | ? | ? |
| `head.weight` | orthogonal(0.5*sqrt(vocab/embd)) | OhMaiHead | ? |
| Block structure | sequential + ln0 at layer 0 | parallel, no ln0 | ❌ |
| emb_scale | None | sqrt(dim) | ❌ |
| Optimizer lr_2x for w0 | Yes | No (uses Muon) | ❌ |
