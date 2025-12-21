#!/usr/bin/env python3
''' RWKV7 for the WIN
- Cấu trúc model giống hệt tools/rwkv7 để tương thích checkpoint
- Sequential block: x = x + att(ln1(x)); x = x + ffn(ln2(x))
- MTP (Multi-Token Prediction) từ WinGPT
- Các kỹ thuật tối ưu khác trong `optimus.py`
'''
import os, math, torch, torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch.utils.cpp_extension import load

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
            ratio_0_to_1 = layer_id / (n_layers - 1)
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

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(dim)

        self.att = RWKV_Tmix(dim, layer_id, n_layers)
        self.ffn = RWKV_CMix(dim, layer_id, n_layers)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_att, v_first = self.att(self.ln1(x), v_first)
        x = x + x_att
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class WinRWKV(nn.Module):
    def __init__(self, vocab_size, n_layers, dim, ctxlen):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.ctxlen = ctxlen
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RwkvBlock(dim, i, n_layers) for i in range(n_layers)])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # MTP components
        self.mtp_head = RwkvBlock(dim, n_layers, n_layers + 1)
        self.mtp_proj = nn.Linear(2 * dim, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        print("\n" + "#" * 76)
        print("# Init model weight")
        print("#" * 76 + "\n")

        n_params = 0
        for n, p in self.named_parameters():
            shape = p.shape
            n_params += p.numel()

            # ln_x.weight special init
            if 'ln_x.weight' in n:
                parts = n.split('.')
                layer_id = int(parts[1]) if parts[0] == 'blocks' else self.n_layers
                layer_scale = (1 + layer_id) / self.n_layers
                p.data.fill_(layer_scale ** 0.7)

            # emb init
            elif n == "emb.weight":
                scale = 1e-4
                nn.init.uniform_(p, -scale, scale)
                print(f"{n}: uniform({-scale}, {scale})")

            # head init (orthogonal like rwkv7)
            elif n == "head.weight":
                if self.vocab_size > self.dim:
                    scale = 0.5 * math.sqrt(self.vocab_size / self.dim)
                else:
                    scale = 0.5
                p_float = p.data.float()
                nn.init.orthogonal_(p_float, gain=scale)
                p.data.copy_(p_float.to(p.dtype))
                print(f"{n}: orthogonal(gain={scale:.4f})")

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

        x = self.emb(input_seq)
        v_first = torch.empty_like(x)

        for blk in self.blocks:
            x, v_first = checkpoint(blk, x, v_first, use_reentrant=False)

        x = self.ln_out(x)
        return x


def fused_loss_fn(model, input_seq, target, n_ignore=1, ignore=-100):
    xn = model(input_seq)
    B, T, C = xn.shape

    def prepare_mtp():
        zeros = torch.zeros_like(xn[:, :1])
        xx = torch.cat([zeros, xn[:, :-1]], dim=1)  # xn shift right
        x0 = model.ln_out(model.emb(input_seq))
        y0 = torch.cat([xx, x0], dim=-1)
        y1 = model.mtp_proj(y0.view(B * T, 2 * C)).view(B, T, C)
        return y1

    y1 = checkpoint(prepare_mtp, use_reentrant=False)
    v_first_mtp = torch.empty_like(y1)
    y2, _ = model.mtp_head(y1, v_first_mtp)
    yn = model.ln_out(y2)

    # Compute logits
    xn_flat = xn.view(B * T, C)
    yn_flat = yn.view(B * T, C)
    target_flat = target.view(B * T).clone()
    target_flat[0] = ignore

    # Use head for logits, standard cross entropy
    logits_ntp = model.head(xn_flat)
    logits_mtp = model.head(yn_flat)

    ntp_loss = F.cross_entropy(logits_ntp, target_flat, ignore_index=ignore, reduction='mean')
    mtp_loss = F.cross_entropy(logits_mtp, target_flat, ignore_index=ignore, reduction='mean')

    return 0.8 * ntp_loss + 0.2 * mtp_loss


########################
##  TESTING  TESTING  ##
########################
if __name__ == "__main__":
    ctxlen = 1024
    vocab_size = 64 * 1024
    dim, n_layers = 256, 8
    print(f"winrwkv config: layers={n_layers}, dim={dim}, heads={dim // HEAD_SIZE}; ctxlen={ctxlen}")

    torch.manual_seed(1981)
    model = WinRWKV(vocab_size, n_layers, dim, ctxlen).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    after_init_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak VRAM after model initialization: {after_init_memory:.2f} MB")
    model.train()

    for step in range(5):
        input_seq = torch.randint(5, vocab_size // 4, (1, ctxlen), dtype=torch.long).cuda()
        target = F.pad(input_seq[:, 1:], (0, 1), mode='constant', value=-100)

        optimizer.zero_grad()

        loss = fused_loss_fn(model, input_seq, target)
        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"step {step}, loss {loss.item():.4f}, Peak VRAM: {current_memory:.2f} MB")

        loss.backward()
        optimizer.step()
