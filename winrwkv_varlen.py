#!/usr/bin/env python3
''' RWKV7 Varlen - Packed sequences support
- Based on winrwkv.py but uses varlen kernel for packed sequences
- Supports variable length sequences without padding waste
- Compatible with Flash Attention varlen interface (cu_seqlens)
'''
import os, math, torch, torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch.utils.cpp_extension import load
from optimus import FusedCE

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.coordinate_descent_tuning = True
torch.set_default_dtype(torch.bfloat16)

HEAD_SIZE = 64
CHUNK_LEN = 16

flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}",
    "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]

# Load varlen kernel
cu_varlen = f'{os.path.dirname(os.path.abspath(__file__))}/wkv7_varlen.cu'
load(name="wind_backstepping_varlen", sources=[cu_varlen], is_python_module=False, verbose=True, extra_cuda_cflags=flags)


class WindBacksteppingVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b, cu_seqlens):
        # w shape: (total_tokens, H, C)
        total_tokens, H, C = w.shape
        num_chunks = (total_tokens + CHUNK_LEN - 1) // CHUNK_LEN
        
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        assert cu_seqlens.dtype == torch.int32
        
        y = torch.empty_like(v)
        s_chunk = torch.empty(H, num_chunks, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(total_tokens, H, C, dtype=torch.float32, device=w.device)
        
        torch.ops.wind_backstepping_varlen.forward_varlen(w, q, k, v, z, b, cu_seqlens, y, s_chunk, sa)
        ctx.save_for_backward(w, q, k, v, z, b, cu_seqlens, s_chunk, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        if dy.dtype != torch.bfloat16:
            dy = dy.to(torch.bfloat16)
        dy = dy.contiguous()
        w, q, k, v, z, b, cu_seqlens, s_chunk, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping_varlen.backward_varlen(w, q, k, v, z, b, dy, cu_seqlens, s_chunk, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db, None  # None for cu_seqlens


def RUN_CUDA_RWKV7_VARLEN(q, w, k, v, a, b, cu_seqlens):
    """Run RWKV7 varlen kernel on packed sequences.
    
    Args:
        q, w, k, v, a, b: (total_tokens, dim) packed tensors
        cu_seqlens: (num_seqs + 1,) cumulative sequence lengths
    """
    total_tokens, HC = q.shape
    H = HC // HEAD_SIZE
    q, w, k, v, a, b = [i.view(total_tokens, H, HEAD_SIZE) for i in [q, w, k, v, a, b]]
    return WindBacksteppingVarlen.apply(w, q, k, v, a, b, cu_seqlens).view(total_tokens, HC)


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


@torch.no_grad()
def ortho_init(x, scale):
    shape = x.shape
    orig_dtype = x.dtype
    x = x.float()
    if len(shape) == 2:
        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
        nn.init.orthogonal_(x, gain=gain * scale)
    elif len(shape) == 3:
        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
        for i in range(shape[0]):
            nn.init.orthogonal_(x[i], gain=gain * scale)
    return x.to(orig_dtype)


class RWKV_Tmix_Varlen(nn.Module):
    """RWKV7 Time-mixing with varlen support."""
    def __init__(self, dim, layer_id, n_layers):
        super().__init__()
        self.layer_id = layer_id
        self.head_size = HEAD_SIZE
        assert dim % HEAD_SIZE == 0
        self.n_head = dim // self.head_size
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

    def forward(self, x, v_first, cu_seqlens):
        """
        Args:
            x: (total_tokens, C) packed sequences
            v_first: (total_tokens, C) or None
            cu_seqlens: (num_seqs + 1,) cumulative lengths
        """
        T, C = x.size()
        H = self.n_head
        num_seqs = cu_seqlens.size(0) - 1
        
        # Time shift per sequence (varlen-aware)
        xx = torch.zeros_like(x)
        for i in range(num_seqs):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            if end > start:
                xx[start + 1:end] = x[start:end - 1] - x[start + 1:end]
                # First token of each seq: xx = 0 - x = -x (but we want 0)
                # Actually time_shift = prev - curr, first token has no prev so use 0
        
        xr = x + xx * self.x_r.squeeze(0)
        xw = x + xx * self.x_w.squeeze(0)
        xk = x + xx * self.x_k.squeeze(0)
        xv = x + xx * self.x_v.squeeze(0)
        xa = x + xx * self.x_a.squeeze(0)
        xg = x + xx * self.x_g.squeeze(0)

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0.squeeze(0) + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v.clone()
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0.squeeze(0) + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0.squeeze(0) + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k.squeeze(0)
        kk = F.normalize(kk.view(T, H, -1), dim=-1, p=2.0).view(T, C)
        k = k * (1 + (a - 1) * self.k_a.squeeze(0))

        x = RUN_CUDA_RWKV7_VARLEN(r, w, k, v, -kk, kk * a, cu_seqlens)
        x = self.ln_x(x)

        x = x + ((r.view(T, H, -1) * k.view(T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(T, H, -1)).view(T, C)
        x = self.output(x * g)
        return x, v_first


class RWKV_CMix_Varlen(nn.Module):
    """RWKV7 Channel-mixing with varlen support."""
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

    def forward(self, x, cu_seqlens):
        T, C = x.size()
        num_seqs = cu_seqlens.size(0) - 1
        
        # Time shift per sequence
        xx = torch.zeros_like(x)
        for i in range(num_seqs):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            if end > start:
                xx[start + 1:end] = x[start:end - 1] - x[start + 1:end]
        
        k = x + xx * self.x_k.squeeze(0)
        k = self.key(k)
        k = torch.relu(k) ** 2
        return self.value(k)


class RwkvBlockVarlen(nn.Module):
    def __init__(self, dim, layer_id, n_layers):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(dim)

        self.att = RWKV_Tmix_Varlen(dim, layer_id, n_layers)
        self.ffn = RWKV_CMix_Varlen(dim, layer_id, n_layers)

    def forward(self, x, v_first, cu_seqlens):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_att, v_first = self.att(self.ln1(x), v_first, cu_seqlens)
        x = x + x_att
        x = x + self.ffn(self.ln2(x), cu_seqlens)
        return x, v_first


class WinRWKVVarlen(nn.Module):
    """WinRWKV with varlen kernel support for packed sequences."""
    def __init__(self, vocab_size, n_layers, dim, ctxlen):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.ctxlen = ctxlen
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RwkvBlockVarlen(dim, i, n_layers) for i in range(n_layers)])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        print("\n" + "#" * 76)
        print("# Init WinRWKV Varlen model weight")
        print("#" * 76 + "\n")

        n_params = 0
        for n, p in self.named_parameters():
            shape = p.shape
            n_params += p.numel()
            scale = 1.0

            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                if 'ln_x.weight' in n:
                    parts = n.split('.')
                    if parts[0] == 'blocks':
                        layer_id = int(parts[1])
                    else:
                        layer_id = self.n_layers
                    layer_scale = (1 + layer_id) / self.n_layers
                    p.data.fill_(layer_scale ** 0.7)
                continue

            if n == "emb.weight":
                scale = 1e-4
                nn.init.uniform_(p, -scale, scale)
                continue

            if n == "head.weight":
                if self.vocab_size > self.dim:
                    scale = 0.5 * math.sqrt(self.vocab_size / self.dim)
                else:
                    scale = 0.5
                p_float = p.data.float()
                nn.init.orthogonal_(p_float, gain=scale)
                p.data.copy_(p_float.to(p.dtype))
                continue

            assert n.endswith('.weight'), f"Unexpected param: {n}"

            zero_patterns = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
            for kk in zero_patterns:
                if kk in n:
                    scale = 0
                    break

            if ".att.key." in n:
                scale = 0.1
            if ".att.gate." in n:
                scale = 0.1

            if len(shape) >= 2:
                p_float = p.data.float()
                if scale == 0:
                    nn.init.zeros_(p_float)
                else:
                    nn.init.orthogonal_(p_float, gain=scale)
                p.data.copy_(p_float.to(p.dtype))

        print(f'Total params: {n_params:,}')

    def forward(self, input_ids, cu_seqlens, return_logits=False):
        """
        Args:
            input_ids: (total_tokens,) packed token ids
            cu_seqlens: (num_seqs + 1,) cumulative sequence lengths
            return_logits: if True, return logits; else return hidden states
        """
        total_tokens = input_ids.shape[0]
        
        x = self.emb(input_ids)  # (total_tokens, dim)
        v_first = None

        for blk in self.blocks:
            x, v_first = checkpoint(blk, x, v_first, cu_seqlens, use_reentrant=False)

        x = self.ln_out(x)
        if return_logits:
            return self.head(x)
        return x


def fused_loss_fn_varlen(model, input_ids, target, cu_seqlens, n_ignore=1, ignore=-100):
    """Compute loss for packed sequences."""
    xn = model(input_ids, cu_seqlens, return_logits=False)
    T, C = xn.shape
    
    # Ignore first token of each sequence
    target_masked = target.clone()
    num_seqs = cu_seqlens.size(0) - 1
    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        target_masked[start] = ignore
    
    # Use FusedCE
    loss = FusedCE.apply(xn, model.head.weight, target_masked, n_ignore, ignore, 1.0)
    return loss


########################
##  TESTING  TESTING  ##
########################
if __name__ == "__main__":
    ctxlen = 1024
    vocab_size = 64 * 1024
    dim, n_layers = 256, 8
    print(f"winrwkv_varlen config: layers={n_layers}, dim={dim}, heads={dim // HEAD_SIZE}; ctxlen={ctxlen}")

    torch.manual_seed(1981)
    model = WinRWKVVarlen(vocab_size, n_layers, dim, ctxlen).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    after_init_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak VRAM after model initialization: {after_init_memory:.2f} MB")
    model.train()

    # Test with packed sequences
    seq_lengths = [256, 384, 384]  # 3 sequences, total 1024 tokens
    total_tokens = sum(seq_lengths)
    
    for step in range(5):
        # Create packed input
        input_ids = torch.randint(5, vocab_size // 4, (total_tokens,), dtype=torch.long).cuda()
        
        # Create target (shift by 1 within each sequence)
        target = torch.full((total_tokens,), -100, dtype=torch.long).cuda()
        offset = 0
        for seq_len in seq_lengths:
            if seq_len > 1:
                target[offset:offset + seq_len - 1] = input_ids[offset + 1:offset + seq_len]
            offset += seq_len
        
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lengths), 0)), dtype=torch.int32).cuda()

        optimizer.zero_grad()

        loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"step {step}, loss {loss.item():.4f}, Peak VRAM: {current_memory:.2f} MB")

        loss.backward()
        optimizer.step()
