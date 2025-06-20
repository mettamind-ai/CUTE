#!/usr/bin/env python3
## GPT for the WIN (cải biên từ modded nanogpt)
import os, math, torch, torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from optimus import Int8MixedLinear, quantize_int8, FusedCE
from ohmai import OhMaiEmbedding, OhMaiHead
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

class ReLuSquareMLP(nn.Module):
    def __init__(self, dim:int, hdim=None, odim=None, expansion_factor=4, zero_out=True):
        super().__init__()
        if not hdim: hdim = int(dim*expansion_factor)
        if not odim: odim = dim

        self.fc1_proj = nn.Linear(dim, hdim, bias=False)
        self.fc2_proj = nn.Linear(hdim, odim, bias=False)

        # Add weight decay multiplier attribute to the weights
        self.fc1_proj.weight.wd_mul = 2.0  # điều chỉnh hệ số weight decay
        self.fc2_proj.weight.wd_mul = 2.0  # gấp đôi so với mặc định 

        with torch.no_grad():
            self.fc1_proj.weight.copy_(init_linear(torch.empty(hdim, dim)))
            if zero_out: self.fc2_proj.weight.zero_() # sẽ đc residual connect nên khởi tạo là 0
            else:        self.fc2_proj.weight.copy_(init_linear(torch.empty(odim, hdim)))

    def forward(self, x):
        def prepare(x):
            x = self.fc1_proj(norm(x))
            return F.relu(x).square()
        x = checkpoint(prepare, x, use_reentrant=False)
        return self.fc2_proj(x)

##########################
## CausalSelfAttention  ##
##########################
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

    def forward(self, x_THD: Tensor):
        seq_len = x_THD.size(-3) # T seq_len, head, dim (of head)
        assert self.cos.size(0) >= seq_len, f"{self.cos.size(0)} >= {seq_len}?"

        cos = self.cos[:seq_len, None, :] # [seq_len, 1, dim]
        sin = self.sin[:seq_len, None, :] # [seq_len, 1, dim]

        x1, x2 = x_THD.to(dtype=torch.float32).chunk(2, dim=-1)

        y1 = x1 * (+cos) + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_THD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim:int, num_heads:int, num_kv_heads:int, seq_len:int, head_dim=128, long=False, layer_id=-1, odim=None):
        super().__init__() # dim = hidden_size = embedding = feature = representation

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.seq_len = seq_len
        if odim == None: odim = dim

        self.qo_dim = num_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim

        self.qk_proj = nn.Linear(dim, self.qo_dim + self.kv_dim, bias=False)
        self. o_proj = nn.Linear(self.qo_dim, odim, bias=False)

        with torch.no_grad():
            self.qk_proj.weight.copy_(init_linear(torch.empty(self.qo_dim + self.kv_dim, dim)))
            self. o_proj.weight.zero_() # zero init

        if long: self.rope, self.window  = False, 1024*4
        else:    self.rope, self.window  = True,  1024

        print(f"Layer {layer_id} => {'RoPE' if self.rope else 'Nope'}, win {self.window}")
        self.attn_scale = 0.12


    def forward(self, x, v_emb, cu_seqlens, max_seqlen, rotary):
        H, Hkv  = self.num_heads, self.num_kv_heads
        D, T    =  self.head_dim, self.seq_len

        def prepare(qk, v_emb):
            q  = qk[..., : self.qo_dim ]
            k  = qk[..., self.qo_dim : ]

            q = q    .view(T, H,   D)
            k = k    .view(T, Hkv, D)
            v = v_emb.view(T, Hkv, D)

            q, k, v = norm(q), norm(k), norm(v)
            if self.rope: q, k = rotary(q), rotary(k)
            return q, k, v

        q, k, v = checkpoint(prepare, self.qk_proj(x), v_emb, use_reentrant=False)
        y = flash_attn_varlen_func(q, k, v,
            cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True, 
            dropout_p=0.0, softmax_scale=self.attn_scale, window_size=(self.window, 0),
        )
        y = y.view(T, H * D)
        z = self.o_proj(y)
        return z


##############################
## Transformer for the WIN  ##
##############################
class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, max_seq_len, head_dim=128, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.long = layer_id % 5 == 4  # 4 ngắn + 1 dài
        self.mlp = ReLuSquareMLP(dim) if layer_id != 0 else None
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, max_seq_len, head_dim, self.long, layer_id)

    def forward(self, x, v_emb, cu_seqlens, max_seqlen, rotary):
        if self.mlp is not None: x = x + self.mlp(x)
        return x + self.attn(x, v_emb, cu_seqlens, max_seqlen, rotary)


class WinGPT(nn.Module):
    def __init__(self, vocab_size, n_layers, num_heads, num_kv_heads, dim, max_seq_len, head_dim = 128, active_vocab=None):
        super().__init__()
        self.n_layers = n_layers

        Embedding = OhMaiEmbedding if active_vocab else nn.Embedding
        Unembedding = OhMaiHead    if active_vocab and vocab_size >= 32*1024 else nn.Linear
        print(f"Using {Embedding.__name__} and {Unembedding.__name__}")
        self.rotary = Rotary(head_dim, max_seq_len)

        blks = [ Block(dim, num_heads, num_kv_heads, max_seq_len, head_dim, layer_id=i) for i in range(n_layers) ]
        self.blocks = nn.ModuleList(blks)
        self.dim, self.kv_dim = dim, num_kv_heads * head_dim
        
        self.tok_dim = dim
        self.embeds  = Embedding(vocab_size, self.tok_dim + self.kv_dim*n_layers, active_vocab)
        self.tok_mlp = ReLuSquareMLP(self.tok_dim, hdim=2*self.tok_dim, odim=dim, zero_out=False)

        ##   head0 chính là trunk (thân chính của model) to predict next token (NTP)
        self.head1_mlp  = ReLuSquareMLP(  dim, hdim=2*dim) # Early exit ở layer giữa, nên mọc thêm head1 to NTP
        self.head2_mlp  = ReLuSquareMLP(2*dim, hdim=4*dim, odim=dim) # head2 to predict next of next token (MTP)

        self.unembeds = Unembedding(dim, vocab_size, bias=False)
        if isinstance(self.unembeds, nn.Linear):  # khởi tạo riêng cho nn.Linear head
            with torch.no_grad(): self.unembeds.weight.zero_()

        self.skip_from = { (n_layers-i): i for i in range(2, (n_layers-1) // 2, 2) }
        print("WinGPT.skip_from", self.skip_from)


    def update_async_weight(self):
        if isinstance(self.embeds, OhMaiEmbedding): self.embeds.update_async_weight()
        if isinstance(self.unembeds, OhMaiHead):  self.unembeds.update_async_weight()


    def forward(self, input_seq, cu_seqlens, max_seqlen):
        def prepare(embs):
            x = x0 = self.tok_mlp(embs[..., : self.tok_dim ]) # thu tok_dim về dim
            v_embs = list(embs[..., self.tok_dim : ].chunk(self.n_layers, dim=-1))
            return x, x0, v_embs
        x, x0, v_embs = checkpoint(prepare, self.embeds(input_seq.long()), use_reentrant=False)
        
        for i, blk in enumerate(self.blocks):
            x = blk(x, v_embs[i], cu_seqlens, max_seqlen, self.rotary)
            if i == self.n_layers//2: xe = x # early exit
        return x, xe, x0


def fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen, n_ignore=0, ignore=-100):
    ohmaihead = isinstance(model.unembeds, OhMaiHead)
    if ohmaihead: target = model.unembeds.activate(target)  # async offload old token weight ...
    x, xe, x0 = model(input_seq, cu_seqlens, max_seqlen)    # tất cả chưa norm
    if ohmaihead: model.unembeds.update_new_tokens_weight() # async upload new token weight ...
 
    ## Prepare to predict next tokens, không sử dụng head riêng cho NTP vì sẽ làm giảm perf
    def prepare(x, x0, target, xe):
        zeros = torch.zeros_like(x[:1])
        xx    = torch.cat([zeros, x[:-1]], dim=0) # x dịch phải
        xx_x0 = torch.cat([xx, x0], dim=-1)
        re    = (xx + x0) * 0.5 # residual
        y     = re + model.head2_mlp(xx_x0)
        xe    = xe + model.head1_mlp(xe)
        ty    = F.pad(target[1:], (1, 0), mode='constant', value=ignore)
        return norm(y), ty, norm(x), norm(xe)
    y, ty, x, xe = checkpoint(prepare, x, x0, target, xe, use_reentrant=False)

    ## Tính loss cho early exit (x_half), NTP (x) và MTP (y) và cộng lại ưu tiên nhiệm vụ chính NTP
    w     = model.unembeds.active_weight if ohmaihead else model.unembeds.weight
    hloss = FusedCE.apply(xe, w, target, n_ignore, ignore, 0.1)  # NTP: Early exit
    xloss = FusedCE.apply(x,  w, target, n_ignore, ignore, 0.6)  # NTP: Next token prediction
    yloss = FusedCE.apply(y,  w, ty,     n_ignore, ignore, 0.3)  # MTP: Next of next token prediction
    return xloss + yloss + hloss


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
    seq_len = 256
    vocab_size = 32*1024
    dim, n_layers = 128, 8
    num_heads, num_kv_heads = 16, 1
    print(f"win config: layers={n_layers}, dim={dim}, heads={num_heads}/{num_kv_heads}; seq_len={seq_len}")

    torch.manual_seed(seed)
    model = WinGPT(vocab_size, n_layers, num_heads, num_kv_heads, dim, seq_len).cuda()
    
    # Khởi tạo model 2 dùng OhMaiEmbedding và OhMaiHead
    torch.manual_seed(seed)
    ohmai = WinGPT(vocab_size, n_layers, num_heads, num_kv_heads, dim, seq_len, active_vocab=vocab_size//2).cuda()

    # Đảm bảo toàn bộ tham số của 2 model là như nhau
    def check_params():
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), ohmai.named_parameters()):
            if n2 == "embeds.active_weight":
                n2 = "embeds.weight"
                p2 = ohmai.embeds.weight.cuda()
            if n2 == "unembeds.active_weight":
                n2 = "unembeds.weight"
                p2 = ohmai.unembeds.weight.cuda()
            assert n1 == n2, f"{n1} != {n2}"
            assert torch.allclose(p1, p2), f"{n1} values are different"
        print("Params của 2 model đã được khởi tạo giống hệt nhau")
    check_params()

    # Đảm bảo toàn bộ 2 models đều là bf16
    for m in [model, ohmai]:
        for n, p in m.named_parameters(): assert p.dtype == torch.bfloat16, f"{n} is not bf16"
        print(f"All {'ohmai' if m == ohmai else 'model'} params are in bfloat16.")

    convert_int8_mixed_precision(model)
    convert_int8_mixed_precision(ohmai)
    # model = torch.compile(model) # chậm !!!

    apara = {n: p for n, p in model.named_parameters() if "proj" not in n}
    opara = [p for n, p in model.named_parameters() if "proj" in n]

    print("\nAdam:", apara.keys())
    apara = list(apara.values())

    # Bổ xung thêm ohmai params vào adam và muon
    for n, p in ohmai.named_parameters():
        if "proj" not in n: apara.append(p)
        else:  opara.append(p)

    aptim = torch.optim.Adam(apara)
    optim = Muon(opara)

    after_init_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    print(f"Peak VRAM after model initialization: {after_init_memory:.2f} MB")

    tok_emb_before = ohmai.embeds.weight.data.clone()
    head_before = ohmai.unembeds.weight.data.clone()

    model.train()
    ohmai.train()

    for step in range(10):
        ## Generate sequences with batch dimension
        vv = vocab_size//4
        input_seq = torch.randint(5, vv, (seq_len,), dtype=torch.long).cuda()
        target    = torch.randint(5, vv, (seq_len,), dtype=torch.long).cuda()
        cu_seqlens, max_seqlen = get_cu_max_seqlens_from(input_seq)

        optim.zero_grad()
        aptim.zero_grad()

        ## Đảm bảo 2 cách lấy embedding là giống nhau
        a = ohmai.embeds(input_seq).cpu()
        b = ohmai.embeds.weight[input_seq.cpu().long()]
        assert torch.allclose(a, b, atol=1e-5), f"2 cách lấy embeddings không trùng khớp\n{a}\n{b}"

        loss_model = fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen)
        loss_ohmai = fused_loss_fn(ohmai, input_seq, target, cu_seqlens, max_seqlen)

        ## Đảm bảo 2 cách lấy head là giống nhau
        a = ohmai.unembeds.weight.cuda()[target]
        inverse = ohmai.unembeds.activate(target)
        b = ohmai.unembeds.active_weight[inverse]
        assert torch.allclose(a, b, atol=1e-5), f"2 cách lấy head không trùng khớp\n{a}\n{b}"
 
        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"step {step}, loss_model {loss_model.item():.4f}, loss_ohmai {loss_ohmai.item():.4f}, ", end="")

        loss_ohmai.backward()
        loss_model.backward()
        optim.step()
        aptim.step()

        print(f"Peak VRAM: {current_memory:.2f} MB")
        ohmai.update_async_weight() # đảm bảo async weights (embeddings/head) đã được cập nhật

    tok_emb_after = ohmai.embeds.weight.data.clone()
    diff = (tok_emb_before != tok_emb_after).sum().item()
    assert diff > 0, f"Số lượng embedding thay đổi {diff}"

    head_after = ohmai.unembeds.weight.data.clone()
    diff = (head_before != head_after).sum().item()
    assert diff > 0, f"Số lượng head thay đổi {diff}"
