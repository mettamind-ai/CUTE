#!/usr/bin/env python3
## GPT for the WIN (cải biên từ modded nanogpt)
import os, math, torch, torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from optimus import Int8MixedLinear, quantize_int8, FusedLinearCrossEntropy
from ohmai import OhMaiEmbedding, OhMaiHead
from flash_attn import flash_attn_varlen_func

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.coordinate_descent_tuning = True
torch.set_float32_matmul_precision('high') # better for f32 head
torch.backends.cuda.matmul.allow_tf32  = True
torch.set_default_dtype(torch.bfloat16)

def norm(x: Tensor): # root mean square của các phần tử theo chiều cuối
    return F.rms_norm(x, (x.size(-1),))

@torch.no_grad()
def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5) # 0.5 is a bit better
    bound = (3 ** 0.5) * std         # than default 1/sqrt(3)
    return w.uniform_(-bound, bound)

class ReLuSquareMLP(nn.Module):
    def __init__(self, dim:int, hdim=None, odim=None):
        super().__init__()
        if not hdim: hdim = int(3 * dim)
        if not odim: odim = dim

        self.fc1_proj = nn.Linear(dim, hdim, bias=False)
        self.fc2_proj = nn.Linear(hdim, odim, bias=False)
        
        with torch.no_grad():
            self.fc1_proj.weight.copy_(init_linear(torch.empty(hdim, dim)))
            self.fc2_proj.weight.zero_()
        
        # Add weight decay multiplier attribute to the weights
        self.fc1_proj.weight.wd_mul = 2.0  # điều chỉnh hệ số weight decay
        self.fc2_proj.weight.wd_mul = 2.0  # gấp đôi so với mặc định 

    def forward(self, x):
        y = self.fc1_proj(x)
        y = F.relu(y).square() 
        y = self.fc2_proj(y)
        return y

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
    def __init__(self, dim:int, num_heads:int, num_kv_heads:int, 
            seq_len:int, head_dim=128, long=False, layer_id=-1):
        super().__init__() # dim = hidden_size = embedding = feature = representation

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id

        qo_inner_dim = num_heads * head_dim
        kv_inner_dim = num_kv_heads * head_dim

        self.kv_proj = nn.Linear(dim, 2*kv_inner_dim, bias=False)
        self. q_proj = nn.Linear(dim,   qo_inner_dim, bias=False)
        self. o_proj = nn.Linear(  qo_inner_dim, dim, bias=False)

        with torch.no_grad(): # init weights
            self.kv_proj.weight.copy_(init_linear(torch.empty(2*kv_inner_dim, dim)))
            self. q_proj.weight.copy_(init_linear(torch.empty(qo_inner_dim, dim)))
            self. o_proj.weight.zero_() # zero init

        if long: self.rope, self.window  = False, 1024*4
        else:    self.rope, self.window  = True,  1024*1

        print(f"Layer {layer_id} => {'RoPE' if self.rope else 'Nope'}, win {self.window}")
        self.attn_scale = 0.12

    def forward(self, x, v_emb, ve_lambdas, cu_seqlens, max_seqlen, rotary):
        q    = self.q_proj(x)
        k, v = self.kv_proj(x).chunk(2, dim=-1) # T, C

        if ve_lambdas is not None and v_emb is not None:
            v = ve_lambdas[0]*v + ve_lambdas[1]*v_emb

        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim
        T, C = k.shape; assert C == Hkv * D

        q = q.contiguous().view(T, H,   D)
        k = k.contiguous().view(T, Hkv, D)
        v = v.contiguous().view(T, Hkv, D)

        q, k, v = norm(q), norm(k), norm(v) # theo chiều D
        if self.rope: q, k = rotary(q), rotary(k)

        y = flash_attn_varlen_func( q, k, v,
            cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True, 
            dropout_p=0.0, softmax_scale=self.attn_scale, window_size=(self.window, 0),
        ).contiguous()
        y = y.reshape(T, H * D)
        return self.o_proj(y)

##############################
## Transformer for the WIN  ##
##############################
class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, max_seq_len, head_dim=128, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.mlp = ReLuSquareMLP(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, max_seq_len, 
                        head_dim=head_dim, long=layer_id % 6 == 5, layer_id=layer_id) # 5 ngắn + 1 dài

    def forward(self, x, x0, ve, te_lambdas, ve_lambdas, cu_seqlens, max_seqlen, rotary):
        x                     = te_lambdas[0] *  x # te_lambdas[0] init là 1
        if x0 is not None: x += te_lambdas[1] * x0 # trộn với tok emb gốc
        x = x + self.attn(x, ve, ve_lambdas, cu_seqlens, max_seqlen, rotary)
        x = x + self.mlp(norm(x))
        return x

class WinGPT(nn.Module):
    def __init__(self, vocab_size:int, n_layers:int, num_heads:int, num_kv_heads:int, dim:int, max_seq_len:int, head_dim=128, active_vocab=None):
        super().__init__()

        self.ohmai = ( active_vocab is not None )
        Embedding, Head = (OhMaiEmbedding, OhMaiHead) if self.ohmai else (nn.Embedding, nn.Linear)
        print(f"OhMai? {self.ohmai}; using {Embedding.__name__} and {Head.__name__}")
        self.rotary = Rotary(head_dim, max_seq_len)

        self.blocks = nn.ModuleList([Block(dim, num_heads, num_kv_heads, max_seq_len, head_dim, layer_id=i) for i in range(n_layers)])
        self.dim, self.kv_dim = dim, num_kv_heads*head_dim
        
        self.ve = n_layers // 2
        self.embeddings = Embedding(vocab_size, dim + self.kv_dim*self.ve, active_vocab)

        self.scalars = nn.Parameter(torch.cat([
          *[torch.tensor([1.0, 0.0 ]) for _ in range(n_layers)], # token emb mix
          *[torch.tensor([0.5, 0.5 ]) for _ in range(n_layers)], # value emb mix
        ]))

        self.lm_head = Head(dim, vocab_size, bias=False)
        if isinstance(self.lm_head, nn.Linear):  # khởi tạo riêng cho nn.Linear head
            with torch.no_grad(): self.lm_head.weight.zero_()

    def update_async_weight(self):
        if isinstance(self.embeddings, OhMaiEmbedding): self.embeddings.update_async_weight()
        if isinstance(self.lm_head, OhMaiHead): self.lm_head.update_async_weight()

    @torch.compile()
    def forward(self, input_seq, cu_seqlens, max_seqlen):
        n_blks = len(self.blocks)
        embs = self.embeddings(input_seq.long())
        x = x0 = norm(embs[..., : self.dim ])

        v_embs = embs[..., -self.kv_dim*self.ve : ]
        v_embs = list(v_embs.chunk(self.ve, dim=-1))

        skips = [None]*(n_blks - 2*len(v_embs))
        v_embs = v_embs + skips + v_embs
        assert len(v_embs) == n_blks

        ## Độn None cho đầy v_embs
        v_embs += [None]*(n_blks - len(v_embs))
        assert len(v_embs) == n_blks

        te_lambdas = self.scalars[0*n_blks : 2*n_blks].view(-1, 2)
        ve_lambdas = self.scalars[2*n_blks : 4*n_blks].view(-1, 2)
        
        for i in range(n_blks):
            if self.ohmai and i == int(n_blks*0.6): self.lm_head.update_new_tokens_weight() # upload ...
            def fwd(blk, x0, ve, tl, vl, c, m): return lambda x: blk(x, x0, ve, tl, vl, c, m, self.rotary)
            f = fwd(self.blocks[i], x0, v_embs[i], te_lambdas[i], ve_lambdas[i], cu_seqlens, max_seqlen)
            x = checkpoint(f, x, use_reentrant=False)
        return norm(x)

def simple_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen):
    if model.ohmai: target = model.lm_head.activate(target)  # async offload head weight ...
    hidden = model(input_seq, cu_seqlens, max_seqlen)        # hidden chưa norm
    w = model.lm_head.active_weight if model.ohmai else model.lm_head.weight
    logits = ( hidden @ w.t() ).float() 
    logits = 15*logits*torch.rsqrt(logits.square() + 15*15)
    return F.cross_entropy(logits, target)
    
def fused_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen, n_ignore=0, ignore=-100):
    if model.ohmai: target = model.lm_head.activate(target)  # async offload head weight ...
    hidden = model(input_seq, cu_seqlens, max_seqlen)        # hidden chưa norm
    w = model.lm_head.active_weight if model.ohmai else model.lm_head.weight
    return FusedLinearCrossEntropy.apply(hidden, w, target, n_ignore, ignore)

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

    seed = 1982
    seq_len = 1024
    vocab_size = 2048
    dim, n_layers = 256, 8
    num_heads, num_kv_heads = 8, 4
    print(f"Model config: layers={n_layers}, dim={dim}, heads={num_heads}/{num_kv_heads}; seq_len={seq_len}")

    torch.manual_seed(seed)
    model = WinGPT(vocab_size, n_layers, num_heads, num_kv_heads, dim, seq_len).cuda()
    
    # Khởi tạo model 2 dùng OhMaiEmbedding và OhMaiHead
    torch.manual_seed(seed)
    ohmai = WinGPT(vocab_size, n_layers, num_heads, num_kv_heads, dim, seq_len, active_vocab=vocab_size//2).cuda()

    # Đảm bảo toàn bộ tham số của 2 model là như nhau
    def check_params():
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), ohmai.named_parameters()):
            if n2 == "embeddings.active_weight":
                n2 = "embeddings.weight"
                p2 = ohmai.embeddings.weight.cuda()
            if n2 == "lm_head.active_weight":
                n2 = "lm_head.weight"
                p2 = ohmai.lm_head.weight.cuda()
            assert n1 == n2, f"{n1} != {n2}"
            assert torch.allclose(p1, p2), f"{n1} values are different"
        print("Params của 2 model đã được khởi tạo giống hệt nhau")
    check_params()

    # Đảm bảo toàn bộ 2 models đều là bf16
    for m in [model, ohmai]:
        for n, p in m.named_parameters(): assert p.dtype == torch.bfloat16, f"{n} is not bf16"
        print(f"All {'ohmai' if m.ohmai else 'model'} params are in bfloat16.")

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

    tok_emb_before = ohmai.embeddings.weight.data.clone()
    head_before = ohmai.lm_head.weight.data.clone()

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
        a = ohmai.embeddings(input_seq).cpu()
        b = ohmai.embeddings.weight[input_seq.cpu().long()]
        assert torch.allclose(a, b, atol=1e-5), f"2 cách lấy embeddings không trùng khớp, {a}\n{b}"

        loss_model = simple_loss_fn(model, input_seq, target, cu_seqlens, max_seqlen)
        loss_ohmai = fused_loss_fn( ohmai, input_seq, target, cu_seqlens, max_seqlen)

        ## Đảm bảo 2 cách lấy head là giống nhau
        a = ohmai.lm_head.weight.cuda()[target]
        inverse = ohmai.lm_head.activate(target)
        b = ohmai.lm_head.active_weight[inverse]
        assert torch.allclose(a, b, atol=1e-5), f"2 cách lấy head không trùng khớp, {a}\n{b}"
 
        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"step {step}, loss_model {loss_model.item():.4f}, loss_ohmai {loss_ohmai.item():.4f}, ", end="")

        loss_ohmai.backward(); loss_model.backward()
        optim.step(); aptim.step()

        print(f"Peak VRAM: {current_memory:.2f} MB")
        ohmai.update_async_weight() # đảm bảo async weights (embeddings/head) đã được cập nhật

    tok_emb_after = ohmai.embeddings.weight.data
    diff = (tok_emb_before != tok_emb_after).sum().item()
    assert diff > 0, f"Số lượng thay đổi {diff}\n{tok_emb_before}\n{tok_emb_after}"

    head_after = ohmai.lm_head.weight.data
    diff = (head_before != head_after).sum().item()
    assert diff > 0, f"Số lượng thay đổi {diff}\n{head_before}\n{head_after}"
