#!/usr/bin/env python3
## GPT for the WIN (cải biên từ modded nanogpt)
import os, math, torch
from torch import Tensor, nn
import torch.nn.functional as F

try: from flash_attn_interface import flash_attn_func, flash_attn_varlen_func; FA3_ENABLED = True
except:        from flash_attn import flash_attn_func, flash_attn_varlen_func; FA3_ENABLED = False
print("FA3_ENABLED?", FA3_ENABLED)

from optimus import Int8MixedLinear
from liger_kernel import LigerFusedLinearCrossEntropyFunction
from OhMai.embedding import OhMaiEmbedding

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


####################################
##  ReLuSquareMLP Channel Mixing  ##
####################################

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

    def forward(self, x:Tensor, te):
        y = self.fc1_proj(x)
        y = F.relu(y).square() 
        y = self.fc2_proj(y)
        if te is not None: y = y*te
        return y.to(x.dtype)



####################################
##  LIMe Layer-Integrated Memory  ##
####################################

# https://github.com/corl-team/lime/blob/main/src/lm/lime.py
class StaticRouter(nn.Module):
    def __init__(self, num_kv_heads, layer_id):
        super().__init__()
        self.R = num_kv_heads
        self.L = layer_id + 1
        self.fan_in = self.L * self.R

        with torch.no_grad():
            bound = math.sqrt(3 / self.fan_in)
            w = torch.zeros(self.R, self.fan_in).uniform_(-bound, bound)
            w[:, -self.R :] = torch.eye(self.R)
            self.static_weights = nn.Parameter(w)

    def extra_repr(self) -> str: return f"n_repr={self.fan_in}, heads={self.R}"
    
    def forward(self, stacked_last_hiddens): # stacked_last_hiddens: (L H) (B T 2 Hd)
        return self.static_weights.mm(stacked_last_hiddens)


class Layer0Router(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, stacked_last_hiddens):
        return stacked_last_hiddens[: self.num_heads]


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
            seq_len:int, head_dim=128, long=False, layer_id=0):
        super().__init__() # dim=hidden_size=embedding=feature=representation

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
        self.lime_router = StaticRouter(num_kv_heads, layer_id) if layer_id > 0 else Layer0Router(num_heads)

        if long:
            self.rope   = False
            self.window = 1024*4 # long
        else:
            self.rope   = True
            self.window = 1024  # short
        print(f"Layer {layer_id} => {'RoPE' if self.rope else 'Nope'}, win {self.window}")
        self.attn_scale = 0.12


    def forward(self, x, v_emb, ve_lambdas, cu_seqlens, max_seqlen, rotary):
        q    = self.q_proj(x)
        k, v = self.kv_proj(x).chunk(2, dim=-1) # T, C

        if ve_lambdas is not None and v_emb is not None:
            v = ve_lambdas[0]*v + ve_lambdas[1]*v_emb

        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim
        T, C = k.shape; assert C == Hkv * D

        ## Chuyển q, k, v thành x_THD
        q = q.contiguous().view(T, H,   D)
        k = k.contiguous().view(T, Hkv, D)
        v = v.contiguous().view(T, Hkv, D)

        ''' https://github.com/corl-team/lime/blob/main/src/lm/lime.py#L127
        # routing KV-cache to each head from all previous layers and heads
        kv_states = self.lime_router(kv_buffer)
        kv_states = kv_states.view(num_kv_heads, B, T, 2 * Hd).permute(1, 0, 2, 3)
        k_states, v_states = ( kv_states[..., :Hd], kv_states[..., Hd:],)  # B H T (2 Hd) -> B H T Hd, B H T Hd
        '''

        q, k, v = norm(q), norm(k), norm(v) # theo chiều D
        if self.rope: q, k = rotary(q), rotary(k)

        # Layer lẻ hoặc nope áp dụng varlen
        if self.layer_id % 3 == 2 or not self.rope: # long attn
            y = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                causal=True, dropout_p=0.0,
                softmax_scale=self.attn_scale,
                window_size=(self.window, 0),
            ).to(x.dtype)
        else:
            assert self.rope
            y = flash_attn_func(
                q=q.view(-1, self.window, H, D),
                k=k.view(-1, self.window, Hkv, D),
                v=v.view(-1, self.window, Hkv, D),
                causal=True, dropout_p=0.0,
                softmax_scale=self.attn_scale,
            )

        y = y.contiguous()
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
            head_dim=head_dim, long=layer_id % 6 == 5, layer_id=layer_id) # 2, 5, 8, 11 ...

    def forward(self, x, x0, te, ve, te_lambdas, ve_lambdas, cu_seqlens, max_seqlen, rotary):
        x                     = te_lambdas[0] *  x # te_lambdas[0] init là 1
        if x0 is not None: x += te_lambdas[1] * x0 # trộn với tok emb gốc

        ## per layer token emb trộn sau block
        # if te is not None: x += te_lambdas[2] * te  # trộn trước block
        x = x + self.attn(x, ve, ve_lambdas, cu_seqlens, max_seqlen, rotary)
        x = x + self.mlp(norm(x), te)
        return x


class Future(nn.Module):
    """ Dự đoán xa hơn 1 token, ideas from Multi-Token Prediction, DeepSeek và MiMo papers """
    def __init__(self, dim, num_heads, num_kv_heads, max_seq_len, head_dim=128):
        super().__init__()
        self.mlp = ReLuSquareMLP(dim, 2*dim) # nhẹ 2/3 MLP bình thường
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, max_seq_len, head_dim=head_dim, long=True)

        self.proj = nn.Linear(2*dim, dim, bias=False)
        with torch.no_grad(): self.proj.weight.copy_(init_linear(torch.empty(dim, 2*dim)))

    def forward(self, x, x0, te, ve, tl, vl, cu_seqlens, max_seqlen, rotary):
        # trộn feat của last layer với token embed gốc (x0)
        x = torch.cat((x, x0), dim=-1)
        x = self.proj(x) # mlp mixer
        x = norm(x)
        # if te is not None: x += tl[2] * te  #~~trộn per layer te trước block~~
        x = x + self.attn(x, ve, vl, cu_seqlens, max_seqlen, rotary)
        x = x + self.mlp(norm(x), te)         #  trộn per layer te sau block
        return norm(x)


class WinGPT(nn.Module):
    def has_future(self):
        return self.future_ratio > 0.009

    def __init__(self, vocab_size:int, n_layers:int, num_heads:int, num_kv_heads:int, dim:int,
        max_seq_len:int, head_dim=128, ve=3, te=1, future_percent=0, active_vocab=None):

        self.ohmai = ( active_vocab is not None )
        Embedding = OhMaiEmbedding if self.ohmai else nn.Embedding
        print(f"OH_MAI? {self.ohmai}; using {Embedding.__name__}")

        super().__init__()
        self.rotary = Rotary(head_dim, max_seq_len)

        self.n_layers = n_layers
        blocks = [ Block(dim, num_heads, num_kv_heads, max_seq_len, head_dim, layer_id=i) for i in range(n_layers) ]

        self.future_ratio = future_percent / 100.0
        if self.has_future():
            blocks.append(Future(dim, num_heads//2, num_kv_heads//2, max_seq_len, head_dim))

        self.blocks = nn.ModuleList(blocks)
        n_blks = len(self.blocks)

        if ve > n_blks: ve = n_blks
        if te > n_blks: te = n_blks

        self.ve, self.te = ve, te
        self.dim, self.kv_dim = dim, num_kv_heads*head_dim
        
        # fused embeddings
        self.embeddings = Embedding(vocab_size, dim*te + self.kv_dim*self.ve, active_vocab)

        self.scalars = nn.Parameter(torch.cat([
          torch.ones(n_blks),  # skip_weights khởi tạo là 1 cho tất cả layers
          *[torch.tensor([1.0, 0.0, 0.0]) for _ in range(n_blks)], # token emb mix
          *[torch.tensor([0.5, 0.5     ]) for _ in range(n_blks)], # value emb mix
        ]))

        self.skip_from = { (n_layers-i): i for i in range(2, (n_layers-1) // 2, 2) }
        print("WinGPT.skip_from", self.skip_from)

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        with torch.no_grad(): self.lm_head.weight.zero_()


    def update_embeddings(self):
        if isinstance(self.embeddings, OhMaiEmbedding):
            self.embeddings.update_embeddings()


    def forward(self, input_seq:Tensor, cu_seqlens, max_seqlen):
        n_blks = len(self.blocks)
        embs = self.embeddings(input_seq.long())
        # print(self.embeddings.__class__.__name__, embs.dtype); input()

        t_embs = embs[..., : self.dim*self.te ]
        t_embs = list(t_embs.chunk(self.te, dim=-1))

        x = x0 = norm(t_embs[0])
        # assert t_embs[-1].size(-1) == self.dim

        v_embs = embs[..., -self.kv_dim*self.ve : ]
        v_embs = list(v_embs.chunk(self.ve, dim=-1))

        ## ve[0],1,2 ... ve[0],1,2 u-shape
        if len(v_embs) < self.n_layers - 3:
            skips = [None]*(self.n_layers - 3 - len(v_embs))
            v_embs += skips + v_embs[:3]
            assert len(v_embs) == self.n_layers

        ## Độn None cho đầy v_embs, t_embs
        v_embs += [None]*(n_blks - len(v_embs))
        t_embs += [None]*(n_blks - len(t_embs))
        assert len(v_embs) == len(t_embs) == n_blks

        skip_weights = self.scalars[ :n_blks]
        te_lambdas   = self.scalars[1*n_blks : 4*n_blks].view(-1, 3)
        ve_lambdas   = self.scalars[4*n_blks : 6*n_blks].view(-1, 2)
        
        layer_outputs = []
        for i in range(self.n_layers):
            if i in self.skip_from:
                k = self.skip_from[i]
                x += skip_weights[k] * layer_outputs[k]
            
            def fwd(blk, x0, te, ve, tl, vl, c, m): return lambda x: blk(x, x0, te, ve, tl, vl, c, m, self.rotary)
            f = fwd(self.blocks[i], t_embs[0], t_embs[i], v_embs[i], te_lambdas[i], ve_lambdas[i], cu_seqlens, max_seqlen)

            x = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)
            layer_outputs.append(x)
        return norm(x), t_embs, v_embs, te_lambdas, ve_lambdas, cu_seqlens, max_seqlen

###################
## Loss function ##
###################

def _loss_fn(_loss_method, model, input_seq, target, future, cu_seqlens, max_seqlen):
    x, te, ve, tl, vl, c, m = model(input_seq, cu_seqlens, max_seqlen) # x đã norm
    loss, _ = _loss_method(x, target.flatten(), model.lm_head)

    if not model.has_future(): return loss
    if torch.rand(1).item() > 0.5: return loss

    assert model.n_layers + 1 == len(model.blocks)
    future_loss, _ = _loss_method(
        model.blocks[-1](x, te[0], te[-1], ve[-1], tl[-1], vl[-1], c, m, model.rotary), # đã norm
        future.flatten(), model.lm_head, # tied với main task head 
    )
    # import gc; del layer_outputs, t, v, l, s; gc.collect(); torch.cuda.empty_cache() # no use
    return loss * (1 - model.future_ratio) + future_loss * model.future_ratio


import math
def simple_loss_fn(model, input_seq, target, future, cu_seqlens, max_seqlen):
    def _loss_method(hidden, target, head, chunk_size=2048):
        total_tokens = hidden.size(0)  # hidden đã được flatten
        num_chunks = math.ceil(total_tokens / chunk_size)
        total_loss = None
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            if end_idx > total_tokens: end_idx = total_tokens
            
            # Lấy chunk của hidden và target
            hidden_chunk = hidden[start_idx:end_idx]
            target_chunk = target[start_idx:end_idx]
            
            logits_chunk = torch.utils.checkpoint.checkpoint(head, hidden_chunk, use_reentrant=False,)                
            logits_chunk = logits_chunk.view(-1, logits_chunk.size(-1))
            logits_chunk = 15 * logits_chunk * torch.rsqrt(logits_chunk.square() + 15*15)
            
            chunk_loss = F.cross_entropy(logits_chunk.float(), target_chunk.long(),)
            if total_loss is None: total_loss = chunk_loss
            else: total_loss += chunk_loss
        return total_loss / num_chunks, None
    return _loss_fn(_loss_method, model, input_seq, target, future, cu_seqlens, max_seqlen)


def fused_loss_fn(model, input_seq, target, future, cu_seqlens, max_seqlen):
    def _loss_method(hidden, target, head):
        hidden = hidden.view(-1, hidden.size(-1))#.bfloat16()
        return LigerFusedLinearCrossEntropyFunction.apply(hidden, head.weight, target)
    return _loss_fn(_loss_method, model, input_seq, target, future, cu_seqlens, max_seqlen)


def get_cu_max_seqlens_from(input_seq, eot=6399):
        mask = (input_seq == eot)
        mask[-1] = True
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=input_seq.device), 
            torch.where(mask)[0].to(torch.int32) + 1,
        ])
        max_seqlen = int(torch.max(torch.diff(cu_seqlens)))
        return cu_seqlens, max_seqlen


########################
##  TESTING  TESTING  ##
########################

if __name__ == "__main__":
    import numpy as np
    from optimus import Muon1GPU as Muon
    from optimus import convert_int8_mixed_precision

    sseed = 1982
    seq_len = 1024
    vocab_size = 512
    dim, n_layers = 256, 4
    num_heads, num_kv_heads = 8, 4
    print(f"Model config: layers={n_layers}, dim={dim}, heads={num_heads}/{num_kv_heads}; seq_len={seq_len}")

    # Khởi tạo model 1 dùng LigerEmbedding
    torch.manual_seed(sseed)
    model = WinGPT(vocab_size, n_layers, num_heads, num_kv_heads, dim, seq_len, 
        ve=3, te=2, future_percent=20).cuda()
    
    # Khởi tạo model 2 dùng OhMaiEmbedding
    torch.manual_seed(sseed)
    ohmai = WinGPT(vocab_size, n_layers, num_heads, num_kv_heads, dim, seq_len, 
        ve=3, te=2, future_percent=20, active_vocab=vocab_size//2).cuda()

    # Đảm bảo toàn bộ tham số của 2 model là như nhau
    def check_params():
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), ohmai.named_parameters()):

            if n2 == "embeddings.active_weight":
                n2 = "embeddings.weight"
                p2 = ohmai.embeddings.weight.cuda()

            assert n1 == n2, f"{n1} != {n2}"
            assert torch.allclose(p1, p2), f"{n1} values are different"
    check_params()

    for m in [model, ohmai]:
        for n, p in m.named_parameters(): assert p.dtype == torch.bfloat16, f"{n} is not bf16"
        print(f"All {'ohmai' if m.ohmai else 'model'} params are in bfloat16.")

    convert_int8_mixed_precision(model)
    # model = torch.compile(model) # chậm !!!

    apara = {n: p for n, p in model.named_parameters() if "fc" not in n and "proj" not in n}
    opara = [p for n, p in model.named_parameters() if "fc" in n or "proj" in n]

    print("\nAdam:", apara.keys())
    apara = list(apara.values())

    # Bổ xung thêm ohmai params vào adam và muon
    for n, p in ohmai.named_parameters():
        if "fc" not in n and "proj" not in n:
                print("Adam:", n)
                apara.append(p)
        else:   opara.append(p)

    aptim = torch.optim.Adam(apara)
    optim = Muon(opara)

    after_init_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    print(f"Peak VRAM after model initialization: {after_init_memory:.2f} MB")

    tok_emb_before = ohmai.embeddings.weight.data.clone()
    model.train()
    ohmai.train()

    ## Generate sequences with batch dimension
    input_seq = torch.randint(5, vocab_size//2, (seq_len,), dtype=torch.int16).cuda()
    target    = torch.randint(5, vocab_size//2, (seq_len,), dtype=torch.int16).cuda()
    future    = torch.randint(5, vocab_size//2, (seq_len,), dtype=torch.int16).cuda()
    cu_seqlens, max_seqlen = get_cu_max_seqlens_from(input_seq)

    for step in range(10):
        optim.zero_grad()
        aptim.zero_grad()
        loss_fn = [ simple_loss_fn, fused_loss_fn ][step % 2]
        loss_ohmai = loss_fn(ohmai, input_seq, target, future, cu_seqlens, max_seqlen)
        loss_model = loss_fn(model, input_seq, target, future, cu_seqlens, max_seqlen)
 
        ## Đảm bảo 2 cách lấy embedding là giống nhau
        a = ohmai.embeddings(input_seq, force=True)
        b = ohmai.embeddings.weight.to(input_seq.device)[input_seq.long()]
        if not torch.allclose(a.cpu(), b.cpu(), atol=1e-5): assert False

        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"step {step}, loss_model {loss_model.item():.4f}, loss_ohmai {loss_ohmai.item():.4f}, Peak VRAM: {current_memory:.2f} MB, {loss_fn.__name__}")
        # assert torch.allclose(loss_model, loss_ohmai, atol=1e-5), f"Loss mismatch: model={loss_model.item():.6f}, ohmai={loss_ohmai.item():.6f}"

        loss_ohmai.backward()
        loss_model.backward()
        optim.step()
        aptim.step()
        ohmai.update_embeddings()
        # check_params()

    ohmai.embeddings.update_stream.synchronize() # đảm bảo weigh đã được cập nhật
    tok_emb_after = ohmai.embeddings.weight.data
    diff = (tok_emb_before != tok_emb_after).sum().item()
    assert diff > 0, f"Số lượng thay đổi {diff}\n{tok_emb_before}\n{tok_emb_after}"
