#!/usr/bin/env python3
## GPT for the WIN (cải biên từ modded nanogpt)
import os, math, torch
from torch import Tensor, nn
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.coordinate_descent_tuning = True
torch.set_float32_matmul_precision('high') # better for f32 head
torch.backends.cuda.matmul.allow_tf32  = True
torch.set_default_dtype(torch.bfloat16)

#######################################################################
USE_FLASH_ATTN = ( os.getenv('flash_attn', '1') == '1' )
if USE_FLASH_ATTN:
    try:# to use flash_attn directly
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        print("!!! Use Flash Attn 2 Directly !!!")
    except: USE_FLASH_ATTN = False
#######################################################################
if not USE_FLASH_ATTN:
    # SDPA IMPL https://medium.com/data-science/fefa6f87b1d6 THROUGHPUT
    torch.backends.cuda.enable_flash_sdp(            True)   # 50kt/sec
    torch.backends.cuda.enable_mem_efficient_sdp(   False)   # 35kt/sec
    torch.backends.cuda.enable_math_sdp(            False)   # 14kt/sec
    torch.backends.cuda.enable_cudnn_sdp(            True)   # 49kt/sec

    print(f""">> scaled_dot_product_attention engine
    flash? {torch.backends.cuda.flash_sdp_enabled()}
    mem__? {torch.backends.cuda.mem_efficient_sdp_enabled()}
    math_? {torch.backends.cuda.math_sdp_enabled()}
    cudnn? {torch.backends.cuda.cudnn_sdp_enabled()}""")
#######################################################################

def norm(x: Tensor):
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

        self.fc = nn.Linear(dim, hdim, bias=False)
        self.proj = nn.Linear(hdim, odim, bias=False)
        
        with torch.no_grad():
            self.fc.weight.copy_(init_linear(torch.empty(hdim, dim)))
            self.proj.weight.zero_()
        
        # Add weight decay multiplier attribute to the weights
        self.fc.weight.wd_mul = 2.0  # điều chỉnh hệ số weight decay
        self.proj.weight.wd_mul = 2.0  # gấp đôi so với mặc định 

    def forward(self, x:Tensor):
        y = self.fc(x)
        y = F.relu(y).square() 
        x = self.proj(y)
        return x


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


    def forward(self, x_BTHD: Tensor):
        seq_len = x_BTHD.size(-3) # batch, T seq_len, head, dim (of head)
        assert self.cos.size(0) >= seq_len, f"{self.cos.size(0)} >= {seq_len}?"

        cos = self.cos[None, :seq_len, None, :] # [1, seq_len, 1, dim]
        sin = self.sin[None, :seq_len, None, :] # [1, seq_len, 1, dim]

        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)

        y1 = x1 * (+cos) + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim:int, num_heads:int, num_kv_heads:int, 
            seq_len:int, head_dim=128, nope=False, layer_id=None):
        super().__init__() # dim=hidden_size=embedding=feature=representation

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim

        qo_inner_dim = num_heads * head_dim
        kv_inner_dim = num_kv_heads * head_dim

        self.kv_proj = nn.Linear(dim, 2*kv_inner_dim, bias=False)
        self.q_proj = nn.Linear(dim, qo_inner_dim, bias=False)
        self.o_proj = nn.Linear(qo_inner_dim, dim, bias=False)

        # Set the weights directly
        with torch.no_grad():
            self.kv_proj.weight.copy_(init_linear(torch.empty(2*kv_inner_dim, dim)))
            self.q_proj.weight.copy_(init_linear(torch.empty(qo_inner_dim, dim)))
            self.o_proj.weight.zero_() # zero init

        if nope: 
            self.rotary = None
            self.window = -1 # full attn
            print(f"Layer {layer_id} => NoPE, win {self.window}")
        else:
            self.rotary = Rotary(head_dim, seq_len)
            self.window = seq_len // 4
            print(f"Layer {layer_id} => RoPE, win {self.window}")
        self.attn_scale = 0.12

    """ Implement casual_conv1d đơn giản
        self.conv_kernel = 3
        self.kv_conv = torch.ones(
            kv_inner_dim,       # số kênh (C / D / hidden dim)
            1,                  # số kênh đầu vào cho mỗi nhóm
            self.conv_kernel,   # kích thước cửa sổ trượt conv
        ).cuda() / self.conv_kernel # khởi tạo 1 / k => avg

    def causal_moving_avg(self, x):
        B, T, C = x.shape
        x = x.view(B, C, T)
        pad_left = self.conv_kernel - 1
        y = F.pad(x, (pad_left, 0))
        y = F.conv1d(y, self.kv_conv, groups=C)
        return (x + y).view(B, T, C)
    # """

    def forward(self, x, v_emb, ve_lambdas, cu_seqlens, max_seqlen):
        q    = self.q_proj(x)
        k, v = self.kv_proj(x).chunk(2, dim=-1) # B, T, C
        # k, v = self.causal_moving_avg(k), self.causal_moving_avg(v)

        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim
        B, T, C   = k.shape; assert C == Hkv * D

        ## Chuyển q, k, v thành x_BTHD
        q = q.view(B, T, H,   D)
        k = k.view(B, T, Hkv, D)
        v = v.view(B, T, Hkv, D)

        q, k, v = norm(q), norm(k), norm(v)
        if self.rotary: q, k = self.rotary(q), self.rotary(k)

        if ve_lambdas is not None and v_emb is not None:
            # Trộn value với value embedding
            v = ve_lambdas[0]*v + ve_lambdas[1]*v_emb.view(B, T, Hkv, D)

        if not USE_FLASH_ATTN:
            # Make tensors contiguous and transpose for attention
            q = q.transpose(1, 2).contiguous()  # BTHD -> BHTD
            k = k.transpose(1, 2).contiguous()  # BTHD -> BHTD
            v = v.transpose(1, 2).contiguous()  # BTHD -> BHTD
            
            # Repeat KV heads to match query head count (GQA)
            k = torch.repeat_interleave(k, repeats=self.num_kv_groups, dim=1)
            v = torch.repeat_interleave(v, repeats=self.num_kv_groups, dim=1)
            
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0, scale=self.attn_scale,)
            y = y.transpose(1, 2) # back to [B, T, H, D]
        else:
            # Make tensors contiguous and transpose for attention
            # q, k, v = q.contiguous(), k.contiguous(), v.contiguous() # BTHD
            y = flash_attn_varlen_func(
                q.reshape(B*T, H, D),
                k.reshape(B*T, Hkv, D), 
                v.reshape(B*T, Hkv, D),
                cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                causal=True, dropout_p=0.0,
                softmax_scale=self.attn_scale,
                window_size=(self.window, 0),
            )
        y = y.contiguous().reshape(B, T, H * D)
        y = self.o_proj(y) # y có shape (B, T, dim)
        return y    # trả về y có shape giống hệt x đầu vào


##############################
## Transformer for the WIN  ##
##############################

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, max_seq_len, head_dim=128, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.mlp = ReLuSquareMLP(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, max_seq_len, 
            head_dim=head_dim, nope=layer_id % 4 == 3, layer_id=layer_id) # 2, 5, 8, 11 ...

    def forward(self, x, x0, te, ve, te_lambdas, ve_lambdas, cu_seqlens, max_seqlen):
        x                     = te_lambdas[0] *  x # te_lambdas[0] init là 1
        if x0 is not None: x += te_lambdas[1] * x0 # trộn với tok emb gốc
        if te is not None: x += te_lambdas[2] * te # trộn với layer tok emb

        x = x + self.attn(x, ve, ve_lambdas, cu_seqlens, max_seqlen)
        x = x + self.mlp(norm(x))
        return x


class Future(nn.Module):
    """ Dự đoán xa hơn 1 token, ideas from Multi-Token Prediction, DeepSeek và MiMo papers """
    def __init__(self, dim, num_heads, num_kv_heads, max_seq_len, head_dim=128):
        super().__init__()
        self.mlp = ReLuSquareMLP(dim, 2*dim) # nhẹ 2/3 MLP bình thường
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, max_seq_len, head_dim=head_dim, nope=True)

        self.proj = nn.Linear(2*dim, dim, bias=False)
        with torch.no_grad(): self.proj.weight.copy_(init_linear(torch.empty(dim, 2*dim)))

    def forward(self, x, x0, te, ve, tl, vl, cu_seqlens, max_seqlen):
        # trộn feat của last layer với token embed gốc (x0)
        x = torch.cat((x, x0), dim=2)
        x = self.proj(x) # mlp mixer
        x = norm(x)
        if te is not None: x += tl[2] * te # trộn với layer tok emb
        x = x + self.attn(x, ve, vl, cu_seqlens, max_seqlen)
        x = x + self.mlp(norm(x))
        return norm(x)


class WinGPT(nn.Module):
    def has_future(self):
        return self.future_ratio > 0.009

    def __init__(self, vocab_size:int, n_layers:int, num_heads:int, num_kv_heads:int,
            dim:int, max_seq_len:int, head_dim=128, ve=3, te=1, future_percent=0):
        super().__init__()

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

        self.tok_emb0 = nn.Embedding(vocab_size, dim) # tok emb gốc

        lte = te - 1 # layer token embeddings
        if lte > 1:
            dd = dim // 4 # giảm 1/2 dim nếu không phải tok emb gốc
            self.tok_embs = nn.Embedding(vocab_size, dd*lte)
            self.tok_proj = nn.Linear(dd*lte, dim*lte, bias=False)
            with torch.no_grad():
                self.tok_proj.weight.copy_(init_linear(torch.empty(dim*lte, dd*lte)))

        kv_dim = num_kv_heads * head_dim # use _proj như tok nếu val_embs quá to
        self.val_embs = nn.Embedding(vocab_size, kv_dim*ve) 

        self.scalars = nn.Parameter(torch.cat([
          torch.ones(n_blks),  # skip_weights khởi tạo là 1 cho tất cả layers
          *[torch.tensor([1.0, 0.0, 0.0]) for _ in range(n_blks)], # token emb mix
          *[torch.tensor([0.5, 0.5     ]) for _ in range(n_blks)], # value emb mix
        ]))

        self.skip_from = { (n_layers-i): i for i in range(2, (n_layers-1) // 2, 2) }
        print("WinGPT.skip_from", self.skip_from)

        self.lm_head = nn.Linear(dim, vocab_size, bias=False).float()
        with torch.no_grad(): self.lm_head.weight.zero_()


    def forward(self, input_seq:Tensor):
        mask = (input_seq == 6399)
        mask[:, -1] = True

        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=input_seq.device), 
            torch.where(mask.flatten())[0].to(torch.int32) + 1,
        ])
        max_seqlen = int(torch.max(torch.diff(cu_seqlens)))
        # print(cu_seqlens, max_seqlen) # DEBUG

        n_blks = len(self.blocks)
        x = x0 = norm(self.tok_emb0(input_seq))

        if self.te > 1:
                t_embs = self.tok_embs(input_seq)
                t_embs = self.tok_proj(t_embs)
                t_embs = [x0] + list(t_embs.chunk(self.te-1, dim=-1))
        else:   t_embs = [x0]

        v_embs = self.val_embs(input_seq)
        v_embs = list(v_embs.chunk(self.ve, dim=-1))

        if len(v_embs) < self.n_layers - 3: # ve[0],1,2 ... ve[0],1,2 u-shape
            skips = [None]*(self.n_layers - 3 - len(v_embs))
            v_embs += skips + v_embs[:3]
            assert len(v_embs) == self.n_layers

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
            
            def fwd(blk, x0, te, ve, tl, vl, c, m): return lambda x: blk(x, x0, te, ve, tl, vl, c, m)
            f = fwd(self.blocks[i], t_embs[0], t_embs[i], v_embs[i], te_lambdas[i], ve_lambdas[i], cu_seqlens, max_seqlen)
            x = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)
            layer_outputs.append(x)

        return norm(x), t_embs, v_embs, te_lambdas, ve_lambdas, cu_seqlens, max_seqlen


###################
## Loss function ##
###################

maximum_seqlen = 0
def _loss_fn(_loss_method, model, input_seq, target, future):
    x, te, ve, tl, vl, c, m = model(input_seq) # x đã norm
    loss, _ = _loss_method(x, target.flatten(), model.lm_head)

    global maximum_seqlen # log lại seqlen lớn nhất
    if maximum_seqlen < m: maximum_seqlen = m; print(f">>> maximum_seqlen {m}")

    if not model.has_future(): return loss
    if torch.rand(1).item() > 0.5: return loss

    assert model.n_layers + 1 == len(model.blocks)
    future_loss, _ = _loss_method(
        model.blocks[-1](x, te[0], te[-1], ve[-1], tl[-1], vl[-1], c, m), # đã norm
        future.flatten(), model.lm_head, # tied với main task head 
    )
    # import gc; del layer_outputs, t, v, l, s; gc.collect(); torch.cuda.empty_cache() # no use
    return loss * (1 - model.future_ratio) + future_loss * model.future_ratio


def simple_loss_fn(model, input_seq, target, future):
    def _loss_method(hidden, target, head):
        logits = head(hidden.float())
        logits = logits.view(-1, logits.size(-1))
        logits = 15*logits*torch.rsqrt(logits.square() + 15*15)
        return F.cross_entropy(logits.float(), target.long()), None
    return _loss_fn(_loss_method, model, input_seq, target, future)


try: # pip install liger_kernel
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    def fused_loss_fn(model, input_seq, target, future):
        def _loss_method(hidden, target, head):
            hidden = hidden.view(-1, hidden.size(-1)).float()
            return LigerFusedLinearCrossEntropyFunction.apply(hidden, head.weight, target)
        return _loss_fn(_loss_method, model, input_seq, target, future)
except: None

## TEST MODEL
if __name__ == "__main__":
    import os
    import numpy as np
    from optimus import Muon1GPU as Muon
    from optimus import convert_int8_mixed_precision

    loss_fn = simple_loss_fn
    # loss_fn = fused_loss_fn
    vocab_size = 1981

    # Clear cache and reset peak memory stats
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    
    # Use a medium-sized model to show memory benefits
    dim, n_layers = 128, 8
    num_heads, num_kv_heads = 8, 4
    print(f"Model config: layers={n_layers}, dim={dim}, heads={num_heads}/{num_kv_heads}")
    
    batch_size, seq_len = 2, 256
    print(f"batch_size {batch_size}, seq_len {seq_len}")
    model = WinGPT(vocab_size, n_layers, num_heads, num_kv_heads, dim, seq_len, 
                        ve=3, te=3, future_percent=20).cuda()

    if os.environ.get('int8', '0') == '1':
        print(convert_int8_mixed_precision(model), "linear converted to int8") # lỗi trên 3050

    ## Generate sequences with batch dimension
    input_seq = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    target    = torch.randint(0, vocab_size, (batch_size, seq_len,)).cuda()
    future    = torch.randint(0, vocab_size, (batch_size, seq_len,)).cuda()

    aptim = torch.optim.Adam([p for n, p in model.named_parameters() if "fc" not in n and "proj" not in n])
    optim = Muon([p for n, p in model.named_parameters() if "fc" in n or "proj" in n])

    # Memory after model initialization
    after_init_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    print(f"Peak VRAM after model initialization: {after_init_memory:.2f} MB")

    for step in range(10):
        loss = loss_fn(model, input_seq, target, future)
        loss.backward()
        optim.step(); optim.zero_grad()
        aptim.step(); aptim.zero_grad()

        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        print(f"step {step}, loss {loss.item():.4f}, Peak VRAM: {current_memory:.2f} MB")
