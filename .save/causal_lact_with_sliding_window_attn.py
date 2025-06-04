import math, torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

try: from flash_attn import flash_attn_func
except: flash_attn_func = None

def silu_backprop(dy, x):
    s = torch.sigmoid(x)
    return dy * s * (1 + x * (1 - s))

def l2_norm(x): return x / (x.norm(dim=-1, keepdim=True) + 1e-5)

def zeropower_via_newtonschulz5(G):
    X = G.bfloat16()
    if G.size(1) > G.size(2): X = X.transpose(1, 2)   # Ensure số cột ≥ số hàng; giúp NS hoạt động tốt
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7) # Ensure spectral norm ≤ 1, điều kiện bắt buộc để NS hội tụ
    for a,b,c in [(4.0848,-6.8946,2.9270),(3.9505,-6.3029,2.6377),(3.7418,-5.5913,2.3037),(2.8769,-3.1427,1.2046),(2.8366,-3.0525,1.2012)]:
        A = X @ X.transpose(1, 2); X = a*X + (b*A + c*A@A) @ X
    return X.transpose(1, 2) if G.size(1) > G.size(2) else X


def block_causal_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, momentum=None, use_muon=True, chunk_size=2048):
    """ Block causal LaCT with SwiGLU fast weight function. Apply then Update => Shifted Block Causal LaCT
    Fast weights cho phép model thích nghi với context hiện tại trong lúc inference
    - w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))
    - w0, w1, w2 are mostly likely fp32. q, k, v are fp16. lr0, lr1, lr2 are fp32.
    - The forward, backward produce bf16 gradients, updated fast weights are fp32.
    - The final output are bf16.

    TTT Test Time Training:
    - Mỗi chunk: Infer với q → Train với k,v → Update weights
    - Không có loss function: Dùng v làm target, flow gradient ngược qua w1
    - Self-supervised: Model tự học cách biểu diễn k sao cho match với v
    - Incremental: Weights evolve dần theo context

    torch.bmm = Batch Matrix Multiplication. Công thức:
    - Input: A[b, n, m], B[b, m, p]
    - Output: C[b, n, p] = A @ B cho mỗi batch
    """
    # Lưu norm ban đầu để stabilize weights sau này
    w_norms = [w.norm(dim=2, keepdim=True) for w in [w0, w1, w2]]
    
    # Khởi tạo momentum buffers nếu dùng momentum
    if momentum is not None: dw_mom = [torch.zeros_like(w0) for _ in range(3)]
    
    # Chuyển q, v sang layout phù hợp cho bmm: [batch, dim, length]
    q, v, output = q.transpose(1, 2), v.transpose(1, 2), torch.zeros_like(v.transpose(1, 2))
    
    # Process từng chunk sequence
    for i in range(0, k.shape[1] - chunk_size, chunk_size):
        s, e = i, i + chunk_size

        # Lấy chunk hiện tại của k, v (để train) và q (để infer)
        ki, vi, qi = k[:, s:e], v[:, :, s:e], q[:, :, s:e]

        # Learning rates cho chunk này
        lrs = [lr[:, s:e] for lr in [lr0, lr1, lr2]]
        
        # ========== INFERENCE PHASE: Dùng weights hiện tại ==========
        # SwiGLU forward: output = w1 @ (silu(w0 @ q) * (w2 @ q))
        output[:, :, s:e] = torch.bmm(w1, F.silu(torch.bmm(w0, qi)) * torch.bmm(w2, qi))
        
        # ========== TRAINING PHASE: Học từ k, v ==========
        # Forward pass với k (thay vì q) để tính gradients
        g_act = torch.bmm(w0, ki.transpose(1, 2))  # Gate activations trước silu
        h_mul = torch.bmm(w2, ki.transpose(1, 2))  # Value branch
        h = F.silu(g_act) * h_mul                  # Hidden state = gate * value
        
        # "Backward" pass: v đóng vai trò learning signal
        dh = torch.bmm(w1.transpose(1, 2), vi)     # Gradient từ v qua w1^T
        
        # Tính gradients cho từng branch
        dg_act = silu_backprop(dh * h_mul, g_act)  # Gradient cho gate branch
        
        # Gradient cho weights (outer product với learning rate)
        dws = [
            torch.bmm(dg_act, (ki * lrs[0]).type_as(dg_act)),        # dw0: gate weights
            torch.bmm(vi, (h.transpose(1, 2) * lrs[1]).type_as(vi)), # dw1: output weights  
            torch.bmm(dh * F.silu(g_act), (ki * lrs[2]).type_as(dh)) # dw2: value weights
        ]
        
        # Apply momentum nếu có
        if momentum is not None:
            m = momentum[:, s:e].mean(dim=1, keepdim=True)  # Momentum scalar
            dws = [dw + dw_m * m for dw, dw_m in zip(dws, dw_mom)]  # Add momentum
            dw_mom = dws  # Update momentum buffer
        
        # Muon optimizer: normalize gradients (giống Adam nhưng không track statistics)
        if use_muon: dws = [zeropower_via_newtonschulz5(dw) for dw in dws]
        
        # Update weights và normalize để giữ scale
        w0, w1, w2 = [(w + dw) / ((w + dw).norm(dim=2, keepdim=True) + 1e-5) * w_n 
                      for w, dw, w_n in zip([w0, w1, w2], dws, w_norms)]
    
    # Process chunk cuối với updated weights (inference only)
    qi = q[:, :, e:]
    output[:, :, e:] = torch.bmm(w1, F.silu(torch.bmm(w0, qi)) * torch.bmm(w2, qi))
    
    return output.transpose(1, 2)


class CausalLaCTSwiGLUWithSlidingWindowAttn(nn.Module):
    def __init__(self, dim, head_dim, attn_head_dim, **kwargs):
        super().__init__()
        self.d, self.hd, self.ahd = dim, head_dim, attn_head_dim
        self.nh, self.nah = dim // head_dim, dim // attn_head_dim
        self.cs,   self.ws  = kwargs.get('lact_chunk_size', 2048), kwargs.get('window_size', 2048)
        self.muon, self.mom = kwargs.get('use_muon', True),        kwargs.get('use_momentum', True)
        
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.lr_proj = nn.Linear(dim, 3 * self.nh, bias=False)
        self.base_lr_inv = kwargs.get('base_lr', 1e-2) + math.log(-math.expm1(-kwargs.get('base_lr', 1e-2)))
        
        dh = int(head_dim * kwargs.get('inter_multi', 1))
        for i, shape in enumerate([(dh, head_dim), (head_dim, dh), (dh, head_dim)]):
            setattr(self, f'w{i}', nn.Parameter(torch.randn(self.nh, *shape) / math.sqrt(shape[1])))
        
        if self.mom: self.momentum_proj = nn.Sequential(nn.Linear(dim, self.nh), nn.Sigmoid())
        self.o_norm = nn.RMSNorm(head_dim) if kwargs.get('use_o_norm', True) else nn.Identity()
        self.qk_scale, self.qk_offset = nn.Parameter(torch.ones(dim, 2)), nn.Parameter(torch.zeros(dim, 2))
        if kwargs.get('ttt_scale_before_sum', True): self.ttt_scale_proj = nn.Linear(dim, self.nh)


    def forward(self, x):
        qkv = self.to_qkv(x)
        tq, tk, tv = rearrange(F.silu(qkv), "b l (qkv h d) -> qkv (b h) l d", qkv=3, h=self.nh, d=self.hd)
        tq, tk = l2_norm(tq), l2_norm(tk)
        
        lr = F.softplus(self.lr_proj(x).float() + self.base_lr_inv)
        lr0, lr1, lr2 = rearrange(lr, "b l (h lrs) -> lrs (b h) l 1", lrs=3, h=self.nh)
        
        ws = [getattr(self, f'w{i}').repeat(x.shape[0], 1, 1) for i in range(3)]
        mom = rearrange(self.momentum_proj(x), 'b s (h d) -> (b h) s d', h=self.nh) if self.mom else None
        
        to = block_causal_lact_swiglu(*ws, tq, tk, tv, lr0, lr1, lr2, mom, self.muon, self.cs)
        to = self.o_norm(to) * rearrange(F.silu(self.ttt_scale_proj(x)), 'b s (h d) -> (b h) s d', h=self.nh)
        to = rearrange(to, "(b h) l d -> b l (h d)", h=self.nh, b=x.shape[0])
        
        aq, ak, av = qkv.chunk(3, dim=-1)
        s, o = self.qk_scale.view(1, 1, -1, 2), self.qk_offset.view(1, 1, -1, 2)
        aq, ak = (aq * s[..., 0] + o[..., 0]).to(aq.dtype), (ak * s[..., 1] + o[..., 1]).to(ak.dtype)
        
        aq, ak, av = [rearrange(t, '... (h d) -> ... h d', d=self.ahd) for t in [aq, ak, av]]
        ao = rearrange(flash_attn_func(aq, ak, av, causal=True, window_size=(self.ws-1, 0) if self.ws else (-1, -1)), '... h d -> ... (h d)')
        
        return self.o_proj(ao + to)


if __name__ == "__main__":
    B, L, D = 1, 4096, 2048
    layer = CausalLaCTSwiGLUWithSlidingWindowAttn(D, 512, 128, inter_multi=1, use_o_norm=True,).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16): out = layer(torch.randn(B, L, D).cuda())
    print(f"Shape: {out.shape}, Norm: {out.norm():.2f}")
