token-shift trong rwkv phiên bản đầu đơn giản là lấy nửa sau của vector nhúng của token trước ghép vào nửa đầu của vector nhúng của token hiện tại.

```
token x:   [a_0, a_1, a_2, ..., a_{n-3}, a_{n-2}, a_{n-1}]
token x-1: [b_0, b_1, b_2, ..., b_{n-3}, b_{n-2}, b_{n-1}]
=>         [a_0, b_1, a_2, ..., b_{n-3}, b_{n-2}, b_{n-1}]
```

# Token-shift time-sift mixing

https://github.com/BlinkDL/RWKV-LM#token-shift-time-shift-mixing

Token-shift (rwkv đời đầu) sử dụng một nửa số channels của token đang xét và một nửa channel của token trước để tạo ra tất cả các vectors (QKV, RWKV, ...)
```py
self.time_shift = nn.ZeroPad2d((0,0,1,-1))

x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
```
Chia đôi số kênh và dịch chuyển 1 (shift-1 có nghĩa là chỉ dùng 1 token phía trước) hoạt động rất tốt với char-level English and char-level Chinese LM. Tuy nhiên với BPE-level English LM, nó chỉ có hiệu quả khi embedding đủ lớn (ít nhất là 1024).

Giả thiết của Peng Bo về hiệu quả của token-shift là: khi huấn luyện GPT, biểu diễn ẩn của một token là để thực hiện 2 nhiệm vụ khác nhau:
1. Dự đoán token tiếp theo. Đôi khi nó rất dễ (obvious next token)
2. Thu thập tất cả thông tin ngữ cảnh trước đó, để các tokens phía sau có thể sử dụng. Điều này luôn khó!

Các kênh được dịch chuyển (shifted channels) có thể tập trung vào (2), và vì thế chúng ta có sự lan truyền thông tin tốt. __Nó giống như residual connection hoặc một rnn nhỏ bên trong tfm__.

Bạn có thể sử dụng token-shift bên trong QKV self-attn thông thường. Peng Bo xem xét các trọng số, và thấy __V thực sự giống như shifted channels__. Make sense if you think about it (??). Peng Bo cũng nhận ra rằng nên dùng ít mixing hơn ở những tầng sâu hơn (I also found you may want to use less mixing in higher layers).

- - -

Sau đó, rwkv-4 nâng cấp cách trộn x và xx thành vector hệ số trộn có thể huấn luyện được cho k,v,r. Xem code: 
```py
attn_sz = n_embd = 8
inits = torch.zeros(1, 1, n_embd) # shape = (1, 1, n_embed) để khớp với 3 chiều dữ liệu đầu vào là (B, T, C)
for i in range(n_embd): inits[0, 0, i] = i / n_embd

layer_id, n_layer = 1, 24
ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0

# fancy time_decay
decay_speed = torch.ones(attn_sz)
for h in range(attn_sz): decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
time_decay = nn.Parameter(decay_speed)
# time_decay => -5.00, -3.16, -1.89, -0.78,  0.23,  1.20,  2.11,  3.00

# fancy time_first
zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
# zigzag     =>  0.00,  0.50, -0.50,  0.00,  0.50, -0.50,  0.00,  0.50
# time_first => -1.20, -0.70, -1.70, -1.20, -0.70, -1.70, -1.20, -0.70

# fancy time_mix
time_mix_k = nn.Parameter(torch.pow(inits, ratio_1_to_almost0))
time_mix_v = nn.Parameter(torch.pow(inits, ratio_1_to_almost0) + 0.3*ratio_0_to_1)
time_mix_r = nn.Parameter(torch.pow(inits, 0.5*ratio_1_to_almost0))
# time_mix_k => 0.00, 0.13, 0.26, 0.39, 0.51, 0.63, 0.75, 0.87
# time_mix_v => 0.01, 0.14, 0.27, 0.40, 0.52, 0.65, 0.77, 0.89
# time_mix_r => 0.00, 0.36, 0.51, 0.62, 0.71, 0.79, 0.87, 0.93

time_shift = nn.ZeroPad2d((0, 0, 1, -1)) # padding zero vào embd vector đầu tiên
key = nn.Linear(n_embd, attn_sz, bias=False)
value = nn.Linear(n_embd, attn_sz, bias=False)
receptance = nn.Linear(n_embd, attn_sz, bias=False)
output = nn.Linear(attn_sz, n_embd, bias=False)

# FORWARD
# Mix x with the previous timestep to produce xk, xv, xr
xx = time_shift(x) # nghĩa là xx[t] là x[t-1], với xx[0] = (0,0,..,0) <= zero padding
xk = x * time_mix_k + xx * (1 - time_mix_k)
xv = x * time_mix_v + xx * (1 - time_mix_v)
xr = x * time_mix_r + xx * (1 - time_mix_r)

# Use xk, xv, xr to produce k, v, r
k = key(xk)
v = value(xv)
r = receptance(xr)

rwkv = torch.sigmoid(r) * WKV.apply(B, T, C, time_decay, time_first, k, v) # WKV.apply(B, T, C, w, u, k, v)
return output(rwkv)
```

![](files/token-shift-00.jpg)
Peng Bo nói rằng khi tỉ lệ vocab_size / emb_size cao thì không cần "more-shift", và more-shift có tác dụng hơn với char-level LM. Điều đó có nghĩa là:
- `vocab_size / emb_size` cao thì nhiều khả năng token dài được sử dụng, như vậy một token chứa nhiều thông tin và thông tin "trọn vẹn" (1 từ hoàn chỉnh) hơn là nhiều char tokens. Với char tokens thì more tokens mới tạo được một từ vì thế cần dùng more-shift?
`vocab_size / emb_size` cao cũng có nghĩa là ctx_len dài hơn được sử dụng, khi đó more-shift sẽ giúp mối liên hệ giữa các tokens xa nhau tốt hơn?

# Token shift GPT
Phi Wang https://github.com/lucidrains/token-shift-gpt

An autoregressive model that relies solely on __shifting along the sequence dimension__ and __feedforwards__.

Không thể giải thích được nhưng nó hoạt động khá tốt. Module feedforward có thiết kế giống [gMLP](./gmlp.md), ngoại trừ kích thước feature của gate tensor được chia thành các đoạn log_2(seq_len) và mean pool của các phân đoạn liên tiếp trước đó (độ dài 1,2,4,8 ... vào quá khứ) được shifted vào thành từng đoạn trước khi project along the feature dimension.

TODO: đọc hiểu code token-shift-gpt.

```py
from math import log2, ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def shift(x, amt, dim = -1): # giống rwkv time_shift
    return F.pad(x, (*((0, 0) * (-dim - 1)), amt, -amt), value = 0.)

def shift_tokens(x, amt, eps = 1e-5):
    n, device = x.shape[1], x.device

    cumsum = x.cumsum(dim = 1) # Returns the cumulative sum of elements of input in the dimension dim.
    *x, x_pass = x.chunk(amt + 1, dim = -1) # split x into amt+1 chunks
    *x_cumsum, _ = cumsum.chunk(amt + 1, dim = -1)

    amts = 2 ** torch.arange(amt)
    amts = amts.tolist()

    shifts = []
    denom = torch.arange(n, device = device)

    for x_chunk, x_cumsum_chunk, amt in zip(x, x_cumsum, amts):
        shifted_chunk = shift(x_cumsum_chunk, amt, dim = -2) - shift(x_cumsum_chunk, 2 * amt, dim = -2)
        shifted_denom = shift(denom, amt, dim = -1) - shift(denom, 2 * amt, dim = -1)
        shifted_denom = rearrange(shifted_denom, 'n -> () n ()')
        normed_shifted_x = shifted_chunk / (shifted_denom + eps)
        shifts.append(normed_shifted_x)

    return torch.cat((*shifts, x_pass), dim = -1)

def discounted_cumsum(t, gamma):
    try:
        from torch_discounted_cumsum import discounted_cumsum_left
    except ImportError:
        print('unable to import torch_discounted_cumsum - please run `pip install torch-discounted-cumsum`')

    b, n, d = t.shape
    t = rearrange(t, 'b n d -> (b d) n')
    t = discounted_cumsum_left(t, gamma)
    t = rearrange(t, '(b d) n -> b n d', b = b)
    return t

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_seq_len,
        num_shifts,
        mult = 4,
        eps = 1e-3,
        use_discounted_cumsum = False,
        discount_gamma = 0.9
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.project_in = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU()
        )

        self.num_shifts = num_shifts
        hidden_dim = dim * mult // 2

        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.to_gate = nn.Linear(hidden_dim, hidden_dim)

        nn.init.constant_(self.to_gate.weight, eps)
        nn.init.constant_(self.to_gate.bias, 1.)

        self.project_out = nn.Linear(hidden_dim, dim)

        # for using discounted cumsum approach

        self.use_discounted_cumsum = use_discounted_cumsum
        self.discount_gamma = discount_gamma

    def forward(self, x):
        x = self.norm(x)

        x = self.project_in(x)

        x, gate = x.chunk(2, dim = -1)

        gate = self.gate_norm(gate)

        if self.use_discounted_cumsum:
            gate = shift(gate, 1, dim = -2)
            gate = discounted_cumsum(gate, self.discount_gamma)
        else:
            gate = shift_tokens(gate, self.num_shifts)

        x = x * self.to_gate(gate)
        return self.project_out(x)

# classes

class TokenShiftGPT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        depth,
        ff_mult = 4,
        use_discounted_cumsum = False,
        discount_gamma = 0.9
    ):
        super().__init__()
        self.seq_len = max_seq_len
        num_shifts = ceil(log2(max_seq_len)) - 1

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.net = nn.Sequential(
            *[Residual(FeedForward(dim = dim, num_shifts = num_shifts, mult = ff_mult, max_seq_len = max_seq_len, use_discounted_cumsum = use_discounted_cumsum, discount_gamma = discount_gamma)) for _ in range(depth)],
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )
    def forward(self, x):
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[1], device = x.device))
        x = x + rearrange(pos_emb, 'n d -> () n d')
        return self.net(x)
```