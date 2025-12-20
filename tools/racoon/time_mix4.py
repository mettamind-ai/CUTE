import torch, os, math
import torch.nn as nn
from torch.utils import cpp_extension
from utils import JITableModule, JITableFunction

MAX_CTX_LEN = int(os.environ.get("RWKV_T_MAX", 256)) 
# Load nhân cuda: MAX_CTX_LEN càng dài càng tốn vram
wkv_cuda = cpp_extension.load(
    name=f"racoon_bf16_wkv4_{MAX_CTX_LEN}", sources=["wkv4.cu"], verbose=True,
    extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", 
        "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={MAX_CTX_LEN}"]
)

# Lớp WKV là giao diện / utils để sử dụng nhân cuda. Chỉ hỗ trợ bf16 để đơn giản mã nguồn
class WKV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B, ctx.T, ctx.C = B, T, C
        assert T <= MAX_CTX_LEN
        if C > 32: assert (B * C) % 32 == 0, "Nếu C > 32 thì B * C phải chia hết cho 32 để tối ưu cho nhân cuda"

        # biến thành f32 để tăng độ chính xác, 
        # và duỗi thành mảng 1 chiều để chuẩn bị feed cho nhân cuda
        w = -torch.exp(w.float().contiguous())

        u = u.contiguous() # giá trị khởi tạo t0
        k = k.contiguous() # k như trong trong KQV
        v = v.contiguous() # v như trong trong KQV

        # Kiểm tra 1 lần nữa để đảm bảo các giá trị là bf16
        if u.dtype == torch.float32:
            u = u.bfloat16()
            k = k.bfloat16()
            v = v.bfloat16()

        # Chuẩn bị bộ nhớ các giá trị đầu ra
        y = torch.empty((B, T, C), device=w.device, 
            memory_format=torch.contiguous_format, dtype=torch.bfloat16)

        wkv_cuda.forward(B, T, C, w, u, k, v, y) # giá trị đầu ra được lưu vào y
        ctx.save_for_backward(w, u, k, v, y) # lưu lại giá trị để tính backward
        return y


    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        w, u, k, v, y = ctx.saved_tensors

        gw = torch.empty((B, C),
            device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)

        gu = torch.empty((B, C),
            device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)

        gk = torch.empty((B, T, C),
            device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)

        gv = torch.empty((B, T, C),
            device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)

        wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
        del w; del u; del k; del v; del y

        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)

        # Vì forward(ctx, B, T, C, w, u, k, v) nên backward cần trả lại từng đấy tham số (trừ ctx)
        # Đầu vào B, T, C không cần tính gradient nên giá trị trả về là None, None, None
        return (None, None, None, gw, gu, gk, gv)


class TimeMix(JITableModule):
    def __init__(self, args, layer_id):
        super().__init__()
        attn_sz = args.n_embd # chọn attention size bằng chiều của vector nhúng

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0

            # fancy time_decay
            decay_speed = [-5 + 8*(h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1) for h in range(attn_sz) ]
            self.time_decay = nn.Parameter(torch.tensor(decay_speed))
            # time_decay => -5.00, -3.16, -1.89, -0.78,  0.23,  1.20,  2.11,  3.00

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
            # zigzag     =>  0.00,  0.50, -0.50,  0.00,  0.50, -0.50,  0.00,  0.50
            # time_first => -1.20, -0.70, -1.70, -1.20, -0.70, -1.70, -1.20, -0.70

            # fancy time_mix
            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd): x[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
            # time_mix_k => 0.00, 0.13, 0.26, 0.39, 0.51, 0.63, 0.75, 0.87
            # time_mix_v => 0.01, 0.14, 0.27, 0.40, 0.52, 0.65, 0.77, 0.89
            # time_mix_r => 0.00, 0.36, 0.51, 0.62, 0.71, 0.79, 0.87, 0.93

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1)) # padding zero trước embd vector đầu tiên trong batch
        self.key = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, args.n_embd, bias=False)

    @JITableFunction
    def jitable(self, x):
        xx = self.time_shift(x) # do token mixing
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)

        # sigmoid(receptance @ xr) can be fused
        r = self.receptance(xr)
        r = torch.sigmoid(r)
        return r, k, v


    def forward(self, x):
        r, k, v = self.jitable(x)
        B, T, C = x.size()
        rwkv = r * WKV.apply(B, T, C, self.time_decay, self.time_first, k, v)
        return self.output(rwkv)
