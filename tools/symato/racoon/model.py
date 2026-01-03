# Rút gọn từ https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/model.py
import torch, os, math, gc, deepspeed
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load as cpp_load
import pytorch_lightning as pl
from deepspeed.ops.adam import FusedAdam

# Load nhân cuda
T_MAX = int(os.environ.get("RWKV_T_MAX", 256)) # T_MAX càng dài càng tốn vram
wkv_cuda = cpp_load(name=f"racoon_bf6_wkv_{T_MAX}", sources=["wkv_cuda.cu"], verbose=True,
    extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", 
        "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])

# Note: Chỉ hỗ trợ bf16 để loại bỏ if then
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B, ctx.T, ctx.C = B, T, C
        assert T <= T_MAX # Độ dài ctx_len phải <= T_MAX
        if C > 32: assert (B * C) % 32 == 0, "Nếu C > 32 thì B * C phải chia hết cho 32 để tối ưu cho nhân cuda"

        # biến thành f32 để tăng độ chính xác, 
        # và duỗi thành mảng 1 chiều để chuẩn bị feed cho nhân cuda
        w = -torch.exp(w.float().contiguous())

        u = u.contiguous() # giá trị khởi tạo t0
        k = k.contiguous() # k,v như trong trong KQV
        v = v.contiguous()

        y = torch.empty((B, T, C), device=w.device, 
            memory_format=torch.contiguous_format, dtype=torch.bfloat16)

        if u.dtype == torch.float32:
            u = u.bfloat16()
            k = k.bfloat16()
            v = v.bfloat16()

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


class RWKV_TimeMix(torch.jit.ScriptModule):
# class RWKV_TimeMix(nn.Module):
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

    @torch.jit.script_method
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


class RWKV_ChannelMix(torch.jit.ScriptModule):
# class RWKV_ChannelMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd): x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            # time_mix_k => 0.00, 0.13, 0.26, 0.39, 0.51, 0.63, 0.75, 0.87
            # time_mix_r => 0.00, 0.13, 0.26, 0.39, 0.51, 0.63, 0.75, 0.87

        hidden_sz = 4 * args.n_embd
        self.key = nn.Linear(args.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, args.n_embd, bias=False)


    @torch.jit.script_method
    def forward(self, x):
        xx = self.time_shift(x) # do token mixing
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # square(relu(key @ xk)) can be fused
        k = self.key(xk)
        k = torch.relu(k)
        k = torch.square(k)

        # sigmoid(receptance @ xr) can be fused
        r = self.receptance(xr)
        r = torch.sigmoid(r)

        rkv = r * self.value(k) # kv
        return rkv


# Các tầng của RWKV
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_TimeMix(args, layer_id) # TimeMix được gọi là Attention (att)
        self.ffn = RWKV_ChannelMix(args, layer_id) # ChannelMix được gọi là Feedforward (ffn)
        # self.ffn = torch.compile(self.ffn)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# Mô hình RWKV
'''Dưới đây là mã nguồn của mô hình RWKV (Receptance Weighted Key-Value) được triển khai trong PyTorch. Đối tượng RWKV là một nn.Module và có ba thành phần chính:

- `self.emb`: một nn.Embedding dùng để ánh xạ các từ trong từ điển thành các vector embedding có kích thước args.n_embd.

- `self.blocks`: một nn.ModuleList chứa args.n_layer đối tượng Block, mỗi Block đại diện cho một tầng của mô hình và có kiến trúc tương tự nhau.

- `self.head`: một lớp tuyến tính nn.Linear dùng để chuyển đổi vector đại diện của mỗi từ thành một vector dự đoán xác suất cho từ tiếp theo.

- Phương thức forward nhận đầu vào là idx là một tensor kích thước (batch_size, sequence_length) chứa chỉ mục của các từ trong câu. Đầu ra của mô hình là một tensor có kích thước (batch_size, sequence_length, vocab_size), trong đó vocab_size là số lượng từ trong từ điển. Mỗi phần tử (i, j, k) của tensor đầu ra là xác suất để từ có chỉ mục k là từ tiếp theo của từ có chỉ mục j trong câu thứ i.
'''
class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.blocks[0].ln0 = nn.LayerNorm(args.n_embd)
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        x = self.blocks[0].ln0(x)

        for i, block in enumerate(self.blocks): 
            if self.args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)
        x = self.ln_out(x) # layernorm
        return self.head(x)


    def init_weight(self):
        print(f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################""")
        m = {}
        for n, p in self.state_dict().items():
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n:
                m[n] = p.bfloat16()
            else:
                shape = p.shape
                gain, scale = 1.0, 1.0
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if n == "head.weight":
                        scale = 0.5
                    else:
                        for kk in [".att.key.", ".att.receptance.", ".att.output.", ".att.key.", 
                                ".ffn.value.", ".ffn.receptance."]:
                            if kk in n: scale = 0; break

                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(8)} {n}")
                x = torch.empty((shape[0], shape[1]))
                if scale == 0:  nn.init.zeros_(x)
                elif scale < 0: nn.init.uniform_(x, a=scale, b=-scale)
                else:           nn.init.orthogonal_(x, gain = gain*scale)
                m[n] = x.bfloat16()

        # Giải phóng bộ nhớ và trả về bộ tham số đã được khởi tạo
        gc.collect(); torch.cuda.empty_cache()
        return m


''' L2Wrap được sử dụng để tính đạo hàm theo phương pháp L2 regularization. Cụ thể, L2Wrap thêm một chi phí bổ sung vào hàm mất mát (loss) hiện tại. Công thức chi phí bổ sung được tính bằng cách lấy giá trị lớn nhất trong ma trận đầu vào y, nhân với một hệ số nhỏ và gán giá trị này trở lại cho ma trận gy (khởi tạo zeros) có cùng kích thước như y. Hệ số factor được sử dụng để giảm thiểu giá trị của chi phí bổ sung trong quá trình huấn luyện.
'''
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y) # khởi tạo gy có kích thước giống y và giá trị là 0
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


''' Lớp Racoon sử dụng lighting để quản lý việc huấn luyện RWKV và điều chỉnh tốc độ học (learning rate)
'''
class Racoon(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = RWKV(args)

    def training_step(self, batch, batch_idx=None):
        idx, targets = batch # idx = [[1,2,3][a,b,c]], targets = [[2,3,4],[b,c,d]]
        logits = self.rwkv(idx) # => (B, T, vocab_size), targets => (B, T)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def configure_optimizers(self):
        args = self.args
        if args.layerwise_lr > 0:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()

            for n, p in self.named_parameters():
                if     "time_mix" in n: lr_1x.add(n)
                elif "time_decay" in n: lr_2x.add(n)
                elif "time_first" in n: lr_3x.add(n)
                else:                   lr_1x.add(n)

            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))

            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = [
                {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
            ]
        else:
            optim_groups = [
                {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},
            ]

        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, \
            eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
