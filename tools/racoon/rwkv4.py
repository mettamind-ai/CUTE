# Rút gọn từ https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/model.py
import torch, os, math, gc, deepspeed
import torch.nn as nn

from channel_mix import ChannelMix
from time_mix4 import TimeMix
ZERO_INIT = ".att.key att.receptance .att.output .ffn.value .ffn.receptance .ffnPre.value .ffnPre.receptance head_q. .oo. .rr.".strip().split()

# Các tầng của RWKV
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = TimeMix(args, layer_id) # TimeMix được gọi là Attention (att)
        self.ffn = ChannelMix(args, layer_id) # ChannelMix được gọi là Feedforward (ffn)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.ln_emb = nn.LayerNorm(args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)


    def forward(self, idx):
        x = self.emb(idx)
        x = self.ln_emb(x)

        if self.args.grad_cp == 1: # add gradient checkpoint tại từng block
            for block in self.blocks:
                x = deepspeed.checkpointing.checkpoint(block, x)
        else:
            for block in self.blocks: 
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
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.5)
                else:
                    m[n] = p
                m[n] = m[n].bfloat16()

            else:
                shape = p.shape
                gain, scale = 1.0, 1.0
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if n == "head.weight":
                        scale = 0.5
                    else:
                        for kk in ZERO_INIT:
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
