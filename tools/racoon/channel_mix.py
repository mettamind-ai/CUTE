from .utils import JITableModule, JITableFunction
import torch
from torch import nn

from functools import partial

class ChannelMix(JITableModule):
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


    @JITableFunction
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


class RWKV_CMix_x070(JITableModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    @JITableFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
