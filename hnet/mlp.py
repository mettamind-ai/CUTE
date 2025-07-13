# Base code imported from
# https://github.com/state-spaces/mamba
import torch.nn as nn
import torch.nn.functional as F

from flash.ops.swiglu import swiglu


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model,
        d_intermediate=None,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if d_intermediate is None: d_intermediate = int(8 * d_model / 3)
        d_intermediate = (d_intermediate + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = swiglu(y)
        y = self.fc2(y)
        return y
