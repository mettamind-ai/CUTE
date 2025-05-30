import torch

import qllmt
from qllmt.functional.hadamard import is_pow2
from .halo_linear import HaloLinear
from copy import deepcopy


class InputFwdHadamardWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        if isinstance(module, torch.nn.Linear):    assert is_pow2(module.in_features),    'Input features should be power of 2!'
        if isinstance(module, torch.nn.Embedding): assert is_pow2(module.num_embeddings), 'Input features should be power of 2!'

    def forward(self, hidden_states, **kwargs):
        x = torch.tensor(hidden_states.shape[-1]).sqrt()
        return self.module(qllmt.power_two_fwd_had(hidden_states, scale=1.0 / x), **kwargs)


def wrap_linear_module(module, config):
    kernel = config.get('kernel', 'simulated')
    if kernel.startswith('halo'):
        print(f'wrapping with {kernel}')
        return HaloLinear.from_unquantized(module, hq_config=config)
    else: # kernel == 'base'
        print(f'not wrapping since kernel is {kernel}')
        return module
