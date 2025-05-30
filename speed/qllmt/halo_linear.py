from copy import deepcopy

import torch
from qllmt import halo1_fns, halo2_fns

class HaloLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None, hq_config=None, **kwargs):
        
        assert bias==False, 'Bias is not supported yet'
        super(HaloLinear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        self.in_features = in_features
        self.out_features = out_features
        
        self.hq_config = deepcopy(hq_config)
        kernel_name = self.hq_config.get('kernel', 'halo0_fp8')
        
        assert 'int8' in kernel_name, f'Unsupported precision: {kernel_name}'
        self.hq_config['halo_precision'] = 'fp8' if 'fp8' in kernel_name else 'int8'
        backward_xH = True if 'bxh' in kernel_name.lower() else False

        if   kernel_name.startswith('halo1'): self._fn = halo1_fns.HaloFnLevel1WithQFSDPBackwardXH
        elif kernel_name.startswith('halo2'): self._fn = halo2_fns.HaloFnLevel2WithQFSDPBackwardXH
        else: raise ValueError(f'Unsupported Halo level: {kernel_name}')


    @staticmethod
    def from_unquantized(module: torch.nn.Linear, hq_config=None, **kwargs):
        assert module.bias is None, 'Bias is not supported yet'
        q_module = HaloLinear( module.in_features, module.out_features,
            device=module.weight.device, dtype=module.weight.dtype,
            bias=False, hq_config=hq_config, **kwargs )
        with torch.no_grad(): q_module.weight.data.copy_(module.weight.data)
        return q_module

    def forward(self, x):
        x_shape = x.shape
        x_view = x.view(-1, x_shape[-1])
        return self._fn.apply( x_view, self.weight, self.hq_config,).view(*x_shape[:-1], -1)
