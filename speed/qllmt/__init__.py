import re
from .halo_linear import HaloLinear

def wrap_linear_module(module, config):
    kernel = config.get('kernel', 'halo')
    if kernel.startswith('halo'):
        print(f'wrapping with {kernel}')
        return HaloLinear.from_unquantized(module, hq_config=config)
    else: # kernel == 'base'
        print(f'not wrapping since kernel is {kernel}')
        return module


def wrap_model(model, config, exceptions='head'):
    exceptions = re.compile(rf'{exceptions}')
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not exceptions.search(name):
            wrapped_module = wrap_linear_module(module, config)
            setattr(wrapped_module, 'layer_name', name)
            swap_module(model, name, wrapped_module)
            print(f"{name} wrapped")
