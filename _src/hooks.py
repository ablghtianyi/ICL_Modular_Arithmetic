import torch
from torch import Tensor
from torch.nn import Module

class Hook:
    def __init__(self, module: Module, backward: bool = False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_full_backward_hook(self.hook_fn)

    def hook_fn(self, module: Module, input: Tensor, output: Tensor):
        with torch.no_grad():
            self.input = input[0]
            self.output = output

    def close(self):
        self.hook.remove()


class CacheHook:
    """Used to cache module output"""
    def __init__(self, module: Module):
        self.hook = module.register_forward_pre_hook(self.cache_fn)

    def cache_fn(self, module: Module, input: Tensor):
        with torch.no_grad():
            self.cache = input[0].detach()
                
    def close(self):
        self.hook.remove()