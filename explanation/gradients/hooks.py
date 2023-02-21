import torch
import models


def define_hook(memory, idx):
    def hook(module, input, output):
        assert type(input) == tuple and len(input) == 1
        input = input[0]
        assert type(output) == torch.Tensor
        if type(module) == models.layers.GlobalAveragePooling2D:
            assert len(input.shape) == 4, f"{input.shape=}"
            memory[idx] = input.shape[2:4]
        elif type(module) in [torch.nn.ReLU, torch.nn.Tanh]:
            memory[idx] = input.detach()
        elif type(module) == torch.nn.Softmax:
            assert len(output.shape) == 2, f"{type(module)=}, {output.shape=}, {idx=}"
            memory[idx] = output.detach()
        else:
            pass
    return hook


def register_hooks(model, memory):
    hooks = [None] * len(list(model.children()))
    assert len(memory) == len(hooks)
    for idx, layer in enumerate(list(model.children())):
        hooks[idx] = layer.register_forward_hook(define_hook(memory, idx))
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
