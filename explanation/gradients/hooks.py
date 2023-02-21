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
            assert type(memory[idx]) == tuple
            assert len(memory[idx]) == 2
        elif type(module) in [torch.nn.ReLU, torch.nn.Tanh]:
            memory[idx] = input.detach()
        elif type(module) == torch.nn.Softmax:
            assert len(output.shape) == 2, f"{type(module)=}, {output.shape=}, {idx=}"
            memory[idx] = output.detach()
        else:
            pass
    return hook


def register_hooks(model, memory, layer_idx=0):
    length = len(list(model.children()))
    hooks = [None] * length
    assert len(memory) == len(hooks)
    for idx in range(layer_idx, length):
        hooks[idx] = list(model.children())[idx].register_forward_hook(define_hook(memory, idx))
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
