import torch
import models


def check_io(input, output):
    assert type(input) == tuple and len(input) == 1 and type(input[0]) == torch.Tensor
    assert type(output) == torch.Tensor
    input = input[0]
    return input, output


def register_hooks(model, memory, layer_idx=0):
    length = len(list(model.children()))
    hooks = [None] * length
    assert len(memory) == len(hooks)
    for idx in range(layer_idx, length):
        layer = list(model.children())[idx]
        if type(layer) not in [
            models.layers.GlobalAveragePooling2D, torch.nn.ReLU, torch.nn.Tanh, torch.nn.Softmax,
            ]:
            continue
        assert len(layer._forward_hooks) == 0, f"{layer_idx=}, {layer._forward_hooks=}"
        if type(layer) == models.layers.GlobalAveragePooling2D:
            def get_forward_hook(memory, idx):
                def hook(module, input, output):
                    input, output = check_io(input, output)
                    assert len(input.shape) == 4, f"{input.shape=}"
                    memory[idx] = input.shape[2:4]
                return hook
        elif type(layer) in [torch.nn.ReLU, torch.nn.Tanh]:
            def get_forward_hook(memory, idx):
                def hook(module, input, output):
                    input, output = check_io(input, output)
                    memory[idx] = input.detach()
                return hook
        elif type(layer) == torch.nn.Softmax:
            def get_forward_hook(memory, idx):
                def hook(module, input, output):
                    input, output = check_io(input, output)
                    assert len(output.shape) == 2, f"{idx=}, {type(module)=}, {output.shape=}"
                    memory[idx] = output.detach()
                return hook
        hooks[idx] = layer.register_forward_hook(get_forward_hook(memory, idx))
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        if hook is not None:
            hook.remove()
