"""
Note that this program is written with LeNet in mind.
It might be too specific for image classification tasks.
For example, I am asserting the output of softmax layer is 1-D.
"""
import torch
import models


def tanh_gradient(x):
    return 1 - torch.tanh(x)**2


def CE_gradient(input, label):
    assert len(input.shape) == 2 and input.shape[0] == 1
    assert type(label) == torch.Tensor
    assert label.shape == (1,) and label.dtype == torch.int64
    label = label.item()
    ans = torch.zeros(size=input.shape)
    ans[0, label] = -label/input[0, label]
    return ans


def register_hooks(model, memory):
    length = len(list(model.children()))
    hooks = [None] * length
    assert len(memory) == length
    for idx, layer in enumerate(list(model.children())):
        def forward_hook(_module, _input, _output):
            assert type(_input) == tuple and len(_input) == 1
            assert type(_output) == torch.Tensor
            if type(layer) in [torch.nn.ReLU, torch.nn.Tanh]:
                memory[idx] = _input[0].detach()
            elif type(layer) == torch.nn.Softmax:
                assert len(_output.shape) == 2, f"{_output.shape=}, {idx=}"
                memory[idx] = _output.detach()
            elif type(layer) == torch.nn.AvgPool2d:
                assert len(_output.shape) == 4, f"{_output.shape=}, {idx=}"
                memory[idx] = layer.kernel_size
            else:
                pass
        hooks[idx] = layer.register_forward_hook(forward_hook)
    return model, hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_backward_model(model, memory):
    length = len(list(model.children()))
    layers = [None] * length
    assert len(memory) == length
    for idx, layer in enumerate(list(model.children())):
        if type(layer) == torch.nn.Conv2d:
            new_weight = layer.weight.data.clone()
            assert len(new_weight.shape) == 4
            new_weight = torch.flip(new_weight, dims=[2, 3])
            new_weight = torch.stack([
                new_weight[:, k, :, :] for k in range(new_weight.shape[1])
            ], dim=0)
            new_layer = torch.nn.Conv2d(
                in_channels=new_weight.shape[1], out_channels=new_weight.shape[0],
                kernel_size=new_weight.shape[2:4], stride=1, bias=False,
                padding=(new_weight.shape[2]-1, new_weight.shape[3]-1),
            )
            new_layer.weights = torch.nn.Parameter(new_weight)
        elif type(layer) == torch.nn.AvgPool2d:
            kernel_size = memory[idx]
            new_layer = lambda x: torch.tile(x, dims=(2 ,2)) / kernel_size**2
        elif type(layer) == torch.nn.Linear:
            new_weight = layer.weight.data.clone()
            assert len(new_weight.shape) == 2
            new_weight = torch.permute(new_weight, dims=(1, 0))
            new_layer = torch.nn.Linear(
                in_features=new_weight.shape[1], out_features=new_weight.shape[0], bias=False,
            )
            new_layer.weights = torch.nn.Parameter(new_weight)
        elif type(layer) == torch.nn.Softmax:
            output = memory[idx]
            assert len(output.shape) == 1
            new_weight = torch.outer(output, output) - torch.diag(output)
            new_layer = torch.nn.Linear(
                in_features=len(output), out_features=len(output), bias=False,
            )
            new_layer.weights = torch.nn.Parameter(new_weight)
        elif type(layer) == torch.nn.Tanh:
            input_tensor = memory[idx]
            new_layer = lambda x: x * tanh_gradient(input_tensor)
        else:
            raise NotImplementedError(f"[ERROR] Layers of type {type(layer)} not implemented.")
        layers.append(new_layer)
    layers.reverse()
    return layers


def compute_gradients(model, image, label):
    assert len(image.shape) == 4 and image.shape[0] == 1
    assert len(label.shape) == 1 and label.shape[0] == 1
    model.eval()
    # get pre-activations and gradients of activation functions
    memory = [None] * len(list(model.children()))
    model, hooks = register_hooks(model, memory)
    output = model(image)
    # backward pass
    backward_model = get_backward_model(model=model, memory=memory)
    gradient_tensor = CE_gradient(input=output, label=label)
    assert gradient_tensor.shape == (1, model.out_features)
    for layer in backward_model:
        gradient_tensor = layer(gradient_tensor)
    # final steps
    remove_hooks(hooks)
    assert gradient_tensor.shape == image.shape
    return gradient_tensor


model = models.LeNet(in_features=1, out_features=10)
print(list(model.children()))
compute_gradients(model, image=torch.zeros(size=(1, 1, 32, 32)), label=torch.zeros(size=(1,)))
