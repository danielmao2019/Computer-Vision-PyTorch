import torch
import models


def tanh_gradient(x):
    return 1 - torch.tanh(x)**2


def register_hooks(model, preact):
    assert len(preact) == len(list(model.children()))
    hooks = []
    for idx, layer in enumerate(list(model.children())):
        if type(layer) in [torch.nn.Conv2d, torch.nn.Linear]:
            def hook(_model, _input, _output):
                preact[idx] = _output.detach()
            hooks.append(layer.register_forward_hook(hook))
        else:
            raise NotImplementedError(f"[ERROR] Layers of type {type(layer)} not implemented.")
    return model, hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_backward_model(model):
    layers = []
    for layer in model.children():
        new_weight = layer.weight.data.clone()
        if type(layer) == torch.nn.Conv2d:
            assert len(new_weight.shape) == 4
            new_weight = torch.flip(new_weight, dims=[2, 3])
            new_weight = torch.stack([
                new_weight[:, k, :, :] for k in range(new_weight.shape[1])
            ], dim=0)
            new_layer = torch.nn.Conv2d(
                in_channels=new_weight.shape[1], out_channels=new_weight.shape[0],
                kernel_size=new_weight.shape[2:4], stride=1, bias=False,
            )
            new_layer.weights = torch.nn.Parameter(new_weight)
            layers.append(new_layer)
    return layers


def compute_initial_gradient_map(model, image, label, criterion):
    pass


def compute_gradients(model, image, label, criterion):
    model.eval()
    # get pre-activations and gradients of activation functions
    preact = [None] * len(list(model.children()))
    model, hooks = register_hooks(model, preact)
    model(image)
    activation_gradients = [tanh_gradient(preact[i]) for i in range(len(preact))]
    # backward pass
    backward_model = get_backward_model(model)
    gradient_map = compute_initial_gradient_map(model, image, label, criterion)
    for idx, layer in enumerate(backward_model):
        gradient_map = layer(gradient_map * activation_gradients[idx])
    # outputs
    remove_hooks(hooks)
    assert gradient_map.shape == image.shape
    return gradient_map


model = models.LeNet(in_features=1, out_features=10)
criterion = torch.nn.CrossEntropyLoss()
compute_gradients(model, input=torch.zeros(size=(1, 1, 32, 32)), criterion=criterion)
