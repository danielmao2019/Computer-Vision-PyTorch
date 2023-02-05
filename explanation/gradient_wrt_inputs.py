"""
Limitations:
* Only designed for image classification tasks.
* Output shape of softmax layer must be 1-D.
* Only accepts batch_size = 1.
* Does not accept batch normalization layers.
"""
import torch
import models
import explanation


def get_backward_model(model, memory, depth):
    device = next(model.parameters()).device
    length = len(list(model.children()))
    length = length if depth is None else min(length, depth)
    layers = [None for _ in range(length)]
    for idx in range(length):
        layer = list(model.children())[-(idx+1)]
        if type(layer) == torch.nn.Conv2d:
            new_weight = layer.weight.data.clone()
            assert len(new_weight.shape) == 4
            new_weight = torch.flip(new_weight, dims=[2, 3])
            new_weight = torch.stack([
                new_weight[:, k, :, :] for k in range(new_weight.shape[1])
            ], dim=0)
            new_layer = torch.nn.Conv2d(
                in_channels=new_weight.shape[1], out_channels=new_weight.shape[0],
                kernel_size=(new_weight.shape[2], new_weight.shape[3]), stride=1,
                padding=(new_weight.shape[2]-1, new_weight.shape[3]-1), bias=False,
            )
            new_layer.weights = torch.nn.Parameter(new_weight)
        elif type(layer) == torch.nn.Linear:
            new_weight = layer.weight.data.clone()
            assert len(new_weight.shape) == 2
            new_weight = torch.permute(new_weight, dims=(1, 0))
            new_layer = torch.nn.Linear(
                in_features=new_weight.shape[1], out_features=new_weight.shape[0], bias=False,
            )
            new_layer.weights = torch.nn.Parameter(new_weight)
        ##################################################
        # pooling layers
        ##################################################
        elif type(layer) == torch.nn.AvgPool2d:
            kernel_size = layer.kernel_size
            new_layer = lambda x: torch.tile(x, dims=(2 ,2)) / kernel_size**2
        elif type(layer) == models.layers.GlobalAveragePooling2D:
            input_size = memory[-(idx+1)]
            new_layer = lambda x: torch.tile(
                torch.unsqueeze(torch.unsqueeze(x, dim=2), dim=3), dims=(1, 1)+input_size,
            ) / (input_size[0] * input_size[1])
        ##################################################
        # activation layers
        ##################################################
        elif type(layer) == torch.nn.Tanh:
            inputs = memory[-(idx+1)]
            assert inputs is not None, f"{idx=}"
            # print(f"{explanation.gradients.tanh_gradient(inputs).shape=}")
            new_layer = tanh_backward(inputs)
        elif type(layer) == torch.nn.Softmax:
            outputs = memory[-(idx+1)]
            assert outputs is not None
            assert len(outputs.shape) == 2, f"{outputs.shape=}"
            assert outputs.shape[0] == 1, f"{outputs.shape=}"
            outputs = outputs[0]
            new_weight = torch.outer(outputs, outputs) - torch.diag(outputs)
            assert len(new_weight.shape) == 2
            new_layer = torch.nn.Linear(
                in_features=len(outputs), out_features=len(outputs), bias=False,
            )
            new_layer.weights = torch.nn.Parameter(new_weight)
        else:
            raise NotImplementedError(f"[ERROR] Layers of type {type(layer)} not implemented.")
        if isinstance(new_layer, torch.nn.Module):
            new_layer = new_layer.to(device)
            new_layer.eval()
        layers[idx] = new_layer
    return layers


def tanh_backward(inputs):
    def new_layer(x):
        assert x.shape == inputs.shape
        return x * explanation.gradients.tanh_gradient(inputs)
    return new_layer


def compute_gradients(model, image, label, depth):
    device = next(model.parameters()).device
    image = image.to(device)
    label = label.to(device)
    assert len(image.shape) == 4 and image.shape[0] == 1
    assert len(label.shape) == 1 and label.shape[0] == 1
    model.eval()
    # get pre-activations and gradients of activation functions
    length = len(list(model.children()))
    memory = [None for _ in range(length)]
    model, hooks = explanation.hooks.register_hooks(model=model, memory=memory)
    output = model(image)
    # backward pass
    backward_model = get_backward_model(model=model, memory=memory, depth=depth)
    gradient_tensor = explanation.gradients.CE_gradient(inputs=output, labels=label)
    assert gradient_tensor.shape == (1, model.out_features)
    for idx, layer in enumerate(backward_model):
        gradient_tensor = layer(gradient_tensor)
    # final steps
    explanation.hooks.remove_hooks(hooks)
    return gradient_tensor
