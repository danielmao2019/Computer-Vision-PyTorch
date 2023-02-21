"""
Limitations:
* Only designed for image classification tasks.
* Output shape of softmax layer must be 1-D.
* Does not accept batch normalization layers.
"""
import torch
import models
import explanation


def tanh_backward(inputs):
    def new_layer(x):
        assert x.shape == inputs.shape
        return x * explanation.gradients.tanh_gradient(inputs)
    return new_layer


class GradientModelInputs(torch.nn.Module):

    def __init__(self, model, layer_idx):
        super(GradientModelInputs, self).__init__()
        model.eval()
        self.model = model
        self.layer_idx = layer_idx
        self.length = len(list(model.children()))
        self.memory = [None] * length
        self.hooks = explanation.gradients.hooks.register_hooks(model=self.model, memory=self.memory)

    def forward(self, images, labels, criterion_gradient_func):
        outputs = self.model(images)
        gradients = criterion_gradient_func(inputs=outputs, labels=labels)
        assert len(gradients.shape) == 2, f"{gradients.shape=}"
        for idx in range(self.length-1, self.layer_idx-1, -1):
            layer = list(self.model.children())[idx]
            new_layer = self.get_new_layer(layer)
            gradients = new_layer(gradients)
        return gradients

    def get_new_layer(self, layer):
        if type(layer) == torch.nn.Conv2d:
            new_weight = layer.weight.data.clone()
            assert len(new_weight.shape) == 4
            new_weight = torch.flip(new_weight, dims=[2, 3])
            new_weight = torch.stack([new_weight[:, k, :, :] for k in range(new_weight.shape[1])], dim=0)
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
            input_size = self.memory[idx]
            new_layer = lambda x: torch.tile(
                torch.unsqueeze(torch.unsqueeze(x, dim=2), dim=3), dims=(1, 1)+input_size,
            ) / (input_size[0] * input_size[1])
        ##################################################
        # activation layers
        ##################################################
        elif type(layer) == torch.nn.Tanh:
            inputs = self.memory[idx]
            assert inputs is not None, f"{idx=}"
            new_layer = tanh_backward(inputs)
        elif type(layer) == torch.nn.Softmax:
            outputs = self.memory[idx]
            assert outputs is not None
            assert len(outputs.shape) == 2, f"{outputs.shape=}"
            new_weight = torch.outer(outputs, outputs) - torch.diag(outputs)
            assert len(new_weight.shape) == 2
            new_layer = torch.nn.Linear(
                in_features=len(outputs), out_features=len(outputs), bias=False,
            )
            new_layer.weights = torch.nn.Parameter(new_weight)
        else:
            raise NotImplementedError(f"[ERROR] Layers of type {type(layer)} not implemented.")
        if isinstance(new_layer, torch.nn.Module):
            new_layer.eval()
            new_layer = new_layer.to(device)
