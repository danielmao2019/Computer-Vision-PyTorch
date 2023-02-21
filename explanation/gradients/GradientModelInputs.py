"""
Limitations:
* Only designed for image classification tasks.
* Output shape of softmax layer must be 2-D.
* Does not accept batch normalization layers.
"""
import torch
import models
import explanation


class GradientModelInputs(torch.nn.Module):

    def __init__(self, model, layer_idx):
        super(GradientModelInputs, self).__init__()
        model.eval()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.layer_idx = layer_idx
        self.length = len(list(self.model.children()))
        self.memory = [None] * self.length
        self.hooks = explanation.gradients.hooks.register_hooks(
            model=self.model, memory=self.memory, layer_idx=layer_idx,
        )

    def forward(self, images, labels, criterion_gradient_func):
        outputs = self.model(images)
        gradients = criterion_gradient_func(inputs=outputs, labels=labels)
        assert len(gradients.shape) == 2, f"{gradients.shape=}"
        for idx in range(self.length-1, self.layer_idx-1, -1):
            layer = list(self.model.children())[idx]
            new_layer = self.get_new_layer(layer, idx)
            gradients = new_layer(gradients)
        return gradients

    def get_new_layer(self, layer, idx):
        if type(layer) == torch.nn.Conv2d:
            new_weights = layer.weight.data.clone()
            assert len(new_weights.shape) == 4
            new_weights = torch.flip(new_weights, dims=[2, 3])
            new_weights = torch.stack([new_weights[:, k, :, :] for k in range(new_weights.shape[1])], dim=0)
            new_layer = torch.nn.Conv2d(
                in_channels=new_weights.shape[1], out_channels=new_weights.shape[0],
                kernel_size=(new_weights.shape[2], new_weights.shape[3]), stride=1,
                padding=(new_weights.shape[2]-1, new_weights.shape[3]-1), bias=False,
            )
            new_layer.weights = torch.nn.Parameter(new_weights)
        elif type(layer) == torch.nn.Linear:
            new_weights = layer.weight.data.clone()
            assert len(new_weights.shape) == 2
            new_weights = torch.permute(new_weights, dims=(1, 0))
            new_layer = torch.nn.Linear(
                in_features=new_weights.shape[1], out_features=new_weights.shape[0], bias=False,
            )
            new_layer.weights = torch.nn.Parameter(new_weights)
        ##################################################
        # pooling layers
        ##################################################
        elif type(layer) == torch.nn.AvgPool2d:
            kernel_size = layer.kernel_size
            print(f"{kernel_size=}")
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
            new_layer = self.tanh_backward(inputs)
        elif type(layer) == torch.nn.Softmax:
            outputs = self.memory[idx]
            assert outputs is not None, f"{idx=}"
            new_layer = self.softmax_backward(outputs)
        else:
            raise NotImplementedError(f"[ERROR] Layers of type {type(layer)} not implemented.")
        if isinstance(new_layer, torch.nn.Module):
            new_layer.eval()
            new_layer = new_layer.to(self.device)

    def tanh_backward(self, inputs):
        def new_layer(x):
            assert x.shape == inputs.shape
            return x * explanation.gradients.tanh_gradient(inputs)
        return new_layer

    def softmax_backward(self, outputs):
        def new_layer(x):
            assert x.shape == outputs.shape
            new_weights = (torch.bmm(torch.unsqueeze(outputs, dim=1), torch.unsqueeze(outputs, dim=2))
                           - torch.diag_embed(outputs))
            assert new_weights.shape == (outputs.shape[0], outputs.shape[1], outputs.shape[1])
            return torch.bmm(new_weights, torch.unsqueeze(x, dim=2))
        return new_layer
