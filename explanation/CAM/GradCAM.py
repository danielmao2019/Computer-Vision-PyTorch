"""
Limitations:
* Assumes image classification model with last layer as softmax.
* Assumes batch size = 1.
"""
import torch
import explanation


def compute_grad_cam(model, layer_idx, image):
    device = next(model.parameters()).device
    image = image.to(device)
    assert len(image.shape) == 4, f"{image.shape=}"
    model.eval()
    length = len(list(model.children()))
    depth = length - layer_idx - 1
    memory = [None] * length
    model, hooks = explanation.gradients.hooks.register_hooks(model=model, memory=memory)
    def get_hook_func(_memory, _idx):
        def hook(_module, _input, _output):
            _memory[_idx] = _output
        return hook
    list(model.children())[layer_idx].register_forward_hook(get_hook_func(memory, layer_idx))
    output = model(image)
    backward_model = explanation.gradients.get_backward_model(model=model, memory=memory, depth=depth)
    activations = memory[layer_idx]
    grad_cams = [None] * model.out_features
    for cls in range(model.out_features):
        gradient_tensor = torch.zeros(size=(1, model.out_features), dtype=torch.float32).to(device)
        gradient_tensor[0, cls] = 1
        for layer in backward_model:
            gradient_tensor = layer(gradient_tensor)
        assert len(gradient_tensor.shape) == 4, f"{gradient_tensor.shape=}"
        coefficients = torch.mean(gradient_tensor, dim=[2, 3], keepdims=True)
        assert coefficients.shape == (1, activations.shape[1], 1, 1)
        cam = activations * coefficients
        cam = torch.sum(cam, dim=1, keepdims=True)
        cam = torch.nn.functional.relu(cam)
        grad_cams[cls] = cam
    return grad_cams
