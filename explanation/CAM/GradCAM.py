"""
Limitations:
* Assumes image classification model with last layer as softmax.
"""
import torch
import explanation


def compute_grad_cam(model, layer_idx, image):
    # setup the gradient model.
    gmi = explanation.gradients.GradientModelInputs(model=model, layer_idx=layer_idx)
    device = next(model.parameters()).device
    def get_hook_func(memory, idx):
        def hook(module, input, output):
            assert type(input) == tuple and len(input) == 1
            assert type(output) == torch.Tensor
            input = input[0]
            memory[idx] = input
        return hook
    gmi = gmi.register_forward_hook(layer_idx=layer_idx, hook=get_hook_func(gmi.memory, layer_idx))
    # compute GradCAM.
    gmi.update(image)
    activations = gmi.memory[layer_idx]
    grad_cam_list = [None] * model.out_features
    for cls in range(model.out_features):
        gradients = torch.zeros(size=(1, model.out_features), dtype=torch.float32).to(device)
        gradients[0, cls] = 1
        gradients = gmi(gradients)
        assert len(gradients.shape) == 4, f"{gradients.shape=}"
        coefficients = torch.mean(gradients, dim=[2, 3], keepdim=True)
        assert coefficients.shape == (activations.shape[0], activations.shape[1], 1, 1), f"{coefficients.shape=}, {activations.shape=}"
        grad_cam = activations * coefficients
        grad_cam = torch.sum(grad_cam, dim=1, keepdim=True)
        grad_cam = torch.nn.functional.relu(grad_cam)
        grad_cam_list[cls] = grad_cam
    explanation.gradients.hooks.remove_hooks(gmi.hooks)
    return grad_cam_list
