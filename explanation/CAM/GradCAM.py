"""
Limitations:
* Assumes image classification model with last layer as softmax.
"""
import torch
import explanation


def compute_grad_cam(gmi, cls):
    """
    Assume gmi has called update on image.
    """
    device = next(gmi.model.parameters()).device
    activations = gmi.memory[gmi.layer_idx]
    gradients = torch.zeros(size=(1, gmi.model.out_features), dtype=torch.float32).to(device)
    gradients[0, cls] = 1
    gradients = gmi(gradients)
    assert len(gradients.shape) == 4, f"{gradients.shape=}"
    coefficients = torch.mean(gradients, dim=[2, 3], keepdim=True)
    assert coefficients.shape == (1, activations.shape[1], 1, 1), f"{coefficients.shape=}, {activations.shape=}"
    grad_cam = activations * coefficients
    grad_cam = torch.sum(grad_cam, dim=1, keepdim=True)
    grad_cam = torch.nn.functional.relu(grad_cam)
    return grad_cam
