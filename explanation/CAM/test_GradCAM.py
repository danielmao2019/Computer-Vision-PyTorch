import pytest
import torch
import models
import explanation

import torchvision


def test_GradCAM():
    in_features = 3
    out_features = 5
    layer_idx = 6
    model = models.LeNet(in_features=in_features, out_features=out_features)
    first_part = torch.nn.Sequential(*list(model.children())[:layer_idx])
    second_part = torch.nn.Sequential(*list(model.children())[layer_idx:])
    gmi = explanation.gradients.GradientModelInputs(model=model, layer_idx=layer_idx)
    image = torch.rand(size=(1, in_features, 32, 32)).requires_grad_()
    output = gmi.update(image)
    for cls in range(out_features):
        # computed Grad-CAM
        grad_cam_computed = explanation.CAM.compute_grad_cam(gmi=gmi, cls=cls)
        # expected Grad-CAM
        inter = first_part(image)
        final = second_part(inter)
        gradient_initial = torch.zeros(size=(1, out_features))
        gradient_initial[:, cls] = 1
        loss = torch.sum(final * gradient_initial)
        gradient = torch.autograd.grad(outputs=loss, inputs=inter)[0]
        gradient = torch.mean(gradient, dim=[2, 3], keepdim=True)
        grad_cam_expected = torch.nn.functional.relu(torch.sum(inter*gradient, dim=1, keepdim=True))
        assert grad_cam_expected.shape == (1, 1,) + inter.shape[2:]
        # compare
        assert torch.allclose(grad_cam_computed, grad_cam_expected), \
            f"{grad_cam_computed.flatten().detach().numpy()=}\n{grad_cam_expected.flatten().detach().numpy()=}"
