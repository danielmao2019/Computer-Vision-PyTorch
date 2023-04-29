import pytest
import torch
import explanation
import losses


def test_CE_gradient():
    num_classes = 5
    for _ in range(10):
        y_pred = torch.softmax(torch.rand(size=(1, num_classes)), dim=1)
        y_true = torch.randint(low=0, high=num_classes, size=(1,)).type(torch.int64)
        mapping = torch.randperm(num_classes)
        gradient = explanation.gradients.CE_gradient(
            y_pred=y_pred, y_true=y_true, mapping=mapping,
        )
        y_pred.requires_grad_()
        criterion = losses.MappedMNISTCEL(mapping=mapping)
        loss = criterion(y_pred=y_pred, y_true=y_true)
        gradient_expected = torch.autograd.grad(outputs=loss, inputs=y_pred)
        assert type(gradient_expected) == tuple and len(gradient_expected) == 1
        assert torch.allclose(gradient, gradient_expected[0]), f"{gradient.numpy()=}, {gradient_expected[0].numpy()=}"
