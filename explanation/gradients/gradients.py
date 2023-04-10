"""
Limitations:
* only supports batch size = 1.
"""
import torch


def tanh_gradient(x):
    return 1 - torch.tanh(x)**2


def CE_gradient(inputs, labels, mapping=None):
    assert len(inputs.shape) == 2 and inputs.shape[0] == 1, f"{inputs.shape=}"
    assert type(labels) == torch.Tensor
    assert labels.shape == (1,), f"{labels.shape=}"
    assert labels.dtype == torch.int64, f"{labels.dtype=}"
    if mapping is not None:
        mapping = mapping.to(labels.device)
        labels = mapping[labels]
    labels = labels.item()
    ans = torch.zeros(size=inputs.shape).to(inputs.device)
    ans[0, labels] = -labels/inputs[0, labels]
    return ans


def MSE_gradient(inputs, labels):
    assert len(inputs.shape) == 2 and inputs.shape[0] == 1
    assert labels.shape == (1,), f"{labels.shape=}"
    assert inputs.dtype == torch.float32, f"{inputs.shape=}"
    assert labels.dtype == torch.int64, f"{labels.dtype=}"
    ans = 2 * inputs / len(inputs)
    labels = labels.item()
    ans[0, labels] -= 2 / len(inputs)
    return ans
