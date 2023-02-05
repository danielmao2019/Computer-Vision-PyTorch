import torch


def tanh_gradient(x):
    return 1 - torch.tanh(x)**2


def CE_gradient(input, label):
    assert len(input.shape) == 2 and input.shape[0] == 1
    assert type(label) == torch.Tensor
    assert label.shape == (1,), f"{label.shape=}"
    assert label.dtype == torch.int64, f"{label.dtype=}"
    label = label.item()
    ans = torch.zeros(size=input.shape).to(input.device)
    ans[0, label] = -label/input[0, label]
    return ans
