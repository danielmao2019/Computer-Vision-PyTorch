"""
Limitations:
* only supports batch size = 1.
"""
import torch


def tanh_gradient(x):
    return 1 - torch.tanh(x)**2


def CE_gradient(y_pred, y_true, mapping=None):
    assert len(y_pred.shape) == 2 and y_pred.shape[0] == 1, f"{y_pred.shape=}"
    assert type(y_true) == torch.Tensor
    assert y_true.shape == (1,), f"{y_true.shape=}"
    assert y_true.dtype == torch.int64, f"{y_true.dtype=}"
    if mapping is not None:
        mapping = mapping.to(y_true.device)
        y_true = mapping[y_true]
    gradient = torch.exp(y_pred)
    gradient = gradient / torch.sum(gradient)
    gradient[:, y_true.item()] -= 1
    return gradient


def MSE_gradient(y_pred, y_true):
    assert len(y_pred.shape) == 2 and y_pred.shape[0] == 1
    assert y_true.shape == (1,), f"{y_true.shape=}"
    assert y_pred.dtype == torch.float32, f"{y_pred.shape=}"
    assert y_true.dtype == torch.int64, f"{y_true.dtype=}"
    ans = 2 * y_pred / len(y_pred)
    ans[0, y_true.item()] -= 2 / len(y_pred)
    return ans
