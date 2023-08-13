import pytest
import torch
import metrics


@pytest.mark.parametrize("reduction, y_pred, y_true, expected", [
    pytest.param(
        None,
        torch.Tensor([
            [0.8895, 0.7085, 0.7907, 0.1265, 0.8119, 0.0256, 0.8282, 0.1713, 0.5923, 0.6070],
            [0.3946, 0.0786, 0.7835, 0.7832, 0.7004, 0.4610, 0.7581, 0.6049, 0.8544, 0.2614],
            [0.0396, 0.0108, 0.0945, 0.2340, 0.1516, 0.3440, 0.6353, 0.6482, 0.8100, 0.9654],
        ]),
        torch.Tensor([0, 6, 3]),
        torch.Tensor([True, False, False]),
    ),
    pytest.param(
        "sum",
        torch.Tensor([
            [0.8895, 0.7085, 0.7907, 0.1265, 0.8119, 0.0256, 0.8282, 0.1713, 0.5923, 0.6070],
            [0.3946, 0.0786, 0.7835, 0.7832, 0.7004, 0.4610, 0.7581, 0.6049, 0.8544, 0.2614],
            [0.0396, 0.0108, 0.0945, 0.2340, 0.1516, 0.3440, 0.6353, 0.6482, 0.8100, 0.9654],
        ]),
        torch.Tensor([0, 6, 3]),
        torch.tensor(1),
    ),
])
def test_acc(reduction, y_pred, y_true, expected):
    acc = metrics.Acc(reduction=reduction)(y_pred=y_pred, y_true=y_true)
    assert torch.equal(acc, expected), f"{acc=}, {expected=}"
