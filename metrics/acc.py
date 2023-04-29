import torch


class Acc(object):

    def __init__(self, reduction=None):
        if reduction is not None and type(reduction) != str:
            raise TypeError(
                f"[ERROR] Argument \"reduction\" must be None type or a string object. "
                "Got {type(reduction)}."
            )
        if reduction is None:
            self.reduce = lambda x: x
        elif reduction == "mean":
            self.reduce = lambda x: torch.mean(x)
        elif reduction == "sum":
            self.reduce = lambda x: torch.sum(x)
        else:
            raise ValueError(
                f"[ERROR] Argument \"reduction\" not handled properly."
            )

    def __call__(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): model outputs (pre- or post-softmax).
            y_true (torch.Tensor): ground-truth probability distribution (one-hot).
        """
        assert len(y_pred.shape) == 2
        assert len(y_true.shape) == 1
        assert y_pred.shape[0] == y_true.shape[0]
        y_pred = torch.argmax(y_pred, dim=1, keepdim=False)
        acc = self.reduce(torch.eq(y_pred, y_true))
        return acc
