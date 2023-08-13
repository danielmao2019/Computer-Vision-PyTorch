import torch


def onehot_encode(shape, labels):
    assert len(shape) == 2, f"{shape=}"
    assert type(labels) == torch.Tensor, f"{type(labels)=}"
    assert len(labels.shape) == 1, f"{labels.shape=}"
    onehot_labels = torch.tile(torch.arange(shape[1]), dims=(shape[0], 1)).to(labels.device)
    onehot_labels = (onehot_labels == torch.unsqueeze(labels, dim=1)).type(torch.float32)
    return onehot_labels
