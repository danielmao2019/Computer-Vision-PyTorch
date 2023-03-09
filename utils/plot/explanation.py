import torch
import matplotlib
import matplotlib.pyplot as plt


def imshow_tensor(ax=None, tensor=None):
    assert tensor.dtype == torch.float32, f"{tensor.dtype=}"
    assert 0 <= torch.min(tensor) <= torch.max(tensor) <= 1, f"{torch.min(tensor)=}, {torch.max(tensor)=}"
    if len(tensor.shape) == 4:
        assert tensor.shape[0] == 1, f"{tensor.shape=}"
        tensor = tensor[0]
    assert len(tensor.shape) == 3, f"{tensor.shape=}"
    # tensor = torch.amax(tensor, dim=0)
    tensor = torch.mean(tensor, dim=0)
    assert len(tensor.shape) == 2, f"{tensor.shape=}"
    tensor = tensor.detach().cpu().numpy()
    if ax:
        ax.imshow(tensor, cmap=matplotlib.colormaps['viridis'])
    else:
        plt.imshow(tensor, cmap=matplotlib.colormaps['viridis'])
