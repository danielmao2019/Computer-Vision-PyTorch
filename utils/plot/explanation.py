import torch
import matplotlib
import matplotlib.pyplot as plt


def rescale(tensor):
    """transforms to the range [0, 1]
    """
    tensor -= torch.min(tensor)
    assert torch.min(tensor) == 0, f"{torch.min(tensor)=}"
    assert torch.max(tensor) >= 0, f"{torch.max(tensor)=}"
    if torch.max(tensor) != 0:
        tensor /= torch.max(tensor)
    return tensor


def imshow_tensor(ax=None, tensor=None):
    tensor = tensor.to(torch.float32)
    tensor = rescale(tensor)
    if len(tensor.shape) == 4:
        assert tensor.shape[0] == 1, f"{tensor.shape=}"
        tensor = tensor[0]
    assert len(tensor.shape) == 3, f"{tensor.shape=}"
    # tensor = torch.amax(tensor, dim=0)
    tensor = torch.mean(tensor, dim=0)
    assert len(tensor.shape) == 2, f"{tensor.shape=}"
    tensor = tensor.detach().cpu().numpy()
    if ax:
        return ax.imshow(tensor, cmap=matplotlib.colormaps['viridis'])
    else:
        return plt.imshow(tensor, cmap=matplotlib.colormaps['viridis'])
