import torch
import matplotlib
import matplotlib.pyplot as plt


def rescale(tensor):
    """transforms to the range [0, 1]
    """
    tensor = tensor.type(torch.float32)
    tensor -= torch.min(tensor)
    assert torch.min(tensor) == 0, f"{torch.min(tensor)=}"
    assert torch.max(tensor) >= 0, f"{torch.max(tensor)=}"
    if torch.max(tensor) != 0:
        tensor /= torch.max(tensor)
    else:
        assert torch.unique(tensor).tolist() == [0]
    return tensor


def imshow_tensor(ax=None, tensor=None):
    tensor = rescale(tensor)
    if len(tensor.shape) == 4:
        assert tensor.shape[0] == 1, f"{tensor.shape=}"
        tensor = tensor[0]
    assert len(tensor.shape) == 3, f"{tensor.shape=}"
    if tensor.shape[0] == 3:
        tensor = torch.permute(tensor, dims=[1, 2, 0])
    else:
        tensor = torch.mean(tensor, dim=0)
        assert len(tensor.shape) == 2, f"{tensor.shape=}"
    tensor = tensor.detach().cpu().numpy()
    if ax:
        return ax.imshow(tensor, cmap=matplotlib.colormaps['viridis'])
    else:
        return plt.imshow(tensor, cmap=matplotlib.colormaps['viridis'])
