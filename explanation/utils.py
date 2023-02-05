import torch


def imshow_tensor(ax, tensor):
    if len(tensor.shape) == 4:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    assert len(tensor.shape) == 3
    tensor = torch.permute(tensor, dims=(1, 2, 0))
    if tensor.shape[-1] != 1:
        tensor = torch.unsqueeze(tensor[:, :, 0], dim=2)
    tensor = tensor.detach().cpu().numpy()
    ax.imshow(tensor, cmap='gray')
