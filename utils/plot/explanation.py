import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt


def rescale(tensor):
    """transforms to the range [0, 1]
    """
    tensor = tensor.type(torch.float32)
    if torch.min(tensor) == torch.max(tensor):
        return tensor, 0
    tmax = torch.max(tensor)
    tmin = torch.min(tensor)
    def affine_transform(x):
        return (x - tmin) / (tmax - tmin)
    return affine_transform(tensor), affine_transform(0).item()


def imshow_tensor(fig=None, ax=None, tensor=None, title=None, show_colorbar=False, show_origin=False):
    tensor, new_origin = rescale(tensor)
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
        im = ax.imshow(tensor, cmap=matplotlib.colormaps['viridis'])
    else:
        im = plt.imshow(tensor, cmap=matplotlib.colormaps['viridis'])
    if title is not None:
        ax.set_title(title)
    if show_colorbar:
        assert fig
        cb = fig.colorbar(im)
    if show_origin:
        assert show_colorbar
        cb.ax.plot([0, 1], [new_origin]*2, 'r')


def rgb2hsv(input, epsilon=1e-10):
    """
    https://linuxtut.com/en/20819a90872275811439/
    """
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)


def hsv2rgb(input):
    """
    https://linuxtut.com/en/20819a90872275811439/
    """
    assert(input.shape[1] == 3)

    h, s, v = input[:, 0], input[:, 1], input[:, 2]
    h_ = (h - torch.floor(h / 360) * 360) / 60
    c = s * v
    x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))

    zero = torch.zeros_like(c)
    y = torch.stack((
        torch.stack((c, x, zero), dim=1),
        torch.stack((x, c, zero), dim=1),
        torch.stack((zero, c, x), dim=1),
        torch.stack((zero, x, c), dim=1),
        torch.stack((x, zero, c), dim=1),
        torch.stack((c, zero, x), dim=1),
    ), dim=0)

    index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
    rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze(0)
    return rgb


def overlay_heatmap(image, heatmap):
    heatmap = heatmap.type(torch.float32)
    heatmap = torchvision.transforms.Resize(image.shape[2:4])(heatmap)
    assert heatmap.shape == (1, 1) + image.shape[2:4], f"{heatmap.shape=}, {image.shape=}"
    heatmap, _ = rescale(heatmap)
    overlay = rgb2hsv(image)
    overlay[:, 1, :, :] = torch.clamp(heatmap*10, min=0, max=1)
    overlay = hsv2rgb(overlay)
    return overlay
