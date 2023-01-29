import torch


class GlobalAveragePooling2D(torch.nn.Module):

    def __init__(self):
        super(GlobalAveragePooling2D, self).__init__()

    def forward(self, x):
        assert len(x.shape) == 4, f"{len(x.shape)=}"
        x = torch.mean(x, dim=[2, 3], keepdim=False)
        assert len(x.shape) == 2, f"{len(x.shape)=}"
        return x
