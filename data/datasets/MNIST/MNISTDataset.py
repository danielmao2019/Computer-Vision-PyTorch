import torch
import torchvision
from data.datasets import Dataset
import os


class MNISTDataset(Dataset):

    NUM_CLASSES = 10
    DISTRIBUTION = {
        'training': torch.Tensor([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]).type(torch.int64),
        'evaluation': torch.Tensor([980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]).type(torch.int64),
    }

    def __init__(self, purpose):
        if purpose not in self.PURPOSE_OPTIONS:
            raise ValueError(
                f"[ERROR] Argument 'purpose' should be one of {self.PURPOSE_OPTIONS}. "
                f"Got {purpose}."
            )
        root = os.path.join('data', 'datasets', 'MNIST', 'downloads')
        download = not os.path.exists(os.path.join(root, 'MNIST', 'raw'))
        self.core = torchvision.datasets.MNIST(root=root, train=purpose=='training', download=download)

    def __getitem__(self, idx):
        image, label = self.core[idx]
        image = torchvision.transforms.ToTensor()(image)
        return image, label
