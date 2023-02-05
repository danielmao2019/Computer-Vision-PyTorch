import torch
import torchvision
from data.datasets import Dataset
import os


class CIFARDataset(Dataset):

    NUM_CLASSES = None
    VERSION_OPTIONS = [10, 100]
    DISTRIBUTION = {
        10: {
            'training': torch.Tensor([]),
            'evaluation': torch.Tensor([]),
        },
        100: {
            'training': torch.Tensor([]),
            'evaluation': torch.Tensor([]),
        },
    }

    def __init__(self, version, purpose):
        super(CIFARDataset, self).__init__()
        if purpose not in self.PURPOSE_OPTIONS:
            raise ValueError(
                f"[ERROR] Argument 'purpose' should be one of {self.PURPOSE_OPTIONS}. "
                f"Got {purpose}."
            )
        self.purpose = purpose
        if version not in self.VERSION_OPTIONS:
            raise ValueError(
                f"[ERROR] Argument 'version' should be one of {self.VERSION_OPTIONS}. "
                f"Got {version}."
            )
        self.version = version
        if version == 10:
            root = os.path.join('data', 'datasets', 'CIFAR', 'downloads')
            download = not os.path.exists(os.path.join(root, 'CIFAR', 'raw'))
            self.core = torchvision.datasets.CIFAR100(root=root, train=purpose=='training', download=download)
        elif version == 100:
            pass
        else:
            raise RuntimeError(f"[ERROR] Argument {version=} not handled properly.")
        self.NUM_CLASSES = version
    
    def __getitem__(self, idx):
        image, label = self.core[idx]
        image = torchvision.transforms.ToTensor()(image)
        return image, label
