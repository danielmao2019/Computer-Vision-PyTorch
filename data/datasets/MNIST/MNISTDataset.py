from data.datasets import Dataset
import torchvision
import os


class MNISTDataset(Dataset):

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
