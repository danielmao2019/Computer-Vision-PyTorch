from data.datasets import Dataset
import torch


class OverfitDataset(Dataset):

    TASK_OPTIONS = ['image_classification']

    def __init__(self, task):
        self.image = torch.rand(1, 3, 224, 224)
        if task not in self.TASK_OPTIONS:
            raise NotImplementedError()
        if task == 'image_classification':
            self.label = 0
        self.core = [(self.image, self.label)]

    def __getitem__(self, idx):
        return self.core[idx]
