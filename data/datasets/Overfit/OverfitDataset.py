from data.datasets import Dataset
import torch


class OverfitDataset(Dataset):

    TASK_OPTIONS = ['image_classification', 'semantic_segmentation']

    def __init__(self, task, image_shape, label_shape):
        self.image = torch.rand(image_shape)
        self.label = torch.zeros(size=label_shape)
        if task not in self.TASK_OPTIONS:
            raise NotImplementedError()
        if task in ['image_classification', 'semantic_segmentation']:
            self.label = self.label.type(torch.int64)
        else:
            raise RuntimeError(f"Argument {task=} not handled properly.")
        self.core = [(self.image, self.label)]

    def __getitem__(self, idx):
        return self.core[idx]
