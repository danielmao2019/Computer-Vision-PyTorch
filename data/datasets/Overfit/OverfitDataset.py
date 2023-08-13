from data.datasets import Dataset
import torch
import data


class OverfitDataset(Dataset):

    def __init__(self, task, image_shape):
        self.image = torch.rand(image_shape)
        if task not in data.TASK_OPTIONS:
            raise NotImplementedError()
        if task == 'image_classification':
            self.label = torch.zeros(size=(), dtype=torch.int64)
        elif task == 'semantic_segmentation':
            self.label = torch.zeros(size=(image_shape[1:]), dtype=torch.float32)
        else:
            raise RuntimeError(f"Argument {task=} not handled properly.")
        self.core = [(self.image, self.label)]

    def get_item(self, idx):
        return self.core[idx]
