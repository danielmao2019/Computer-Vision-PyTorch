import torch
import torchvision
import data


class Resize(torch.nn.Module):

    def __init__(self, new_size):
        super(Resize, self).__init__()
        if type(new_size) != tuple:
            raise TypeError(
                f"[ERROR] Argument 'new_size' should be of type 'tuple'. "
                f"Got {type(new_size)}."
            )
        if len(new_size) != 2:
            raise ValueError(
                f"[ERROR] Argument 'new_size' should be a tuple of length 2. "
                f"Got length {len(new_size)}."
            )
        self.new_size = new_size
        self.image_transform = torchvision.transforms.Resize(size=new_size)
        self.label_transform = None

    def set_task(self, task):
        if task not in data.TASK_OPTIONS:
            raise ValueError(
                f"Argument 'task' should be one of {data.TASK_OPTIONS}. "
                f"Got {task=}."
            )
        if task == 'image_classification':
            self.label_transform = lambda x: x
        elif task == 'semantic_segmentation':
            self.label_transform = torchvision.transforms.Resize(size=self.new_size, method='nearest')
        else:
            raise ValueError(f"Argument 'task' not handled properly.")
        return self

    def __call__(self, element):
        image, label = element
        image = self.image_transform(image)
        label = self.label_transform(label)
        return image, label
