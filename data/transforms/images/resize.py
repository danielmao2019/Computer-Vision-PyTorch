import torch
import torchvision
import data


class Resize(torch.nn.Module):

    def __init__(self, new_size, task, label_transform):
        super(Resize, self).__init__()
        if type(new_size) == int:
            new_size = (new_size,) * 2
        if type(new_size) != tuple:
            raise TypeError(
                f"[ERROR] Argument 'new_size' must be of type 'tuple'. "
                f"Got {type(new_size)}."
            )
        if len(new_size) != 2:
            raise ValueError(
                f"[ERROR] Argument 'new_size' must be a tuple of length 2. "
                f"Got length {len(new_size)}."
            )
        if type(task) != str:
            raise TypeError(
                f"[ERROR] Argument 'task' must be of type 'str'. "
                f"Got {type(task)}."
            )
        self.new_size = new_size
        self.task = task
        self.image_transform = torchvision.transforms.Resize(size=new_size)
        self.label_transform = self._set_label_transform(label_transform)

    def _set_label_transform(self, label_transform):
        assert (self.task is None) ^ (label_transform is None), f"[ERROR] Exactly one of the two arguments 'task' and 'label_transform' should be specified."
        if label_transform is not None:
            return label_transform
        if self.task not in data.TASK_OPTIONS:
            raise ValueError(
                f"Argument 'task' should be one of {data.TASK_OPTIONS}. "
                f"Got {self.task=}."
            )
        if self.task == 'image_classification':
            self.label_transform = lambda x: x
        elif self.task == 'semantic_segmentation':
            self.label_transform = torchvision.transforms.Resize(size=self.new_size, method='nearest')
        else:
            raise ValueError(f"[ERROR] Argument {self.task=} not handled properly.")
        return self

    def __call__(self, element):
        image, label = element
        image = self.image_transform(image)
        label = self.label_transform(label)
        return image, label
