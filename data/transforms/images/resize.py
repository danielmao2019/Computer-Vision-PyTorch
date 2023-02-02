import torch
import torchvision


class Resize(object):

    def __init__(self, task, new_size):
        if type(new_size) != tuple:
            raise TypeError(
                f"[ERROR] Argument 'new_size' should be of type 'tuple'. "
                "Got {type(new_size)}."
            )
        if len(new_size) != 2:
            raise ValueError(
                f"[ERROR] Argument 'new_size' should be a tuple of length 2. "
                "Got length {len(new_size)}."
            )
        self.image_transform = torchvision.transforms.Resize(size=new_size)

    def __call__(self, element):
        image, label = element
        image = self.image_transform(image)
        return image, label
