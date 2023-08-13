import torch
import torchvision


class Dataloader(object):

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        num_workers=0,
        transforms=None,
        sampler=None,
    ):
        """
        Args:
            transforms (list).
        """
        if transforms is not None and type(transforms) != list:
            raise TypeError(
                f"Argument 'transforms' should be of type 'list'. "
                f"Got {type(transforms)=}."
            )
        self.core = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler,
        )
        self.transform = torchvision.transforms.Compose(transforms)
        self.sampler = sampler
        self.num_examples = len(dataset)
        self.batch_size = batch_size
        self.num_batches = len(self.core)

    def __len__(self):
        return len(self.core)

    def __iter__(self):
        self.iter_core = iter(self.core)
        return self
    
    def __next__(self):
        element = tuple(next(self.iter_core))
        element = self.transform(element)
        return element
