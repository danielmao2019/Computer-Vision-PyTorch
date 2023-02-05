import torch
import torchvision


class Dataloader(object):

    def __init__(self, task, dataset, transforms=None, batch_size=1, shuffle=None, num_workers=0, sampler=None):
        """
        Parameters:
            transforms (list).
        """
        if transforms is not None and type(transforms) != list:
            raise TypeError(
                f"Argument 'transforms' should be of type 'list'. "
                f"Got {type(transforms)=}."
            )
        #TODO: enable this line
        # dataset = dataset.set_task(task)
        self.core = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler,
        )
        self.iter_core = iter(self.core)
        self.batch_size = batch_size
        self.sampler = sampler
        for t in transforms:
            t = t.set_task(task)
        self.transform = torchvision.transforms.Compose(transforms)

    def __len__(self):
        return len(self.core)

    def __iter__(self):
        return self
    
    def __next__(self):
        element = tuple(next(self.iter_core))
        element = self.transform(element)
        return element
