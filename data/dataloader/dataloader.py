import torch


class Dataloader(object):

    def __init__(self, dataset, batch_size=1, shuffle=None, num_workers=0, sampler=None, transform=None):
        self.core = iter(torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler,
        ))
        self.batch_size = batch_size
        self.sampler = sampler
        self.transform = transform if transform else lambda x: x

    def __iter__(self):
        return self
    
    def __next__(self):
        element = next(self.core)
        element = self.transform(element)
        return element
