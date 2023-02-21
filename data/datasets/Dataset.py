import numpy as np
import torch
import copy


class Dataset(object):

    PURPOSE_OPTIONS = ['training', 'evaluation']

    def __init__(self):
        pass

    def __len__(self):
        return len(self.core)

    def subset(self, indices):
        assert type(indices) in [list, np.ndarray, torch.Tensor], f"{type(indices)=}"
        indices = torch.Tensor(indices).type(torch.int64)
        assert len(indices.shape) == 1, f"{indices.shape=}"
        assert indices.dtype == torch.int64, f"{indices.dtype=}"
        new_dataset = copy.deepcopy(self)
        new_dataset.core = torch.utils.data.Subset(dataset=new_dataset.core, indices=indices)
        return new_dataset
