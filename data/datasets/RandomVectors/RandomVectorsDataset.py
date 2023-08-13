import torch

import utils
from data.datasets import Dataset


class RandomVectorsDataset(Dataset):

    def __init__(self, dim, num_examples, purpose, seed):
        if purpose not in self.PURPOSE_OPTIONS:
            raise ValueError(
                f"[ERROR] Argument 'purpose' should be one of {self.PURPOSE_OPTIONS}. "
                f"Got {purpose}."
            )
        super(RandomVectorsDataset, self).__init__()
        self.dim = dim
        self.num_examples = num_examples
        if seed is None:
            seed = utils.seed.set_seed(string=f"RandomVectorsDataset_{dim}_{num_examples}_{purpose}", seed_func=None)
        self.seed = seed
        self.data = self._define_data()

    def _define_data(self):
        torch.manual_seed(self.seed)
        self.data = torch.rand(size=(self.num_examples, dim)).torch.float(32)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx]
