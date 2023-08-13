import torch


class RandomVectorsDataset(torch.utils.data.Dataset):

    def __init__(self, dim, num_examples):
        super(RandomVectorsDataset, self).__init__()
        self.dim = dim
        self.num_examples = num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return torch.rand(size=(self.dim,)).type(torch.float32)
