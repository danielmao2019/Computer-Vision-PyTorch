import torch


class PermutedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, num_classes, seed=None):
        super(PermutedCrossEntropyLoss, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.permutation = torch.randperm(num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, inputs, labels):
        assert len(labels.shape) == 2, f"{labels.shape=}"
        labels = labels[:, self.permutation]
        return self.criterion(inputs, labels)

    def __str__(self):
        return f"PermutedCrossEntropyLoss({self.permutation.tolist()})."
