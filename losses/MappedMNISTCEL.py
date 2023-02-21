import torch


class MappedMNISTCEL(torch.nn.Module):

    def __init__(self, mapping=None, num_classes=None, seed=None):
        super(MappedMNISTCEL, self).__init__()
        if mapping is None:
            if seed is not None:
                torch.manual_seed(seed)
            mapping = torch.randperm(num_classes)
            # seed 0 should give [4, 1, 7, 5, 3, 9, 0, 8, 6, 2]
        elif mapping == 'circle':
            mapping = torch.Tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 1])
        elif mapping == 'horiz':
            mapping = torch.Tensor([0, 0, 1, 0, 1, 1, 0, 1, 0, 0])
        elif mapping == 'vert':
            mapping = torch.Tensor([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
        elif type(mapping) == str:
            raise ValueError(f"[ERROR] Argument {mapping=} not handled properly.")
        else:
            assert type(mapping) == torch.Tensor, f"{type(mapping)=}"
        self.mapping = mapping.type(torch.int64)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, inputs, labels):
        device = labels.device
        labels = torch.argmax(labels, dim=1, keepdims=False)
        labels = self.mapping[labels].to(device)
        return self.criterion(inputs, labels)

    def __str__(self):
        return f"MappedMNISTCEL({self.mapping.tolist()})."
