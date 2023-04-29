import torch


class MappedMNISTCEL(torch.nn.Module):

    def __init__(self, mapping=None, num_classes=None, seed=None):
        super(MappedMNISTCEL, self).__init__()
        if mapping is None:
            assert num_classes is not None
            mapping = torch.arange(num_classes)
        elif mapping == 'random':
            assert num_classes is not None
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
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, inputs, labels):
        """
        Args:
            inputs (torch.Tensor): 4D tensor of dtype torch.float32.
            labels (torch.Tensor): 2D tensor of dtype torch.int64.
        Returns:
            loss (torch.Tensor): 0D tensor of dtype torch.float32.
        """
        assert inputs.device == labels.device
        self.mapping = self.mapping.to(labels.device)
        labels = self.mapping[labels]
        loss = self.criterion(inputs, labels)
        assert type(loss) == torch.Tensor
        assert len(loss.shape) == 0
        return loss

    def __str__(self):
        return f"MappedMNISTCEL({self.mapping.tolist()})."
