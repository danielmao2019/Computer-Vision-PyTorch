import torch


class Acc(object):

    def __call__(self, inputs, labels):
        return torch.mean(
            torch.eq(torch.argmax(inputs, dim=1, keepdim=False), labels).type(torch.float32)
        )
