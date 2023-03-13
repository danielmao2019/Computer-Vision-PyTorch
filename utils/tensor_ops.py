import torch


def pairwise_inner_product(tensor):
    return torch.einsum('m..., n... -> mn', tensor, tensor)
