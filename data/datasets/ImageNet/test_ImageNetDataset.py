import pytest
import torch
import torchvision
import random
import engine


@pytest.mark.parametrize("split", [
    pytest.param("train"),
    # pytest.param("test"),
])
def test_ImageNetDataset(split):
    dataset = engine.datasets.source.ImageNetDataset(
        split=split, transform=torchvision.transforms.ToTensor(),
    )
    example = dataset[random.choice(range(len(dataset)))]
    assert type(example) == tuple
    assert len(example) == 2
    image, label = example
    assert type(image) == torch.Tensor
    assert len(image.shape) == 3
    assert image.dtype == torch.float32
    assert 0 <= torch.min(image) <= torch.max(image) <= 1
    assert type(label) == int
