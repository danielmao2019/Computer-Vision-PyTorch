import pytest
import torch
import data


@pytest.mark.parametrize("task", [
    pytest.param('image_classification'),
])
def test_OverfitDataset(task):
    dataset = data.datasets.OverfitDataset(task=task)
    assert len(dataset) == 1
    example = next(iter(dataset))
    assert type(example) == tuple
    assert len(example) == 2
    image, label = example
    assert type(image) == torch.Tensor
    assert image.dtype == torch.float32
    assert type(label) == int
