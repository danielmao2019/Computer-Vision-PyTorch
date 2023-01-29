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
    # check image
    assert type(image) == torch.Tensor
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    # check label
    assert type(label) == int
