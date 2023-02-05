import pytest
import torch
import data


@pytest.mark.parametrize("task, image_shape, label_shape", [
    pytest.param('image_classification', (1, 224, 224), ()),
    pytest.param('image_classification', (3, 224, 224), ()),
])
def test_OverfitDataset(task, image_shape, label_shape):
    dataset = data.datasets.OverfitDataset(task=task, image_shape=image_shape)
    assert len(dataset) == 1
    example = next(iter(dataset))
    assert type(example) == tuple
    assert len(example) == 2
    image, label = example
    # check image
    assert type(image) == torch.Tensor
    assert image.shape == image_shape
    assert image.dtype == torch.float32
    # check label
    assert type(label) == torch.Tensor
    assert label.shape == label_shape
