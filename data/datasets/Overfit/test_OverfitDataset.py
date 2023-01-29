import pytest
import torch
import data


@pytest.mark.parametrize("task, input_shape, output_shape", [
    pytest.param('image_classification', (1, 224, 224), (10,)),
    pytest.param('image_classification', (3, 224, 224), (10,)),
])
def test_OverfitDataset(task, input_shape, output_shape):
    dataset = data.datasets.OverfitDataset(task=task, image_shape=input_shape, label_shape=output_shape)
    assert len(dataset) == 1
    example = next(iter(dataset))
    assert type(example) == tuple
    assert len(example) == 2
    image, label = example
    # check image
    assert type(image) == torch.Tensor
    assert image.shape == input_shape
    assert image.dtype == torch.float32
    # check label
    assert type(label) == torch.Tensor
    assert label.shape == output_shape
