import pytest
import torch
import data
import random


@pytest.mark.parametrize("purpose", [
    pytest.param('training'),
    pytest.param('evaluation'),
])
def test_data_structure(purpose):
    dataset = data.datasets.MNISTDataset(purpose=purpose)
    assert isinstance(dataset.core, torch.utils.data.Dataset), f"{type(dataset.core)=}"
    assert len(dataset) == 60000 if purpose == 'training' else 10000, f"{len(dataset)=}"
    example = dataset[random.choice(range(len(dataset)))]
    assert type(example) == tuple, f"{type(example)=}"
    assert len(example) == 2, f"{len(example)=}"
    image, label = example
    # check image
    assert type(image) == torch.Tensor, f"{type(image)=}"
    assert image.shape == (1, 28, 28), f"{image.shape=}"
    assert image.dtype == torch.float32, f"{image.dtype=}"
    # check label
    assert type(label) == int, f"{type(label)=}"


@pytest.mark.parametrize("purpose", [
    pytest.param('training'),
    pytest.param('evaluation'),
])
def test_class_count(purpose):
    dataset = data.datasets.MNISTDataset(purpose=purpose)
    class_count = [0] * dataset.NUM_CLASSES
    for _, label in dataset:
        class_count[label] += 1
    assert torch.equal(torch.Tensor(class_count), dataset.DISTRIBUTION[purpose]), f"{class_count=}"
