import pytest
import torch
import data
import random


@pytest.mark.parametrize("version, purpose", [
    pytest.param(10, 'training'),
    pytest.param(10, 'evaluation'),
    pytest.param(100, 'training'),
    pytest.param(100, 'evaluation'),
])
def test_data_structure(version, purpose):
    dataset = data.datasets.CIFARDataset(version=version, purpose=purpose)
    assert isinstance(dataset.core, torch.utils.data.Dataset), f"{type(dataset.core)=}"
    assert len(dataset) == 50000 if purpose == 'training' else 10000
    example = dataset[random.choice(range(len(dataset)))]
    assert type(example) == tuple, f"{type(example)=}"
    assert len(example) == 2, f"{len(example)=}"
    image, label = example
    # check image
    assert type(image) == torch.Tensor, f"{type(image)=}"
    assert image.shape == (3, 32, 32), f"{image.shape=}"
    assert image.dtype == torch.float32, f"{image.dtype=}"
    # check label
    assert type(label) == int, f"{type(label)=}"


@pytest.mark.parametrize("version, purpose", [
    pytest.param(10, 'training'),
    pytest.param(10, 'evaluation'),
    pytest.param(100, 'training'),
    pytest.param(100, 'evaluation'),
])
def test_class_distribution(version, purpose):
    dataset = data.datasets.CIFARDataset(version=version, purpose=purpose)
    class_count = torch.zeros(size=(dataset.NUM_CLASSES,), dtype=torch.int64)
    for _, label in dataset:
        class_count[label] += 1
    assert torch.equal(class_count, dataset.DISTRIBUTION[dataset.version][dataset.purpose]), f"{class_count=}"
