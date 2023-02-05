import pytest
import torch
import data


@pytest.mark.parametrize("purpose", [
    pytest.param('training'),
    pytest.param('evaluation'),
])
def test_MNIST_dataloader(purpose):
    dataset = data.datasets.MNISTDataset(purpose=purpose)
    dataloader = data.Dataloader(task='image_classification', dataset=dataset, transforms=[], batch_size=1, shuffle=True)
    class_count = [0] * dataset.NUM_CLASSES
    for image, label in dataloader:
        assert len(image.shape) == 4
        assert len(label.shape) == 1
        class_count[label.item()] += 1
    assert torch.equal(torch.Tensor(class_count), dataset.DISTRIBUTION[purpose]), f"{class_count=}"
