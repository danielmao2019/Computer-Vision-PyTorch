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
    for _ in range(2):
        # mimic using 2 epochs
        class_count = torch.zeros(size=(dataset.NUM_CLASSES,), dtype=torch.int64)
        for image, label in dataloader:
            assert len(image.shape) == 4
            assert len(label.shape) == 1
            class_count[label.item()] += 1
        assert torch.equal(class_count, dataset.DISTRIBUTION[purpose]), f"{class_count=}"
