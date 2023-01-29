import pytest
import torch
import data
import models
import training


@pytest.mark.parametrize("task, input_shape, output_shape", [
    pytest.param('image_classification', (1, 224, 224), (10,)),
    pytest.param('image_classification', (3, 224, 224), (10,)),
    # pytest.param('semantic_segmentation', (1, 224, 224), (10, 224, 224)),
    # pytest.param('semantic_segmentation', (3, 224, 224), (10, 224, 224)),
])
def test_forward_pass(task, input_shape, output_shape):
    model = models.ExperimentalModel(
        task=task, in_features=input_shape[0], out_features=output_shape[0],
    )
    batch_size = 1
    fake_input = torch.zeros(size=(batch_size,)+input_shape)
    fake_output = model(fake_input)
    assert fake_output.shape == (batch_size,)+output_shape, f"{fake_output.shape=}, {output_shape=}"


@pytest.mark.parametrize("task", [
    pytest.param('image_classification'),
])
def test_overfit(task):
    model = models.ExperimentalModel(
        task=task, in_features=3, out_features=10,
    )
    dataset = data.datasets.OverfitDataset(task=task)
    dataloader = torch.utils.data.DataLoader(dataset)
    criterion = torch.nn.CrossEntropyLoss()
    specs = {
        'tag': 'ExperimentalModel',
        'model': model,
        'dataloader': dataloader,
        'epochs': 100,
        'criterion': criterion,
        'optimizer': torch.optim.SGD(model.parameters(), lr=1.0e-03),
        'save_model': False,
        'load_model': None,
    }
    training.train_model(specs=specs)
    model.eval()
    for image, label in dataloader:
        error = criterion(input=model(image), target=label)
        assert error <= 1.0e-03
