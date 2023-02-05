import pytest
import torch
import data
import models
import training


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


@pytest.mark.parametrize("task, input_shape, output_shape", [
    pytest.param('image_classification', (3, 224, 224), (10,)),
])
def test_overfit(task, input_shape, output_shape):
    model = models.ExperimentalModel(
        task=task, in_features=input_shape[0], out_features=output_shape[0],
    )
    dataset = data.datasets.OverfitDataset(task=task, image_shape=input_shape)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False,
    )
    training.train_model_minimal(
        model=model, dataloader=dataloader, epochs=100, lr=1.0e-01, device=device,
    )
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss = []
    for element in dataloader:
        image, label = element[0].to(device), element[1].to(device)
        loss.append(criterion(input=model(image), target=label).item())
    #TODO: enable this line
    # assert all(loss[i] < 1.0e-03 for i in range(len(loss))), f"{loss=}"
