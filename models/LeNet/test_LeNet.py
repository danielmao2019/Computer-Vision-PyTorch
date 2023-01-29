import pytest
import torch
import data
import models
import training


@pytest.mark.parametrize("input_shape, output_shape", [
    pytest.param((1, 32, 32), (10,)),
    pytest.param((3, 32, 32), (10,)),
])
def test_forward_pass(input_shape, output_shape):
    model = models.LeNet(in_features=input_shape[0], out_features=output_shape[0])
    batch_size = 1
    fake_input = torch.zeros(size=(batch_size,)+input_shape)
    fake_output = model(fake_input)
    assert fake_output.shape == (batch_size,)+output_shape, f"{fake_output.shape=}, {output_shape=}"


@pytest.mark.parametrize("input_shape, output_shape", [
    pytest.param((1, 32, 32), (10,)),
    # pytest.param((3, 32, 32), (10,)),
])
def test_overfit(input_shape, output_shape):
    model = models.LeNet(in_features=input_shape[0], out_features=output_shape[0])
    dataset = data.datasets.OverfitDataset(task='image_classification', image_shape=input_shape, label_shape=())
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False,
    )
    training.train_model_minimal(
        model=model, dataloader=dataloader, epochs=100, lr=1.0e-01,
    )
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss = []
    for image, label in dataloader:
        loss.append(criterion(input=model(image), target=label).item())
        # assert False, f"{model(image)=}, {label=}"
    # assert all(loss[i] < 1.0e-03 for i in range(len(loss))), f"{loss=}"
