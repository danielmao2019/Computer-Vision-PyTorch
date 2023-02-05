import pytest
import torch
import models


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@pytest.mark.parametrize("input_shape, output_shape", [
    pytest.param((3, 224, 224), (10,)),
    pytest.param((1, 224, 224), (10,)),
])
def test_forward_pass(input_shape, output_shape):
    model = models.AlexNet(in_features=input_shape[0], out_features=output_shape[0])
    batch_size = 1
    fake_input = torch.zeros(size=(batch_size,)+input_shape, dtype=torch.float32)
    fake_output = model(fake_input)
    assert fake_output.shape == (batch_size,)+output_shape, f"{fake_output.shape=}, {output_shape=}"
