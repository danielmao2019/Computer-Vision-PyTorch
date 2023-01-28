import pytest
import torch
import models


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
