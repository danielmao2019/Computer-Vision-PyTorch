import pytest
import torch
import utils


@pytest.mark.parametrize('tensor', [
    pytest.param(torch.rand(2)),
    pytest.param(torch.rand(2, 8)),
    pytest.param(torch.rand(2, 8, 1, 28, 28)),
    pytest.param(torch.rand(2, 8, 3, 224, 224)),
])
def test_p_inner(tensor):
    expected = torch.zeros(size=(len(tensor), len(tensor)), dtype=torch.float32)
    for i in range(len(tensor)):
        for j in range(len(tensor)):
            expected[i, j] = torch.sum(tensor[i]*tensor[j])
    output = utils.tensor_ops.pairwise_inner_product(tensor)
    assert torch.allclose(output, expected), f"{output=}, {expected=}"
