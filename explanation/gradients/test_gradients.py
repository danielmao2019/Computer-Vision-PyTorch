import pytest
import torch
import explanation
import models
import losses


def test_CE_gradient():
    num_classes = 5
    for _ in range(10):
        y_pred = torch.softmax(torch.rand(size=(1, num_classes)), dim=1)
        y_true = torch.randint(low=0, high=num_classes, size=(1,)).type(torch.int64)
        mapping = torch.randperm(num_classes)
        gradient = explanation.gradients.CE_gradient(
            y_pred=y_pred, y_true=y_true, mapping=mapping,
        )
        y_pred.requires_grad_()
        criterion = losses.MappedMNISTCEL(mapping=mapping)
        loss = criterion(y_pred=y_pred, y_true=y_true)
        gradient_expected = torch.autograd.grad(outputs=loss, inputs=y_pred)
        assert type(gradient_expected) == tuple and len(gradient_expected) == 1
        assert torch.allclose(gradient, gradient_expected[0]), \
            f"{gradient.detach().numpy()=}, {gradient_expected[0].detach().numpy()=}"


def test_GradientModelInputs_Linear():
    in_features=3
    out_features=5
    model = torch.nn.Sequential(torch.nn.Linear(
        in_features=in_features, out_features=out_features,
    ))
    model.eval()
    inputs = torch.rand(size=(1, in_features)).requires_grad_()
    labels = torch.randint(low=0, high=out_features, size=(1,)).type(torch.int64)
    # computed gradient
    gmi = explanation.gradients.GradientModelInputs(model=model, layer_idx=0)
    outputs = gmi.update(inputs)
    gradient_computed = gmi(explanation.gradients.CE_gradient(y_pred=outputs, y_true=labels))
    # expected gradient
    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(input=outputs, target=labels)
    gradient_expected = torch.autograd.grad(outputs=loss, inputs=inputs, retain_graph=True)[0]
    # compare
    assert torch.allclose(gradient_computed, gradient_expected), \
        f"{gradient_computed.flatten().detach().numpy()=}\n{gradient_expected.flatten().detach().numpy()=}"


def test_GradientModelInputs_tanh():
    model = torch.nn.Sequential(torch.nn.Tanh())
    model.eval()
    inputs = torch.rand(size=(1, 3, 32, 32)).requires_grad_()
    gmi = explanation.gradients.GradientModelInputs(model=model, layer_idx=0)
    outputs = gmi.update(inputs)
    for i in range(outputs.shape[1]):
        for j in range(outputs.shape[2]):
            for k in range(outputs.shape[3]):
                print(f"{i=}, {j=}, {k=}")
                # computed gradient
                initial_gradient = torch.zeros(size=outputs.shape)
                initial_gradient[0, i, j, k] = 1
                gradient_computed = gmi(initial_gradient)
                # expected gradient
                loss = torch.sum(outputs * initial_gradient)
                gradient_expected = torch.autograd.grad(outputs=loss, inputs=inputs, retain_graph=True)[0]
                # compare
                assert torch.allclose(gradient_computed, gradient_expected), \
                    f"{gradient_computed.flatten().detach().numpy()=}\n{gradient_expected.flatten().detach().numpy()=}"


# TODO: add test for each type of layer individually


def test_GradientModelInputs_LeNet():
    in_features=3
    out_features=5
    model = models.LeNet(in_features=in_features, out_features=out_features)
    model.eval()
    inputs = torch.rand(size=(1, in_features, 32, 32)).requires_grad_()
    labels = torch.randint(low=0, high=out_features, size=(1,)).type(torch.int64)
    for layer_idx in [10]:
        # computed gradient
        gmi = explanation.gradients.GradientModelInputs(model=model, layer_idx=layer_idx)
        outputs = gmi.update(inputs)
        gradient_computed = gmi(explanation.gradients.CE_gradient(y_pred=outputs, y_true=labels))
        # expected gradient
        first_part = torch.nn.Sequential(*list(model.children())[:layer_idx])
        second_part = torch.nn.Sequential(*list(model.children())[layer_idx:])
        inter = first_part(inputs)
        final = second_part(inter)
        loss = torch.nn.CrossEntropyLoss()(input=final, target=labels)
        gradient_expected = torch.autograd.grad(outputs=loss, inputs=inter, retain_graph=True)[0]
        # compare
        assert torch.allclose(gradient_computed, gradient_expected), \
            f"{gradient_computed.flatten().detach().numpy()=}\n{gradient_expected.flatten().detach().numpy()=}"
