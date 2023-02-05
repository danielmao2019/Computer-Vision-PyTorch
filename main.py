import torch
import torchvision

import data
import models
import training
import evaluation
import explanation

import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.LeNet(in_features=1, out_features=10)

##################################################
# training
##################################################

train_dataset = data.datasets.MNISTDataset(purpose='training')
train_dataloader = data.Dataloader(
    task='image_classification', dataset=train_dataset,
    batch_size=8, shuffle=True, transforms=[
        data.transforms.Resize(new_size=(32, 32)),
])
train_specs = {
    'tag': 'LeNet',
    'model': model,
    'dataloader': train_dataloader,
    'epochs': 100,
    'criterion': torch.nn.CrossEntropyLoss(),
    'optimizer': torch.optim.SGD(model.parameters(), lr=1.0e-02, momentum=0.9),
    'save_model': True,
    'load_model': None#"checkpoint_010.pt",
}
training.train_model(
    tag=train_specs['tag'], model=train_specs['model'], dataloader=train_specs['dataloader'], epochs=train_specs['epochs'],
    criterion=train_specs['criterion'], optimizer=train_specs['optimizer'],
    save_model=train_specs['save_model'], load_model=train_specs['load_model'], device=device,
)

##################################################
# evaluation
##################################################

eval_dataset = data.datasets.MNISTDataset(purpose='evaluation')
eval_dataloader = data.Dataloader(
    task='image_classification', dataset=eval_dataset,
    batch_size=1, shuffle=True, transforms=[
        data.transforms.Resize(new_size=(32, 32)),
])
# acc = lambda output, label: torch.equal(torch.argmax(output, dim=1, keepdim=False), label)
# scores = evaluation.eval_model(
#     model=model, dataloader=eval_dataloader, metrics=[acc], device=device,
# )
# print(scores)
model.eval()
for _ in range(3):
    image, label = next(iter(eval_dataloader))
    gradient_tensor = explanation.compute_gradients(
        model=model, image=image, label=label, depth=None,
    )
    print(f"{gradient_tensor.shape=}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    explanation.utils.imshow_tensor(ax1, gradient_tensor)
    ax1.set_title("Gradient Map")
    explanation.utils.imshow_tensor(ax2, image)
    ax2.set_title("Original Image")
    plt.show()