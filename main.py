import torch
import torchvision

import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import data
import models
import losses
import metrics
import training
import evaluation
import explanation
import utils


"""
* what do we do as of network architecture (additional layer)
* which samples (spatial location) are conflicting
"""


criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    ], weights=[3, 1],
)
metric = metrics.Acc()

##################################################
# model
##################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.LeNet(in_features=1, out_features=10)
model.to(device)

##################################################
# datasets
##################################################

train_dataset = data.datasets.MNISTDataset(purpose='training')
train_dataloader = data.Dataloader(
    task='image_classification', dataset=train_dataset,
    batch_size=8, shuffle=True, transforms=[
        data.transforms.Resize(new_size=(32, 32)),
    ])
eval_dataset = data.datasets.MNISTDataset(purpose='evaluation')
eval_dataloader = data.Dataloader(
    task='image_classification', dataset=eval_dataset,
    batch_size=1, shuffle=True, transforms=[
        data.transforms.Resize(new_size=(32, 32)),
    ])

##################################################
# training
##################################################

train_specs = {
    'tag': 'LeNet_MNIST_Multi_6',
    'epochs': 200,
    'save_model': True,
    'load_model': None,#"checkpoint_100.pt",
    'model': model,
    'train_dataloader': train_dataloader,
    'eval_dataloader': eval_dataloader,
    'criterion': criterion,
    'optimizer': torch.optim.SGD(model.parameters(), lr=1.0e-03, momentum=0.9),
    'metric': metric,
}
training.train_model(
    tag=train_specs['tag'], model=train_specs['model'], epochs=train_specs['epochs'],
    train_dataloader=train_specs['train_dataloader'], eval_dataloader=train_specs['eval_dataloader'],
    criterion=train_specs['criterion'], optimizer=train_specs['optimizer'], metric=train_specs['metric'],
    save_model=train_specs['save_model'], load_model=train_specs['load_model'],
)


num_examples = 1
for idx in range(num_examples):
    print(f"{idx=}")
    image, label = next(iter(eval_dataloader))
    image, label = image.to(device), label.to(device)
    output = model(image)
    print(f"{label=}")
    print(f"{output=}")
    print(f"{criterion(output, label)=}")

##################################################
# VOG
##################################################

# scores = evaluation.eval_model(
#     model=model, dataloader=eval_dataloader, metrics=[metric],
# )
# print(scores)


# def rescale(tensor):
#     """transforms to the range [0, 1]
#     """
#     tensor -= torch.min(tensor)
#     assert torch.min(tensor) == 0, f"{torch.min(tensor)=}"
#     assert torch.max(tensor) >= 0, f"{torch.max(tensor)=}"
#     if torch.max(tensor) != 0:
#         tensor /= torch.max(tensor)
#     return tensor


# model.eval()
# num_examples = 100
# count = [0] * eval_dataset.NUM_CLASSES
# for idx in range(num_examples):
#     print(f"{idx=}")
#     fig, axs = plt.subplots(nrows=1, ncols=3)
#     image, label = next(iter(eval_dataloader))
#     image, label = image.to(device), label.to(device)
#     gradient_tensor = explanation.compute_gradients(
#         model=model, image=image, label=label, depth=None,
#     )
#     utils.explanation.imshow_tensor(ax=axs[0], tensor=rescale(gradient_tensor))
#     axs[0].set_title("Gradient Map")
#     utils.explanation.imshow_tensor(ax=axs[1], tensor=image)
#     axs[1].set_title("Original Image")
#     utils.explanation.imshow_tensor(ax=axs[2], tensor=gradient_tensor*image)
#     axs[2].set_title("E.w. Product")
#     label = label.item()
#     filepath = os.path.join("saved_images", train_specs['tag'], f"class_{label}", f"instance_{count[label]}.png")
#     plt.savefig(filepath)
#     count[label] += 1

#TODO: multi-task

##################################################
# Grad-CAM
##################################################

# model.eval()
# num_examples = 100
# count = [0] * eval_dataset.NUM_CLASSES
# for idx in tqdm(range(num_examples)):
#     nrows, ncols = 4, 3
#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9))
#     image, label = next(iter(eval_dataloader))
#     image, label = image.to(device), label.to(device)
#     grad_cams = explanation.CAM.compute_grad_cam(model=model, layer_idx=4, image=image)
#     for cls in range(model.out_features):
#         utils.explanation.imshow_tensor(ax=axs[cls//ncols, cls%ncols], tensor=rescale(grad_cams[cls]))
#         axs[cls//ncols, cls%ncols].set_title(f"{cls=}")
#     label = label.item()
#     utils.explanation.imshow_tensor(ax=axs[model.out_features//ncols, model.out_features%ncols], tensor=rescale(image))
#     axs[model.out_features//ncols, model.out_features%ncols].set_title(f"{label=}")
#     filepath = os.path.join("saved_images", train_specs['tag'], f"class_{label}", f"instance_{count[label]}.png")
#     plt.savefig(filepath)
#     count[label] += 1
