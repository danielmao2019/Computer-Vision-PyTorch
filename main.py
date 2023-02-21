import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

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


def main(args):
    criterion = losses.MultiTaskCriterion(criteria=[
        torch.nn.CrossEntropyLoss(),
        losses.MappedMNISTCEL(num_classes=10, seed=0),
        ], weights=[1, 1],
    )
    criterion_gradient_list = [
        explanation.gradients.CE_gradient,
        lambda inputs, labels: explanation.gradients.CE_gradient(inputs, labels, criterion.criteria[1].mapping)
    ]
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
    train_dataset_easy = train_dataset.subset(indices=np.loadtxt("data/datasets/MNIST/easy.txt"))
    train_dataset_hard = train_dataset.subset(indices=np.loadtxt("data/datasets/MNIST/hard.txt"))
    train_dataloader = data.Dataloader(
        task='image_classification', dataset=train_dataset,
        batch_size=8, shuffle=False, transforms=[
            data.transforms.Resize(new_size=(32, 32)),
        ])

    eval_dataset = data.datasets.MNISTDataset(purpose='evaluation')
    eval_dataloader = data.Dataloader(
        task='image_classification', dataset=eval_dataset,
        batch_size=1, shuffle=False, transforms=[
            data.transforms.Resize(new_size=(32, 32)),
        ])

    ##################################################
    # training
    ##################################################

    train_specs = {
        'tag': 'LeNet_MNIST_Multi_3',
        'epochs': 0,
        'save_model': False,
        'load_model': args.checkpoint,
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

    ##################################################
    # VOG
    ##################################################

    print('#' * 50)
    print('### eval')
    print('#' * 50)

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


    exp_dataloader = data.Dataloader(
        task='image_classification', dataset=train_dataset,
        batch_size=1, shuffle=False, transforms=[
            data.transforms.Resize(new_size=(32, 32)),
        ])
    model.eval()
    num_examples = len(exp_dataloader)
    inner_products = np.zeros(shape=(num_examples,))
    # count = [0] * eval_dataset.NUM_CLASSES
    for idx in tqdm(range(num_examples)):
        # fig, axs = plt.subplots(nrows=1, ncols=3)
        image, label = next(iter(exp_dataloader))
        image, label = image.to(device), label.to(device)
        gradient_tensor_list = torch.stack([explanation.gradients.compute_gradients(
            model=model, image=image, label=label, criterion_gradient=criterion_gradient, depth=None,
        ) for criterion_gradient in criterion_gradient_list], dim=0)
        assert len(gradient_tensor_list.shape) == 5, f"{gradient_tensor_list.shape=}"
        inner_products[idx] = utils.tensors.pairwise_inner_product(gradient_tensor_list)[0, 1].item()
    # np.savetxt(fname="data/datasets/MNIST/easy.txt", X=np.where(inner_products >= 0)[0].astype(np.int64), fmt='%d')
    # np.savetxt(fname="data/datasets/MNIST/hard.txt", X=np.where(inner_products < 0)[0].astype(np.int64), fmt='%d')
    plt.figure()
    plt.hist(inner_products, bins=50, range=[-10, +10])
    plt.savefig(os.path.join("saved_images", f'{args.checkpoint}.png'))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    main(args)
