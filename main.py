import torch
import torchvision

import math
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
        losses.MappedMNISTCEL(mapping=None, num_classes=10),  # identity mapping
        losses.MappedMNISTCEL(mapping=torch.Tensor([0, 1, 2, 5, 4, 3, 6, 7, 8, 9])),  # swap cats and dogs
        ], weights=[1, 1],
    )
    criterion_gradient_list = [
        lambda y_pred, y_true: explanation.gradients.CE_gradient(
            y_pred=y_pred, y_true=y_true, mapping=sub_criterion.mapping,
        ) for sub_criterion in criterion.criteria
    ]
    metric = metrics.Acc(reduction="sum")

    ##################################################
    # model
    ##################################################

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.LeNetLarge(in_features=3, out_features=10)
    # model = models.LeNet(in_features=3, out_features=10)
    model.to(device)

    ##################################################
    # datasets
    ##################################################

    root = os.path.join("data", "datasets", "downloads", "STL10")
    download = not os.path.exists(root)
    train_dataset = torchvision.datasets.STL10(
        root=root, split='train', download=download,
        transform=torchvision.transforms.ToTensor(),
    )
    train_dataloader = data.Dataloader(
        task='image_classification', dataset=train_dataset,
        batch_size=8, shuffle=True, transforms=[
            # data.transforms.Resize(new_size=(32, 32)),
        ])
    eval_dataset = torchvision.datasets.STL10(
        root=root, split='test', download=download,
        transform=torchvision.transforms.ToTensor(),
    )
    eval_dataloader = data.Dataloader(
        task='image_classification', dataset=eval_dataset,
        batch_size=8, shuffle=True, transforms=[
            # data.transforms.Resize(new_size=(32, 32)),
        ])

    ##################################################
    # training
    ##################################################

    train_specs = {
        'tag': 'LeNetLarge_STL10_Multi_2',
        'epochs': 0,
        'save_model': False,
        'load_model': args.checkpoint,
        'model': model,
        'train_dataloader': train_dataloader,
        'eval_dataloader': eval_dataloader,
        'criterion': criterion,
        'optimizer': torch.optim.SGD(model.parameters(), lr=1.0e-03, momentum=0.9),
        # 'optimizer': torch.optim.Adam(model.parameters()),
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

    # print('#' * 50)
    # print('### eval')
    # print('#' * 50)

    # scores = evaluation.eval_model(
    #     model=model, dataloader=eval_dataloader, metrics=[metric],
    # )
    # print(scores)

    exp_dataloader = data.Dataloader(
        task='image_classification', dataset=train_dataset,
        batch_size=1, shuffle=True, transforms=[
            # data.transforms.Resize(new_size=(32, 32)),
        ])
    model.eval()
    num_examples = 100

    layer_idx = int(args.layer_idx)
    gmi = explanation.gradients.GradientModelInputs(model=model, layer_idx=layer_idx)
    iterator = iter(exp_dataloader)
    idx = 0
    pbar = tqdm(total=num_examples, leave=False)
    while idx < num_examples:
        image, label = next(iterator)
        if label.item() not in [3, 5]:
            continue
        image, label = image.to(device), label.to(device)
        output = gmi.update(image)
        activations = gmi.memory[layer_idx]
        # get conflict map
        gradient_tensor_list = torch.cat([
            gmi(criterion_gradient(y_pred=output, y_true=label))
            for criterion_gradient in criterion_gradient_list
        ], dim=0)
        assert gradient_tensor_list.shape == (len(criterion_gradient_list),) + activations.shape[1:4], f"{gradient_tensor_list.shape=}"
        inner_products = torch.sum(torch.prod(gradient_tensor_list, dim=0, keepdim=True), dim=[2, 3], keepdim=True)
        norm_products = torch.prod(torch.sqrt(torch.sum(gradient_tensor_list**2, dim=[2, 3], keepdim=True)), dim=0, keepdim=True)
        conflict_scores = (inner_products / norm_products) if torch.prod(norm_products) != 0 else inner_products
        assert conflict_scores.shape == (1, activations.shape[1], 1, 1), f"{conflict_scores.shape}"
        conflict_map = torch.sum(activations * conflict_scores, dim=1, keepdim=True)
        assert conflict_map.shape == (1, 1) + activations.shape[2:4], f"{conflict_map.shape=}"
        # get Grad-CAM True
        class_score_grad = torch.zeros(size=(1, model.out_features), dtype=torch.float32).to(device)
        class_score_grad[0, label.item()] = 1
        class_score_grad = gmi(class_score_grad)
        grad_weights = torch.mean(class_score_grad, dim=[2, 3], keepdim=True)
        assert grad_weights.shape == (1, activations.shape[1], 1, 1), f"{grad_weights.shape=}, {activations.shape=}"
        grad_cam_true = torch.sum(activations * grad_weights, dim=1, keepdim=True)
        grad_cam_true = torch.nn.functional.relu(grad_cam_true)
        assert grad_cam_true.shape == (1, 1) + activations.shape[2:4], f"{grad_cam_true.shape=}"
        # get Grad-CAM Pert
        class_score_grad = torch.zeros(size=(1, model.out_features), dtype=torch.float32).to(device)
        class_score_grad[0, criterion.criteria[1].mapping[label.item()]] = 1
        class_score_grad = gmi(class_score_grad)
        grad_weights = torch.mean(class_score_grad, dim=[2, 3], keepdim=True)
        assert grad_weights.shape == (1, activations.shape[1], 1, 1), f"{grad_weights.shape=}, {activations.shape=}"
        grad_cam_pert = torch.sum(activations * grad_weights, dim=1, keepdim=True)
        grad_cam_pert = torch.nn.functional.relu(grad_cam_pert)
        assert grad_cam_pert.shape == (1, 1) + activations.shape[2:4], f"{grad_cam_pert.shape=}"
        # mask the conflict map
        conflict_map_true = conflict_map * grad_cam_true
        conflict_map_pert = conflict_map * grad_cam_pert
        # plots
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
        utils.plot.imshow_tensor(
            ax=axs[0, 0], tensor=image,
            title="Original Image",
        )
        # utils.plot.imshow_tensor(ax=axs[1, 0], tensor=conflict_map)
        # axs[1, 0].set_title("Conflict Map")
        utils.plot.imshow_tensor(
            fig=fig, ax=axs[0, 1], tensor=conflict_map_true,
            title="Conflict Map True", show_colorbar=True, show_origin=True,
        )
        utils.plot.imshow_tensor(
            ax=axs[1, 1], tensor=utils.plot.overlay_heatmap(image=image, heatmap=(conflict_map_true < 0)),
            title="Negative Region True", show_colorbar=False, show_origin=False,
        )
        utils.plot.imshow_tensor(
            fig=fig, ax=axs[2, 1], tensor=utils.plot.overlay_heatmap(image=image, heatmap=grad_cam_true),
            title="Grad-CAM True", show_colorbar=False, show_origin=False,
        )
        utils.plot.imshow_tensor(
            fig=fig, ax=axs[0, 2], tensor=conflict_map_pert,
            title="Conflict Map Pert", show_colorbar=True, show_origin=True,
        )
        utils.plot.imshow_tensor(
            ax=axs[1, 2], tensor=utils.plot.overlay_heatmap(image=image, heatmap=(conflict_map_pert < 0)),
            title="Negative Region Pert", show_colorbar=False, show_origin=False,
        )
        utils.plot.imshow_tensor(
            fig=fig, ax=axs[2, 2], tensor=utils.plot.overlay_heatmap(image=image, heatmap=grad_cam_pert),
            title="Grad-CAM Pert", show_colorbar=False, show_origin=False,
        )
        filepath = os.path.join(args.image_dir, f"class_{label.item()}", f"example_{idx:03d}.png")
        plt.savefig(filepath)
        plt.close()
        idx += 1
        pbar.update(1)
    pbar.close()
    print("Done generating images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--layer_idx')
    parser.add_argument('--image_dir')
    args = parser.parse_args()
    main(args)
