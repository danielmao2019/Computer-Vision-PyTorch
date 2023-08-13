import numpy as np
import torch
from tqdm import tqdm
import os
import time

import metrics
import evaluation
import utils


def train_loop(model, dataloader, criterion, optimizer, metric, loss_graph, score_graph):
    """
    Args:
        loss_graph (list of float): Full list of loss for all iterations during training. Will be modified.
    Returns:
        avg_loss (float): average loss across all samples in the dataloader for current epoch.
        avg_score (float): average score across all samples in the dataloader for current epoch.
    """
    device = next(model.parameters()).device
    tot_loss = 0
    tot_score = 0
    for element in tqdm(dataloader, leave=False):
        images, labels = element[0].to(device), element[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        score = metric(outputs, labels)
        ###
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        tot_loss += loss.item()
        tot_score += score.item()
        loss_graph.append(loss.item())
        score_graph.append(score.item())
    # assume reductions are both 'sum'
    avg_loss = tot_loss / dataloader.num_examples
    avg_score = tot_score / dataloader.num_examples
    return avg_loss, avg_score


def train_model(tag, model, train_dataloader, eval_dataloader, epochs, criterion, optimizer, metric,
                save_model, load_model,
                ):
    """
    Args:
        tag (str):
        model (torch.nn.Module):
        train_dataloader (torch.utils.data.DataLoader):
        eval_dataloader (torch.utils.data.DataLoader):
        epochs (int):
        criterion (torch.nn.Module):
        optimizer (torch.optim.Optimizer):
        metric (torch.nn.Module):
        save_model (bool):
        load_model (str|None):
    """
    models_root = os.path.join('saved_models', tag)
    logger = utils.logging.get_logger(filename=os.path.join(models_root, "training.log") if epochs else None)
    device = next(model.parameters()).device
    logger.info(f"Using device {device}.")
    # get model, optimizer, and epoch
    start_epoch = 0
    if load_model is not None:
        filepath = os.path.join(models_root, "checkpoints", load_model)
        model, optimizer, start_epoch = utils.training.load_model(model, optimizer, filepath=filepath)
        logger.info(f"Loaded model from {filepath}.")
    model.train()
    end_epoch = start_epoch + epochs
    #######################################################################
    loss_graph = []
    score_graph = []
    #######################################################################
    # log info
    if epochs:
        logger.info(f"Experiment tag: \"{tag}\".")
        logger.info(f"Number of trainable parameters: {utils.training.trainable_params(model)}.")
        utils.logging.log_criterion_info(logger=logger, criterion=criterion)
        utils.logging.log_optimizer_info(logger=logger, optimizer=optimizer)
        logger.info(f"epochs={epochs}.")
        logger.info(f"batch_size={train_dataloader.batch_size}.")
    #######################################################################
    # main loop
    for cur_epoch in range(start_epoch, end_epoch):
        start_time = time.time()
        train_loss, train_score = train_loop(
            model=model, dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, metric=metric,
            loss_graph=loss_graph, score_graph=score_graph,
        )
        logger.info("Epoch {:03d}/{}, loss={:.6f}, score={:.6f}, time={:.2f}sec.".format(
            cur_epoch+1, end_epoch, train_loss, train_score, time.time()-start_time))
        if save_model and (cur_epoch+1) % (5 ** np.floor(np.log10(cur_epoch+1))) == 0:
            eval_scores = evaluation.eval_model(model, dataloader=eval_dataloader, metrics=[metric])
            logger.info(f"{eval_scores=}")
            filepath = os.path.join(models_root, "checkpoints", f'checkpoint_{cur_epoch+1:03d}.pt')
            utils.training.save_model(model=model, optimizer=optimizer, epoch=cur_epoch, filepath=filepath)
            logger.info(f"Saved model to {filepath}.")
        np.savetxt(
            fname=os.path.join(models_root, f"loss_graph.txt"),
            X=torch.Tensor(loss_graph).cpu().numpy(),
        )
        np.savetxt(
            fname=os.path.join(models_root, f"score_graph.txt"),
            X=torch.Tensor(score_graph).cpu().numpy(),
        )
