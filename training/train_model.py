import numpy as np
from tqdm import tqdm
import os
import time

import training
import evaluation
import metrics

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def train_loop(model, dataloader, criterion, optimizer, metric, loss_graph, score_graph):
    """
    Parameters:
        loss_graph (list of float): Full list of loss for all iterations during training. Will be modified.
    Returns:
        avg_loss (float): average loss across all samples in the dataloader for current epoch.
        avg_score (float): average score across all samples in the dataloader for current epoch.
    """
    device = next(model.parameters()).device
    tot_loss = 0
    tot_score = 0
    for element in tqdm(dataloader):
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
        tot_score += score
        loss_graph.append(loss.item())
        score_graph.append(score)
    avg_loss = tot_loss / len(dataloader)
    avg_score = tot_score / len(dataloader)
    return avg_loss, avg_score


def train_model(tag, model, train_dataloader, eval_dataloader, epochs, criterion, optimizer, metric,
                save_model, load_model,
                ):
    """
    Parameters:
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
    device = next(model.parameters()).device
    logging.info(f"Using device {device}.")
    models_root = os.path.join('saved_models', tag)
    # get model, optimizer, and epoch
    start_epoch = 0
    if load_model is not None:
        filepath = os.path.join(models_root, load_model)
        model, optimizer, start_epoch = training.utils.load_model(model, optimizer, filepath=filepath)
        logging.info(f"Loaded model from {filepath}.")
    model.train()
    end_epoch = start_epoch + epochs
    #######################################################################
    loss_graph = []
    score_graph = []
    save_interval = 5
    #######################################################################
    # log info
    logging.info(f"Training model \"{tag}\".")
    logging.info(f"Number of trainable parameters: {training.utils.trainable_params(model)}.")
    # training.utils.log_criterion_info(criterion)
    training.utils.log_optimizer_info(optimizer)
    logging.info(f"epochs={epochs}.")
    logging.info(f"batch_size={train_dataloader.batch_size}.")
    #######################################################################
    # main loop
    for cur_epoch in range(start_epoch, end_epoch):
        start_time = time.time()
        train_loss, train_score = train_loop(
            model=model, dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, metric=metric,
            loss_graph=loss_graph, score_graph=score_graph,
        )
        logging.info("Epoch {:03d}/{}, loss={:.6f}, score={:.6f}, time={:.2f}sec.".format(
            cur_epoch+1, end_epoch, train_loss, train_score, time.time()-start_time))
        if save_model and (cur_epoch+1) % save_interval == 0:
            eval_scores = evaluation.eval_model(model, dataloader=eval_dataloader, metrics=[metrics.Acc()])
            eval_scores = [score.item() for score in eval_scores]
            logging.info(f"{eval_scores=}")
            filepath = os.path.join(models_root, f'checkpoint_{cur_epoch+1:03d}.pt')
            training.utils.save_model(model=model, optimizer=optimizer, epoch=cur_epoch, filepath=filepath)
            logging.info(f"Saved model to {filepath}.")
        #TODO: there are some bugs with copying data between cpu and gpu.
        # np.savetxt(
        #     fname=os.path.join(models_root, f"loss_graph_{tag}.txt"),
        #     X=loss_graph.cpu().numpy(),
        # )
        # np.savetxt(
        #     fname=os.path.join(models_root, f"score_graph_{tag}.txt"),
        #     X=score_graph.cpu().numpy(),
        # )
