import torch
import numpy as np
from tqdm import tqdm
import os
import time

import training

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def train_loop(model, dataloader, criterion, optimizer, loss_graph, score_graph, device):
    """
    Parameters:
        loss_graph (list of float): Full list of loss for all iterations during training. Will be modified.
    Returns:
        avg_loss (float): average loss across all samples in the dataloader for current epoch.
        avg_score (float): average score across all samples in the dataloader for current epoch.
    """
    tot_loss = 0
    tot_score = 0
    for data in tqdm(dataloader):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        score = 0
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


def train_model(specs):
    """
    specs should be a dictionary with the following arguments:
    {
        tag (str):
        model (torch.nn.Module):
        dataloader (torch.utils.data.DataLoader):
        epochs (int):
        criterion (torch.nn.Module):
        optimizer (torch.optim.Optimizer):
        save_model (bool):
    }
    """
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}.")
    tag = specs['tag']
    models_root = os.path.join('saved_models', tag)
    model = specs['model']
    dataloader = specs['dataloader']
    epochs = specs['epochs']
    criterion = specs['criterion']
    optimizer = specs['optimizer']
    # get model, optimizer, and epoch
    start_epoch = 0
    if isinstance(specs.get('load_model', None), str):
        filepath = os.path.join(models_root, specs['load_model'])
        model, optimizer, start_epoch = training.utils.load_model(model, optimizer, filepath=filepath)
    model.train()
    model.to(device)
    end_epoch = start_epoch + epochs
    #######################################################################
    loss_graph = []
    score_graph = []
    save_interval = 5
    #######################################################################
    # log info
    logging.info(f"Training model \"{tag}\".")
    logging.info(f"Number of trainable parameters: {training.utils.trainable_params(model)}.")
    training.utils.log_optimizer_info(optimizer)
    logging.info(f"criterion={criterion.__class__.__name__}.")
    logging.info(f"epochs={epochs}.")
    logging.info(f"batch_size={dataloader.batch_size}.")
    #######################################################################
    # main loop
    for cur_epoch in range(start_epoch, end_epoch):
        start_time = time.time()
        loss, score = train_loop(
            model=model, dataloader=dataloader, criterion=criterion, optimizer=optimizer,
            loss_graph=loss_graph, score_graph=score_graph, device=device,
        )
        logging.info("Epoch: {:03d}/{}, loss={:.6f}, score={:.6f}, time={:.2f}seconds.".format(
            cur_epoch+1, end_epoch, loss, score, time.time()-start_time))
        if (cur_epoch+1) % save_interval == 0 and specs.get('save_model', False):
            filepath = os.path.join(models_root, f'checkpoint_{cur_epoch+1:03d}.pt')
            training.utils.save_model(model=model, optimizer=optimizer, epoch=cur_epoch, filepath=filepath)
            logging.info(f"Saved model to {filepath}.")
        np.savetxt(
            fname=os.path.join(models_root, f"loss_graph_{tag}.txt"),
            X=np.array(loss_graph),
        )
        np.savetxt(
            fname=os.path.join(models_root, f"score_graph_{tag}.txt"),
            X=np.array(score_graph),
        )
