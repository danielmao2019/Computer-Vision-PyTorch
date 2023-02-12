import torch

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


INDENT = ' ' * 4


def save_model(model, optimizer, epoch, filepath):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_model(model, optimizer, filepath):
    """
    Parameters:
        optimizer (None|torch.optim): could be None, in which case loading the model is for inference use rather than training.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def log_criterion_info(criterion):
    logging.info(f"criterion={criterion.__class__.__name__}")
    for name, param in criterion.named_parameters():
        logging.info(INDENT + f"{name}={param.data}")


def log_optimizer_info(optimizer):
    logging.info(f"optimizer={optimizer.__class__.__name__}")
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    for key in sorted(group):
        if key != 'params':
            logging.info(INDENT + f"{key}={group[key]}")


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
