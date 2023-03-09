import torch
import os


def save_model(model, optimizer, epoch, filepath):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if os.path.exists(filepath):
        os.chmod(filepath, '0o600')
    torch.save(checkpoint, filepath)
    os.chmod(filepath, '0o400')


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


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
