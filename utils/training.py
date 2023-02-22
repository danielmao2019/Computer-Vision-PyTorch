import torch


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


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
