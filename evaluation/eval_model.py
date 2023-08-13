import torch
from tqdm import tqdm


def eval_model(model, dataloader, metrics):
    device = next(model.parameters()).device
    model.eval()
    tot_scores = [0] * len(metrics)
    for element in tqdm(dataloader, leave=False):
        images, labels = element[0].to(device), element[1].to(device)
        outputs = model(images)
        for idx, metric in enumerate(metrics):
            # assume a reduction of "sum" so score should be scalar tensor.
            tot_scores[idx] += metric(y_pred=outputs, y_true=labels).item()
    # assume a reduction of "sum" so divide by total number of examples
    avg_scores = [tot_scores[idx] / dataloader.num_examples for idx in range(len(tot_scores))]
    return avg_scores
