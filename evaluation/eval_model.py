from tqdm import tqdm


def eval_model(model, dataloader, metrics):
    device = next(model.parameters()).device
    model.eval()
    tot_scores = [0] * len(metrics)
    for element in tqdm(dataloader):
        image, label = element[0].to(device), element[1].to(device)
        output = model(image)
        for idx, metric in enumerate(metrics):
            tot_scores[idx] += metric(output, label)
    avg_scores = [tot_scores[idx] / len(dataloader) for idx in range(len(tot_scores))]
    return avg_scores
