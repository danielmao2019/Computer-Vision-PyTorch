import torch


def train_model_minimal(model, dataloader, epochs, lr):
    model.train()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for element in dataloader:
            image, label = element[0].to(device), element[1].to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(input=output, target=label)
            loss.backward()
            optimizer.step()
