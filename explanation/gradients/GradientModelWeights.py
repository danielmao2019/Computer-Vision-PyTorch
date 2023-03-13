import torch


class GradientModelWeights(torch.nn.Module):

    def __init__(self, model):
        super(GradientModelWeights, self).__init__()
        model.eval()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, image, label, criterion):
        loss = criterion(self.model(image), label)
        loss.backward()
        gradients = []
        for param in self.model.parameters():
            gradients.append(param.grad)
        gradients = torch.cat([grad.view(-1) for grad in gradients])
        for param in self.model.parameters():
            param.grad.zero_()
        return gradients
