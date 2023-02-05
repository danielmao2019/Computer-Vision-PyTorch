import torch
import models


class LeNet(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(LeNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_features, out_channels=6, kernel_size=5, stride=1,
        )
        self.tanh1 = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1,
        )
        self.tanh2 = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=5, stride=1,
        )
        self.tanh3 = torch.nn.Tanh()
        self.pool3 = models.layers.GlobalAveragePooling2D()
        self.linear1 = torch.nn.Linear(
            in_features=120, out_features=84,
        )
        self.tanh4 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(
            in_features=84, out_features=out_features,
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape[2:] == (32, 32), f"{x.shape=}"

        x = self.conv1(x)
        x = self.tanh1(x)
        assert x.shape[1:] == (6, 28, 28), f"{x.shape=}"
        x = self.pool1(x)
        assert x.shape[1:] == (6, 14, 14), f"{x.shape=}"

        x = self.conv2(x)
        x = self.tanh2(x)
        assert x.shape[1:] == (16, 10, 10), f"{x.shape=}"
        x = self.pool2(x)
        assert x.shape[1:] == (16, 5 ,5), f"{x.shape=}"

        x = self.conv3(x)
        x = self.tanh3(x)
        assert x.shape[1:] == (120, 1, 1), f"{x.shape=}"
        x = self.pool3(x)
        assert x.shape[1:] == (120,), f"{x.shape=}"

        x = self.linear1(x)
        x = self.tanh4(x)
        assert x.shape[1:] == (84,), f"{x.shape=}"

        x = self.linear2(x)
        x = self.softmax(x)
        assert x.shape[1:] == (self.out_features,), f"{x.shape=}"
        return x
