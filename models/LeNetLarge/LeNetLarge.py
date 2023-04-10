import torch
import models


class LeNetLarge(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(LeNetLarge, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_features, out_channels=12, kernel_size=3, stride=1,
        )
        self.tanh1 = torch.nn.Tanh()
        self.conv2 = torch.nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=3, stride=1,
        )
        self.tanh2 = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(
            in_channels=12, out_channels=32, kernel_size=3, stride=1,
        )
        self.tanh3 = torch.nn.Tanh()
        self.conv4 = torch.nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1,
        )
        self.tanh4 = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5 = torch.nn.Conv2d(
            in_channels=32, out_channels=240, kernel_size=3, stride=1,
        )
        self.tanh5 = torch.nn.Tanh()
        self.conv6 = torch.nn.Conv2d(
            in_channels=240, out_channels=240, kernel_size=3, stride=1,
        )
        self.tanh6 = torch.nn.Tanh()
        self.pool3 = models.layers.GlobalAveragePooling2D()

        self.linear1 = torch.nn.Linear(
            in_features=240, out_features=84,
        )
        self.tanh7 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(
            in_features=84, out_features=out_features,
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool1(x)  # layer_idx = 4

        x = self.conv3(x)
        x = self.tanh3(x)
        x = self.conv4(x)
        x = self.tanh4(x)
        x = self.pool2(x)  # layer_idx = 9

        x = self.conv5(x)
        x = self.tanh5(x)  # layer_idx = 11
        x = self.conv6(x)
        x = self.tanh6(x)
        x = self.pool3(x)  # layer_idx = 14

        x = self.linear1(x)
        x = self.tanh7(x)  # layer_idx = 16

        x = self.linear2(x)
        x = self.softmax(x)
        return x
