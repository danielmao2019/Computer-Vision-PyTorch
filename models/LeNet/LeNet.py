import torch


class LeNet(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_features, out_channels=6, kernel_size=5, stride=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=5, stride=1,
        )
        self.linear1 = torch.nn.Linear(
            in_features=120, out_features=84,
        )
        self.linear2 = torch.nn.Linear(
            in_features=84, out_features=out_features,
        )

    def forward(self, x):
        assert x.shape[2:] == (32, 32), f"{x.shape=}"
        x = self.conv1(x)
        x = torch.tanh(x)
        assert x.shape[1:] == (6, 28, 28), f"{x.shape=}"
        x = torch.nn.functional.avg_pool2d(
            x, kernel_size=2, stride=2,
        )
        assert x.shape[1:] == (6, 14, 14), f"{x.shape=}"
        x = self.conv2(x)

        x = torch.tanh(x)
        assert x.shape[1:] == (16, 10, 10), f"{x.shape=}"
        x = torch.nn.functional.avg_pool2d(
            x, kernel_size=2, stride=2,
        )
        assert x.shape[1:] == (16, 5 ,5), f"{x.shape=}"
        x = self.conv3(x)
        assert x.shape[1:] == (120, 1, 1), f"{x.shape=}"
        x = torch.squeeze(torch.squeeze(x, dim=3), dim=2)
        assert x.shape[1:] == (120,), f"{x.shape=}"
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x
