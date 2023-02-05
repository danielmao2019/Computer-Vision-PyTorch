import torch


class AlexNet(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(AlexNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_features, out_channels=96, kernel_size=11, stride=4, padding=0,
        )
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=3, stride=2,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2,
        )
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=3, stride=2,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1,
        )
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1,
        )
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1,
        )
        self.relu5 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(
            kernel_size=3, stride=2,
        )
        self.linear1 = torch.nn.Linear(
            in_features=6 * 6 * 256, out_features=4096,
        )
        self.relu6 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.linear2 = torch.nn.Linear(
            in_features=4096, out_features=4096,
        )
        self.relu7 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.linear3 = torch.nn.Linear(
            in_features=4096, out_features=out_features,
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape[1:] == (self.in_features, 224, 224)

        x = self.conv1(x)
        x = self.relu1(x)
        assert x.shape[1:] == (96, 55, 55)
        x = self.poo1(x)
        assert x.shape[1:] == (96, 27, 27)

        x = self.conv2(x)
        x = self.relu2(x)
        assert x.shape[1:] == (256, 27, 27)
        x = self.pool2(x)
        assert x.shape[1:] == (256, 13, 13)

        x = self.conv3(x)
        x = self.relu3(x)
        assert x.shape[1:] == (384, 13, 13)
        x = self.conv4(x)
        x = self.relu4(x)
        assert x.shape[1:] == (384, 13, 13)
        x = self.conv5(x)
        x = self.relu5(x)
        assert x.shape[1:] == (256, 13, 13)
        x = self.pool3(x)
        assert x.shape[1:] == (256, 6, 6)

        x = x.view(x.size(0), -1)
        assert x.shape[1:] == (9216,)

        x = self.linear1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        assert x.shape[1:] == 4096

        x = self.linear2(x)
        x = self.relu7(x)
        x = self.dropout2(x)
        assert x.shape[1:] == 4096

        x = self.linear3(x)
        x = self.softmax(x)
        assert x.shape[1:] == self.out_features

        return x
