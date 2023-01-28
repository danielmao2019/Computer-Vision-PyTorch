import torch
import models


class ExperimentalModel(torch.nn.Module):

    def __init__(self, task, in_features, out_features):
        super(ExperimentalModel, self).__init__()
        if task not in ['image_classification']:
            raise NotImplementedError(f"[ERROR] Predictor for task {task} not implemented.")
        self.task = task
        self.extractor = torch.nn.Sequential(
            self.get_conv_layer(in_channels=in_features, out_channels=128),
        )
        self.predictor = self.get_predictor(in_features=128, out_features=out_features)

    def get_conv_layer(self, in_channels, out_channels):
        conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same',
        )
        batch_norm = torch.nn.BatchNorm2d(num_features=out_channels)
        relu = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(
            conv, batch_norm, relu,
        )

    def get_predictor(self, in_features, out_features):
        if self.task == 'image_classification':
            return torch.nn.Sequential(
                models.layers.GlobalAveragePooling2D(),
                torch.nn.Linear(in_features=in_features, out_features=out_features),
                torch.nn.Softmax(dim=1),
            )
        assert False, f"Program should not reach here."

    def forward(self, x):
        return self.predictor(self.extractor(x))
