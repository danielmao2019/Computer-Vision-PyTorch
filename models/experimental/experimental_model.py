import torch

import models


class ExperimentalModel(torch.nn.Module):

    def __init__(self, task, in_features, out_features):
        super(ExperimentalModel, self).__init__()
        if task not in ['image_classification']:
            raise NotImplementedError(f"[ERROR] Predictor for task {task} not implemented.")
        self.task = task
        self.extractor = torch.nn.Sequential(
            models.builder.units.get_conv_unit(in_channels=in_features, out_channels=128),
            models.builder.units.get_conv_unit(in_channels=128, out_channels=256),
            models.builder.units.get_conv_unit(in_channels=256, out_channels=512),
        )
        self.predictor = self._get_predictor(in_features=512, out_features=out_features)

    def _get_predictor(self, in_features, out_features):
        if self.task == 'image_classification':
            return torch.nn.Sequential(
                models.layers.GlobalAveragePooling2D(),
                torch.nn.Linear(in_features=in_features, out_features=out_features),
                torch.nn.Softmax(dim=1),
            )
        assert False, f"Program should not reach here."

    def forward(self, x):
        return self.predictor(self.extractor(x))
