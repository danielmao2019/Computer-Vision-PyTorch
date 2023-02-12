import torch
import losses


class MultiTaskCriterion(torch.nn.Module):
    """Only supports classification task.
    """

    def __init__(self, criteria, weights=None):
        super(MultiTaskCriterion, self).__init__()
        if weights is None:
            weights = [1] * len(criteria)
        if len(criteria) != len(weights):
            raise ValueError(
                f"[ERROR] Argument 'criteria' and 'weights' should have the same length. "
                f"Got {len(criteria)=} and {len(weights)=}."
            )
        weights = [w / len(weights) for w in weights]
        self.criteria = criteria
        self.weights = weights

    def forward(self, inputs, labels):
        tot_loss = 0
        onehot_labels = losses.utils.onehot_encode(shape=inputs.shape, labels=labels)
        for criterion, weight in zip(self.criteria, self.weights):
            tot_loss += weight * criterion(inputs, onehot_labels)
        return tot_loss

    def __str__(self):
        return '\n'.join([
            f"weight={weight:.4f}, criterion={criterion.__str__()}"
            for criterion, weight in zip(self.criteria, self.weights)
        ])
