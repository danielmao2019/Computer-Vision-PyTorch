"""
LOSSES APIs.
"""
from losses import utils
from losses.MultiTaskCriterion import MultiTaskCriterion
from losses.PermutedCrossEntropyLoss import PermutedCrossEntropyLoss


__all__ = (
    "utils",
    "MultiTaskCriterion",
    "PermutedCrossEntropyLoss",
)
