"""
DATASETS APIs.
"""
from data.datasets.Dataset import Dataset
from data.datasets.MNIST.MNISTDataset import MNISTDataset
from data.datasets.Overfit.OverfitDataset import OverfitDataset


__all__ = (
    "Dataset",
    "OverfitDataset",
    "MNISTDataset",
)
