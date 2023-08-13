"""
DATASETS API.
"""
from data.datasets.Dataset import Dataset
from data.datasets.Overfit.OverfitDataset import OverfitDataset
from data.datasets.RandomVectors.RandomVectorsDataset import RandomVectorsDataset
from data.datasets.MNIST.MNISTDataset import MNISTDataset
from data.datasets.CIFAR.CIFARDataset import CIFARDataset


__all__ = (
    "Dataset",
    "OverfitDataset",
    "RandomVectorsDataset",
    "MNISTDataset",
    "CIFARDataset",
)
