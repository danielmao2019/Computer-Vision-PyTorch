"""
DATA APIs.
"""
from data import datasets
from data.dataloader.dataloader import Dataloader


__all__ = (
    "datasets",
    "Dataloader",
)

TASK_OPTIONS = [
    'image_classification',
    'object_detection',
    'semantic_segmentation',
]
