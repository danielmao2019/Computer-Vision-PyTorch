"""
DATA APIs.
"""
from data import datasets
from data.dataloader.dataloader import Dataloader
from data import transforms


__all__ = (
    "datasets",
    "Dataloader",
    "transforms",
)

TASK_OPTIONS = [
    'image_classification',
    'object_detection',
    'semantic_segmentation',
]
