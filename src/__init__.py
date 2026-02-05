"""Multispectral Habitat Segmentation package."""

from .model import make_deeplabv3, predict
from .dataset import GeoTiffSegmentationDataset, SegSample
from .metrics import SegmentationMetrics, compute_confusion_matrix, iou_from_confusion_matrix
from .augmentations import (
    SegmentationAugmentation,
    get_train_augmentation,
    get_val_transform,
)
from .utils import split_dataset, ensure_dir, device_from_arg

CLASS_NAMES = [
    "Background",
    "Seagrass",
    "Coral",
    "Sand",
    "Rock",
    "Algae",
    "Deep Water",
]

__all__ = [
    # Model
    "make_deeplabv3",
    "predict",
    # Dataset
    "GeoTiffSegmentationDataset",
    "SegSample",
    # Metrics
    "SegmentationMetrics",
    "compute_confusion_matrix",
    "iou_from_confusion_matrix",
    # Augmentations
    "SegmentationAugmentation",
    "get_train_augmentation",
    "get_val_transform",
    # Utils
    "split_dataset",
    "ensure_dir",
    "device_from_arg",
    # Constants
    "CLASS_NAMES",
]
