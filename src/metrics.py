"""Evaluation metrics for semantic segmentation."""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        preds: Predicted labels [N, H, W] or flattened
        targets: Ground truth labels [N, H, W] or flattened
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape [num_classes, num_classes]
        where cm[i, j] = count of pixels with true label i predicted as j
    """
    preds = preds.flatten()
    targets = targets.flatten()

    # Filter out ignored pixels (if any have value >= num_classes)
    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    return cm


def iou_from_confusion_matrix(cm: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute per-class IoU and mIoU from confusion matrix.

    Args:
        cm: Confusion matrix [num_classes, num_classes]

    Returns:
        Tuple of (per_class_iou, mean_iou)
    """
    # IoU = TP / (TP + FP + FN)
    # TP for class i = cm[i, i]
    # FP for class i = sum(cm[:, i]) - cm[i, i]  (predicted as i but not i)
    # FN for class i = sum(cm[i, :]) - cm[i, i]  (is i but predicted as other)

    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

    # Avoid division by zero
    valid = union > 0
    iou = np.zeros(cm.shape[0], dtype=np.float64)
    iou[valid] = intersection[valid] / union[valid]

    # mIoU only over classes that exist in the dataset
    miou = iou[valid].mean() if valid.any() else 0.0

    return iou, float(miou)


def precision_recall_f1_from_confusion_matrix(
    cm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute per-class precision, recall, F1, and macro F1.

    Args:
        cm: Confusion matrix [num_classes, num_classes]

    Returns:
        Tuple of (precision, recall, f1, macro_f1)
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    # Precision = TP / (TP + FP)
    precision = np.zeros(cm.shape[0], dtype=np.float64)
    denom_p = tp + fp
    valid_p = denom_p > 0
    precision[valid_p] = tp[valid_p] / denom_p[valid_p]

    # Recall = TP / (TP + FN)
    recall = np.zeros(cm.shape[0], dtype=np.float64)
    denom_r = tp + fn
    valid_r = denom_r > 0
    recall[valid_r] = tp[valid_r] / denom_r[valid_r]

    # F1 = 2 * precision * recall / (precision + recall)
    f1 = np.zeros(cm.shape[0], dtype=np.float64)
    denom_f1 = precision + recall
    valid_f1 = denom_f1 > 0
    f1[valid_f1] = 2 * precision[valid_f1] * recall[valid_f1] / denom_f1[valid_f1]

    # Macro F1 (average over classes that exist)
    valid = denom_r > 0  # Classes that have ground truth samples
    macro_f1 = f1[valid].mean() if valid.any() else 0.0

    return precision, recall, f1, float(macro_f1)


def pixel_accuracy_from_confusion_matrix(cm: np.ndarray) -> float:
    """Compute overall pixel accuracy from confusion matrix."""
    correct = np.diag(cm).sum()
    total = cm.sum()
    return float(correct / max(total, 1))


class SegmentationMetrics:
    """
    Accumulator for segmentation metrics.

    Usage:
        metrics = SegmentationMetrics(num_classes=7)
        for pred, target in dataloader:
            metrics.update(pred, target)
        results = metrics.compute()
    """

    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset accumulated confusion matrix."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(
        self,
        preds: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            preds: Predicted class IDs [B, H, W]
            targets: Ground truth class IDs [B, H, W]
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        batch_cm = compute_confusion_matrix(preds, targets, self.num_classes)
        self.confusion_matrix += batch_cm

    def compute(self) -> Dict[str, float | np.ndarray]:
        """
        Compute all metrics from accumulated confusion matrix.

        Returns:
            Dictionary with:
                - pixel_accuracy: Overall pixel accuracy
                - miou: Mean IoU
                - per_class_iou: Array of per-class IoU
                - macro_f1: Macro-averaged F1 score
                - per_class_f1: Array of per-class F1
                - per_class_precision: Array of per-class precision
                - per_class_recall: Array of per-class recall
                - confusion_matrix: The confusion matrix
        """
        cm = self.confusion_matrix

        pixel_acc = pixel_accuracy_from_confusion_matrix(cm)
        per_class_iou, miou = iou_from_confusion_matrix(cm)
        precision, recall, f1, macro_f1 = precision_recall_f1_from_confusion_matrix(cm)

        return {
            "pixel_accuracy": pixel_acc,
            "miou": miou,
            "per_class_iou": per_class_iou,
            "macro_f1": macro_f1,
            "per_class_f1": f1,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "confusion_matrix": cm,
        }

    def summary(self) -> str:
        """Return a formatted summary string."""
        results = self.compute()

        lines = [
            "=" * 60,
            "SEGMENTATION METRICS",
            "=" * 60,
            f"Pixel Accuracy: {results['pixel_accuracy']:.4f}",
            f"Mean IoU:       {results['miou']:.4f}",
            f"Macro F1:       {results['macro_f1']:.4f}",
            "",
            "Per-Class Results:",
            "-" * 60,
            f"{'Class':<15} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}",
            "-" * 60,
        ]

        for i, name in enumerate(self.class_names):
            lines.append(
                f"{name:<15} "
                f"{results['per_class_iou'][i]:>8.4f} "
                f"{results['per_class_f1'][i]:>8.4f} "
                f"{results['per_class_precision'][i]:>8.4f} "
                f"{results['per_class_recall'][i]:>8.4f}"
            )

        lines.append("=" * 60)

        return "\n".join(lines)
