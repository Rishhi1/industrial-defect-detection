"""
evaluation.py
=============
Computes proper evaluation metrics for the anomaly detection system, at
both the image level (NORMAL vs ANOMALOUS classification) and the pixel
level (how well the heatmap localizes the actual defect region).

Image-level metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC —
computed the standard way via scikit-learn, using the threshold already
selected in Phase 6.

Pixel-level metrics: pixel-wise ROC-AUC and average precision, computed by
treating every pixel in every test image as one classification instance
(is this pixel part of a defect or not?). Normal test images contribute an
all-zero ground-truth mask (MVTec doesn't ship one, since there's no defect
to mark) — this is necessary so pixel-level metrics reflect false positives
on clean images, not just localization quality within already-known
defective regions.

When aggregating across categories, two different things are reported and
labeled explicitly, since silently averaging metrics can be misleading:
- "pooled": every test image from every category is combined into one
  giant set before computing metrics — categories with more test images
  contribute proportionally more.
- "macro" (per-category mean): each category's metric is computed
  independently, then the categories are averaged with equal weight
  regardless of how many test images each had.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ImageLevelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    tp: int
    tn: int
    fp: int
    fn: int
    num_images: int
    threshold: float


def compute_image_level_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> ImageLevelMetrics:
    """
    scores: (N,) anomaly scores. labels: (N,) 0=normal, 1=anomalous.
    threshold: the decision boundary (from Phase 6) used to convert scores
    into predictions.
    """
    predictions = (scores >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # ROC-AUC / PR-AUC need both classes present to be meaningful.
    both_classes_present = len(np.unique(labels)) == 2
    roc_auc = float(roc_auc_score(labels, scores)) if both_classes_present else float("nan")
    pr_auc = float(average_precision_score(labels, scores)) if both_classes_present else float("nan")

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return ImageLevelMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        num_images=len(labels),
        threshold=threshold,
    )


@dataclass
class PixelLevelMetrics:
    pixel_roc_auc: float
    pixel_average_precision: float
    num_pixels: int
    num_defect_pixels: int


def compute_pixel_level_metrics(pixel_scores: np.ndarray, pixel_labels: np.ndarray) -> PixelLevelMetrics:
    """
    pixel_scores, pixel_labels: flattened arrays, one entry per pixel across
    every test image (normal images contribute all-zero labels). Both must
    be the same length.

    Returns NaN metrics (with a note in the caller) if no defect pixels are
    present at all — this shouldn't happen with real MVTec data but is
    handled defensively.
    """
    num_defect_pixels = int(pixel_labels.sum())
    if num_defect_pixels == 0 or num_defect_pixels == len(pixel_labels):
        # Degenerate: all-normal or all-defect pixels, AUC undefined.
        return PixelLevelMetrics(
            pixel_roc_auc=float("nan"),
            pixel_average_precision=float("nan"),
            num_pixels=len(pixel_labels),
            num_defect_pixels=num_defect_pixels,
        )

    roc_auc = float(roc_auc_score(pixel_labels, pixel_scores))
    ap = float(average_precision_score(pixel_labels, pixel_scores))

    return PixelLevelMetrics(
        pixel_roc_auc=roc_auc,
        pixel_average_precision=ap,
        num_pixels=len(pixel_labels),
        num_defect_pixels=num_defect_pixels,
    )


def get_roc_curve_data(scores: np.ndarray, labels: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}


def get_pr_curve_data(scores: np.ndarray, labels: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    ap = float(average_precision_score(labels, scores))
    return {"precision": precision, "recall": recall, "thresholds": thresholds, "ap": ap}


def get_confusion_matrix(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    predictions = (scores >= threshold).astype(int)
    return confusion_matrix(labels, predictions, labels=[0, 1])


# -----------------------------------------------------------------------
# Cross-category aggregation
# -----------------------------------------------------------------------
def aggregate_pooled(all_scores: list[np.ndarray], all_labels: list[np.ndarray], threshold_per_category: list[float]) -> dict:
    """
    Pool every category's test images together and compute metrics as if
    they were one combined test set. NOTE: this requires a single shared
    threshold — since thresholds differ per category, pooled metrics here
    use each image's own category threshold to form predictions, then pool
    the resulting binary predictions/labels (this is the closest meaningful
    interpretation of "pooled" when categories have different score scales
    and different thresholds).
    """
    all_predictions = []
    all_labels_flat = []
    for scores, labels, threshold in zip(all_scores, all_labels, threshold_per_category):
        all_predictions.append((scores >= threshold).astype(int))
        all_labels_flat.append(labels)

    predictions = np.concatenate(all_predictions)
    labels_flat = np.concatenate(all_labels_flat)

    return {
        "accuracy": float(accuracy_score(labels_flat, predictions)),
        "precision": float(precision_score(labels_flat, predictions, zero_division=0)),
        "recall": float(recall_score(labels_flat, predictions, zero_division=0)),
        "f1": float(f1_score(labels_flat, predictions, zero_division=0)),
        "num_images": len(labels_flat),
    }


def aggregate_macro(per_category_metrics: list[ImageLevelMetrics]) -> dict:
    """Simple unweighted mean of each category's metrics — every category counts equally."""
    if not per_category_metrics:
        return {}

    def safe_mean(values):
        valid = [v for v in values if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    return {
        "accuracy": safe_mean([m.accuracy for m in per_category_metrics]),
        "precision": safe_mean([m.precision for m in per_category_metrics]),
        "recall": safe_mean([m.recall for m in per_category_metrics]),
        "f1": safe_mean([m.f1 for m in per_category_metrics]),
        "roc_auc": safe_mean([m.roc_auc for m in per_category_metrics]),
        "pr_auc": safe_mean([m.pr_auc for m in per_category_metrics]),
        "num_categories": len(per_category_metrics),
    }
