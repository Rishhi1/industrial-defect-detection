"""
visualization.py
=================
Plotting utilities used throughout the project: sample image grids,
preprocessing sanity checks, and (in later phases) score distributions,
ROC/PR curves, confusion matrices, and anomaly heatmap overlays.

This phase only implements the pieces needed for Phase 3 (preprocessing
sanity checks). Evaluation-related plots are added in Phase 8.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; scripts may run without a display
import matplotlib.pyplot as plt
import numpy as np

from src.utils import ensure_dir


def plot_preprocessing_check(
    originals: list[np.ndarray],
    preprocessed: list[np.ndarray],
    titles: list[str],
    save_path: str | Path,
) -> Path:
    """
    Plot a grid comparing original (raw, resized-for-display-only) images
    against their fully preprocessed-then-denormalized counterparts.

    This is a sanity check, not a claim of pixel-perfect equivalence: the
    "original" column is only resized for consistent display, while the
    "preprocessed" column has gone through the full resize -> tensor ->
    normalize -> denormalize round trip. If normalization/denormalization
    is implemented correctly, the two should look visually very close
    (same content, same orientation, no color corruption or distortion).

    Returns the path the figure was saved to.
    """
    n = len(originals)
    if n == 0:
        raise ValueError("No images provided to plot_preprocessing_check.")

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(originals[i])
        axes[0, i].set_title(f"{titles[i]}\n(original, resized)", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(preprocessed[i])
        axes[1, i].set_title("preprocessed\n(resize+normalize+denormalize)", fontsize=10)
        axes[1, i].axis("off")

    fig.suptitle("Preprocessing Sanity Check: Original vs. Round-Tripped", fontsize=13)
    fig.tight_layout()

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_category_sample_grid(
    images: list[np.ndarray],
    labels: list[str],
    save_path: str | Path,
    ncols: int = 4,
) -> Path:
    """
    Plot a grid of sample images with text labels underneath each
    (e.g. defect type or "normal"/"anomalous"). Used for a quick visual
    overview of a category's train/test images.
    """
    n = len(images)
    if n == 0:
        raise ValueError("No images provided to plot_category_sample_grid.")

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)  # flatten regardless of shape

    for i in range(len(axes)):
        if i < n:
            axes[i].imshow(images[i])
            axes[i].set_title(labels[i], fontsize=10)
        axes[i].axis("off")

    fig.tight_layout()

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return save_path


# -----------------------------------------------------------------------
# Phase 8: evaluation plots
# -----------------------------------------------------------------------
def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: str | Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", color="#1f77b4", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve {title}".strip())
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, ap: float, save_path: str | Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})", color="#d62728", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve {title}".strip())
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrix_fig(cm: np.ndarray, save_path: str | Path, title: str = "") -> Path:
    """cm: 2x2 array from sklearn.confusion_matrix(labels=[0,1]) -> [[TN,FP],[FN,TP]]."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["NORMAL", "ANOMALOUS"], yticklabels=["NORMAL", "ANOMALOUS"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix {title}".strip())
    fig.tight_layout()

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_score_distribution(
    scores: np.ndarray, labels: np.ndarray, threshold: float, save_path: str | Path, title: str = ""
) -> Path:
    """Histogram of anomaly scores, split by true label, with the decision threshold marked."""
    fig, ax = plt.subplots(figsize=(8, 5))

    normal_scores = scores[labels == 0]
    anomalous_scores = scores[labels == 1]

    bins = np.linspace(scores.min(), scores.max(), 40)
    if len(normal_scores) > 0:
        ax.hist(normal_scores, bins=bins, alpha=0.6, label="Normal", color="#2ca02c")
    if len(anomalous_scores) > 0:
        ax.hist(anomalous_scores, bins=bins, alpha=0.6, label="Anomalous", color="#d62728")

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Anomaly Score Distribution {title}".strip())
    ax.legend()
    fig.tight_layout()

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_category_comparison(
    categories: list[str], metric_values: list[float], metric_name: str, save_path: str | Path
) -> Path:
    """Simple bar chart comparing one metric across categories."""
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.2), 5))
    bars = ax.bar(categories, metric_values, color="#1f77b4")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by Category")
    ax.set_ylim(0, max(1.05, max(metric_values) * 1.1 if metric_values else 1.0))

    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path
