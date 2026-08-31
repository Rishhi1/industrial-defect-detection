"""
localization.py
================
Converts a patch-level anomaly score map (e.g. 28x28, one score per spatial
feature location) into a full-resolution heatmap overlaid on the original
image — the visual "here's where the defect is" output of the system.

Pipeline:
    patch score map (h, w)
        -> upsample to (image_size, image_size) via bilinear interpolation
        -> normalize to [0, 1] (see normalize_heatmap for the two modes)
        -> apply a colormap (JET: blue=low anomaly, red=high anomaly)
        -> alpha-blend onto the original image

Normalization mode matters for how interpretable the heatmap is:
- "threshold_relative" (default): scales scores relative to the category's
  decision threshold, so red consistently means "meaningfully past the
  anomaly cutoff" across different images, not just "the worst pixel in
  THIS image" — the latter would make an entirely normal image's mildest
  texture variation look just as red as a genuinely defective image's
  actual defect, which is misleading.
- "minmax": simple per-image min-max normalization. Always shows some red
  region even on a perfectly normal image, since it stretches contrast
  regardless of actual anomaly magnitude. Useful for a rough "what is the
  network looking at" check, but should not be read as a severity signal.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import ensure_dir


def upsample_score_map(score_map: np.ndarray, target_size: int) -> np.ndarray:
    """
    Upsample a (h, w) patch-level score map to (target_size, target_size)
    using bilinear interpolation, matching the original image resolution.
    """
    return cv2.resize(
        score_map.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )


def normalize_heatmap(
    upsampled_map: np.ndarray,
    mode: str = "threshold_relative",
    threshold: float | None = None,
) -> np.ndarray:
    """
    Normalize an upsampled score map to [0, 1] for colormap display.

    mode="threshold_relative" requires `threshold` (the category's selected
    anomaly threshold). Maps score=0 -> 0.0 and score=2*threshold -> 1.0,
    clipped to [0, 1]. This means the heatmap turns solidly red only once
    scores are well past the actual decision boundary, not just "locally
    highest in this image."

    mode="minmax" ignores `threshold` and stretches purely from this
    image's own min/max score.
    """
    if mode == "threshold_relative":
        if threshold is None or threshold <= 0:
            raise ValueError("normalize_heatmap(mode='threshold_relative') requires a positive threshold.")
        upper_bound = threshold * 2.0
        normalized = upsampled_map / upper_bound
        return np.clip(normalized, 0.0, 1.0)

    elif mode == "minmax":
        mn, mx = upsampled_map.min(), upsampled_map.max()
        span = max(mx - mn, 1e-8)
        return np.clip((upsampled_map - mn) / span, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown normalization mode '{mode}'. Valid: 'threshold_relative', 'minmax'.")


def apply_colormap(normalized_map: np.ndarray) -> np.ndarray:
    """
    Apply a JET colormap (blue=low, green/yellow=mid, red=high anomaly) to
    a [0, 1]-normalized map. Returns an (H, W, 3) uint8 RGB array.
    """
    heat_uint8 = (normalized_map * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return colored_rgb


def create_overlay(original_rgb: np.ndarray, colored_heatmap_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Alpha-blend a colored heatmap onto the original image.
    original_rgb, colored_heatmap_rgb: (H, W, 3) uint8, same shape.
    alpha: heatmap opacity (0=invisible, 1=heatmap only, no original image).
    """
    if original_rgb.shape != colored_heatmap_rgb.shape:
        raise ValueError(
            f"Shape mismatch: original {original_rgb.shape} vs heatmap {colored_heatmap_rgb.shape}. "
            f"Both must be the same (H, W, 3)."
        )
    return cv2.addWeighted(original_rgb, 1 - alpha, colored_heatmap_rgb, alpha, 0)


def localize_anomaly(
    patch_score_map: np.ndarray,
    original_rgb: np.ndarray,
    image_size: int,
    threshold: float | None = None,
    normalization_mode: str = "threshold_relative",
    alpha: float = 0.45,
) -> dict:
    """
    Full localization pipeline for a single image: patch score map -> full
    pipeline -> {raw_upsampled, heatmap_colored, overlay}.

    Returns a dict with:
        "raw_upsampled": (image_size, image_size) float32 — actual anomaly
            scores at full resolution, useful for pixel-level metrics.
        "heatmap_colored": (image_size, image_size, 3) uint8 — colormapped.
        "overlay": (image_size, image_size, 3) uint8 — blended onto original.
    """
    upsampled = upsample_score_map(patch_score_map, image_size)
    normalized = normalize_heatmap(upsampled, mode=normalization_mode, threshold=threshold)
    colored = apply_colormap(normalized)
    overlay = create_overlay(original_rgb, colored, alpha=alpha)

    return {
        "raw_upsampled": upsampled,
        "heatmap_colored": colored,
        "overlay": overlay,
    }


def plot_localization_result(
    original_rgb: np.ndarray,
    heatmap_colored: np.ndarray,
    overlay: np.ndarray,
    save_path: str | Path,
    title: str = "",
    ground_truth_mask: np.ndarray | None = None,
    image_score: float | None = None,
    threshold: float | None = None,
    predicted_label: str | None = None,
) -> Path:
    """
    Save a comparison figure: Original | Heatmap | Overlay, and — when a
    ground-truth defect mask is available (MVTec provides these for
    defective test images) — a fourth panel showing it for visual
    validation of the localization quality.
    """
    n_panels = 4 if ground_truth_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_colored)
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if ground_truth_mask is not None:
        axes[3].imshow(ground_truth_mask, cmap="gray")
        axes[3].set_title("Ground Truth Defect Mask")
        axes[3].axis("off")

    subtitle_parts = []
    if image_score is not None:
        subtitle_parts.append(f"Anomaly Score: {image_score:.3f}")
    if threshold is not None:
        subtitle_parts.append(f"Threshold: {threshold:.3f}")
    if predicted_label is not None:
        subtitle_parts.append(f"Prediction: {predicted_label}")
    subtitle = "  |  ".join(subtitle_parts)

    full_title = f"{title}\n{subtitle}" if subtitle else title
    fig.suptitle(full_title, fontsize=12)
    fig.tight_layout()

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return save_path
