"""
evaluate.py
===========
Phase 8: full evaluation suite. For each category:

1. Load the saved memory bank + threshold.
2. Score every test image (reusing Phase 6's saved scores when available,
   to avoid redundant computation).
3. Compute image-level metrics (accuracy, precision, recall, F1, ROC-AUC,
   PR-AUC) and pixel-level metrics (pixel ROC-AUC, pixel average precision).
4. Generate and save: ROC curve, PR curve, confusion matrix, score
   distribution plots.
5. Print a per-category results table.

After all categories: a cross-category comparison table and bar charts,
with pooled and macro-averaged overall metrics (both reported explicitly —
see evaluation.py's module docstring for why these differ).

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --category bottle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.anomaly_detector import AnomalyDetector, MemoryBank
from src.dataset import MVTecDatasetError, MVTecTestDataset
from src.evaluation import (
    aggregate_macro,
    aggregate_pooled,
    compute_image_level_metrics,
    compute_pixel_level_metrics,
    get_confusion_matrix,
    get_pr_curve_data,
    get_roc_curve_data,
)
from src.feature_extractor import FeatureExtractor
from src.preprocessing import build_inference_transform
from src.utils import discover_categories, ensure_dir, get_device, load_config, log_error, log_info
from src.visualization import (
    plot_category_comparison,
    plot_confusion_matrix_fig,
    plot_pr_curve,
    plot_roc_curve,
    plot_score_distribution,
)


def _test_collate(batch):
    """Module-level collate function — see select_threshold.py for why this can't be a nested closure."""
    images = torch.stack([b["image"] for b in batch])
    labels = [b["label"] for b in batch]
    defect_types = [b["defect_type"] for b in batch]
    masks = [b["mask"] for b in batch]
    return images, labels, defect_types, masks


def score_and_localize_test_set(
    dataset_root: Path, category: str, config: dict, detector: AnomalyDetector, device: torch.device
) -> dict:
    """
    Run every test image through the detector, collecting both image-level
    scores AND full-resolution pixel score maps (needed for pixel-level
    metrics). Ground-truth masks are resized to match; normal images
    contribute an all-zero mask.
    """
    from src.localization import upsample_score_map

    transform = build_inference_transform(config)
    test_ds = MVTecTestDataset(dataset_root, category, transform=transform)
    image_size = config["preprocessing"]["image_size"]

    batch_size = config["dataloader"]["batch_size"]
    num_workers = config["dataloader"]["num_workers"]
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_test_collate
    )

    all_image_scores = []
    all_labels = []
    all_defect_types = []
    all_pixel_scores = []
    all_pixel_labels = []

    for images, labels, defect_types, masks in loader:
        images = images.to(device)
        patch_maps = detector.compute_patch_score_map(images)  # (B, h, w)
        image_scores = detector.compute_image_scores(images)   # (B,)

        all_image_scores.append(image_scores)
        all_labels.extend(labels)
        all_defect_types.extend(defect_types)

        for i in range(len(labels)):
            upsampled = upsample_score_map(patch_maps[i], image_size)
            all_pixel_scores.append(upsampled.flatten())

            if masks[i] is not None:
                import cv2
                gt_resized = cv2.resize(
                    masks[i].astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST
                )
                all_pixel_labels.append(gt_resized.flatten())
            else:
                # Normal image: no defect anywhere -> all-zero ground truth.
                all_pixel_labels.append(np.zeros(image_size * image_size, dtype=np.uint8))

    return {
        "image_scores": np.concatenate(all_image_scores),
        "labels": np.array(all_labels),
        "defect_types": all_defect_types,
        "pixel_scores": np.concatenate(all_pixel_scores),
        "pixel_labels": np.concatenate(all_pixel_labels),
    }


def evaluate_category(
    dataset_root: Path, category: str, config: dict, extractor: FeatureExtractor, device: torch.device,
    models_dir: Path, plots_dir: Path, metrics_dir: Path,
) -> dict | None:
    memory_bank_path = models_dir / f"{category}_memory_bank.pt"
    threshold_path = models_dir / f"{category}_threshold.json"

    if not memory_bank_path.exists():
        log_error(f"[{category}] No memory bank found. Run scripts/build_memory_bank.py first.")
        return None
    if not threshold_path.exists():
        log_error(f"[{category}] No threshold found. Run scripts/select_threshold.py first.")
        return None

    memory_bank = MemoryBank.load(memory_bank_path)
    detector = AnomalyDetector(extractor, memory_bank, config)

    with open(threshold_path) as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]

    log_info(f"[{category}] Scoring test set and computing pixel-level score maps...")
    try:
        data = score_and_localize_test_set(dataset_root, category, config, detector, device)
    except MVTecDatasetError as e:
        log_error(f"[{category}] {e}")
        return None

    image_metrics = compute_image_level_metrics(data["image_scores"], data["labels"], threshold)
    pixel_metrics = compute_pixel_level_metrics(data["pixel_scores"], data["pixel_labels"])

    log_info(
        f"[{category}] Image-level: Acc={image_metrics.accuracy:.3f} P={image_metrics.precision:.3f} "
        f"R={image_metrics.recall:.3f} F1={image_metrics.f1:.3f} ROC-AUC={image_metrics.roc_auc:.3f} "
        f"PR-AUC={image_metrics.pr_auc:.3f}"
    )
    log_info(
        f"[{category}] Pixel-level: ROC-AUC={pixel_metrics.pixel_roc_auc:.3f} "
        f"AP={pixel_metrics.pixel_average_precision:.3f} "
        f"({pixel_metrics.num_defect_pixels}/{pixel_metrics.num_pixels} defect pixels)"
    )

    # --- Plots ---
    category_plots_dir = plots_dir / category
    roc_data = get_roc_curve_data(data["image_scores"], data["labels"])
    plot_roc_curve(roc_data["fpr"], roc_data["tpr"], roc_data["auc"], category_plots_dir / "roc_curve.png", title=f"— {category}")

    pr_data = get_pr_curve_data(data["image_scores"], data["labels"])
    plot_pr_curve(pr_data["precision"], pr_data["recall"], pr_data["ap"], category_plots_dir / "pr_curve.png", title=f"— {category}")

    cm = get_confusion_matrix(data["image_scores"], data["labels"], threshold)
    plot_confusion_matrix_fig(cm, category_plots_dir / "confusion_matrix.png", title=f"— {category}")

    plot_score_distribution(
        data["image_scores"], data["labels"], threshold, category_plots_dir / "score_distribution.png", title=f"— {category}"
    )

    # --- Save metrics JSON ---
    ensure_dir(metrics_dir)
    metrics_output = {
        "category": category,
        "image_level": image_metrics.__dict__,
        "pixel_level": pixel_metrics.__dict__,
    }
    with open(metrics_dir / f"{category}_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)

    return {
        "category": category,
        "image_metrics": image_metrics,
        "pixel_metrics": pixel_metrics,
        "image_scores": data["image_scores"],
        "labels": data["labels"],
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full evaluation across one, several, or all categories.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        log_error(str(e))
        sys.exit(1)

    dataset_root = Path(config["dataset"]["root_path"])
    available = discover_categories(dataset_root)
    if not available:
        log_error(f"No dataset categories found under '{dataset_root}'.")
        sys.exit(1)

    if args.category:
        if args.category not in available:
            log_error(f"Category '{args.category}' not found. Available: {available}")
            sys.exit(1)
        categories = [args.category]
    else:
        selected = config["dataset"].get("selected_categories") or []
        categories = [c for c in selected if c in available] if selected else available

    device = get_device(config["model"]["device"])
    log_info(f"Using device: {device}")
    log_info(f"Evaluating categories: {categories}")

    log_info("Loading feature extractor...")
    extractor = FeatureExtractor(config).to(device)

    models_dir = Path(config["output"]["models_dir"])
    plots_dir = Path(config["output"]["plots_dir"])
    metrics_dir = Path(config["output"]["metrics_dir"])

    results = []
    for category in categories:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {category}")
        print("=" * 70)
        result = evaluate_category(dataset_root, category, config, extractor, device, models_dir, plots_dir, metrics_dir)
        if result is not None:
            results.append(result)

    if not results:
        log_error("No categories were successfully evaluated.")
        sys.exit(1)

    # --- Cross-category summary ---
    print(f"\n{'=' * 100}")
    print("CATEGORY-WISE EVALUATION SUMMARY")
    print("=" * 100)
    header = (
        f"{'Category':<12}{'Images':<9}{'Accuracy':<11}{'Precision':<12}{'Recall':<10}"
        f"{'F1':<9}{'ROC-AUC':<10}{'PixROC-AUC':<12}"
    )
    print(header)
    print("-" * 100)
    for r in results:
        im = r["image_metrics"]
        pm = r["pixel_metrics"]
        print(
            f"{r['category']:<12}{im.num_images:<9}{im.accuracy:<11.4f}{im.precision:<12.4f}"
            f"{im.recall:<10.4f}{im.f1:<9.4f}{im.roc_auc:<10.4f}{pm.pixel_roc_auc:<12.4f}"
        )
    print("=" * 100)

    # --- Overall aggregation: pooled and macro, both reported explicitly ---
    all_scores = [r["image_scores"] for r in results]
    all_labels = [r["labels"] for r in results]
    all_thresholds = [r["threshold"] for r in results]

    pooled = aggregate_pooled(all_scores, all_labels, all_thresholds)
    macro = aggregate_macro([r["image_metrics"] for r in results])

    print("\nOVERALL METRICS — two aggregation methods (see evaluation.py docstring for the distinction):")
    print(f"  POOLED (all {pooled['num_images']} test images combined, weighted by category size):")
    print(f"    Accuracy={pooled['accuracy']:.4f}  Precision={pooled['precision']:.4f}  "
          f"Recall={pooled['recall']:.4f}  F1={pooled['f1']:.4f}")
    print(f"  MACRO (simple mean across {macro['num_categories']} categories, equal weight each):")
    print(f"    Accuracy={macro['accuracy']:.4f}  Precision={macro['precision']:.4f}  "
          f"Recall={macro['recall']:.4f}  F1={macro['f1']:.4f}  ROC-AUC={macro['roc_auc']:.4f}")

    # --- Category comparison bar charts ---
    categories_list = [r["category"] for r in results]
    plot_category_comparison(
        categories_list, [r["image_metrics"].roc_auc for r in results], "ROC-AUC", plots_dir / "category_comparison_roc_auc.png"
    )
    plot_category_comparison(
        categories_list, [r["image_metrics"].f1 for r in results], "F1 Score", plots_dir / "category_comparison_f1.png"
    )

    # --- Save overall summary JSON ---
    summary_output = {
        "per_category": {r["category"]: {**r["image_metrics"].__dict__, "pixel_roc_auc": r["pixel_metrics"].pixel_roc_auc} for r in results},
        "pooled": pooled,
        "macro": macro,
    }
    with open(metrics_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary_output, f, indent=2)

    log_info(f"\nAll metrics saved to '{metrics_dir}/', all plots saved to '{plots_dir}/'.")


if __name__ == "__main__":
    main()
