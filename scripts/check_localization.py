"""
check_localization.py
======================
Phase 7 sanity check: runs anomaly localization on a handful of real test
images per category (both normal and defective, across different defect
types where possible) and saves Original | Heatmap | Overlay | Ground-Truth
comparison figures.

This is a visual/qualitative check — proper pixel-level localization
metrics (pixel ROC-AUC etc.) come in Phase 8's full evaluation. Here we're
just confirming the heatmaps actually highlight defective regions, and that
normal images don't light up everywhere.

Usage:
    python scripts/check_localization.py
    python scripts/check_localization.py --category bottle --num-per-type 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.anomaly_detector import AnomalyDetector, MemoryBank
from src.dataset import MVTecDatasetError, MVTecTestDataset
from src.feature_extractor import FeatureExtractor
from src.localization import localize_anomaly, plot_localization_result
from src.preprocessing import build_inference_transform, denormalize
from src.utils import discover_categories, get_device, load_config, log_error, log_info


def pick_samples_per_defect_type(test_ds: MVTecTestDataset, num_per_type: int) -> list[int]:
    """Return dataset indices covering each defect type (including 'good'), up to num_per_type each."""
    by_type: dict[str, list[int]] = {}
    for idx, sample in enumerate(test_ds.samples):
        by_type.setdefault(sample.defect_type, []).append(idx)

    selected = []
    for defect_type, indices in sorted(by_type.items()):
        selected.extend(indices[:num_per_type])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly localization on sample test images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--num-per-type", type=int, default=2, help="Samples per defect type (including 'good').")
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

    models_dir = Path(config["output"]["models_dir"])
    predictions_dir = Path(config["output"]["predictions_dir"])

    log_info("Loading feature extractor...")
    extractor = FeatureExtractor(config).to(device)

    image_size = config["preprocessing"]["image_size"]
    transform = build_inference_transform(config)

    for category in categories:
        print(f"\n{'=' * 70}")
        print(f"Localization check: {category}")
        print("=" * 70)

        memory_bank_path = models_dir / f"{category}_memory_bank.pt"
        threshold_path = models_dir / f"{category}_threshold.json"

        if not memory_bank_path.exists():
            log_error(f"[{category}] No memory bank found. Run scripts/build_memory_bank.py first.")
            continue
        if not threshold_path.exists():
            log_error(f"[{category}] No threshold found. Run scripts/select_threshold.py first.")
            continue

        memory_bank = MemoryBank.load(memory_bank_path)
        detector = AnomalyDetector(extractor, memory_bank, config)

        with open(threshold_path) as f:
            threshold_data = json.load(f)
        threshold = threshold_data["threshold"]

        try:
            test_ds = MVTecTestDataset(dataset_root, category, transform=transform)
        except MVTecDatasetError as e:
            log_error(f"[{category}] {e}")
            continue

        indices = pick_samples_per_defect_type(test_ds, args.num_per_type)
        log_info(f"[{category}] Running localization on {len(indices)} sample(s) across "
                  f"{len(set(test_ds.samples[i].defect_type for i in indices))} defect type(s)...")

        category_out_dir = predictions_dir / category
        correct = 0

        for idx in indices:
            raw_sample = test_ds[idx]
            image_tensor = raw_sample["image"].unsqueeze(0).to(device)
            true_label = raw_sample["label"]
            defect_type = raw_sample["defect_type"]
            gt_mask = raw_sample["mask"]

            score_map = detector.compute_patch_score_map(image_tensor)[0]  # (h, w)
            image_score = detector.compute_image_scores(image_tensor)[0]
            predicted_label = "ANOMALOUS" if image_score >= threshold else "NORMAL"
            true_label_str = "ANOMALOUS" if true_label == 1 else "NORMAL"
            is_correct = (predicted_label == true_label_str)
            correct += int(is_correct)

            original_rgb = denormalize(raw_sample["image"], config)

            result = localize_anomaly(
                score_map, original_rgb, image_size, threshold=threshold, normalization_mode="threshold_relative"
            )

            gt_display = None
            if gt_mask is not None:
                gt_display = cv2_resize_mask(gt_mask, image_size)

            save_path = category_out_dir / f"{defect_type}_{idx}_{'correct' if is_correct else 'WRONG'}.png"
            plot_localization_result(
                original_rgb,
                result["heatmap_colored"],
                result["overlay"],
                save_path,
                title=f"{category} / {defect_type} (true: {true_label_str})",
                ground_truth_mask=gt_display,
                image_score=image_score,
                threshold=threshold,
                predicted_label=predicted_label,
            )

        accuracy = correct / len(indices) if indices else 0.0
        log_info(f"[{category}] Sample accuracy on this small check set: {correct}/{len(indices)} ({accuracy*100:.0f}%)")
        log_info(f"[{category}] Saved localization figures to '{category_out_dir}/'")
        log_info(
            f"[{category}] NOTE: this is a small qualitative sample, not a full evaluation "
            f"(that's Phase 8). Visually inspect the saved figures — heatmaps for defective "
            f"images should highlight regions overlapping the ground-truth mask; heatmaps for "
            f"normal images should stay mostly blue/cool colored."
        )


def cv2_resize_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    import cv2
    return cv2.resize(mask.astype(np.uint8) * 255, (target_size, target_size), interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    main()
