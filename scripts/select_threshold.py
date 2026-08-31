"""
select_threshold.py
====================
Phase 6: choose the NORMAL/ANOMALOUS decision threshold for each category.

Loads the category's saved memory bank, runs every TEST image (both 'good'
and every defect type) through the anomaly detector to get an image-level
score, then selects a threshold using the method configured in
config.yaml (Youden's J on the ROC curve by default).

Saves the result to models/<category>_threshold.json, which downstream
scripts (evaluate.py, run_inference.py, the Streamlit app) load rather than
recomputing.

Usage:
    python scripts/select_threshold.py
    python scripts/select_threshold.py --category bottle
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.anomaly_detector import AnomalyDetector, MemoryBank, select_threshold
from src.dataset import MVTecDatasetError, MVTecTestDataset
from src.feature_extractor import FeatureExtractor
from src.preprocessing import build_inference_transform
from src.utils import discover_categories, ensure_dir, get_device, load_config, log_error, log_info


def _test_collate(batch):
    """
    Custom collate function for MVTecTestDataset batches.

    Defined at module level (not nested inside another function) because
    DataLoader worker processes need to pickle the collate function when
    num_workers > 0 — a nested/local function can't be pickled and causes
    a PicklingError on some platforms (notably macOS with recent Python
    versions, which default to a spawn-based multiprocessing start method).
    """
    images = torch.stack([b["image"] for b in batch])
    labels = [b["label"] for b in batch]
    defect_types = [b["defect_type"] for b in batch]
    return images, labels, defect_types


def score_test_set(
    dataset_root: Path, category: str, config: dict, detector: AnomalyDetector, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Run every test image through the detector. Returns (scores, labels,
    defect_types) as parallel arrays/lists, one entry per test image.
    """
    transform = build_inference_transform(config)
    test_ds = MVTecTestDataset(dataset_root, category, transform=transform)

    batch_size = config["dataloader"]["batch_size"]
    num_workers = config["dataloader"]["num_workers"]

    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_test_collate
    )

    all_scores = []
    all_labels = []
    all_defect_types = []

    for images, labels, defect_types in loader:
        images = images.to(device)
        scores = detector.compute_image_scores(images)
        all_scores.append(scores)
        all_labels.extend(labels)
        all_defect_types.extend(defect_types)

    return np.concatenate(all_scores), np.array(all_labels), all_defect_types


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the anomaly-detection threshold per category.")
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

    models_dir = Path(config["output"]["models_dir"])

    log_info("Loading feature extractor...")
    extractor = FeatureExtractor(config).to(device)

    summary = []

    for category in categories:
        print(f"\n{'=' * 70}")
        print(f"Selecting threshold: {category}")
        print("=" * 70)

        memory_bank_path = models_dir / f"{category}_memory_bank.pt"
        if not memory_bank_path.exists():
            log_error(
                f"[{category}] No memory bank found at '{memory_bank_path}'. "
                f"Run scripts/build_memory_bank.py --category {category} first."
            )
            continue

        memory_bank = MemoryBank.load(memory_bank_path)
        detector = AnomalyDetector(extractor, memory_bank, config)

        try:
            scores, labels, defect_types = score_test_set(dataset_root, category, config, detector, device)
        except MVTecDatasetError as e:
            log_error(f"[{category}] {e}")
            continue

        log_info(f"[{category}] Scored {len(scores)} test images "
                  f"({int((labels == 0).sum())} normal, {int((labels == 1).sum())} anomalous).")

        result = select_threshold(scores, labels, config)

        log_info(
            f"[{category}] Threshold={result.threshold:.4f} (method={result.method}) — "
            f"ROC-AUC={result.roc_auc:.4f}, TPR={result.tpr_at_threshold:.3f}, FPR={result.fpr_at_threshold:.3f}"
        )

        # Save threshold + full raw scores (for reuse in Phase 8 evaluation
        # without having to rescore everything from scratch).
        output = {
            "category": category,
            **asdict(result),
            "raw_scores": scores.tolist(),
            "raw_labels": labels.tolist(),
            "raw_defect_types": defect_types,
        }
        save_path = models_dir / f"{category}_threshold.json"
        ensure_dir(save_path.parent)
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        log_info(f"[{category}] Saved threshold + scores to '{save_path}'")

        summary.append(result)

    if not summary:
        log_error("No thresholds were successfully selected.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("THRESHOLD SELECTION SUMMARY")
    print("=" * 70)
    header = f"{'Category':<15}{'Threshold':<12}{'Method':<12}{'ROC-AUC':<10}{'TPR':<8}{'FPR'}"
    print(header)
    print("-" * 70)
    successful_categories = [c for c in categories if (models_dir / f"{c}_threshold.json").exists()]
    for category, r in zip(successful_categories, summary):
        print(f"{category:<15}{r.threshold:<12.4f}{r.method:<12}{r.roc_auc:<10.4f}{r.tpr_at_threshold:<8.3f}{r.fpr_at_threshold:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
