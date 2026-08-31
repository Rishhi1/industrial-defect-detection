"""
build_memory_bank.py
=====================
Builds the reference memory bank(s) for one or more MVTec categories:
loads all normal ('good') training images, extracts patch embeddings via
the pretrained backbone, applies coreset subsampling, and saves the result
to disk so it doesn't need rebuilding every time the app or eval scripts run.

Usage:
    python scripts/build_memory_bank.py
    python scripts/build_memory_bank.py --category bottle
    python scripts/build_memory_bank.py --category bottle --coreset-ratio 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly_detector import build_memory_bank
from src.dataset import MVTecDatasetError
from src.feature_extractor import FeatureExtractor
from src.utils import discover_categories, ensure_dir, get_device, load_config, log_error, log_info, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference memory bank(s) from normal training images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--category", type=str, default=None, help="Build for a single category only.")
    parser.add_argument("--coreset-ratio", type=float, default=None, help="Override config's memory_bank.coreset_ratio.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        log_error(str(e))
        sys.exit(1)

    if args.coreset_ratio is not None:
        config["memory_bank"]["coreset_ratio"] = args.coreset_ratio

    set_seed(config["memory_bank"]["random_seed"])

    dataset_root = Path(config["dataset"]["root_path"])
    available = discover_categories(dataset_root)
    if not available:
        log_error(f"No dataset categories found under '{dataset_root}'. Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    if args.category:
        if args.category not in available:
            log_error(f"Category '{args.category}' not found. Available: {available}")
            sys.exit(1)
        categories = [args.category]
    else:
        selected = config["dataset"].get("selected_categories") or []
        categories = [c for c in selected if c in available] if selected else available

    if not categories:
        log_error("No valid categories to build memory banks for.")
        sys.exit(1)

    device = get_device(config["model"]["device"])
    log_info(f"Using device: {device}")
    log_info(f"Backbone: {config['model']['backbone']}, feature layers: {config['model']['feature_layers']}")
    log_info(f"Building memory banks for: {categories}")

    log_info("Loading feature extractor (downloads ImageNet weights on first run)...")
    extractor = FeatureExtractor(config).to(device)

    models_dir = Path(config["output"]["models_dir"])
    ensure_dir(models_dir)

    summary_rows = []

    for category in categories:
        print(f"\n{'=' * 70}")
        print(f"Building memory bank: {category}")
        print("=" * 70)

        try:
            memory_bank = build_memory_bank(dataset_root, category, config, extractor, device)
        except MVTecDatasetError as e:
            log_error(f"[{category}] {e}")
            continue

        save_path = models_dir / f"{category}_memory_bank.pt"
        memory_bank.save(save_path)

        meta = memory_bank.metadata
        summary_rows.append(meta)

        log_info(
            f"[{category}] Done in {meta.build_time_seconds:.1f}s — "
            f"{meta.num_training_images} images, {meta.total_patches_before_coreset} patches "
            f"-> {meta.num_patches_after_coreset} after coreset "
            f"({meta.embedding_dim}-d embeddings, grid={meta.patch_grid_size})."
        )

    if not summary_rows:
        log_error("No memory banks were successfully built.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("MEMORY BANK BUILD SUMMARY")
    print("=" * 70)
    header = f"{'Category':<15}{'Train imgs':<12}{'Patches (raw)':<16}{'Patches (coreset)':<19}{'Dim':<6}{'Time (s)'}"
    print(header)
    print("-" * 70)
    for m in summary_rows:
        print(
            f"{m.category:<15}{m.num_training_images:<12}{m.total_patches_before_coreset:<16}"
            f"{m.num_patches_after_coreset:<19}{m.embedding_dim:<6}{m.build_time_seconds:.1f}"
        )
    print("=" * 70)
    log_info(f"Memory banks saved under '{models_dir}/'.")


if __name__ == "__main__":
    main()
