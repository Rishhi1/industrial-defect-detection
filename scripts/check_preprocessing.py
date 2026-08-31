"""
check_preprocessing.py
=======================
Sanity-check script for Phase 3. Loads a handful of sample images from the
dataset, runs them through the exact preprocessing pipeline used everywhere
else in the project (resize -> tensor -> normalize), then denormalizes them
back for display. Saves a comparison figure to results/plots/ so you can
visually confirm nothing is being distorted, mis-colored, or corrupted
before we build the feature extractor on top of this pipeline.

Usage:
    python scripts/check_preprocessing.py
    python scripts/check_preprocessing.py --category bottle --num-samples 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.dataset import MVTecDatasetError, MVTecTrainDataset
from src.preprocessing import build_inference_transform, denormalize
from src.utils import discover_categories, load_config, log_error, log_info
from src.visualization import plot_preprocessing_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a preprocessing sanity check and save a comparison plot.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--category", type=str, default=None, help="Category to sample from (default: first configured category).")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of sample images to check.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        log_error(str(e))
        sys.exit(1)

    dataset_root = Path(config["dataset"]["root_path"])
    available = discover_categories(dataset_root)
    if not available:
        log_error(
            f"No dataset categories found under '{dataset_root}'. "
            f"Run scripts/prepare_dataset.py first to diagnose."
        )
        sys.exit(1)

    category = args.category
    if category is None:
        selected = config["dataset"].get("selected_categories") or []
        category = selected[0] if selected else available[0]

    if category not in available:
        log_error(f"Category '{category}' not found. Available: {available}")
        sys.exit(1)

    log_info(f"Running preprocessing sanity check on category '{category}'...")

    try:
        train_ds = MVTecTrainDataset(dataset_root, category, transform=None)
    except MVTecDatasetError as e:
        log_error(str(e))
        sys.exit(1)

    num_samples = min(args.num_samples, len(train_ds))
    if num_samples < args.num_samples:
        log_info(f"Only {len(train_ds)} training images available; using {num_samples} instead of {args.num_samples}.")

    image_size = config["preprocessing"]["image_size"]
    transform = build_inference_transform(config)

    originals = []
    preprocessed = []
    titles = []

    for i in range(num_samples):
        raw_image = train_ds[i]  # PIL Image, RGB, untransformed
        # "Original, resized" column: resize only, no normalization, purely for
        # fair visual comparison against the preprocessed column.
        display_original = np.array(raw_image.resize((image_size, image_size)))

        tensor = transform(raw_image)
        display_preprocessed = denormalize(tensor, config)

        originals.append(display_original)
        preprocessed.append(display_preprocessed)
        titles.append(f"{category} #{i}")

    plots_dir = Path(config["output"]["plots_dir"])
    save_path = plots_dir / f"preprocessing_check_{category}.png"
    saved_to = plot_preprocessing_check(originals, preprocessed, titles, save_path)

    log_info(f"Saved sanity-check plot to '{saved_to}'")
    log_info(
        "Visually inspect the plot: the 'preprocessed' row should look like the same "
        "images as the 'original' row (same content, orientation, and roughly the same "
        "colors) after the round trip through normalization and denormalization. "
        "Minor color/contrast shifts are expected from float rounding; structural "
        "distortion, misalignment, or color channel swaps are NOT expected and would "
        "indicate a bug."
    )


if __name__ == "__main__":
    main()
