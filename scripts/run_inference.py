"""
run_inference.py
=================
Phase 9: run the trained anomaly detection system on a single image.

Loads the saved memory bank + threshold for a category, preprocesses the
given image, scores it, and reports:

    Prediction: ANOMALOUS
    Anomaly Score: 0.82
    Threshold: 0.51
    Processing Time: 0.43 seconds

Also saves the heatmap and overlay visualization to disk.

Usage:
    python scripts/run_inference.py --image path/to/image.jpg --category bottle
    python scripts/run_inference.py --image path/to/image.jpg --category bottle --output-dir results/predictions/my_run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.anomaly_detector import AnomalyDetector, MemoryBank
from src.feature_extractor import FeatureExtractor
from src.localization import localize_anomaly, plot_localization_result
from src.preprocessing import build_inference_transform, denormalize
from src.utils import discover_categories, ensure_dir, get_device, load_config, log_error, log_info

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_and_validate_image(image_path: Path) -> Image.Image:
    """
    Load an arbitrary user-supplied image file with clear, specific errors
    for the failure modes an end user is likely to hit: missing file,
    unsupported format, corrupted file.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: '{image_path}'.")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{image_path.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}."
        )

    try:
        img = Image.open(image_path)
        img.load()  # force-read now so a corrupt file fails here with a clear message
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Could not read image '{image_path}' — the file may be corrupted: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly detection inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to inspect.")
    parser.add_argument("--category", type=str, default=None, help="MVTec category (determines which memory bank/threshold to use).")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save the heatmap/overlay (default: results/predictions/inference/).")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        log_error(str(e))
        sys.exit(1)

    image_path = Path(args.image)
    try:
        raw_image = load_and_validate_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        log_error(str(e))
        sys.exit(1)

    dataset_root = Path(config["dataset"]["root_path"])
    available = discover_categories(dataset_root)

    category = args.category
    if category is None:
        selected = config["dataset"].get("selected_categories") or []
        if len(selected) == 1:
            category = selected[0]
        else:
            log_error(
                f"Multiple categories are configured ({selected or available}) — "
                f"please specify which one with --category."
            )
            sys.exit(1)

    models_dir = Path(config["output"]["models_dir"])
    memory_bank_path = models_dir / f"{category}_memory_bank.pt"
    threshold_path = models_dir / f"{category}_threshold.json"

    if not memory_bank_path.exists():
        log_error(
            f"No memory bank found for category '{category}' at '{memory_bank_path}'. "
            f"Run: python scripts/build_memory_bank.py --category {category}"
        )
        sys.exit(1)
    if not threshold_path.exists():
        log_error(
            f"No threshold found for category '{category}' at '{threshold_path}'. "
            f"Run: python scripts/select_threshold.py --category {category}"
        )
        sys.exit(1)

    device = get_device(config["model"]["device"])

    log_info(f"Category: {category}")
    log_info(f"Using device: {device}")
    log_info("Loading feature extractor and memory bank...")

    extractor = FeatureExtractor(config).to(device)
    memory_bank = MemoryBank.load(memory_bank_path)
    detector = AnomalyDetector(extractor, memory_bank, config)

    with open(threshold_path) as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]

    # --- Timed inference: preprocessing + forward pass + scoring ---
    start_time = time.time()

    transform = build_inference_transform(config)
    image_tensor = transform(raw_image).unsqueeze(0).to(device)

    patch_score_map = detector.compute_patch_score_map(image_tensor)[0]  # (h, w)
    image_score = float(detector.compute_image_scores(image_tensor)[0])

    processing_time = time.time() - start_time

    prediction = "ANOMALOUS" if image_score >= threshold else "NORMAL"

    # --- Localization / heatmap ---
    image_size = config["preprocessing"]["image_size"]
    original_rgb = denormalize(image_tensor[0].cpu(), config)

    result = localize_anomaly(
        patch_score_map, original_rgb, image_size, threshold=threshold, normalization_mode="threshold_relative"
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["predictions_dir"]) / "inference"
    ensure_dir(output_dir)

    stem = image_path.stem
    figure_path = output_dir / f"{stem}_result.png"
    plot_localization_result(
        original_rgb,
        result["heatmap_colored"],
        result["overlay"],
        figure_path,
        title=f"Inference: {image_path.name} ({category})",
        image_score=image_score,
        threshold=threshold,
        predicted_label=prediction,
    )

    # --- Report ---
    print()
    print(f"Prediction: {prediction}")
    print(f"Anomaly Score: {image_score:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Processing Time: {processing_time:.3f} seconds")
    print(f"Heatmap/overlay saved to: {figure_path}")
    print()

    # --- Also save a machine-readable result alongside the figure ---
    result_json_path = output_dir / f"{stem}_result.json"
    with open(result_json_path, "w") as f:
        json.dump(
            {
                "image_path": str(image_path),
                "category": category,
                "prediction": prediction,
                "anomaly_score": image_score,
                "threshold": threshold,
                "processing_time_seconds": processing_time,
            },
            f,
            indent=2,
        )
    log_info(f"Result JSON saved to: {result_json_path}")


if __name__ == "__main__":
    main()
