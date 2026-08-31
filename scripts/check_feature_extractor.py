"""
check_feature_extractor.py
===========================
Sanity-check script for Phase 4. Loads the configured backbone, runs a
handful of real dataset images through it, and verifies:

1. Forward hooks fire correctly and return feature maps of sane shape.
2. Multi-layer patch embeddings are correctly aligned and concatenated.
3. Feature values are not degenerate (all-zero, NaN, or constant), which
   would indicate a hook wired to the wrong layer or a frozen/broken model.

Also saves a quick visualization: the mean activation map per layer,
upsampled back to image size — a very rough preview of what localization
will eventually look like (developed properly in Phase 7).

Usage:
    python scripts/check_feature_extractor.py
    python scripts/check_feature_extractor.py --category bottle --num-samples 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import MVTecDatasetError, MVTecTrainDataset
from src.feature_extractor import FeatureExtractor
from src.preprocessing import build_inference_transform, denormalize
from src.utils import discover_categories, get_device, load_config, log_error, log_info, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the feature extractor on real dataset images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=2)
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

    category = args.category
    if category is None:
        selected = config["dataset"].get("selected_categories") or []
        category = selected[0] if selected else available[0]
    if category not in available:
        log_error(f"Category '{category}' not found. Available: {available}")
        sys.exit(1)

    device = get_device(config["model"]["device"])
    log_info(f"Using device: {device}")
    log_info(f"Backbone: {config['model']['backbone']}, feature layers: {config['model']['feature_layers']}")

    log_info("Loading backbone (downloads ImageNet weights on first run)...")
    extractor = FeatureExtractor(config).to(device)

    try:
        train_ds = MVTecTrainDataset(dataset_root, category, transform=None)
    except MVTecDatasetError as e:
        log_error(str(e))
        sys.exit(1)

    num_samples = min(args.num_samples, len(train_ds))
    transform = build_inference_transform(config)

    batch = torch.stack([transform(train_ds[i]) for i in range(num_samples)]).to(device)
    log_info(f"Input batch shape: {tuple(batch.shape)}")

    # --- Check 1: raw per-layer feature maps ---
    raw_features = extractor.extract(batch)
    for layer_name, feat in raw_features.items():
        has_nan = torch.isnan(feat).any().item()
        is_all_zero = torch.all(feat == 0).item()
        log_info(
            f"  [{layer_name}] shape={tuple(feat.shape)}  "
            f"mean={feat.mean().item():.4f}  std={feat.std().item():.4f}  "
            f"NaN={has_nan}  all_zero={is_all_zero}"
        )
        if has_nan or is_all_zero:
            log_error(f"Layer '{layer_name}' produced degenerate output — check the backbone/hook wiring.")
            sys.exit(1)

    # --- Check 2: combined patch embedding ---
    patch_embed = extractor.get_patch_embeddings(batch)
    expected_channels = extractor.total_embedding_channels()
    log_info(f"Combined patch embedding shape: {tuple(patch_embed.shape)} (expected channels: {expected_channels})")
    assert patch_embed.shape[1] == expected_channels, "Channel count mismatch in combined embedding!"

    # --- Visualization: mean activation heatmap per layer, upsampled to image size ---
    image_size = config["preprocessing"]["image_size"]
    plots_dir = Path(config["output"]["plots_dir"])
    ensure_dir(plots_dir)

    n_layers = len(raw_features)
    fig, axes = plt.subplots(num_samples, n_layers + 1, figsize=(4 * (n_layers + 1), 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        original = denormalize(batch[i].cpu(), config)
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"{category} #{i}\n(input image)")
        axes[i, 0].axis("off")

        for j, (layer_name, feat) in enumerate(raw_features.items()):
            activation_map = feat[i].mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W), stay in torch
            upsampled = torch.nn.functional.interpolate(
                activation_map,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()
            axes[i, j + 1].imshow(original)
            axes[i, j + 1].imshow(upsampled, cmap="jet", alpha=0.5)
            axes[i, j + 1].set_title(f"{layer_name}\nmean activation")
            axes[i, j + 1].axis("off")

    fig.suptitle(f"Feature Extractor Sanity Check — {config['model']['backbone']}", fontsize=13)
    fig.tight_layout()
    save_path = plots_dir / f"feature_extractor_check_{category}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    log_info(f"Saved feature activation visualization to '{save_path}'")
    log_info(
        "This is NOT the final anomaly localization (that's Phase 7) — it's just a raw "
        "'what is each layer paying attention to' check. Expect it to highlight edges/"
        "textures/structure broadly, not defects specifically, since these are normal "
        "training images and the model hasn't been compared against any reference yet."
    )
    log_info("Feature extractor verification passed.")


if __name__ == "__main__":
    main()
