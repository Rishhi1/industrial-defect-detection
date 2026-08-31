"""
preprocessing.py
=================
Image preprocessing pipeline: resizing, RGB conversion, normalization, and
tensor conversion. All parameters (image size, normalization stats) come
from config.yaml — nothing here is hardcoded.

Two transform builders are provided:

- build_inference_transform: deterministic resize + normalize, used for
  building the memory bank and for scoring test/inference images. No
  randomness, so results are reproducible.

- build_augmentation_transform: adds light, industrially-sensible
  augmentation (small rotations/flips) for cases where the reference set
  benefits from a bit more variety. Off by default — MVTec normal images
  are already fairly consistent, and heavy augmentation can distort
  fine-grained texture/defect cues, which matters a lot for anomaly
  detection. Only enable this deliberately.

A denormalize() helper is included so preprocessed tensors can be converted
back to viewable images for the sanity-check visualizations in
visualization.py.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms


def build_inference_transform(config: dict) -> transforms.Compose:
    """
    Standard deterministic preprocessing pipeline used for both building the
    memory bank (normal training images) and for scoring test/inference
    images. Must be identical for both, or feature distances become
    meaningless.
    """
    image_size = config["preprocessing"]["image_size"]
    mean = config["preprocessing"]["normalize_mean"]
    std = config["preprocessing"]["normalize_std"]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),                 # HWC [0,255] uint8 -> CHW [0,1] float
        transforms.Normalize(mean=mean, std=std),
    ])


def build_augmentation_transform(config: dict) -> transforms.Compose:
    """
    Optional, lightly-augmented pipeline. NOT used by default anywhere in
    the pipeline. Only reach for this if the reference memory bank needs
    more variety for a specific category — e.g. if defects in that category
    tend to appear at rotated/mirrored orientations relative to training
    images.
    """
    image_size = config["preprocessing"]["image_size"]
    mean = config["preprocessing"]["normalize_mean"]
    std = config["preprocessing"]["normalize_std"]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),   # small — industrial parts are rarely wildly rotated
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def denormalize(tensor: torch.Tensor, config: dict) -> np.ndarray:
    """
    Reverse ImageNet normalization on a preprocessed CHW tensor and return an
    HWC uint8 numpy array suitable for display with matplotlib/OpenCV.

    Used purely for sanity-check visualization — never for feeding back
    into the model.
    """
    mean = torch.tensor(config["preprocessing"]["normalize_mean"]).view(3, 1, 1)
    std = torch.tensor(config["preprocessing"]["normalize_std"]).view(3, 1, 1)

    denorm = tensor.clone().detach().cpu()
    denorm = denorm * std + mean
    denorm = denorm.clamp(0, 1)

    array = (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return array
