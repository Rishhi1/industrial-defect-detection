"""Tests for src/preprocessing.py: transform pipeline shape/range and denormalize round-trip."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.preprocessing import build_augmentation_transform, build_inference_transform, denormalize


class TestBuildInferenceTransform:
    def test_output_shape_matches_config(self, minimal_config):
        transform = build_inference_transform(minimal_config)
        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        tensor = transform(img)
        image_size = minimal_config["preprocessing"]["image_size"]
        assert tensor.shape == (3, image_size, image_size)

    def test_output_is_float_tensor(self, minimal_config):
        transform = build_inference_transform(minimal_config)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.dtype == torch.float32

    def test_deterministic_no_randomness(self, minimal_config):
        """Same input through the inference transform twice should give identical output."""
        transform = build_inference_transform(minimal_config)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        t1 = transform(img)
        t2 = transform(img)
        assert torch.allclose(t1, t2)


class TestBuildAugmentationTransform:
    def test_output_shape_matches_config(self, minimal_config):
        transform = build_augmentation_transform(minimal_config)
        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        tensor = transform(img)
        image_size = minimal_config["preprocessing"]["image_size"]
        assert tensor.shape == (3, image_size, image_size)


class TestDenormalize:
    def test_output_shape_is_hwc(self, minimal_config):
        tensor = torch.zeros(3, 64, 64)
        result = denormalize(tensor, minimal_config)
        assert result.shape == (64, 64, 3)

    def test_output_dtype_is_uint8(self, minimal_config):
        tensor = torch.randn(3, 64, 64)
        result = denormalize(tensor, minimal_config)
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self, minimal_config):
        tensor = torch.randn(3, 64, 64) * 10  # deliberately extreme values
        result = denormalize(tensor, minimal_config)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_roundtrip_preserves_approximate_content(self, minimal_config):
        """Preprocess a real image then denormalize — result should resemble the original."""
        transform = build_inference_transform(minimal_config)
        original = np.full((64, 64, 3), 128, dtype=np.uint8)  # flat mid-gray image
        img = Image.fromarray(original)

        tensor = transform(img)
        recovered = denormalize(tensor, minimal_config)

        # Allow generous tolerance for resize/normalize/denormalize float rounding.
        assert np.abs(int(recovered.mean()) - 128) < 10
