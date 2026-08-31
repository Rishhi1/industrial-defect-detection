"""Tests for src/localization.py: upsampling, normalization, colormap, overlay."""

from __future__ import annotations

import numpy as np
import pytest

from src.localization import (
    apply_colormap,
    create_overlay,
    localize_anomaly,
    normalize_heatmap,
    upsample_score_map,
)


class TestUpsampleScoreMap:
    def test_output_shape_matches_target(self):
        score_map = np.random.rand(7, 7).astype(np.float32)
        result = upsample_score_map(score_map, target_size=64)
        assert result.shape == (64, 64)

    def test_preserves_relative_ordering_at_corners(self):
        """A score map with a clear hot corner should still have that corner hottest after upsampling."""
        score_map = np.zeros((4, 4), dtype=np.float32)
        score_map[0, 0] = 100.0  # top-left hot spot
        result = upsample_score_map(score_map, target_size=32)
        # Top-left region should have higher values than bottom-right region
        assert result[:8, :8].mean() > result[-8:, -8:].mean()


class TestNormalizeHeatmap:
    def test_threshold_relative_output_in_unit_range(self):
        score_map = np.array([[0, 5], [10, 20]], dtype=np.float32)
        result = normalize_heatmap(score_map, mode="threshold_relative", threshold=10.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_threshold_relative_requires_positive_threshold(self):
        score_map = np.array([[1, 2]], dtype=np.float32)
        with pytest.raises(ValueError, match="requires a positive threshold"):
            normalize_heatmap(score_map, mode="threshold_relative", threshold=None)

        with pytest.raises(ValueError, match="requires a positive threshold"):
            normalize_heatmap(score_map, mode="threshold_relative", threshold=0)

    def test_minmax_output_in_unit_range(self):
        score_map = np.array([[3, 7], [1, 15]], dtype=np.float32)
        result = normalize_heatmap(score_map, mode="minmax")
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_minmax_constant_map_does_not_crash(self):
        """A flat score map (all same value) shouldn't divide by zero."""
        score_map = np.full((4, 4), 5.0, dtype=np.float32)
        result = normalize_heatmap(score_map, mode="minmax")
        assert not np.isnan(result).any()

    def test_unknown_mode_raises(self):
        score_map = np.array([[1, 2]], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            normalize_heatmap(score_map, mode="rainbow_sparkle", threshold=1.0)


class TestApplyColormap:
    def test_output_shape_and_dtype(self):
        normalized = np.random.rand(32, 32).astype(np.float32)
        result = apply_colormap(normalized)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_low_and_high_values_get_different_colors(self):
        normalized = np.zeros((10, 10), dtype=np.float32)
        normalized[:, 5:] = 1.0  # right half is "hot"
        result = apply_colormap(normalized)
        left_color = result[:, :5].mean(axis=(0, 1))
        right_color = result[:, 5:].mean(axis=(0, 1))
        assert not np.allclose(left_color, right_color)


class TestCreateOverlay:
    def test_output_shape_matches_input(self):
        original = np.zeros((32, 32, 3), dtype=np.uint8)
        heatmap = np.full((32, 32, 3), 255, dtype=np.uint8)
        result = create_overlay(original, heatmap, alpha=0.5)
        assert result.shape == (32, 32, 3)

    def test_alpha_zero_is_essentially_original(self):
        original = np.full((16, 16, 3), 100, dtype=np.uint8)
        heatmap = np.full((16, 16, 3), 255, dtype=np.uint8)
        result = create_overlay(original, heatmap, alpha=0.0)
        assert np.allclose(result, original, atol=1)

    def test_shape_mismatch_raises(self):
        original = np.zeros((32, 32, 3), dtype=np.uint8)
        heatmap = np.zeros((16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Shape mismatch"):
            create_overlay(original, heatmap)


class TestLocalizeAnomalyPipeline:
    def test_full_pipeline_output_shapes(self):
        score_map = np.random.rand(7, 7).astype(np.float32) * 5
        original = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = localize_anomaly(score_map, original, image_size=64, threshold=2.0)

        assert result["raw_upsampled"].shape == (64, 64)
        assert result["heatmap_colored"].shape == (64, 64, 3)
        assert result["overlay"].shape == (64, 64, 3)
        assert result["heatmap_colored"].dtype == np.uint8
        assert result["overlay"].dtype == np.uint8

    def test_defective_region_produces_higher_scores_than_normal_region(self):
        """A score map with an obvious hot region should upsample to a hot region
        in roughly the same relative location — the core promise of localization."""
        score_map = np.ones((8, 8), dtype=np.float32) * 0.5
        score_map[1:3, 1:3] = 10.0  # hot spot near top-left
        original = np.zeros((64, 64, 3), dtype=np.uint8)

        result = localize_anomaly(score_map, original, image_size=64, threshold=1.0)
        raw = result["raw_upsampled"]

        top_left_region = raw[:20, :20].mean()
        bottom_right_region = raw[-20:, -20:].mean()
        assert top_left_region > bottom_right_region
