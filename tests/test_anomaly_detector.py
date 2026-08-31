"""
Tests for src/anomaly_detector.py: coreset subsampling, MemoryBank
save/load, k-NN scoring, threshold selection, and AnomalyDetector.

These tests use a lightweight FakeExtractor instead of the real pretrained
FeatureExtractor, so they run in milliseconds with no network access or
ImageNet weight download required — the pipeline LOGIC is what's under
test here, independent of which specific CNN backbone is plugged in.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.anomaly_detector import (
    AnomalyDetector,
    MemoryBank,
    MemoryBankMetadata,
    greedy_kcenter_coreset,
    random_coreset,
    select_threshold,
)


class FakeExtractor:
    """Returns a deterministic, fixed patch-embedding grid regardless of input,
    so AnomalyDetector's scoring logic can be tested without a real CNN."""

    def __init__(self, embedding_grid: torch.Tensor):
        self.embedding_grid = embedding_grid  # (C, h, w), reused for every image in the batch

    def get_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        return self.embedding_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)


# ---------------------------------------------------------------------
# Coreset subsampling
# ---------------------------------------------------------------------
class TestRandomCoreset:
    def test_ratio_controls_output_size(self):
        embeddings = np.random.randn(1000, 8).astype(np.float32)
        result = random_coreset(embeddings, ratio=0.1, seed=42)
        assert result.shape == (100, 8)

    def test_ratio_one_keeps_everything(self):
        embeddings = np.random.randn(50, 8).astype(np.float32)
        result = random_coreset(embeddings, ratio=1.0, seed=42)
        assert result.shape == (50, 8)

    def test_deterministic_with_same_seed(self):
        embeddings = np.random.randn(200, 8).astype(np.float32)
        r1 = random_coreset(embeddings, ratio=0.2, seed=7)
        r2 = random_coreset(embeddings, ratio=0.2, seed=7)
        assert np.array_equal(r1, r2)

    def test_minimum_one_point_kept(self):
        embeddings = np.random.randn(5, 8).astype(np.float32)
        result = random_coreset(embeddings, ratio=0.01, seed=42)
        assert result.shape[0] >= 1


class TestGreedyKCenterCoreset:
    def test_output_size_matches_ratio(self):
        embeddings = np.random.randn(100, 4).astype(np.float32)
        result = greedy_kcenter_coreset(embeddings, ratio=0.1, seed=42)
        assert result.shape == (10, 4)

    def test_covers_separated_clusters(self):
        """Greedy k-center should pick points from BOTH clusters, not just one."""
        cluster1 = np.random.randn(50, 4).astype(np.float32) + 0
        cluster2 = np.random.randn(50, 4).astype(np.float32) + 50
        data = np.concatenate([cluster1, cluster2])

        selected = greedy_kcenter_coreset(data, ratio=0.1, seed=42)
        near_c1 = (np.linalg.norm(selected - cluster1.mean(0), axis=1) < 25).sum()
        near_c2 = (np.linalg.norm(selected - cluster2.mean(0), axis=1) < 25).sum()
        assert near_c1 > 0 and near_c2 > 0


# ---------------------------------------------------------------------
# MemoryBank
# ---------------------------------------------------------------------
def _make_metadata(**overrides) -> MemoryBankMetadata:
    defaults = dict(
        category="widget", backbone="resnet18", feature_layers=["layer2", "layer3"],
        image_size=64, num_training_images=5, total_patches_before_coreset=100,
        num_patches_after_coreset=10, coreset_ratio=0.1, coreset_method="random",
        embedding_dim=8, patch_grid_size=(4, 4), random_seed=42,
    )
    defaults.update(overrides)
    return MemoryBankMetadata(**defaults)


class TestMemoryBank:
    def test_save_and_load_roundtrip(self, tmp_path):
        embeddings = np.random.randn(20, 8).astype(np.float32)
        metadata = _make_metadata()
        mb = MemoryBank(embeddings, metadata)

        save_path = tmp_path / "test_bank.pt"
        mb.save(save_path)
        loaded = MemoryBank.load(save_path)

        assert np.allclose(loaded.embeddings, embeddings)
        assert loaded.metadata.category == "widget"
        assert loaded.metadata.embedding_dim == 8

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            MemoryBank.load(tmp_path / "nonexistent.pt")

    def test_query_returns_low_score_for_points_in_bank(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 4)).astype(np.float32)
        mb = MemoryBank(embeddings, _make_metadata())

        # Points that ARE in the bank should score much lower than points that
        # are deliberately far away — a relative comparison, not an arbitrary
        # absolute cutoff, since "low" only means anything relative to "high" here.
        in_bank_scores = mb.query(embeddings[:5], k=3)
        far_points = rng.standard_normal((5, 4)).astype(np.float32) * 1000
        far_scores = mb.query(far_points, k=3)

        assert in_bank_scores.mean() < far_scores.mean() / 100

    def test_query_returns_high_score_for_far_points(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 4)).astype(np.float32)
        mb = MemoryBank(embeddings, _make_metadata())
        far_points = rng.standard_normal((5, 4)).astype(np.float32) * 1000
        scores = mb.query(far_points, k=3)
        assert (scores > 100).all()


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------
class TestSelectThreshold:
    def test_youden_j_separates_well_separated_classes(self, minimal_config):
        scores = np.array([1, 2, 3, 4, 5, 20, 21, 22, 23, 24], dtype=float)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = select_threshold(scores, labels, minimal_config)
        assert result.method == "youden_j"
        # sklearn's roc_curve picks thresholds from the actual data values, so the
        # optimal Youden's J threshold can legitimately land exactly on the smallest
        # anomalous score (20.0) when classes are perfectly separated — not just
        # strictly between the two clusters.
        assert 5 <= result.threshold <= 20
        assert result.roc_auc == 1.0

    def test_falls_back_to_percentile_when_only_normal_present(self, minimal_config):
        scores = np.array([1, 2, 3, 2, 1, 3, 2, 1], dtype=float)
        labels = np.zeros(8, dtype=int)
        result = select_threshold(scores, labels, minimal_config)
        assert result.method == "percentile"

    def test_explicit_percentile_method(self, minimal_config):
        config = dict(minimal_config)
        config["threshold"] = dict(minimal_config["threshold"])
        config["threshold"]["method"] = "percentile"
        config["threshold"]["fallback_percentile"] = 50.0

        scores = np.array([1, 2, 3, 4, 5, 20, 21, 22, 23, 24], dtype=float)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = select_threshold(scores, labels, config)
        assert result.method == "percentile"

    def test_threshold_result_has_correct_counts(self, minimal_config):
        scores = np.array([1, 2, 3, 20, 21], dtype=float)
        labels = np.array([0, 0, 0, 1, 1])
        result = select_threshold(scores, labels, minimal_config)
        assert result.num_normal == 3
        assert result.num_anomalous == 2


# ---------------------------------------------------------------------
# AnomalyDetector (using FakeExtractor, no real CNN needed)
# ---------------------------------------------------------------------
class TestAnomalyDetector:
    def test_patch_score_map_shape(self, minimal_config):
        embedding_grid = torch.randn(8, 4, 4)  # (C, h, w)
        extractor = FakeExtractor(embedding_grid)
        memory_bank = MemoryBank(np.random.randn(50, 8).astype(np.float32), _make_metadata())
        detector = AnomalyDetector(extractor, memory_bank, minimal_config)

        images = torch.zeros(2, 3, 64, 64)
        score_map = detector.compute_patch_score_map(images)
        assert score_map.shape == (2, 4, 4)

    def test_image_scores_max_aggregation(self, minimal_config):
        embedding_grid = torch.randn(8, 4, 4)
        extractor = FakeExtractor(embedding_grid)
        memory_bank = MemoryBank(np.random.randn(50, 8).astype(np.float32), _make_metadata())

        config = dict(minimal_config)
        config["threshold"] = dict(minimal_config["threshold"])
        config["threshold"]["aggregation"] = "max"
        detector = AnomalyDetector(extractor, memory_bank, config)

        images = torch.zeros(2, 3, 64, 64)
        scores = detector.compute_image_scores(images)
        score_map = detector.compute_patch_score_map(images)
        expected_max = score_map.reshape(2, -1).max(axis=1)
        assert np.allclose(scores, expected_max)

    def test_image_scores_topk_mean_aggregation(self, minimal_config):
        embedding_grid = torch.randn(8, 4, 4)
        extractor = FakeExtractor(embedding_grid)
        memory_bank = MemoryBank(np.random.randn(50, 8).astype(np.float32), _make_metadata())

        config = dict(minimal_config)
        config["threshold"] = dict(minimal_config["threshold"])
        config["threshold"]["aggregation"] = "topk_mean"
        config["threshold"]["topk_mean_k"] = 3
        detector = AnomalyDetector(extractor, memory_bank, config)

        images = torch.zeros(1, 3, 64, 64)
        scores = detector.compute_image_scores(images)
        assert scores.shape == (1,)

    def test_unknown_aggregation_raises(self, minimal_config):
        embedding_grid = torch.randn(8, 4, 4)
        extractor = FakeExtractor(embedding_grid)
        memory_bank = MemoryBank(np.random.randn(50, 8).astype(np.float32), _make_metadata())

        config = dict(minimal_config)
        config["threshold"] = dict(minimal_config["threshold"])
        config["threshold"]["aggregation"] = "banana"
        detector = AnomalyDetector(extractor, memory_bank, config)

        images = torch.zeros(1, 3, 64, 64)
        with pytest.raises(ValueError, match="Unknown aggregation"):
            detector.compute_image_scores(images)
