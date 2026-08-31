"""Tests for src/evaluation.py: image-level metrics, pixel-level metrics, aggregation."""

from __future__ import annotations

import numpy as np

from src.evaluation import (
    ImageLevelMetrics,
    aggregate_macro,
    aggregate_pooled,
    compute_image_level_metrics,
    compute_pixel_level_metrics,
    get_confusion_matrix,
    get_pr_curve_data,
    get_roc_curve_data,
)


class TestComputeImageLevelMetrics:
    def test_perfect_separation_gives_perfect_scores(self):
        scores = np.array([1, 2, 3, 10, 11, 12], dtype=float)
        labels = np.array([0, 0, 0, 1, 1, 1])
        metrics = compute_image_level_metrics(scores, labels, threshold=5.0)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.roc_auc == 1.0
        assert metrics.tp == 3 and metrics.tn == 3 and metrics.fp == 0 and metrics.fn == 0

    def test_confusion_matrix_components_sum_to_total(self):
        scores = np.array([1, 5, 3, 8, 2, 9], dtype=float)
        labels = np.array([0, 1, 0, 1, 0, 1])
        metrics = compute_image_level_metrics(scores, labels, threshold=4.0)
        assert metrics.tp + metrics.tn + metrics.fp + metrics.fn == 6

    def test_single_class_gives_nan_auc(self):
        scores = np.array([1, 2, 3, 4], dtype=float)
        labels = np.zeros(4, dtype=int)
        metrics = compute_image_level_metrics(scores, labels, threshold=2.5)
        assert np.isnan(metrics.roc_auc)
        assert np.isnan(metrics.pr_auc)

    def test_threshold_is_recorded(self):
        scores = np.array([1, 2, 3], dtype=float)
        labels = np.array([0, 0, 1])
        metrics = compute_image_level_metrics(scores, labels, threshold=2.5)
        assert metrics.threshold == 2.5


class TestComputePixelLevelMetrics:
    def test_all_normal_pixels_gives_nan(self):
        scores = np.random.rand(1000)
        labels = np.zeros(1000, dtype=int)
        metrics = compute_pixel_level_metrics(scores, labels)
        assert np.isnan(metrics.pixel_roc_auc)

    def test_all_defect_pixels_gives_nan(self):
        scores = np.random.rand(1000)
        labels = np.ones(1000, dtype=int)
        metrics = compute_pixel_level_metrics(scores, labels)
        assert np.isnan(metrics.pixel_roc_auc)

    def test_perfect_pixel_separation(self):
        normal_scores = np.random.rand(500) * 0.1
        defect_scores = np.random.rand(500) * 0.1 + 10
        scores = np.concatenate([normal_scores, defect_scores])
        labels = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])

        metrics = compute_pixel_level_metrics(scores, labels)
        assert metrics.pixel_roc_auc == 1.0
        assert metrics.num_defect_pixels == 500

    def test_counts_are_correct(self):
        scores = np.random.rand(100)
        labels = np.zeros(100, dtype=int)
        labels[:20] = 1
        metrics = compute_pixel_level_metrics(scores, labels)
        assert metrics.num_pixels == 100
        assert metrics.num_defect_pixels == 20


class TestCurveDataAndConfusionMatrix:
    def test_roc_curve_data_shapes_match(self):
        scores = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        labels = np.array([0, 0, 0, 1, 1, 1])
        data = get_roc_curve_data(scores, labels)
        assert len(data["fpr"]) == len(data["tpr"])
        assert 0.0 <= data["auc"] <= 1.0

    def test_pr_curve_data_shapes_match(self):
        scores = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        labels = np.array([0, 0, 0, 1, 1, 1])
        data = get_pr_curve_data(scores, labels)
        assert len(data["precision"]) == len(data["recall"])

    def test_confusion_matrix_shape(self):
        scores = np.array([1, 2, 3, 4], dtype=float)
        labels = np.array([0, 0, 1, 1])
        cm = get_confusion_matrix(scores, labels, threshold=2.5)
        assert cm.shape == (2, 2)
        assert cm.sum() == 4


class TestAggregation:
    def test_aggregate_pooled_combines_all_images(self):
        all_scores = [np.array([1, 2, 8, 9]), np.array([1, 8])]
        all_labels = [np.array([0, 0, 1, 1]), np.array([0, 1])]
        thresholds = [5.0, 5.0]

        result = aggregate_pooled(all_scores, all_labels, thresholds)
        assert result["num_images"] == 6
        assert result["accuracy"] == 1.0  # perfectly separated in both categories

    def test_aggregate_macro_equal_weight_per_category(self):
        m1 = ImageLevelMetrics(
            accuracy=1.0, precision=1.0, recall=1.0, f1=1.0, roc_auc=1.0, pr_auc=1.0,
            tp=1, tn=1, fp=0, fn=0, num_images=2, threshold=5.0,
        )
        m2 = ImageLevelMetrics(
            accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5, pr_auc=0.5,
            tp=1, tn=0, fp=1, fn=0, num_images=100, threshold=5.0,  # much bigger category
        )
        result = aggregate_macro([m1, m2])
        # Macro = simple mean regardless of size: (1.0 + 0.5) / 2 = 0.75, NOT weighted toward m2
        assert result["accuracy"] == 0.75
        assert result["num_categories"] == 2

    def test_aggregate_macro_empty_list(self):
        assert aggregate_macro([]) == {}

    def test_aggregate_macro_skips_nan_values(self):
        m1 = ImageLevelMetrics(
            accuracy=1.0, precision=1.0, recall=1.0, f1=1.0, roc_auc=float("nan"), pr_auc=float("nan"),
            tp=1, tn=1, fp=0, fn=0, num_images=2, threshold=5.0,
        )
        m2 = ImageLevelMetrics(
            accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.8, pr_auc=0.8,
            tp=1, tn=0, fp=1, fn=0, num_images=2, threshold=5.0,
        )
        result = aggregate_macro([m1, m2])
        assert result["roc_auc"] == 0.8  # NaN excluded, not averaged in as 0
