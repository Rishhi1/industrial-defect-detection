"""
conftest.py
===========
Shared pytest fixtures. Tests build small synthetic MVTec-style datasets on
disk (via tmp_path) rather than depending on the real downloaded dataset —
this keeps the test suite fast, deterministic, and runnable without the
~5GB MVTec AD download or any network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_image(size: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((size, size, 3)) * 255).astype(np.uint8)


@pytest.fixture
def mvtec_category(tmp_path) -> tuple[Path, str]:
    """
    Builds a minimal valid MVTec-style category directory:
        <root>/<category>/train/good/            (5 images)
        <root>/<category>/test/good/              (3 images)
        <root>/<category>/test/broken/            (2 images)
        <root>/<category>/ground_truth/broken/    (2 masks)

    Returns (dataset_root, category_name).
    """
    root = tmp_path / "dataset"
    category = "widget"
    cat_dir = root / category

    (cat_dir / "train" / "good").mkdir(parents=True)
    (cat_dir / "test" / "good").mkdir(parents=True)
    (cat_dir / "test" / "broken").mkdir(parents=True)
    (cat_dir / "ground_truth" / "broken").mkdir(parents=True)

    for i in range(5):
        Image.fromarray(_make_image(seed=i)).save(cat_dir / "train" / "good" / f"{i:03d}.png")
    for i in range(3):
        Image.fromarray(_make_image(seed=100 + i)).save(cat_dir / "test" / "good" / f"{i:03d}.png")
    for i in range(2):
        Image.fromarray(_make_image(seed=200 + i)).save(cat_dir / "test" / "broken" / f"{i:03d}.png")
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        Image.fromarray(mask).save(cat_dir / "ground_truth" / "broken" / f"{i:03d}_mask.png")

    return root, category


@pytest.fixture
def minimal_config() -> dict:
    """A minimal but complete config dict, matching config.yaml's structure, for tests
    that need to pass a config without reading the real configs/config.yaml file."""
    return {
        "dataset": {
            "root_path": "data/mvtec_anomaly_detection",
            "selected_categories": [],
            "train_subdir": "train/good",
            "test_subdir": "test",
            "ground_truth_subdir": "ground_truth",
        },
        "preprocessing": {
            "image_size": 64,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "model": {
            "backbone": "resnet18",
            "feature_layers": ["layer2", "layer3"],
            "device": "cpu",
        },
        "memory_bank": {
            "coreset_ratio": 0.5,
            "coreset_method": "random",
            "k_nearest_neighbors": 3,
            "random_seed": 42,
        },
        "threshold": {
            "method": "youden_j",
            "fallback_percentile": 99.0,
            "aggregation": "max",
            "topk_mean_k": 5,
        },
        "dataloader": {"batch_size": 4, "num_workers": 0},
        "output": {
            "models_dir": "models",
            "results_dir": "results",
            "plots_dir": "results/plots",
            "predictions_dir": "results/predictions",
            "metrics_dir": "results/metrics",
        },
        "mode": "demo",
    }
