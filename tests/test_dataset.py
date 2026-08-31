"""Tests for src/dataset.py: MVTecTrainDataset, MVTecTestDataset, and their error handling."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.dataset import MVTecDatasetError, MVTecTestDataset, MVTecTrainDataset


class TestMVTecTrainDataset:
    def test_valid_dataset_loads(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTrainDataset(root, category)
        assert len(ds) == 5

    def test_getitem_returns_pil_image(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTrainDataset(root, category)
        image = ds[0]
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    def test_missing_category_raises(self, tmp_path):
        (tmp_path / "root").mkdir()
        with pytest.raises(MVTecDatasetError, match="not found"):
            MVTecTrainDataset(tmp_path / "root", "nonexistent")

    def test_missing_train_good_folder_raises(self, tmp_path):
        root = tmp_path / "root"
        (root / "category" / "train").mkdir(parents=True)  # no 'good' subfolder
        with pytest.raises(MVTecDatasetError, match="train/good"):
            MVTecTrainDataset(root, "category")

    def test_empty_train_good_folder_raises(self, tmp_path):
        root = tmp_path / "root"
        (root / "category" / "train" / "good").mkdir(parents=True)
        with pytest.raises(MVTecDatasetError, match="No training images"):
            MVTecTrainDataset(root, "category")

    def test_transform_is_applied(self, mvtec_category):
        root, category = mvtec_category
        calls = []

        def fake_transform(img):
            calls.append(img)
            return "transformed"

        ds = MVTecTrainDataset(root, category, transform=fake_transform)
        result = ds[0]
        assert result == "transformed"
        assert len(calls) == 1


class TestMVTecTestDataset:
    def test_valid_dataset_loads_all_images(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTestDataset(root, category)
        assert len(ds) == 5  # 3 good + 2 broken

    def test_normal_images_labeled_zero(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTestDataset(root, category)
        good_samples = [s for s in ds.samples if s.defect_type == "good"]
        assert len(good_samples) == 3
        assert all(s.label == 0 for s in good_samples)

    def test_defective_images_labeled_one(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTestDataset(root, category)
        broken_samples = [s for s in ds.samples if s.defect_type == "broken"]
        assert len(broken_samples) == 2
        assert all(s.label == 1 for s in broken_samples)

    def test_defective_images_have_mask_path(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTestDataset(root, category)
        broken_samples = [s for s in ds.samples if s.defect_type == "broken"]
        assert all(s.mask_path is not None for s in broken_samples)
        assert all(s.mask_path.exists() for s in broken_samples)

    def test_normal_images_have_no_mask_path(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTestDataset(root, category)
        good_samples = [s for s in ds.samples if s.defect_type == "good"]
        assert all(s.mask_path is None for s in good_samples)

    def test_getitem_returns_expected_keys(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTestDataset(root, category)
        sample = ds[0]
        assert set(sample.keys()) == {"image", "label", "defect_type", "mask", "image_path"}

    def test_getitem_mask_is_binary_array_for_defective(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTestDataset(root, category)
        defective_idx = next(i for i, s in enumerate(ds.samples) if s.label == 1)
        sample = ds[defective_idx]
        assert sample["mask"] is not None
        assert set(np.unique(sample["mask"])).issubset({0, 1})

    def test_getitem_mask_is_none_for_normal(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTestDataset(root, category)
        normal_idx = next(i for i, s in enumerate(ds.samples) if s.label == 0)
        sample = ds[normal_idx]
        assert sample["mask"] is None

    def test_missing_test_folder_raises(self, tmp_path):
        root = tmp_path / "root"
        (root / "category" / "train" / "good").mkdir(parents=True)
        with pytest.raises(MVTecDatasetError, match="test"):
            MVTecTestDataset(root, "category")

    def test_summary_counts_per_defect_type(self, mvtec_category):
        root, category = mvtec_category
        ds = MVTecTestDataset(root, category)
        summary = ds.summary()
        assert summary == {"good": 3, "broken": 2}

    def test_missing_ground_truth_mask_tolerated(self, tmp_path):
        """A defective image with no matching mask file should get mask_path=None, not crash."""
        root = tmp_path / "root"
        cat_dir = root / "category"
        (cat_dir / "train" / "good").mkdir(parents=True)
        (cat_dir / "test" / "broken").mkdir(parents=True)
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(cat_dir / "train" / "good" / "0.png")
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(cat_dir / "test" / "broken" / "0.png")
        # deliberately no ground_truth folder at all

        ds = MVTecTestDataset(root, "category")
        assert ds.samples[0].mask_path is None


class TestCorruptedAndUnsupportedFiles:
    def test_corrupted_image_raises_clear_error(self, tmp_path):
        root = tmp_path / "root"
        (root / "category" / "train" / "good").mkdir(parents=True)
        bad_file = root / "category" / "train" / "good" / "corrupted.png"
        bad_file.write_bytes(b"this is not a valid png file")

        ds = MVTecTrainDataset(root, "category")
        with pytest.raises(MVTecDatasetError, match="Could not read image"):
            ds[0]

    def test_unsupported_extension_is_skipped_not_crashed(self, tmp_path):
        root = tmp_path / "root"
        good_dir = root / "category" / "train" / "good"
        good_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(good_dir / "valid.png")
        (good_dir / "readme.txt").write_text("not an image")

        ds = MVTecTrainDataset(root, "category")
        assert len(ds) == 1  # only the .png counted, .txt silently skipped
