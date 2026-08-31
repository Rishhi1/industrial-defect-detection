"""Tests for src/utils.py: config loading, device selection, category discovery."""

from __future__ import annotations

import pytest

from src.utils import discover_categories, ensure_dir, get_device, load_config


class TestLoadConfig:
    def test_missing_file_raises_clear_error(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(missing_path)

    def test_empty_file_raises_error(self, tmp_path):
        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config(empty_path)

    def test_valid_config_loads(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("dataset:\n  root_path: some/path\n")
        config = load_config(config_path)
        assert config["dataset"]["root_path"] == "some/path"

    def test_malformed_yaml_raises_clear_error(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("dataset: [unclosed bracket\n")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_config(config_path)


class TestGetDevice:
    def test_explicit_cpu_returns_cpu(self):
        assert get_device("cpu").type == "cpu"

    def test_auto_falls_back_to_cpu_without_cuda(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert get_device("auto").type == "cpu"

    def test_unknown_preference_raises(self):
        with pytest.raises(ValueError, match="Unknown device preference"):
            get_device("quantum_gpu")


class TestDiscoverCategories:
    def test_missing_root_returns_empty_list(self, tmp_path):
        assert discover_categories(tmp_path / "nonexistent") == []

    def test_empty_root_returns_empty_list(self, tmp_path):
        root = tmp_path / "empty_root"
        root.mkdir()
        assert discover_categories(root) == []

    def test_folder_without_train_good_is_excluded(self, tmp_path):
        root = tmp_path / "root"
        (root / "incomplete_category").mkdir(parents=True)
        assert discover_categories(root) == []

    def test_valid_category_is_detected(self, tmp_path):
        root = tmp_path / "root"
        (root / "bottle" / "train" / "good").mkdir(parents=True)
        assert discover_categories(root) == ["bottle"]

    def test_multiple_categories_sorted(self, tmp_path):
        root = tmp_path / "root"
        for name in ["zebra", "apple"]:
            (root / name / "train" / "good").mkdir(parents=True)
        assert discover_categories(root) == ["apple", "zebra"]


class TestEnsureDir:
    def test_creates_nested_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert target.exists()
        assert result == target

    def test_no_error_if_already_exists(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        ensure_dir(target)  # should not raise
        assert target.exists()
