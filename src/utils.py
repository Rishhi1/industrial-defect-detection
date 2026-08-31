"""
utils.py
========
Shared utility functions used across the project: config loading, random
seeding, device selection, directory management, and simple logging helpers.

Keeping these in one place means every script (build_memory_bank.py,
evaluate.py, run_inference.py, the Streamlit app, etc.) behaves consistently
and nothing about paths / devices / seeds is hardcoded elsewhere.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# -----------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------
def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """
    Load the YAML configuration file into a plain dict.

    Raises a clear, actionable error if the file is missing instead of
    letting a raw FileNotFoundError/yaml error surface to the user.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            f"Make sure you're running this command from the project root, "
            f"or pass --config with the correct path."
        )

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse config file '{config_path}': {e}")

    if not config:
        raise ValueError(f"Config file '{config_path}' is empty or invalid.")

    return config


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------
def get_device(preferred: str = "auto") -> torch.device:
    """
    Resolve the compute device.

    preferred:
        "auto" -> cuda if available, else cpu
        "cuda" -> cuda if available, else falls back to cpu with a warning
        "cpu"  -> always cpu
    """
    if preferred == "cpu":
        return torch.device("cpu")

    if preferred in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "cuda":
            print(
                "[WARNING] CUDA was requested but is not available on this "
                "machine. Falling back to CPU. Training/inference will be slower.",
                file=sys.stderr,
            )
        return torch.device("cpu")

    raise ValueError(f"Unknown device preference: '{preferred}'. Use 'auto', 'cuda', or 'cpu'.")


# -----------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    """
    Return the project root directory, assumed to be the parent of the
    'src' package this file lives in. Lets scripts resolve config/data
    paths correctly regardless of the current working directory.
    """
    return Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------
def discover_categories(dataset_root: str | Path) -> list[str]:
    """
    Auto-detect available MVTec AD category folders under dataset_root.

    A folder is considered a valid category if it contains a
    'train/good' subdirectory, matching the official MVTec AD layout.
    Returns an empty list (never raises) if the root doesn't exist yet,
    so callers can give a friendly "please download the dataset" message.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        return []

    categories = []
    for entry in sorted(dataset_root.iterdir()):
        if entry.is_dir() and (entry / "train" / "good").exists():
            categories.append(entry.name)
    return categories


# -----------------------------------------------------------------------
# Simple console logging (kept dependency-free — no logging config needed)
# -----------------------------------------------------------------------
def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warning(message: str) -> None:
    print(f"[WARNING] {message}", file=sys.stderr)


def log_error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
