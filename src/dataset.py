"""
dataset.py
==========
PyTorch Dataset classes for loading the MVTec AD dataset.

MVTec AD's on-disk layout (per category) looks like:

    <category>/
        train/
            good/                     <- ONLY normal images, used for the
                                          reference memory bank. No defects.
        test/
            good/                     <- normal test images
            <defect_type_1>/          <- e.g. "broken_large", "contamination"
            <defect_type_2>/
            ...
        ground_truth/
            <defect_type_1>/          <- binary masks for each defective
            <defect_type_2>/             test image (not for "good")

Two dataset classes are provided:

- MVTecTrainDataset: yields ONLY normal images from train/good. This is what
  builds the reference memory bank. Test defective images must never leak
  into this set.
- MVTecTestDataset: yields every test image (both "good" and every defect
  subfolder) along with an image-level label (0=normal, 1=anomalous) and,
  for defective images, the path to its ground-truth mask (None for normal
  images, since MVTec doesn't provide a mask for "good" test images).

Both classes fail loudly and specifically (not with a generic KeyError or
IndexError) when the dataset is missing, a category doesn't exist, or an
image file cannot be read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Formats we know PIL can open reliably for this project. Anything else is
# skipped with a warning rather than crashing the whole pipeline.
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class MVTecDatasetError(Exception):
    """Raised for structural problems with the MVTec dataset on disk."""


@dataclass
class TestSample:
    """One row of the test set: an image path, its label, and mask path."""
    image_path: Path
    label: int              # 0 = normal, 1 = anomalous
    defect_type: str        # "good" or e.g. "broken_large"
    mask_path: Path | None  # None for normal images


def _validate_category_dir(category_dir: Path, category: str) -> None:
    if not category_dir.exists():
        raise MVTecDatasetError(
            f"Category '{category}' not found at '{category_dir}'. "
            f"Check the category name and that the dataset has been "
            f"downloaded and extracted correctly."
        )
    train_good = category_dir / "train" / "good"
    if not train_good.exists():
        raise MVTecDatasetError(
            f"Expected '{train_good}' to exist (standard MVTec AD layout: "
            f"<category>/train/good/). This category's folder looks "
            f"incomplete or non-standard."
        )


def _list_images(folder: Path) -> list[Path]:
    """Return sorted image file paths in a folder, skipping unsupported files."""
    if not folder.exists():
        return []
    files = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def _safe_load_image(path: Path) -> Image.Image:
    """
    Load an image as RGB, raising a clear, specific error on failure instead
    of letting a raw PIL exception (or a silently wrong image) propagate.
    """
    try:
        img = Image.open(path)
        img.load()  # force-read now so corrupt files fail here, not later
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise MVTecDatasetError(f"Could not read image '{path}': {e}")


# -----------------------------------------------------------------------
# Train dataset — normal images only
# -----------------------------------------------------------------------
class MVTecTrainDataset(Dataset):
    """
    Loads ONLY the normal ('good') training images for one MVTec category.
    This is the set used to build the reference memory bank — it must never
    contain defective images.
    """

    def __init__(self, dataset_root: str | Path, category: str, transform=None):
        self.dataset_root = Path(dataset_root)
        self.category = category
        self.transform = transform

        category_dir = self.dataset_root / category
        _validate_category_dir(category_dir, category)

        self.image_paths = _list_images(category_dir / "train" / "good")
        if len(self.image_paths) == 0:
            raise MVTecDatasetError(
                f"No training images found in '{category_dir / 'train' / 'good'}'. "
                f"The 'good' folder exists but is empty."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        image = _safe_load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image


# -----------------------------------------------------------------------
# Test dataset — all defect types + "good", with labels and masks
# -----------------------------------------------------------------------
class MVTecTestDataset(Dataset):
    """
    Loads every test image for one MVTec category: normal ('good') and every
    defect-type subfolder under test/. Each sample carries an image-level
    label (0=normal, 1=anomalous), the defect type name, and (for defective
    images) the path to its ground-truth binary mask.
    """

    def __init__(self, dataset_root: str | Path, category: str, transform=None):
        self.dataset_root = Path(dataset_root)
        self.category = category
        self.transform = transform

        category_dir = self.dataset_root / category
        _validate_category_dir(category_dir, category)

        test_dir = category_dir / "test"
        if not test_dir.exists():
            raise MVTecDatasetError(
                f"Expected '{test_dir}' to exist (standard MVTec AD layout: "
                f"<category>/test/<defect_type>/)."
            )

        gt_dir = category_dir / "ground_truth"

        self.samples: list[TestSample] = []
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            is_normal = defect_type == "good"
            images = _list_images(defect_dir)

            for img_path in images:
                mask_path = None
                if not is_normal:
                    # MVTec ground-truth masks are named "<id>_mask.png"
                    mask_path = gt_dir / defect_type / f"{img_path.stem}_mask.png"
                    if not mask_path.exists():
                        mask_path = None  # tolerate missing mask, just flag as None

                self.samples.append(
                    TestSample(
                        image_path=img_path,
                        label=0 if is_normal else 1,
                        defect_type=defect_type,
                        mask_path=mask_path,
                    )
                )

        if len(self.samples) == 0:
            raise MVTecDatasetError(
                f"No test images found under '{test_dir}'. The folder exists "
                f"but contains no readable images."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = _safe_load_image(sample.image_path)
        if self.transform is not None:
            image = self.transform(image)

        mask = None
        if sample.mask_path is not None:
            mask_img = Image.open(sample.mask_path).convert("L")
            mask = (np.array(mask_img) > 0).astype(np.uint8)

        return {
            "image": image,
            "label": sample.label,
            "defect_type": sample.defect_type,
            "mask": mask,
            "image_path": str(sample.image_path),
        }

    def summary(self) -> dict:
        """Quick per-defect-type counts, useful for verification/logging."""
        counts: dict[str, int] = {}
        for s in self.samples:
            counts[s.defect_type] = counts.get(s.defect_type, 0) + 1
        return counts
