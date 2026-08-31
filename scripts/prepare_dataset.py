"""
prepare_dataset.py
===================
Verifies that the MVTec AD dataset is correctly downloaded and structured,
and prints per-category statistics (train/test image counts, defect types).

This script does NOT download the dataset — MVTec AD must be downloaded
manually from https://www.mvtec.com/company/research/datasets/mvtec-ad
(registration required) and extracted so that each category folder sits
directly under the configured dataset root, e.g.:

    data/mvtec_anomaly_detection/bottle/
    data/mvtec_anomaly_detection/carpet/
    ...

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --config configs/config.yaml
    python scripts/prepare_dataset.py --category bottle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/prepare_dataset.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import MVTecDatasetError, MVTecTestDataset, MVTecTrainDataset
from src.utils import discover_categories, load_config, log_error, log_info, log_warning


def check_category(dataset_root: Path, category: str) -> dict | None:
    """
    Validate a single category and return a stats dict, or None if the
    category is broken/unusable (with the reason already logged).
    """
    try:
        train_ds = MVTecTrainDataset(dataset_root, category)
    except MVTecDatasetError as e:
        log_error(f"[{category}] Training set problem: {e}")
        return None

    try:
        test_ds = MVTecTestDataset(dataset_root, category)
    except MVTecDatasetError as e:
        log_error(f"[{category}] Test set problem: {e}")
        return None

    test_breakdown = test_ds.summary()
    num_normal_test = test_breakdown.get("good", 0)
    num_defect_test = len(test_ds) - num_normal_test
    defect_types = sorted(t for t in test_breakdown if t != "good")

    return {
        "category": category,
        "train_normal": len(train_ds),
        "test_normal": num_normal_test,
        "test_anomalous": num_defect_test,
        "defect_types": defect_types,
        "test_breakdown": test_breakdown,
    }


def print_report(stats_list: list[dict]) -> None:
    if not stats_list:
        log_warning("No categories were successfully validated.")
        return

    print("\n" + "=" * 78)
    print("MVTEC AD DATASET VERIFICATION REPORT")
    print("=" * 78)

    header = f"{'Category':<15}{'Train (normal)':<17}{'Test (normal)':<16}{'Test (anomalous)':<18}{'#Defect types'}"
    print(header)
    print("-" * 78)

    total_train = total_test_normal = total_test_anom = 0
    for s in stats_list:
        print(
            f"{s['category']:<15}{s['train_normal']:<17}{s['test_normal']:<16}"
            f"{s['test_anomalous']:<18}{len(s['defect_types'])}"
        )
        total_train += s["train_normal"]
        total_test_normal += s["test_normal"]
        total_test_anom += s["test_anomalous"]

    print("-" * 78)
    print(f"{'TOTAL':<15}{total_train:<17}{total_test_normal:<16}{total_test_anom:<18}")
    print("=" * 78)

    for s in stats_list:
        print(f"\n[{s['category']}] defect types found in test set:")
        for defect_type in s["defect_types"]:
            count = s["test_breakdown"][defect_type]
            print(f"    - {defect_type}: {count} images")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MVTec AD dataset structure and print statistics.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file.")
    parser.add_argument(
        "--category", type=str, default=None,
        help="Check only this single category (overrides config's selected_categories)."
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        log_error(str(e))
        sys.exit(1)

    dataset_root = Path(config["dataset"]["root_path"])

    if not dataset_root.exists():
        log_error(
            f"Dataset root '{dataset_root}' does not exist. Download MVTec AD "
            f"from https://www.mvtec.com/company/research/datasets/mvtec-ad "
            f"and extract category folders directly under '{dataset_root}'."
        )
        sys.exit(1)

    available = discover_categories(dataset_root)
    if not available:
        log_error(
            f"'{dataset_root}' exists but no valid MVTec category folders were "
            f"found inside it (each category needs a 'train/good/' subfolder). "
            f"Check that the dataset was extracted correctly."
        )
        sys.exit(1)

    log_info(f"Found {len(available)} category folder(s) under '{dataset_root}': {available}")

    if args.category:
        if args.category not in available:
            log_error(
                f"Requested category '{args.category}' not found. "
                f"Available categories: {available}"
            )
            sys.exit(1)
        categories_to_check = [args.category]
    else:
        selected = config["dataset"].get("selected_categories") or []
        if selected:
            missing = [c for c in selected if c not in available]
            if missing:
                log_warning(
                    f"Categories listed in config but not found on disk: {missing}. "
                    f"They will be skipped."
                )
            categories_to_check = [c for c in selected if c in available]
        else:
            categories_to_check = available

    if not categories_to_check:
        log_error("No valid categories to check. Update 'selected_categories' in config.yaml.")
        sys.exit(1)

    log_info(f"Verifying {len(categories_to_check)} category/categories: {categories_to_check}\n")

    stats_list = []
    for category in categories_to_check:
        log_info(f"Checking category '{category}'...")
        stats = check_category(dataset_root, category)
        if stats is not None:
            stats_list.append(stats)

    print_report(stats_list)

    if len(stats_list) < len(categories_to_check):
        log_warning(
            f"{len(categories_to_check) - len(stats_list)} of "
            f"{len(categories_to_check)} categories failed validation (see errors above)."
        )
        sys.exit(1)

    log_info("All checked categories passed verification.")


if __name__ == "__main__":
    main()
