import os
import json
from collections import Counter, defaultdict

import pandas as pd

from src.utils.io import (
    load_configs,
    get_datasets,
)


# -----------------------------
# Helpers
# -----------------------------
def count_from_split_map(csv_path):
    """
    Count images per species from split_map CSV
    """
    df = pd.read_csv(csv_path)
    species_counter = Counter()

    for path in df["new_path"]:
        parts = path.split(os.sep)
        if len(parts) >= 2:
            species = parts[-2]
            species_counter[species] += 1

    return species_counter, len(df)


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = load_configs()

    dataset_cfg = cfg["dataset_cfg"]
    splits_root = dataset_cfg["splits_root"]
    stats_root = dataset_cfg["stats_root"]

    os.makedirs(stats_root, exist_ok=True)

    summary_stats = {}

    print("=== Dataset Statistics ===")

    for dataset_name in get_datasets(cfg):
        print(f"\nProcessing dataset: {dataset_name}")

        train_csv = os.path.join(splits_root, f"{dataset_name}_train_split_map.csv")
        test_csv = os.path.join(splits_root, f"{dataset_name}_test_split_map.csv")

        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"⚠️ Split files not found for {dataset_name}, skipping.")
            continue

        train_counter, train_total = count_from_split_map(train_csv)
        test_counter, test_total = count_from_split_map(test_csv)

        all_species = set(train_counter.keys()) | set(test_counter.keys())

        dataset_stats = {
            "num_classes": len(all_species),
            "num_train_images": train_total,
            "num_test_images": test_total,
            "total_images": train_total + test_total,
            "train_per_class": dict(train_counter),
            "test_per_class": dict(test_counter),
        }

        summary_stats[dataset_name] = dataset_stats

        print(f"  Classes: {dataset_stats['num_classes']}")
        print(f"  Train images: {train_total}")
        print(f"  Test images: {test_total}")
        print(f"  Total images: {dataset_stats['total_images']}")

    # Save statistics
    stats_path = os.path.join(stats_root, "dataset_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(summary_stats, f, indent=4)

    print(f"\n✅ Statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
