import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from src.utils.io import get_datasets, load_configs


def make_clusterer(threshold, metric, linkage):
    kwargs = {"n_clusters": None, "distance_threshold": threshold, "linkage": linkage}
    try:
        return AgglomerativeClustering(metric=metric, **kwargs)
    except TypeError:
        return AgglomerativeClustering(affinity=metric, **kwargs)


def split_cluster_ids(cluster_ids, sizes, train_ratio, val_ratio, seed):
    rng = np.random.RandomState(seed)
    ids = list(cluster_ids)
    rng.shuffle(ids)
    total = sum(sizes[cid] for cid in ids)
    train_target = total * train_ratio
    val_target = total * val_ratio
    train, val, test = set(), set(), set()
    train_n, val_n = 0, 0
    for cid in ids:
        if train_n < train_target:
            train.add(cid)
            train_n += sizes[cid]
        elif val_n < val_target:
            val.add(cid)
            val_n += sizes[cid]
        else:
            test.add(cid)
    return train, val, test


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test CSV splits from extracted features.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    dataset_cfg = cfg["dataset_cfg"]
    split_cfg = dataset_cfg["split"]
    cluster_cfg = dataset_cfg["clustering"]
    paths_cfg = cfg["experiments_cfg"].get("paths", {})
    features_dir = paths_cfg.get("features_dir", "features")
    data_root = paths_cfg.get("data_root", dataset_cfg.get("data_root", "data/raw"))
    threshold = args.threshold if args.threshold is not None else cluster_cfg.get("distance_threshold", 0.10)
    train_ratio = split_cfg.get("train_ratio", 0.65)
    val_ratio = split_cfg.get("val_ratio", 0.15)
    seed = dataset_cfg.get("seed", 42)

    for dataset_name in args.datasets or get_datasets(cfg):
        feature_path = os.path.join(features_dir, f"{dataset_name}_features.npz")
        if not os.path.exists(feature_path):
            print(f"[skip] missing {feature_path}")
            continue
        out_dir = os.path.join(data_root, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        split_paths = {split: os.path.join(out_dir, f"{split}.csv") for split in ("train", "val", "test")}
        if all(os.path.exists(p) for p in split_paths.values()) and not args.force:
            print(f"[skip] existing CSV splits for {dataset_name}")
            continue

        data = np.load(feature_path, allow_pickle=True)
        features = data["features"]
        paths = data["paths"]
        labels = data["labels"]
        label_idx = data["label_idx"]
        split_rows = {"train": [], "val": [], "test": []}

        for label in sorted(set(labels)):
            idx = np.where(labels == label)[0]
            if len(idx) < split_cfg.get("min_images_for_clustering", 10):
                rng = np.random.RandomState(seed)
                shuffled = idx.copy()
                rng.shuffle(shuffled)
                n_train = int(len(shuffled) * train_ratio)
                n_val = int(len(shuffled) * val_ratio)
                buckets = {
                    "train": shuffled[:n_train],
                    "val": shuffled[n_train:n_train + n_val],
                    "test": shuffled[n_train + n_val:],
                }
            else:
                clusters = make_clusterer(
                    threshold,
                    cluster_cfg.get("metric", "cosine"),
                    cluster_cfg.get("linkage", "average"),
                ).fit_predict(features[idx])
                sizes = defaultdict(int)
                for cid in clusters:
                    sizes[int(cid)] += 1
                train_ids, val_ids, test_ids = split_cluster_ids(set(map(int, clusters)), sizes, train_ratio, val_ratio, seed)
                buckets = {
                    "train": idx[[int(c) in train_ids for c in clusters]],
                    "val": idx[[int(c) in val_ids for c in clusters]],
                    "test": idx[[int(c) in test_ids for c in clusters]],
                }

            for split, row_idx in buckets.items():
                for i in row_idx:
                    split_rows[split].append({
                        "path": str(paths[i]),
                        "label": str(labels[i]),
                        "label_idx": int(label_idx[i]),
                        "threshold": threshold,
                    })

        for split, rows in split_rows.items():
            pd.DataFrame(rows).to_csv(split_paths[split], index=False)
            print(f"Saved {split_paths[split]} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
