import argparse
import os

import numpy as np
import pandas as pd

from src.datasets.threshold_analysis import cluster_species
from src.utils.io import load_configs


def parse_args():
    parser = argparse.ArgumentParser(description="VN99-specific clustering threshold sensitivity.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    paths_cfg = cfg["experiments_cfg"].get("paths", {})
    features_dir = paths_cfg.get("features_dir", "features")
    out_dir = os.path.join(paths_cfg.get("results_dir", "results"), "threshold_analysis")
    os.makedirs(out_dir, exist_ok=True)
    feature_path = os.path.join(features_dir, "vn99_features.npz")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"{feature_path} not found. Run feature extraction first.")

    cluster_cfg = cfg["dataset_cfg"]["clustering"]
    data = np.load(feature_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    rows = []

    for threshold in args.thresholds:
        per_species = []
        for species in sorted(set(labels)):
            idx = np.where(labels == species)[0]
            if len(idx) < 2:
                continue
            clusters = cluster_species(
                features[idx],
                threshold,
                metric=cluster_cfg.get("metric", "cosine"),
                linkage=cluster_cfg.get("linkage", "average"),
            )
            n_clusters = len(set(clusters))
            per_species.append({
                "n_images": len(idx),
                "n_clusters": n_clusters,
                "mean_cluster_size": len(idx) / max(1, n_clusters),
            })
        df = pd.DataFrame(per_species)
        rows.append({
            "dataset": "vn99",
            "threshold": threshold,
            "species_count": len(df),
            "mean_clusters_per_species": df["n_clusters"].mean(),
            "median_clusters_per_species": df["n_clusters"].median(),
            "mean_cluster_size": df["mean_cluster_size"].mean(),
            "min_cluster_size_proxy": df["mean_cluster_size"].min(),
        })

    out_csv = os.path.join(out_dir, "vn99_threshold_sensitivity.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
