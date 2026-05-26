import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from src.utils.io import get_datasets, load_configs


def cluster_species(features, threshold, metric="cosine", linkage="average"):
    kwargs = {
        "n_clusters": None,
        "distance_threshold": threshold,
        "linkage": linkage,
    }
    try:
        model = AgglomerativeClustering(metric=metric, **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity=metric, **kwargs)
    return model.fit_predict(features)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate clustering thresholds for leakage-aware splits.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    paths_cfg = cfg["experiments_cfg"].get("paths", {})
    features_dir = paths_cfg.get("features_dir", "features")
    out_dir = os.path.join(paths_cfg.get("results_dir", "results"), "threshold_analysis")
    os.makedirs(out_dir, exist_ok=True)
    cluster_cfg = cfg["dataset_cfg"]["clustering"]
    datasets = args.datasets or get_datasets(cfg)
    rows, species_rows = [], []

    for dataset_name in datasets:
        feature_path = os.path.join(features_dir, f"{dataset_name}_features.npz")
        if not os.path.exists(feature_path):
            print(f"[skip] missing {feature_path}")
            continue
        data = np.load(feature_path, allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        for threshold in args.thresholds:
            cluster_counts = []
            silhouettes = []
            for species in sorted(set(labels)):
                idx = np.where(labels == species)[0]
                if len(idx) < 3:
                    continue
                clusters = cluster_species(
                    features[idx],
                    threshold,
                    metric=cluster_cfg.get("metric", "cosine"),
                    linkage=cluster_cfg.get("linkage", "average"),
                )
                n_clusters = len(set(clusters))
                cluster_counts.append(n_clusters)
                sil = np.nan
                if 1 < n_clusters < len(idx):
                    sil = silhouette_score(features[idx], clusters, metric=cluster_cfg.get("metric", "cosine"))
                    silhouettes.append(sil)
                species_rows.append({
                    "dataset": dataset_name,
                    "threshold": threshold,
                    "species": species,
                    "n_images": len(idx),
                    "n_clusters": n_clusters,
                    "mean_cluster_size": len(idx) / max(1, n_clusters),
                    "silhouette": sil,
                })
            rows.append({
                "dataset": dataset_name,
                "threshold": threshold,
                "mean_clusters_per_species": float(np.mean(cluster_counts)) if cluster_counts else 0.0,
                "mean_silhouette": float(np.nanmean(silhouettes)) if silhouettes else np.nan,
            })

    agg_csv = os.path.join(out_dir, "threshold_analysis.csv")
    species_csv = os.path.join(out_dir, "threshold_analysis_species.csv")
    pd.DataFrame(rows).to_csv(agg_csv, index=False)
    pd.DataFrame(species_rows).to_csv(species_csv, index=False)
    print(f"Saved {agg_csv} and {species_csv}")


if __name__ == "__main__":
    main()
