# src/datasets/download.py
"""
Download and extract datasets based on configs/dataset.yaml
(using unified config loader in src/utils/io.py)

Usage:
    python src/datasets/download.py
"""

import os
import zipfile
import gdown

from src.utils.io import load_configs, get_dataset_info, get_datasets


def download_and_extract(dataset_name, dataset_info, data_root):
    url = dataset_info.get("url", None)
    zip_name = dataset_info.get("zip_name", f"{dataset_name}.zip")

    if url is None or url == "MANUAL_DOWNLOAD":
        print(f"⚠️ Dataset '{dataset_name}' requires manual download. Skipping.")
        return

    os.makedirs(data_root, exist_ok=True)

    zip_path = os.path.join(data_root, zip_name)
    extract_dir = os.path.join(data_root, dataset_name)

    # Download
    if not os.path.exists(zip_path):
        print(f"⬇️ Downloading {dataset_name}")
        gdown.download(url, zip_path, quiet=False)
    else:
        print(f"✅ {zip_name} already exists")

    # Extract
    if not os.path.exists(extract_dir):
        print(f"📦 Extracting {dataset_name}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        print(f"✅ {dataset_name} already extracted")


def main():
    cfg = load_configs()

    dataset_cfg = cfg["dataset_cfg"]
    data_root = dataset_cfg.get("data_root", "data/raw")

    datasets = get_datasets(cfg)

    print("=== Dataset Download Script ===")
    print(f"Data root: {data_root}")

    for dataset_name in datasets:
        dataset_info = get_dataset_info(cfg, dataset_name)
        print(f"\nProcessing dataset: {dataset_name}")
        download_and_extract(dataset_name, dataset_info, data_root)

    print("\n✅ All datasets processed.")


if __name__ == "__main__":
    main()
