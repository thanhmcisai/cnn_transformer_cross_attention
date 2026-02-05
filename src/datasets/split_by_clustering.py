import os
import gc
import random
import shutil
from collections import defaultdict

import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import AgglomerativeClustering

from src.utils.io import (
    load_configs,
    get_datasets,
)


# -----------------------------
# Dataset for feature extraction
# -----------------------------
class FeatureExtractorDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0)
    return torch.stack(batch)


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(model, loader, device):
    model.eval()
    features_list = []

    with torch.no_grad():
        for images in tqdm(loader, desc="Extracting features"):
            if images.shape[0] == 0:
                continue
            images = images.to(device)
            feats = model(images)
            features_list.append(feats.cpu().numpy())

    if not features_list:
        return np.empty((0, model.num_features))

    features = np.vstack(features_list)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norm + 1e-6)
    return features


def create_feature_loader(image_paths, transform, batch_size, num_workers):
    ds = FeatureExtractorDataset(image_paths, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# -----------------------------
# Clustering pipeline
# -----------------------------
def run_clustering_pipeline(image_paths, model, transform, batch_size, num_workers, device, cluster_cfg):
    loader = create_feature_loader(image_paths, transform, batch_size, num_workers)
    features = extract_features(model, loader, device)

    if features.shape[0] == 0:
        return {}, 0

    cluster_model = AgglomerativeClustering(
        n_clusters=None,
        metric=cluster_cfg.get("metric", "cosine"),
        linkage=cluster_cfg.get("linkage", "average"),
        distance_threshold=cluster_cfg["distance_threshold"],
    )

    cluster_labels = cluster_model.fit_predict(features)
    n_clusters = len(np.unique(cluster_labels))

    cluster_to_images = defaultdict(list)
    for path, label in zip(image_paths, cluster_labels):
        cluster_to_images[label].append(path)

    return cluster_to_images, n_clusters


# -----------------------------
# Split helpers
# -----------------------------
def split_clusters_stratified(cluster_map, train_ratio, val_ratio):
    total_imgs = sum(len(v) for v in cluster_map.values())
    target_train = int(total_imgs * train_ratio)
    target_val = int(total_imgs * val_ratio)
    
    current_train = 0
    current_val = 0
    
    train_clusters = set()
    val_clusters = set()
    test_clusters = set()

    items = list(cluster_map.items())
    random.shuffle(items)

    for cid, imgs in items:
        n_imgs = len(imgs)
        # Fill Train first
        if current_train < target_train:
            train_clusters.add(cid)
            current_train += n_imgs
        # Then Fill Val
        elif current_val < target_val:
            val_clusters.add(cid)
            current_val += n_imgs
        # Rest goes to Test
        else:
            test_clusters.add(cid)

    return train_clusters, val_clusters, test_clusters


def random_split_populator(image_paths, train_ratio, val_ratio, original_structure,
                           train_list, val_list, test_list, dataset_root, filename_prefix=""):
    random.shuffle(image_paths)
    n_total = len(image_paths)
    
    idx_train = int(n_total * train_ratio)
    # idx_val is cumulative: train_ratio + val_ratio
    idx_val = int(n_total * (train_ratio + val_ratio))

    # Basic safety for very small datasets
    if n_total > 1:
        if idx_train == 0: idx_train = 1
        if idx_val <= idx_train and n_total > idx_train: idx_val = idx_train + 1
    
    train_paths = image_paths[:idx_train]
    val_paths = image_paths[idx_train:idx_val]
    test_paths = image_paths[idx_val:]

    def add_to_list(paths, target_list, split_name):
        for path in paths:
            if path not in original_structure: continue
            _, species = original_structure[path]
            filename = os.path.basename(path)
            filename = f"{filename_prefix}_{filename}" if filename_prefix else filename
            new_path = os.path.join(split_name, species, filename)
            target_list.append({"old_path": path, "new_path": new_path})

    add_to_list(train_paths, train_list, "train")
    add_to_list(val_paths, val_list, "val")
    add_to_list(test_paths, test_list, "test")


def populate_data_lists(cluster_map, cluster_set, split_name,
                        dataset_root, original_structure, output_list, filename_prefix=""):
    for cid in cluster_set:
        for old_path in cluster_map[cid]:
            if old_path not in original_structure:
                continue

            _, species_name = original_structure[old_path]
            original_filename = os.path.basename(old_path)

            if filename_prefix:
                new_unique_filename = f"{filename_prefix}_{original_filename}"
            else:
                new_unique_filename = original_filename

            new_path = os.path.join(split_name, species_name, new_unique_filename)
            output_list.append({"old_path": old_path, "new_path": new_path})


# -----------------------------
# Standard dataset processing
# -----------------------------
def process_dataset_standard(dataset_root, model, transform, cfg, device, num_workers):
    split_cfg = cfg["dataset_cfg"]["split"]
    cluster_cfg = cfg["dataset_cfg"]["clustering"]

    # Hardcoded ratios as requested: 70% Train, 15% Val, 15% Test
    train_ratio = split_cfg.get("train_ratio", 0.7)
    val_ratio = split_cfg.get("val_ratio", 0.15)
    # test_ratio is implicit (remaining)
    
    min_images = split_cfg.get("min_images_for_clustering", 10)
    # Check logic: if val+test is too small, fallback to random
    min_test_ratio = split_cfg.get("min_test_ratio_from_cluster", 0.25) 
    batch_size = cluster_cfg["batch_size"]

    species_to_paths = defaultdict(list)
    original_structure = {}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            continue
        try:
            dataset = datasets.ImageFolder(split_dir)
            for path, _ in dataset.imgs:
                species = os.path.basename(os.path.dirname(path))
                species_to_paths[species].append(path)
                original_structure[path] = (split, species)
        except:
            continue

    all_train, all_val, all_test = [], [], []

    for species, image_paths in species_to_paths.items():
        print(f"  Species: {species} ({len(image_paths)} images)")

        # Case 1: Too few images -> Random Split
        if len(image_paths) < min_images:
            random_split_populator(image_paths, train_ratio, val_ratio,
                                    original_structure, all_train, all_val, all_test, dataset_root)
            continue

        # Case 2: Clustering
        cluster_map, n_clusters = run_clustering_pipeline(
            image_paths, model, transform,
            batch_size, num_workers, device, cluster_cfg)

        if n_clusters <= 1:
            random_split_populator(image_paths, train_ratio, val_ratio,
                                    original_structure, all_train, all_val, all_test, dataset_root)
            continue

        train_clusters, val_clusters, test_clusters = split_clusters_stratified(cluster_map, train_ratio, val_ratio)

        # Check if split was effective (e.g. ensure we didn't put everything in train)
        num_val_test_images = sum(len(cluster_map[c]) for c in val_clusters) + sum(len(cluster_map[c]) for c in test_clusters)
        min_required = int(len(image_paths) * min_test_ratio) # reusing config parameter name logic

        if num_val_test_images < min_required:
            random_split_populator(image_paths, train_ratio, val_ratio,
                                    original_structure, all_train, all_val, all_test, dataset_root)
        else:
            populate_data_lists(cluster_map, train_clusters, "train",
                                 dataset_root, original_structure, all_train)
            populate_data_lists(cluster_map, val_clusters, "val",
                                 dataset_root, original_structure, all_val)
            populate_data_lists(cluster_map, test_clusters, "test",
                                 dataset_root, original_structure, all_test)

    return all_train, all_val, all_test


# -----------------------------
# VN26 special processing
# -----------------------------
def process_dataset_vn26(dataset_root, model, transform, cfg, device, num_workers):
    split_cfg = cfg["dataset_cfg"]["split"]
    cluster_cfg = cfg["dataset_cfg"]["clustering"]
    vn26_cfg = cfg["dataset_cfg"]["vn26"]

    train_ratio = 0.70
    val_ratio = 0.15
    
    min_images = split_cfg.get("min_images_for_clustering", 10)
    min_test_ratio = split_cfg.get("min_test_ratio_from_cluster", 0.25)
    batch_size = cluster_cfg["batch_size"]

    species_mag_to_paths = defaultdict(lambda: defaultdict(list))
    original_structure = {}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            continue
        try:
            dataset = datasets.ImageFolder(split_dir)
            for path, _ in dataset.imgs:
                species = os.path.basename(os.path.dirname(path))
                filename = os.path.basename(path)

                # Match logic for VN26 magnifications
                matched = False
                for mag in vn26_cfg["magnifications"]:
                    if filename.startswith(f"{mag}_"):
                        species_mag_to_paths[species][mag].append(path)
                        original_structure[path] = (split, species)
                        matched = True
                        break
                if not matched:
                    # Fallback or ignore if not matching mag pattern
                    pass
        except:
            continue

    all_train, all_val, all_test = [], [], []

    for species, mag_map in species_mag_to_paths.items():
        print(f"Species: {species}")
        for mag, image_paths in mag_map.items():
            print(f"  Mag {mag}: {len(image_paths)} images")

            if len(image_paths) < min_images:
                random_split_populator(image_paths, train_ratio, val_ratio,
                                        original_structure, all_train, all_val, all_test, dataset_root)
                continue

            cluster_map, n_clusters = run_clustering_pipeline(
                image_paths, model, transform,
                batch_size, num_workers, device, cluster_cfg)

            if n_clusters <= 1:
                random_split_populator(image_paths, train_ratio, val_ratio,
                                        original_structure, all_train, all_val, all_test, dataset_root)
                continue

            train_clusters, val_clusters, test_clusters = split_clusters_stratified(cluster_map, train_ratio, val_ratio)

            num_val_test_images = sum(len(cluster_map[c]) for c in val_clusters) + sum(len(cluster_map[c]) for c in test_clusters)
            min_required = int(len(image_paths) * min_test_ratio)

            if num_val_test_images < min_required:
                random_split_populator(image_paths, train_ratio, val_ratio,
                                        original_structure, all_train, all_val, all_test, dataset_root)
            else:
                populate_data_lists(cluster_map, train_clusters, "train",
                                     dataset_root, original_structure, all_train)
                populate_data_lists(cluster_map, val_clusters, "val",
                                     dataset_root, original_structure, all_val)
                populate_data_lists(cluster_map, test_clusters, "test",
                                     dataset_root, original_structure, all_test)

    return all_train, all_val, all_test


# -----------------------------
# Copy data to processed folder
# -----------------------------
def copy_split_data(train_data, val_data, test_data, dataset_name, cfg):
    dataset_cfg = cfg["dataset_cfg"]
    processed_root = dataset_cfg["processed_root"]
    target_root = os.path.join(processed_root, dataset_name)

    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    os.makedirs(target_root, exist_ok=True)

    print(f"Copying files to processed folder: {target_root}")

    def copy_records(records, split_name):
        for record in tqdm(records, desc=f"Copying {split_name}"):
            old_path = record["old_path"]
            rel_path = record["new_path"]  # e.g., train/class/img.jpg
            new_path = os.path.join(target_root, rel_path)

            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            try:
                shutil.copy2(old_path, new_path)
            except Exception as e:
                print(f"Error copying {old_path} -> {new_path}: {e}")

    copy_records(train_data, "train")
    copy_records(val_data, "val")
    copy_records(test_data, "test")


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = load_configs()

    dataset_cfg = cfg["dataset_cfg"]
    cluster_cfg = dataset_cfg["clustering"]

    device = cfg["device"]
    num_workers = cfg["num_workers"]

    random.seed(dataset_cfg["seed"])
    np.random.seed(dataset_cfg["seed"])
    torch.manual_seed(dataset_cfg["seed"])

    print(f"Using device: {device}")
    print("Loading feature extractor:", cluster_cfg["feature_extractor"])

    model = timm.create_model(
        cluster_cfg["feature_extractor"],
        pretrained=True,
        num_classes=0
    ).to(device)
    model.eval()

    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    splits_root = dataset_cfg["splits_root"]
    os.makedirs(splits_root, exist_ok=True)

    for dataset_name in get_datasets(cfg):
        dataset_root = os.path.join(dataset_cfg["data_root"], dataset_name)

        print(f"\nProcessing dataset: {dataset_name}")

        if dataset_name.lower() == "vn26":
            train_data, val_data, test_data = process_dataset_vn26(
                dataset_root, model, transform, cfg, device, num_workers
            )
        else:
            train_data, val_data, test_data = process_dataset_standard(
                dataset_root, model, transform, cfg, device, num_workers
            )

        train_csv = os.path.join(splits_root, f"{dataset_name}_train_split_map.csv")
        val_csv = os.path.join(splits_root, f"{dataset_name}_val_split_map.csv")
        test_csv = os.path.join(splits_root, f"{dataset_name}_test_split_map.csv")

        pd.DataFrame(train_data).to_csv(train_csv, index=False)
        pd.DataFrame(val_data).to_csv(val_csv, index=False)
        pd.DataFrame(test_data).to_csv(test_csv, index=False)

        print(f"Saved split maps for {dataset_name}")

        copy_split_data(train_data, val_data, test_data, dataset_name, cfg)

        gc.collect()
        torch.cuda.empty_cache()

    print("\n✅ All datasets processed successfully.")


if __name__ == "__main__":
    main()