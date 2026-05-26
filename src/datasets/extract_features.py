import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.utils.io import get_datasets, load_configs


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NON_SPECIES_DIRS = {"train", "val", "validation", "test"}


def discover_images(dataset_dir):
    records = []
    root = Path(dataset_dir)
    for top in sorted(root.iterdir()):
        if not top.is_dir() or top.name.startswith("."):
            continue
        if top.name.lower() in NON_SPECIES_DIRS:
            for species_dir in sorted(top.iterdir()):
                if not species_dir.is_dir() or species_dir.name.startswith("."):
                    continue
                for path in sorted(species_dir.rglob("*")):
                    if path.suffix.lower() in IMG_EXTENSIONS:
                        records.append((str(path), species_dir.name))
        else:
            for path in sorted(top.rglob("*")):
                if path.suffix.lower() in IMG_EXTENSIONS:
                    records.append((str(path), top.name))
    return records


def magnification(path):
    name = Path(path).name.lower()
    for mag in ("10x_", "20x_", "50x_"):
        if name.startswith(mag):
            return mag[:-1]
    return "none"


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform, img_size):
        self.paths = paths
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.img_size, self.img_size))
        return self.transform(image), idx


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ConvNeXtV2 features for clustering-based splitting.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    dataset_cfg = cfg["dataset_cfg"]
    paths_cfg = cfg["experiments_cfg"].get("paths", {})
    data_root = paths_cfg.get("data_root", dataset_cfg.get("data_root", "data/raw"))
    features_dir = paths_cfg.get("features_dir", "features")
    os.makedirs(features_dir, exist_ok=True)

    cluster_cfg = dataset_cfg["clustering"]
    extractor_name = cluster_cfg.get("feature_extractor", "convnextv2_tiny.fcmae_ft_in22k_in1k_384")
    img_size = 384 if extractor_name.endswith("_384") else 256
    batch_size = int(cluster_cfg.get("batch_size", 64))
    device = torch.device(cfg["device"])
    datasets = args.datasets or get_datasets(cfg)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = timm.create_model(extractor_name, pretrained=True, num_classes=0, global_pool="avg").to(device).eval()

    for dataset_name in datasets:
        out_path = os.path.join(features_dir, f"{dataset_name}_features.npz")
        if os.path.exists(out_path) and not args.force:
            print(f"[skip] {out_path}")
            continue
        records = discover_images(os.path.join(data_root, dataset_name))
        if not records:
            print(f"[skip] no images found for {dataset_name}")
            continue
        image_paths = [r[0] for r in records]
        labels = [r[1] for r in records]
        label_names = sorted(set(labels))
        label_map = {name: idx for idx, name in enumerate(label_names)}
        label_idx = np.asarray([label_map[label] for label in labels], dtype=np.int32)
        loader = DataLoader(
            ImagePathDataset(image_paths, transform, img_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
        feats = []
        order = []
        with torch.no_grad():
            for images, idxs in tqdm(loader, desc=dataset_name):
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    batch = model(images.to(device))
                feats.append(F.normalize(batch.float(), p=2, dim=1).cpu().numpy())
                order.extend(idxs.numpy().tolist())
        features = np.vstack(feats)
        inverse = np.argsort(order)
        features = features[inverse]
        np.savez_compressed(
            out_path,
            features=features.astype(np.float32),
            paths=np.asarray(image_paths),
            labels=np.asarray(labels),
            label_idx=label_idx,
            mags=np.asarray([magnification(path) for path in image_paths]),
        )
        print(f"Saved {out_path}: {features.shape}")


if __name__ == "__main__":
    main()
