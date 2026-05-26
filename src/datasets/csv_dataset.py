import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    """Dataset backed by train/val/test CSV files.

    Required columns:
        path, label_idx
    Optional:
        label
    """

    def __init__(self, csv_path, transform=None, path_root=None):
        self.csv_path = csv_path
        self.path_root = path_root
        self.transform = transform
        df = pd.read_csv(csv_path)
        if "path" not in df.columns or "label_idx" not in df.columns:
            raise ValueError(f"{csv_path} must contain 'path' and 'label_idx' columns")
        self.paths = df["path"].astype(str).tolist()
        labels = df["label_idx"].astype(int).tolist()
        classes = sorted(set(labels))
        remap = {label: idx for idx, label in enumerate(classes)}
        self.labels = [remap[label] for label in labels]
        if "label" in df.columns:
            self.classes = [str(v) for _, v in sorted(zip(labels, df["label"].astype(str).tolist()))]
            self.classes = list(dict.fromkeys(self.classes))
        else:
            self.classes = [str(label) for label in classes]
        self.num_classes = len(classes)

    def __len__(self):
        return len(self.paths)

    def _resolve_path(self, path):
        if os.path.isabs(path) or self.path_root is None:
            return path
        return os.path.join(self.path_root, path)

    def __getitem__(self, idx):
        path = self._resolve_path(self.paths[idx])
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (256, 256))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def find_split_csv(dataset_dir, split):
    for name in (f"{split}.csv", f"{split}_fold_0.csv"):
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No {split}.csv found in {dataset_dir}")
