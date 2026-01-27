# src/visualization/tsne_umap.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import normalize

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils.io import load_configs, get_datasets, get_models
from src.utils.model_loader import get_working_model


# -------------------------------------------------
# Transform
# -------------------------------------------------
def get_val_transform(model_name):
    img_size = 224 if model_name == "coatnet_0_rw_224" else 256
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])


# -------------------------------------------------
# Feature extractor (works for baseline + A1–A7)
# -------------------------------------------------
@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []

    for imgs, lbls in loader:
        imgs = imgs.to(device)

        # baseline timm models
        if hasattr(model, "forward_features"):
            f = model.forward_features(imgs)
            if isinstance(f, (list, tuple)):
                f = f[-1]
            f = torch.flatten(f, 1)
        else:
            # proposed models: use penultimate layer by removing classifier
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            f = out

        feats.append(f.cpu().numpy())
        labels.extend(lbls.numpy())

    feats = np.vstack(feats)
    feats = normalize(feats)  # L2 normalize

    return feats, np.array(labels)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    cfg = load_configs()
    dataset_cfg = cfg["dataset_cfg"]
    exp_cfg = cfg["experiments_cfg"]

    DEVICE = cfg["device"]
    num_workers = cfg["num_workers"]

    weights_cfg = exp_cfg["weights"]
    weight_root = weights_cfg["root_dir"]
    weight_map = weights_cfg["mapping"]

    processed_root = dataset_cfg["processed_root"]

    save_dir = "results/visualization/embedding"
    os.makedirs(save_dir, exist_ok=True)

    max_samples = 2000  # avoid TSNE crash on large datasets

    for dataset_name in get_datasets(cfg):
        print(f"\n=== Dataset: {dataset_name} ===")

        test_dir = os.path.join(processed_root, dataset_name, "test")
        if not os.path.exists(test_dir):
            continue

        if dataset_name not in weight_map:
            continue

        for model_name in get_models(cfg):
            if model_name not in weight_map[dataset_name]:
                continue

            ckpt_path = os.path.join(weight_root, weight_map[dataset_name][model_name])
            if not os.path.exists(ckpt_path):
                continue

            print(f"--- {dataset_name} | {model_name} ---")

            transform = get_val_transform(model_name)
            test_ds = datasets.ImageFolder(test_dir, transform=transform)

            # subsample if too large
            if len(test_ds) > max_samples:
                idx = np.random.choice(len(test_ds), max_samples, replace=False)
                test_ds = torch.utils.data.Subset(test_ds, idx)

            loader = DataLoader(
                test_ds,
                batch_size=64,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            num_classes = len(test_ds.dataset.classes) if hasattr(test_ds, "dataset") else len(test_ds.classes)

            model = get_working_model(model_name, cfg, num_classes, ckpt_path, DEVICE)
            if model is None:
                continue

            feats, labels = extract_features(model, loader, DEVICE)

            # ---- TSNE ----
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            emb_tsne = tsne.fit_transform(feats)

            plt.figure(figsize=(7,6))
            plt.scatter(emb_tsne[:,0], emb_tsne[:,1], c=labels, cmap="tab20", s=5)
            plt.title(f"{dataset_name} - {model_name} (t-SNE)")
            plt.colorbar()
            plt.tight_layout()

            tsne_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_tsne.png")
            plt.savefig(tsne_path, dpi=300)
            plt.close()
            print("Saved:", tsne_path)

            # ---- UMAP ----
            reducer = umap.UMAP(n_components=2, random_state=42)
            emb_umap = reducer.fit_transform(feats)

            plt.figure(figsize=(7,6))
            plt.scatter(emb_umap[:,0], emb_umap[:,1], c=labels, cmap="tab20", s=5)
            plt.title(f"{dataset_name} - {model_name} (UMAP)")
            plt.colorbar()
            plt.tight_layout()

            umap_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_umap.png")
            plt.savefig(umap_path, dpi=300)
            plt.close()
            print("Saved:", umap_path)

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
