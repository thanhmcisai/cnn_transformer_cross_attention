# src/visualization/confusion_matrix.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
# Inference helper
# -------------------------------------------------
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


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

    save_dir = "results/visualization/confusion_matrix"
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name in get_datasets(cfg):
        print(f"\n=== Dataset: {dataset_name} ===")

        test_dir = os.path.join(processed_root, dataset_name, "test")
        if not os.path.exists(test_dir):
            print(f"Test dir not found: {test_dir}")
            continue

        if dataset_name not in weight_map:
            print(f"No weights defined for {dataset_name}")
            continue

        for model_name in get_models(cfg):
            if model_name not in weight_map[dataset_name]:
                continue

            ckpt_path = os.path.join(weight_root, weight_map[dataset_name][model_name])
            if not os.path.exists(ckpt_path):
                print(f"Weight not found: {ckpt_path}")
                continue

            print(f"--- {dataset_name} | {model_name} ---")

            transform = get_val_transform(model_name)
            test_ds = datasets.ImageFolder(test_dir, transform=transform)

            loader = DataLoader(
                test_ds,
                batch_size=32,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            num_classes = len(test_ds.classes)

            model = get_working_model(model_name, cfg, num_classes, ckpt_path, DEVICE)
            if model is None:
                print("Skip model due to load failure.")
                continue

            y_true, y_pred = run_inference(model, loader, DEVICE)

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(8, 8))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=test_ds.classes
            )
            disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)

            plt.title(f"{dataset_name} - {model_name}")
            plt.tight_layout()

            out_path = os.path.join(
                save_dir, f"{dataset_name}_{model_name}_confusion_matrix.png"
            )
            plt.savefig(out_path, dpi=300)
            plt.close()

            print("Saved:", out_path)

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
