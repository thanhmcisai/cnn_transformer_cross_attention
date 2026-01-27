# src/analysis/hardest_class_analysis.py
"""
Hardest Class Analysis (Per-class F1 improvement)

Usage:
    python src/analysis/hardest_class_analysis.py
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

from src.utils.io import (
    load_configs,
    get_datasets,
    get_models,
)

from src.models.models import build_model


# -------------------------------------------------
# Transforms (same as validation)
# -------------------------------------------------
def get_val_transform(model_name):
    if model_name == "coatnet_0_rw_224":
        img_size = 224
    else:
        img_size = 256

    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])


# -------------------------------------------------
# Evaluate model (per-class predictions)
# -------------------------------------------------
@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

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

    processed_root = dataset_cfg["processed_root"]

    # ---- WEIGHTS CONFIG ----
    weights_cfg = exp_cfg["weights"]
    weight_root = weights_cfg["root_dir"]
    weight_map = weights_cfg["mapping"]

    baseline_model_name = exp_cfg.get("baseline_model", get_models(cfg)[0])
    top_k_hard = exp_cfg.get("top_k_hard_classes", 5)

    os.makedirs("results/analysis", exist_ok=True)
    save_csv = "results/analysis/hardest_class_analysis.csv"

    final_rows = []

    print("\n" + "="*70)
    print("HARDEST CLASS ANALYSIS")
    print(f"Baseline model: {baseline_model_name}")
    print(f"Top-K hardest classes: {top_k_hard}")
    print("="*70)

    for dataset_name in get_datasets(cfg):
        print(f"\n>>> Dataset: {dataset_name}")

        if dataset_name not in weight_map:
            print("⚠️ No weights defined, skipping.")
            continue

        test_dir = os.path.join(processed_root, dataset_name, "test")
        if not os.path.exists(test_dir):
            print(f"⚠️ Test directory not found: {test_dir}")
            continue

        # ------------------------
        # Load test dataset (baseline transform)
        # ------------------------
        baseline_transform = get_val_transform(baseline_model_name)

        test_dataset = datasets.ImageFolder(test_dir, transform=baseline_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        num_classes = len(test_dataset.classes)

        # ------------------------
        # Load baseline model
        # ------------------------
        if baseline_model_name not in weight_map[dataset_name]:
            print(f"⚠️ Baseline weight not found for {dataset_name}")
            continue

        baseline_weight = os.path.join(
            weight_root, weight_map[dataset_name][baseline_model_name]
        )

        baseline_model = build_model(
            baseline_model_name, cfg, num_classes
        ).to(DEVICE)

        baseline_model.load_state_dict(
            torch.load(baseline_weight, map_location=DEVICE)
        )

        y_true, y_pred = evaluate_model(baseline_model, test_loader, DEVICE)

        per_class_f1 = f1_score(y_true, y_pred, average=None)

        hardest_indices = np.argsort(per_class_f1)[:top_k_hard]
        hardest_classes = [test_dataset.classes[i] for i in hardest_indices]

        print("Hardest classes:", hardest_classes)

        del baseline_model
        torch.cuda.empty_cache()

        # ------------------------
        # Evaluate all models on hardest classes
        # ------------------------
        for model_name in get_models(cfg):

            if model_name not in weight_map[dataset_name]:
                continue

            ckpt_path = os.path.join(
                weight_root, weight_map[dataset_name][model_name]
            )

            if not os.path.exists(ckpt_path):
                continue

            val_transform = get_val_transform(model_name)

            test_dataset_m = datasets.ImageFolder(test_dir, transform=val_transform)
            test_loader_m = DataLoader(
                test_dataset_m,
                batch_size=64,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            model = build_model(model_name, cfg, num_classes).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

            y_true_m, y_pred_m = evaluate_model(model, test_loader_m, DEVICE)

            per_class_f1_m = f1_score(y_true_m, y_pred_m, average=None)

            avg_hard_f1 = per_class_f1_m[hardest_indices].mean()

            final_rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "avg_hard_class_f1": float(avg_hard_f1),
                "hard_classes": ",".join(hardest_classes)
            })

            print(f"{model_name}: Avg Hard F1 = {avg_hard_f1:.4f}")

            del model
            torch.cuda.empty_cache()

    # ------------------------
    # Save CSV
    # ------------------------
    df = pd.DataFrame(final_rows)
    df.to_csv(save_csv, index=False)

    print("\n✅ Hardest class analysis saved to:", save_csv)


if __name__ == "__main__":
    main()
