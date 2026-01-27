# src/evaluation/evaluate.py
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.utils.io import (
    load_configs,
    get_datasets,
    get_models,
)

from src.utils.model_loader import get_working_model


# -------------------------------------------------
# Transforms
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
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


# -------------------------------------------------
# Evaluate one model
# -------------------------------------------------
@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    cm = confusion_matrix(all_targets, all_preds)

    return acc, prec, rec, f1, cm


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

    results_dir = "results"
    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    summary_results = {}

    for dataset_name in get_datasets(cfg):
        print(f"\n=== Evaluating dataset: {dataset_name} ===")

        test_dir = os.path.join(processed_root, dataset_name, "test")
        if not os.path.exists(test_dir):
            print(f"⚠️ Test directory not found: {test_dir}, skipping.")
            continue

        if dataset_name not in weight_map:
            print(f"⚠️ No pretrained weights defined for {dataset_name}")
            continue

        summary_results[dataset_name] = {}

        for model_name in get_models(cfg):
            print(f"\n--- Model: {model_name} ---")

            if model_name not in weight_map[dataset_name]:
                print(f"⚠️ No weight for {dataset_name} - {model_name}")
                continue

            ckpt_path = os.path.join(weight_root, weight_map[dataset_name][model_name])

            if not os.path.exists(ckpt_path):
                print(f"⚠️ Weight file not found: {ckpt_path}")
                continue

            val_transform = get_val_transform(model_name)

            test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            num_classes = len(test_dataset.classes)

            # ---- SMART LOAD MODEL ----
            model = get_working_model(
                model_name, cfg, num_classes, ckpt_path, DEVICE
            )

            if model is None:
                print(f"❌ Skip {model_name}")
                continue

            acc, prec, rec, f1, cm = evaluate_model(model, test_loader, DEVICE)

            print(f"Acc={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

            summary_results[dataset_name][model_name] = {
                "weight_path": weight_map[dataset_name][model_name],
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }

            # ---- Confusion Matrix ----
            fig, ax = pl
