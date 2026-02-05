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
def get_eval_transform(model_name):
    """
    Standard transform for evaluation (no augmentation)
    Used for Train (eval mode), Val, and Test
    """
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
def evaluate_model(model, loader, device, desc="Evaluating"):
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
    # Usually eval batch size can be larger than train
    batch_size = 32 

    processed_root = dataset_cfg["processed_root"]

    # ---- WEIGHTS CONFIG ----
    weights_cfg = exp_cfg["weights"]
    weight_root = weights_cfg["root_dir"]
    weight_map = weights_cfg["mapping"]

    results_dir = "results"
    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    summary_results = {}

    # Define splits to evaluate
    TARGET_SPLITS = ["train", "val", "test"]

    for dataset_name in get_datasets(cfg):
        print(f"\n==================== Evaluating Dataset: {dataset_name} ====================")

        if dataset_name not in weight_map:
            print(f"⚠️ No pretrained weights defined for {dataset_name} in config.")
            continue

        summary_results[dataset_name] = {}

        for model_name in get_models(cfg):
            print(f"\n--- Model: {model_name} ---")

            if model_name not in weight_map[dataset_name]:
                print(f"⚠️ No weight map found for {dataset_name} -> {model_name}")
                continue

            ckpt_path = os.path.join(weight_root, weight_map[dataset_name][model_name])

            # Determine classes from test set (assuming consistent classes across splits)
            # We need to instantiate at least one dataset to get class names/count
            dummy_path = os.path.join(processed_root, dataset_name, "test")
            if not os.path.exists(dummy_path):
                # Fallback to val or train if test missing
                dummy_path = os.path.join(processed_root, dataset_name, "train")
            
            try:
                temp_ds = datasets.ImageFolder(dummy_path)
                class_names = temp_ds.classes
                num_classes = len(class_names)
            except Exception as e:
                print(f"❌ Could not determine classes for {dataset_name}: {e}")
                continue

            # ---- SMART LOAD MODEL (Load once per model) ----
            model = get_working_model(
                model_name, cfg, num_classes, ckpt_path, DEVICE
            )

            if model is None:
                print(f"❌ Failed to load model {model_name}, skipping.")
                continue

            summary_results[dataset_name][model_name] = {}
            
            # Use same transform for all splits during evaluation
            eval_transform = get_eval_transform(model_name)

            # ---- LOOP THROUGH SPLITS (Train, Val, Test) ----
            for split in TARGET_SPLITS:
                split_dir = os.path.join(processed_root, dataset_name, split)
                
                if not os.path.exists(split_dir):
                    print(f"⚠️ Split directory not found: {split_dir}. Skipping {split}.")
                    continue

                print(f"   > Running evaluation on [{split}] set...")

                ds = datasets.ImageFolder(split_dir, transform=eval_transform)
                loader = DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )

                acc, prec, rec, f1, cm = evaluate_model(model, loader, DEVICE, desc=split)

                print(f"     [{split.upper()}] Acc={acc:.4f}, F1={f1:.4f}")

                # Save metrics to dict
                summary_results[dataset_name][model_name][split] = {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                }

                # ---- Plot & Save Confusion Matrix ----
                # Create a specific folder for plots if needed, or put in eval_dir
                # Naming: {dataset}_{model}_{split}_cm.png
                
                fig, ax = plt.subplots(figsize=(10, 10))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                
                # Rotate labels if there are many classes
                disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
                
                plt.title(f"CM: {dataset_name} - {model_name} ({split})")
                plt.tight_layout()
                
                save_name = f"{dataset_name}_{model_name}_{split}_cm.png"
                plt.savefig(os.path.join(eval_dir, save_name))
                plt.close(fig) # Close plot to free memory

    # Save all summary results to JSON
    json_path = os.path.join(eval_dir, "evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_results, f, indent=4)

    print(f"\n✅ Evaluation completed. Results saved to {eval_dir}")


if __name__ == "__main__":
    main()