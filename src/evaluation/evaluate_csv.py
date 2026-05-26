import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.datasets.csv_dataset import CSVDataset, find_split_csv
from src.models.models import build_model
from src.train.train_multiseed import get_transforms
from src.utils.io import get_datasets, get_model_train_cfg, get_models, load_configs


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        y_pred.extend(outputs.argmax(1).cpu().numpy())
        y_true.extend(labels.numpy())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints on CSV splits.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    paths = cfg["experiments_cfg"].get("paths", {})
    data_root = paths.get("data_root", cfg["dataset_cfg"].get("data_root", "data/raw"))
    ckpt_dir = paths.get("checkpoints_dir", "checkpoints")
    results_dir = os.path.join(paths.get("results_dir", "results"), "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device(cfg["device"])
    datasets = args.datasets or get_datasets(cfg)
    models = args.models or get_models(cfg)
    summary = {}

    for dataset_name in datasets:
        dataset_dir = os.path.join(data_root, dataset_name)
        try:
            csv_path = find_split_csv(dataset_dir, args.split)
        except FileNotFoundError as exc:
            print(f"[skip] {dataset_name}: {exc}")
            continue
        summary[dataset_name] = {}
        for model_name in models:
            ckpt_path = os.path.join(ckpt_dir, f"{dataset_name}_{model_name}_s{args.seed}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[skip] missing checkpoint: {ckpt_path}")
                continue
            model_cfg = get_model_train_cfg(cfg, model_name)
            img_size = int(model_cfg.get("img_size", model_cfg.get("resize_to", 256)))
            _, eval_tf = get_transforms(img_size)
            ds = CSVDataset(csv_path, transform=eval_tf)
            loader = DataLoader(ds, batch_size=int(model_cfg.get("batch_size", 64)), shuffle=False, num_workers=cfg["num_workers"])
            model = build_model(model_name, cfg, ds.num_classes, pretrained=args.pretrained).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            metrics = evaluate_model(model, loader, device)
            summary[dataset_name][model_name] = metrics
            print(f"{dataset_name}/{model_name}: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

    out_path = os.path.join(results_dir, f"{args.split}_summary_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
