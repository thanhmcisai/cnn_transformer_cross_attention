import argparse
import gc
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.csv_dataset import CSVDataset, find_split_csv
from src.models.models import build_model
from src.utils.io import get_datasets, get_model_train_cfg, get_models, load_configs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_transforms(img_size=256):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.9),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(9, (0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.3), value="random"),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def soft_cross_entropy(outputs, targets, criterion):
    if targets.ndim == 2 or targets.dtype.is_floating_point:
        return -(targets * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
    return criterion(outputs, targets)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
        y_true.extend(targets.numpy())
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_one_run(model, train_loader, val_loader, num_classes, cfg, model_cfg, ckpt_path, device):
    global_train = cfg["experiments_cfg"]["global_train"]
    epochs = int(global_train.get("epochs", 100))
    warmup_epochs = int(global_train.get("warmup_epochs", 3))
    lr = float(model_cfg["lr"])
    patience = int(model_cfg.get("patience", global_train.get("early_stop", 5)))
    weight_decay = float(global_train.get("weight_decay", 0.05))
    mixup_prob = float(model_cfg.get("mixup_prob", global_train.get("mixup_prob", 1.0)))
    label_smoothing = float(global_train.get("label_smoothing", 0.1))

    model = model.to(device)
    mixup = Mixup(
        mixup_alpha=float(global_train.get("mixup_alpha", 0.4)),
        cutmix_alpha=float(global_train.get("cutmix_alpha", 0.4)),
        prob=mixup_prob,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.0,
        num_classes=num_classes,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_f1, best_epoch, stale = -1.0, 0, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        if epoch < warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = lr * (epoch + 1) / max(1, warmup_epochs)

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            if images.shape[0] % 2 == 0 and mixup_prob > 0:
                images, targets = mixup(images, targets)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = soft_cross_entropy(model(images), targets, criterion)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch >= warmup_epochs:
            scheduler.step()

        val = evaluate(model, val_loader, device)
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_epoch = epoch + 1
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
        else:
            stale += 1
            if epoch >= warmup_epochs and stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, best_epoch


def bootstrap_ci(values, n=10000):
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        value = float(arr[0]) if len(arr) else 0.0
        return value, 0.0, value, value
    samples = np.random.randint(0, len(arr), size=(n, len(arr)))
    means = arr[samples].mean(axis=1)
    return float(arr.mean()), float(arr.std(ddof=1)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(raw_csv, summary_csv):
    df = pd.read_csv(raw_csv)
    rows = []
    for (dataset, model), grp in df.groupby(["dataset", "model"]):
        for metric in ("test_acc", "test_precision", "test_recall", "test_f1"):
            mean, std, lo, hi = bootstrap_ci(grp[metric].tolist())
            rows.append({
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "mean": round(mean * 100, 2),
                "std": round(std * 100, 2),
                "ci95_lo": round(lo * 100, 2),
                "ci95_hi": round(hi * 100, 2),
                "formatted": f"{mean * 100:.2f}+/-{std * 100:.2f}",
            })
    pd.DataFrame(rows).to_csv(summary_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train WiT experiments with CSV splits and multiple seeds.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    exp_cfg = cfg["experiments_cfg"]
    paths = exp_cfg.get("paths", {})
    data_root = paths.get("data_root", cfg["dataset_cfg"].get("data_root", "data/raw"))
    results_dir = paths.get("results_dir", "results")
    ckpt_dir = paths.get("checkpoints_dir", "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(cfg["device"])
    datasets = args.datasets or get_datasets(cfg)
    models = args.models or get_models(cfg)
    seeds = args.seeds or exp_cfg["global_train"].get("seeds", [exp_cfg["global_train"].get("seed", 42)])
    raw_csv = os.path.join(results_dir, "results_raw.csv")
    summary_csv = os.path.join(results_dir, "results_summary.csv")

    rows = []
    completed = set()
    if os.path.exists(raw_csv) and not args.force:
        old = pd.read_csv(raw_csv)
        rows = old.to_dict("records")
        completed = {(r["dataset"], r["model"], int(r["seed"])) for r in rows}

    for dataset_name in datasets:
        dataset_dir = os.path.join(data_root, dataset_name)
        try:
            train_csv = find_split_csv(dataset_dir, "train")
            val_csv = find_split_csv(dataset_dir, "val")
            test_csv = find_split_csv(dataset_dir, "test")
        except FileNotFoundError as exc:
            print(f"[skip] {dataset_name}: {exc}")
            continue

        for model_name in models:
            model_cfg = get_model_train_cfg(cfg, model_name)
            img_size = int(model_cfg.get("img_size", model_cfg.get("resize_to", 256)))
            train_tf, eval_tf = get_transforms(img_size)
            train_ds = CSVDataset(train_csv, transform=train_tf)
            val_ds = CSVDataset(val_csv, transform=eval_tf)
            test_ds = CSVDataset(test_csv, transform=eval_tf)
            batch_size = int(model_cfg.get("batch_size", exp_cfg["global_train"].get("batch_size", 64)))
            num_workers = int(cfg["num_workers"])
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

            for seed in seeds:
                key = (dataset_name, model_name, int(seed))
                if key in completed:
                    print(f"[resume] {dataset_name}/{model_name}/s{seed}")
                    continue
                set_seed(seed)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                ckpt_path = os.path.join(ckpt_dir, f"{dataset_name}_{model_name}_s{seed}.pt")
                print(f"[train] {dataset_name} | {model_name} | seed={seed}")
                model = build_model(model_name, cfg, train_ds.num_classes, pretrained=args.pretrained)
                model, best_f1, best_epoch = train_one_run(
                    model, train_loader, val_loader, train_ds.num_classes, cfg, model_cfg, ckpt_path, device
                )
                test = evaluate(model, test_loader, device)
                rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "test_acc": test["acc"],
                    "test_precision": test["precision"],
                    "test_recall": test["recall"],
                    "test_f1": test["f1"],
                    "best_val_f1": best_f1,
                    "best_epoch": best_epoch,
                    "checkpoint": ckpt_path,
                })
                pd.DataFrame(rows).to_csv(raw_csv, index=False)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if rows:
        summarize(raw_csv, summary_csv)
        print(f"Saved {raw_csv} and {summary_csv}")


if __name__ == "__main__":
    main()
