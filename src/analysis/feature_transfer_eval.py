import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from src.datasets.csv_dataset import CSVDataset, find_split_csv
from src.models.models import build_model
from src.train.train_multiseed import get_transforms
from src.utils.io import get_datasets, get_model_train_cfg, get_models, load_configs


def embed_from_logits(logits):
    return F.normalize(logits.float(), p=2, dim=1)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, labels = [], []
    for images, y in loader:
        images = images.to(device)
        if hasattr(model, "fused_tokens"):
            emb = model.fused_tokens(images).mean(1)
        elif hasattr(model, "tokens"):
            tokens = model.tokens(images)
            emb = (tokens[0] if isinstance(tokens, tuple) else tokens).mean(1)
        else:
            emb = embed_from_logits(model(images))
        embs.append(F.normalize(emb.float(), p=2, dim=1).cpu().numpy())
        labels.extend(y.numpy())
    return np.vstack(embs), np.asarray(labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset transfer with frozen embeddings and kNN.")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--k-values", nargs="*", type=int, default=[1, 3, 5, 7])
    parser.add_argument("--hard-baseline", default="A1_CNN")
    parser.add_argument("--hard-top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    paths = cfg["experiments_cfg"].get("paths", {})
    data_root = paths.get("data_root", cfg["dataset_cfg"].get("data_root", "data/raw"))
    ckpt_dir = paths.get("checkpoints_dir", "checkpoints")
    out_dir = os.path.join(paths.get("results_dir", "results"), "transfer")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg["device"])
    datasets = args.datasets or get_datasets(cfg)
    models = args.models or get_models(cfg)
    rows = []
    per_class_rows = []
    k_rows = []

    for model_name in models:
        model_cfg = get_model_train_cfg(cfg, model_name)
        img_size = int(model_cfg.get("img_size", model_cfg.get("resize_to", 256)))
        _, eval_tf = get_transforms(img_size)
        for source in datasets:
            source_dir = os.path.join(data_root, source)
            try:
                source_train = find_split_csv(source_dir, "train")
            except FileNotFoundError:
                continue
            source_ds = CSVDataset(source_train, transform=eval_tf)
            ckpt = os.path.join(ckpt_dir, f"{source}_{model_name}_s{args.seed}.pt")
            if not os.path.exists(ckpt):
                print(f"[skip] missing {ckpt}")
                continue
            model = build_model(model_name, cfg, source_ds.num_classes, pretrained=False).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            for target in datasets:
                if target == source:
                    continue
                target_dir = os.path.join(data_root, target)
                try:
                    target_train = find_split_csv(target_dir, "train")
                    target_test = find_split_csv(target_dir, "test")
                except FileNotFoundError:
                    continue
                target_train_ds = CSVDataset(target_train, transform=eval_tf)
                target_ds = CSVDataset(target_test, transform=eval_tf)
                target_train_loader = DataLoader(target_train_ds, batch_size=int(model_cfg.get("batch_size", 64)), shuffle=False, num_workers=cfg["num_workers"])
                target_loader = DataLoader(target_ds, batch_size=int(model_cfg.get("batch_size", 64)), shuffle=False, num_workers=cfg["num_workers"])
                train_emb, train_y = extract_embeddings(model, target_train_loader, device)
                target_emb, target_y = extract_embeddings(model, target_loader, device)
                pred_by_k = {}
                for k in sorted(set(args.k_values + [args.k])):
                    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
                    knn.fit(train_emb, train_y)
                    pred_by_k[k] = knn.predict(target_emb)
                    k_rows.append({
                        "model": model_name,
                        "source": source,
                        "target": target,
                        "k": k,
                        "top1_acc": accuracy_score(target_y, pred_by_k[k]),
                        "macro_f1": f1_score(target_y, pred_by_k[k], average="macro", zero_division=0),
                        "seed": args.seed,
                    })
                pred = pred_by_k[args.k]
                per_class_f1 = f1_score(target_y, pred, average=None, zero_division=0)
                for class_idx, class_f1 in enumerate(per_class_f1):
                    per_class_rows.append({
                        "model": model_name,
                        "source": source,
                        "target": target,
                        "class_idx": class_idx,
                        "class_f1": class_f1,
                        "seed": args.seed,
                        "k": args.k,
                    })
                rows.append({
                    "model": model_name,
                    "source": source,
                    "target": target,
                    "top1_acc": accuracy_score(target_y, pred),
                    "macro_f1": f1_score(target_y, pred, average="macro", zero_division=0),
                    "seed": args.seed,
                    "k": args.k,
                })
                print(f"{model_name}: {source}->{target}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "transfer_matrix.csv")
    df.to_csv(out_csv, index=False)
    summary_csv = os.path.join(out_dir, "transfer_summary.csv")
    if not df.empty:
        (
            df.groupby("model")
            .agg(
                mean_top1_acc=("top1_acc", "mean"),
                std_top1_acc=("top1_acc", "std"),
                mean_macro_f1=("macro_f1", "mean"),
                std_macro_f1=("macro_f1", "std"),
                n_pairs=("target", "count"),
            )
            .reset_index()
            .to_csv(summary_csv, index=False)
        )
    pd.DataFrame(k_rows).to_csv(os.path.join(out_dir, "transfer_k_sensitivity.csv"), index=False)

    per_class_df = pd.DataFrame(per_class_rows)
    hard_rows = []
    if not per_class_df.empty:
        for (source, target), grp in per_class_df.groupby(["source", "target"]):
            base = grp[grp["model"] == args.hard_baseline]
            if base.empty:
                continue
            hard_classes = base.nsmallest(args.hard_top_k, "class_f1")["class_idx"].tolist()
            for model_name, model_grp in grp.groupby("model"):
                hard_f1 = model_grp[model_grp["class_idx"].isin(hard_classes)]["class_f1"].mean()
                hard_rows.append({
                    "source": source,
                    "target": target,
                    "model": model_name,
                    "hard_baseline": args.hard_baseline,
                    "hard_classes": ",".join(str(c) for c in hard_classes),
                    "avg_hard_class_f1": hard_f1,
                    "seed": args.seed,
                    "k": args.k,
                })
    pd.DataFrame(hard_rows).to_csv(os.path.join(out_dir, "transfer_hardclass.csv"), index=False)
    print(f"Saved {out_csv}")
    print(f"Saved {summary_csv}")


if __name__ == "__main__":
    main()
