import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.csv_dataset import CSVDataset, find_split_csv
from src.models.models import build_model
from src.train.train_multiseed import get_transforms
from src.utils.io import get_datasets, get_model_train_cfg, load_configs


def entropy(prob, axis=-1):
    prob = np.clip(prob, 1e-12, 1.0)
    return float(-(prob * np.log(prob)).sum(axis=axis).mean())


def token_metrics(tokens, top_k=8):
    arr = tokens.detach().float().cpu().numpy()
    energy = np.linalg.norm(arr, axis=-1)
    prob = energy / (energy.sum(axis=1, keepdims=True) + 1e-12)
    sorted_prob = np.sort(prob, axis=1)
    n = sorted_prob.shape[1]
    gini = float(((2 * np.arange(1, n + 1) - n - 1) * sorted_prob).sum(axis=1).mean() / n)
    uniform = 1.0 / n
    kl = float((prob * np.log(np.clip(prob / uniform, 1e-12, None))).sum(axis=1).mean())
    top = -np.sort(-prob, axis=1)[:, : min(top_k, n)].sum(axis=1).mean()
    tie = entropy(prob)
    return {
        "spatial_activation_entropy": tie,
        "token_interaction_entropy": tie,
        "topk_token_concentration": float(top),
        "gini": gini,
        "kl_from_uniform": kl,
    }


@torch.no_grad()
def evaluate(model, model_name, loader, device, limit_batches=None):
    model.eval()
    rows = []
    for batch_idx, (images, _) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device)
        if model_name == "WiT" and hasattr(model, "fused_tokens"):
            tokens, attn = model.fused_tokens(images, need_weights=True)
            metrics = token_metrics(tokens)
            if attn is not None:
                attn_np = attn.detach().float().cpu().numpy()
                attn_prob = attn_np / (attn_np.sum(axis=-1, keepdims=True) + 1e-12)
                metrics["cross_attention_entropy"] = entropy(attn_prob)
        elif hasattr(model, "tokens"):
            output = model.tokens(images)
            tokens = output[0] if isinstance(output, tuple) else output
            metrics = token_metrics(tokens)
            metrics["cross_attention_entropy"] = np.nan
        else:
            _ = model(images)
            metrics = {
                "spatial_activation_entropy": np.nan,
                "token_interaction_entropy": np.nan,
                "topk_token_concentration": np.nan,
                "gini": np.nan,
                "kl_from_uniform": np.nan,
                "cross_attention_entropy": np.nan,
            }
        rows.append(metrics)
    if not rows:
        return {
            "spatial_activation_entropy": np.nan,
            "token_interaction_entropy": np.nan,
            "topk_token_concentration": np.nan,
            "gini": np.nan,
            "kl_from_uniform": np.nan,
            "cross_attention_entropy": np.nan,
        }
    return {key: float(pd.Series([row[key] for row in rows]).mean()) for key in rows[0]}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute quantitative attention/token concentration metrics.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=["A1_CNN", "A2_Transformer", "WiT"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    paths = cfg["experiments_cfg"].get("paths", {})
    data_root = paths.get("data_root", cfg["dataset_cfg"].get("data_root", "data/raw"))
    ckpt_dir = paths.get("checkpoints_dir", "checkpoints")
    out_dir = os.path.join(paths.get("results_dir", "results"), "attn_metrics")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg["device"])
    rows = []

    for dataset_name in args.datasets or get_datasets(cfg):
        dataset_dir = os.path.join(data_root, dataset_name)
        try:
            test_csv = find_split_csv(dataset_dir, "test")
        except FileNotFoundError as exc:
            print(f"[skip] {dataset_name}: {exc}")
            continue
        for model_name in args.models:
            model_cfg = get_model_train_cfg(cfg, model_name)
            img_size = int(model_cfg.get("img_size", model_cfg.get("resize_to", 256)))
            _, eval_tf = get_transforms(img_size)
            ds = CSVDataset(test_csv, transform=eval_tf)
            ckpt = os.path.join(ckpt_dir, f"{dataset_name}_{model_name}_s{args.seed}.pt")
            if not os.path.exists(ckpt):
                print(f"[skip] missing {ckpt}")
                continue
            model = build_model(model_name, cfg, ds.num_classes, pretrained=False).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            loader = DataLoader(ds, batch_size=int(model_cfg.get("batch_size", 64)), shuffle=False, num_workers=cfg["num_workers"])
            metrics = evaluate(model, model_name, loader, device, args.limit_batches)
            metrics.update({"dataset": dataset_name, "model": model_name})
            rows.append(metrics)
            print(f"{dataset_name}/{model_name}: {metrics}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["SAE"] = df["spatial_activation_entropy"]
        df["CAE"] = df["cross_attention_entropy"]
        df["TKC"] = df["topk_token_concentration"]
        df["Gini"] = df["gini"]
        df["KL"] = df["kl_from_uniform"]
    out_csv = os.path.join(out_dir, "attention_entropy_results.csv")
    df.to_csv(out_csv, index=False)
    df.to_csv(os.path.join(out_dir, "attention_metrics.csv"), index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
