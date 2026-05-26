import argparse
import io
import os

import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.csv_dataset import CSVDataset, find_split_csv
from src.models.models import build_model
from src.utils.io import get_datasets, get_model_train_cfg, get_models, load_configs


class GaussianNoise:
    def __init__(self, severity):
        self.std = 0.04 * severity

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0, 1)


class MotionBlur:
    def __init__(self, severity):
        self.radius = severity

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


class Illumination:
    def __init__(self, severity):
        self.factor = max(0.2, 1.0 - 0.12 * severity)

    def __call__(self, image):
        return ImageEnhance.Brightness(image).enhance(self.factor)


class ContrastShift:
    def __init__(self, severity):
        self.factor = max(0.2, 1.0 - 0.12 * severity)

    def __call__(self, image):
        return ImageEnhance.Contrast(image).enhance(self.factor)


class JPEGCompression:
    def __init__(self, severity):
        self.quality = max(5, 95 - severity * 15)

    def __call__(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def make_transform(img_size, corruption, severity):
    pre = [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
    if corruption == "blur":
        pre.append(MotionBlur(severity))
    if corruption == "illumination":
        pre.append(Illumination(severity))
    if corruption == "contrast":
        pre.append(ContrastShift(severity))
    if corruption == "jpeg":
        pre.append(JPEGCompression(severity))
    post = [transforms.ToTensor()]
    if corruption == "noise":
        post.append(GaussianNoise(severity))
    post.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(pre + post)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_pred.extend(logits.argmax(1).cpu().numpy())
        y_true.extend(labels.numpy())
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", zero_division=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate robustness under controlled corruptions.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=["A1_CNN", "A2_Transformer", "A6_Par_CrossAttn", "WiT"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severity", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    paths = cfg["experiments_cfg"].get("paths", {})
    data_root = paths.get("data_root", cfg["dataset_cfg"].get("data_root", "data/raw"))
    ckpt_dir = paths.get("checkpoints_dir", "checkpoints")
    out_dir = os.path.join(paths.get("results_dir", "results"), "corruption")
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
        for model_name in args.models or get_models(cfg):
            model_cfg = get_model_train_cfg(cfg, model_name)
            img_size = int(model_cfg.get("img_size", model_cfg.get("resize_to", 256)))
            clean_ds = CSVDataset(test_csv, transform=make_transform(img_size, "clean", 0))
            ckpt = os.path.join(ckpt_dir, f"{dataset_name}_{model_name}_s{args.seed}.pt")
            if not os.path.exists(ckpt):
                print(f"[skip] missing {ckpt}")
                continue
            model = build_model(model_name, cfg, clean_ds.num_classes, pretrained=False).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            clean_acc, clean_f1 = None, None
            for corruption in ["clean", "noise", "blur", "illumination", "contrast", "jpeg"]:
                severities = [0] if corruption == "clean" else args.severity
                for severity in severities:
                    ds = CSVDataset(test_csv, transform=make_transform(img_size, corruption, severity))
                    loader = DataLoader(ds, batch_size=int(model_cfg.get("batch_size", 64)), shuffle=False, num_workers=cfg["num_workers"])
                    acc, f1 = eval_model(model, loader, device)
                    if corruption == "clean":
                        clean_acc, clean_f1 = acc, f1
                    rows.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "corruption": corruption,
                        "severity": severity,
                        "severity_label": "clean" if severity == 0 else f"severity_{severity}",
                        "accuracy": acc,
                        "macro_f1": f1,
                        "clean_accuracy": clean_acc,
                        "clean_macro_f1": clean_f1,
                        "accuracy_drop": None if clean_acc is None else clean_acc - acc,
                        "macro_f1_drop": None if clean_f1 is None else clean_f1 - f1,
                    })
                    print(f"{dataset_name}/{model_name}/{corruption}{severity}: acc={acc:.4f} f1={f1:.4f}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "corruption_results.csv")
    df.to_csv(out_csv, index=False)
    summary = (
        df[df["corruption"] != "clean"]
        .groupby(["model", "dataset"])
        .agg(
            clean_accuracy=("clean_accuracy", "max"),
            mean_corrupt_accuracy=("accuracy", "mean"),
            cad_accuracy=("accuracy_drop", "mean"),
            clean_macro_f1=("clean_macro_f1", "max"),
            mean_corrupt_macro_f1=("macro_f1", "mean"),
            cad_macro_f1=("macro_f1_drop", "mean"),
        )
        .reset_index()
    )
    summary_csv = os.path.join(out_dir, "corruption_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved {out_csv}")
    print(f"Saved {summary_csv}")


if __name__ == "__main__":
    main()
