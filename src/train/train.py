import os
import gc
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from timm.data import Mixup
import timm

from src.utils.io import (
    load_configs,
    get_datasets,
    get_models,
    get_model_train_cfg,
)

from src.models.models import build_model
from src.train.engine import train_one_epoch, validate, compute_metrics

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------
# Transforms (GIỮ NGUYÊN LOGIC CỦA BẠN)
# -------------------------------------------------
def get_transforms_for_model(model_name):
    if model_name == 'coatnet_0_rw_224':
        img_size = 224
    else:
        img_size = 256

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.RandomAffine(
            degrees=45,
            translate=(0.15, 0.15),
            scale=(0.7, 1.3),
            shear=15
        ),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
        ], p=0.9),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomApply([transforms.RandomAdjustSharpness(3)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.3), ratio=(0.3,3.0), value='random')
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    return train_transform, val_transform


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    cfg = load_configs()

    dataset_cfg = cfg["dataset_cfg"]
    exp_cfg = cfg["experiments_cfg"]

    seed = dataset_cfg.get("seed", 42)
    set_seed(seed)

    DEVICE = cfg["device"]
    n_cores = cfg["num_workers"]

    NUM_EPOCHS = exp_cfg["global_train"]["epochs"]
    EARLY_STOP = exp_cfg["global_train"]["early_stop"]
    WARMUP_EPOCHS = exp_cfg["global_train"]["warmup_epochs"]

    processed_root = dataset_cfg["processed_root"]
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "final_results.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "model", "tag", "acc", "precision", "recall", "f1"])

    for dataset_name in get_datasets(cfg):
        dataset_path = os.path.join(processed_root, dataset_name)
        print(f"\n==================== Dataset: {dataset_name} ====================")

        train_dir = os.path.join(dataset_path, "train")
        val_dir = os.path.join(dataset_path, "test")

        for model_name in get_models(cfg):

            train_transform, val_transform = get_transforms_for_model(model_name)

            train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
            val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

            num_classes = len(train_ds.classes)
            print("Classes:", num_classes)

            train_cfg = get_model_train_cfg(cfg, model_name)
            batch_size = train_cfg["batch_size"]
            lr = train_cfg["lr"]

            for use_mix in [False]:  # giữ đúng logic bạn đang dùng
                tag = "mixup" if use_mix else "nomix"
                print(f"\n🚀 Training {model_name} | {tag}")

                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=n_cores,
                    pin_memory=True
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=n_cores,
                    pin_memory=True
                )

                model = build_model(model_name, cfg, num_classes).to(DEVICE)

                mixup_fn = None
                if use_mix:
                    mixup_fn = Mixup(
                        mixup_alpha=0.8,
                        cutmix_alpha=1.0,
                        prob=0.8,
                        switch_prob=0.5,
                        label_smoothing=0.1,
                        num_classes=num_classes
                    )

                optimizer = optim.AdamW(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
                )

                criterion = nn.CrossEntropyLoss()
                scaler = torch.cuda.amp.GradScaler()

                best_acc = 0
                patience = 0

                save_dir = os.path.join(results_dir, f"{dataset_name}_{model_name}_{tag}")
                os.makedirs(save_dir, exist_ok=True)
                best_path = os.path.join(save_dir, "best.pth")

                for epoch in range(NUM_EPOCHS):

                    # Warmup
                    if epoch < WARMUP_EPOCHS:
                        warmup_lr = lr * (epoch + 1) / WARMUP_EPOCHS
                        for g in optimizer.param_groups:
                            g['lr'] = warmup_lr
                        lr_now = warmup_lr
                    else:
                        scheduler.step()
                        lr_now = optimizer.param_groups[0]['lr']

                    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | LR={lr_now:.6f}")

                    tl, ta = train_one_epoch(
                        model, train_loader, optimizer,
                        criterion, scaler, mixup_fn, DEVICE
                    )
                    vl, va = validate(model, val_loader, criterion, DEVICE)

                    print(f"TrainAcc={ta:.4f} | ValAcc={va:.4f}")

                    # ----- EARLY STOP AFTER WARMUP -----
                    if epoch < WARMUP_EPOCHS:
                        if va > best_acc:
                            best_acc = va
                            torch.save(model.state_dict(), best_path)
                        continue

                    if va > best_acc:
                        best_acc = va
                        torch.save(model.state_dict(), best_path)
                        patience = 0
                    else:
                        patience += 1
                        if patience >= EARLY_STOP:
                            print("⛔ Early stopping triggered.")
                            break

                print("🔥 Reloading best model...")
                model.load_state_dict(torch.load(best_path))

                acc, prec, rec, f1 = compute_metrics(model, val_loader, DEVICE)
                print(f"FINAL METRICS: Acc={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([dataset_name, model_name, tag, acc, prec, rec, f1])

                del model, optimizer, scaler, criterion, scheduler, train_loader, val_loader
                gc.collect()
                torch.cuda.empty_cache()

    print("\n🎉 ALL EXPERIMENTS COMPLETED 🎉")


if __name__ == "__main__":
    main()
