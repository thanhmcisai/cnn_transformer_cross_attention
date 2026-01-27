# src/train/engine.py
"""
Training and evaluation engine (refactored from original code)

Contains:
- train_one_epoch
- validate
- compute_metrics
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------------------------------------------------
# Train one epoch
# -------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, criterion,
                    scaler, mixup_fn, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # Mixup
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        # Accuracy (only valid when not mixup)
        if mixup_fn is None:
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        else:
            total += targets.size(0)

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total

    if mixup_fn is None:
        acc = correct / total
    else:
        acc = 0.0  # not meaningful with mixup

    return avg_loss, acc


# -------------------------------------------------
# Validate one epoch
# -------------------------------------------------
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="Validating", leave=False)

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


# -------------------------------------------------
# Compute full metrics (Acc, Precision, Recall, F1)
# -------------------------------------------------
@torch.no_grad()
def compute_metrics(model, val_loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    pbar = tqdm(val_loader, desc="Computing metrics", leave=False)

    for images, targets in pbar:
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

    return acc, prec, rec, f1
