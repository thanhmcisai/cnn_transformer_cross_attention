# src/visualization/attention_map.py
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils.io import load_configs, get_datasets, get_models
from src.utils.model_loader import get_working_model


# -------------------------------------------------
# Transform
# -------------------------------------------------
def get_val_transform(model_name):
    img_size = 224 if model_name == "coatnet_0_rw_224" else 256
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])


# -------------------------------------------------
# Find last conv layer automatically
# -------------------------------------------------
def get_last_conv_layer(model):
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None


# -------------------------------------------------
# Grad-CAM
# -------------------------------------------------
def grad_cam(model, img_tensor, target_layer):
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    pred = output.argmax(dim=1)

    model.zero_grad()
    output[:, pred].backward()

    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]

    weights = grad.mean(axis=(1, 2))
    cam = np.sum(weights[:, None, None] * act, axis=0)
    cam = np.maximum(cam, 0)

    cam = cam / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cam


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    cfg = load_configs()
    dataset_cfg = cfg["dataset_cfg"]
    exp_cfg = cfg["experiments_cfg"]

    DEVICE = cfg["device"]

    weights_cfg = exp_cfg["weights"]
    weight_root = weights_cfg["root_dir"]
    weight_map = weights_cfg["mapping"]

    processed_root = dataset_cfg["processed_root"]

    save_dir = "results/visualization/attention"
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name in get_datasets(cfg):
        print(f"\n=== Dataset: {dataset_name} ===")

        test_dir = os.path.join(processed_root, dataset_name, "test")
        if not os.path.exists(test_dir):
            continue

        if dataset_name not in weight_map:
            continue

        for model_name in get_models(cfg):
            if model_name not in weight_map[dataset_name]:
                continue

            print(f"--- Model: {model_name} ---")

            ckpt_path = os.path.join(weight_root, weight_map[dataset_name][model_name])
            if not os.path.exists(ckpt_path):
                print("Weight not found:", ckpt_path)
                continue

            transform = get_val_transform(model_name)
            ds = datasets.ImageFolder(test_dir, transform=transform)
            num_classes = len(ds.classes)

            model = get_working_model(model_name, cfg, num_classes, ckpt_path, DEVICE)
            if model is None:
                continue

            target_layer = get_last_conv_layer(model)
            if target_layer is None:
                print(f"No conv layer found for {model_name}, skipping.")
                del model
                torch.cuda.empty_cache()
                continue

            img, _ = ds[0]
            img_tensor = img.unsqueeze(0).to(DEVICE)

            cam = grad_cam(model, img_tensor, target_layer)

            img_np = img.permute(1,2,0).cpu().numpy()
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = 0.6 * img_np + 0.4 * heatmap / 255.0
            overlay = np.clip(overlay, 0, 1)

            plt.figure(figsize=(4,4))
            plt.imshow(overlay)
            plt.axis("off")
            plt.title(f"{dataset_name} - {model_name}")

            out_path = os.path.join(
                save_dir, f"{dataset_name}_{model_name}_cam.png"
            )
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()

            print("Saved:", out_path)

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
