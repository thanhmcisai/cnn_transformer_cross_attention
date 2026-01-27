# src/visualization/plot_loss.py
import os, json
import matplotlib.pyplot as plt
from src.utils.io import load_configs, get_datasets, get_models

def main():
    cfg = load_configs()
    results_dir = "results"
    out_dir = os.path.join(results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    for dataset in get_datasets(cfg):
        for model in get_models(cfg):
            log_path = os.path.join(results_dir, f"{dataset}_{model}_nomix_history.json")
            if not os.path.exists(log_path):
                continue

            with open(log_path) as f:
                history = json.load(f)

            epochs = [h["epoch"] for h in history]
            train_loss = [h["train_loss"] for h in history]
            val_loss = [h["val_loss"] for h in history]

            plt.figure()
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset} - {model}")
            plt.legend()
            plt.tight_layout()

            save_path = os.path.join(out_dir, f"{dataset}_{model}_loss.png")
            plt.savefig(save_path)
            plt.close()

            print("Saved:", save_path)

if __name__ == "__main__":
    main()
