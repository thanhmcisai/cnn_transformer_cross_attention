# src/visualization/plot_accuracy.py
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
            train_acc = [h["train_acc"] for h in history]
            val_acc = [h["val_acc"] for h in history]

            plt.figure()
            plt.plot(epochs, train_acc, label="Train Acc")
            plt.plot(epochs, val_acc, label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dataset} - {model}")
            plt.legend()
            plt.tight_layout()

            save_path = os.path.join(out_dir, f"{dataset}_{model}_accuracy.png")
            plt.savefig(save_path)
            plt.close()

            print("Saved:", save_path)

if __name__ == "__main__":
    main()
