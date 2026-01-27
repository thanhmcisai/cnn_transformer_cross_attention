# src/utils/io.py
"""
Utility functions for loading and validating YAML configuration files.

This module loads:
- configs/dataset.yaml
- configs/experiments.yaml

and provides a unified configuration dictionary for the pipeline:
    cfg = load_configs()

Author: Your Name
"""

import os
import yaml
import torch


# -----------------------------
# Basic YAML loader
# -----------------------------
def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Resolve device and workers
# -----------------------------
def resolve_device(device_cfg: str):
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def resolve_num_workers(num_workers_cfg):
    if num_workers_cfg == "auto":
        return os.cpu_count()
    return int(num_workers_cfg)


# -----------------------------
# Validation helpers
# -----------------------------
def validate_datasets(exp_cfg, dataset_cfg):
    available_datasets = set(dataset_cfg["datasets"].keys())
    for d in exp_cfg["datasets"]:
        if d not in available_datasets:
            raise ValueError(
                f"Dataset '{d}' not found in dataset.yaml. "
                f"Available: {available_datasets}"
            )


def validate_models(exp_cfg):
    available_models = set(exp_cfg["models"].keys())
    if len(available_models) == 0:
        raise ValueError("No models defined in experiments.yaml")


# -----------------------------
# Merge global + model configs
# -----------------------------
def build_model_train_config(global_train_cfg: dict, model_cfg: dict) -> dict:
    """
    Merge global training config with per-model config.
    Per-model values override global values.
    """
    train_cfg = global_train_cfg.copy()

    # Override with model-specific hyperparameters if provided
    for key in ["batch_size", "lr"]:
        if key in model_cfg:
            train_cfg[key] = model_cfg[key]

    return train_cfg


# -----------------------------
# Main loader
# -----------------------------
def load_configs(
    dataset_cfg_path="configs/dataset.yaml",
    experiments_cfg_path="configs/experiments.yaml"
):
    """
    Load and validate dataset.yaml and experiments.yaml

    Returns:
        cfg (dict) with keys:
            - dataset_cfg
            - experiments_cfg
            - device
            - num_workers
    """

    dataset_cfg = load_yaml(dataset_cfg_path)
    experiments_cfg = load_yaml(experiments_cfg_path)

    # Resolve device & workers (priority: experiments.yaml > dataset.yaml)
    device_cfg = experiments_cfg.get("global_train", {}).get(
        "device",
        dataset_cfg.get("device", "auto")
    )
    num_workers_cfg = experiments_cfg.get("global_train", {}).get(
        "num_workers",
        dataset_cfg.get("num_workers", "auto")
    )

    device = resolve_device(device_cfg)
    num_workers = resolve_num_workers(num_workers_cfg)

    # Validation
    validate_datasets(experiments_cfg, dataset_cfg)
    validate_models(experiments_cfg)

    cfg = {
        "dataset_cfg": dataset_cfg,
        "experiments_cfg": experiments_cfg,
        "device": device,
        "num_workers": num_workers,
    }

    return cfg


# -----------------------------
# Helper APIs for trainer
# -----------------------------
def get_datasets(cfg: dict):
    """Return list of dataset names to run experiments on."""
    return cfg["experiments_cfg"]["datasets"]


def get_models(cfg: dict):
    """Return dict of models from experiments.yaml."""
    return cfg["experiments_cfg"]["models"]


def get_global_train_cfg(cfg: dict):
    """Return global training config."""
    return cfg["experiments_cfg"]["global_train"]


def get_model_cfg(cfg: dict, model_name: str):
    models_cfg = get_models(cfg)
    if model_name not in models_cfg:
        raise ValueError(f"Model '{model_name}' not found in experiments.yaml")
    return models_cfg[model_name]


def get_model_train_cfg(cfg: dict, model_name: str):
    """
    Return merged training config for a given model:
    global_train + model-specific (batch_size, lr)
    """
    global_train = get_global_train_cfg(cfg)
    model_cfg = get_model_cfg(cfg, model_name)
    return build_model_train_config(global_train, model_cfg)


def get_dataset_info(cfg: dict, dataset_name: str):
    dataset_cfg = cfg["dataset_cfg"]
    if dataset_name not in dataset_cfg["datasets"]:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset.yaml")
    return dataset_cfg["datasets"][dataset_name]


# -----------------------------
# Debug / CLI test
# -----------------------------
if __name__ == "__main__":
    cfg = load_configs()

    print("Device:", cfg["device"])
    print("Num workers:", cfg["num_workers"])

    print("\nDatasets:")
    for d in get_datasets(cfg):
        print(" -", d)

    print("\nModels:")
    for m in get_models(cfg):
        m_cfg = get_model_cfg(cfg, m)
        train_cfg = get_model_train_cfg(cfg, m)
        print(f" - {m}: batch_size={train_cfg.get('batch_size')}, lr={train_cfg.get('lr')}")
