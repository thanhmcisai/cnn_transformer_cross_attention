# src/utils/model_loader.py
import os
import torch
import torch.nn as nn

from src.models.models import build_model


def load_weight_safe(model, weight_path, device):
    """
    Safely load pretrained weights.
    Returns:
        True        -> success
        False       -> failed
        "MISMATCH"  -> size mismatch
    """
    if not weight_path or not os.path.exists(weight_path):
        return False

    try:
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # Fix prefix: model.xxx
        model_keys = list(model.state_dict().keys())
        if model_keys and model_keys[0].startswith("model.") and not list(state_dict.keys())[0].startswith("model."):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        # Fix pos_emb if needed
        if hasattr(model, "pos_emb") and model.pos_emb is None and "pos_emb" in state_dict:
            model.pos_emb = nn.Parameter(torch.zeros_like(state_dict["pos_emb"]).to(device))

        model.load_state_dict(state_dict, strict=True)
        return True

    except RuntimeError as e:
        if "size mismatch" in str(e):
            return "MISMATCH"
        return False
    except Exception:
        return False


def get_working_model(model_name, cfg, num_classes, weight_path, device):
    """
    Build model from experiments.yaml and load pretrained weight safely.
    """
    try:
        model = build_model(model_name, cfg, num_classes).to(device)

        status = load_weight_safe(model, weight_path, device)

        if status is True:
            print(f"Loaded {model_name}")
            return model
        else:
            del model
            torch.cuda.empty_cache()
            return None

    except Exception as e:
        print(f"[FAIL] Could not load {model_name}: {e}")
        return None
