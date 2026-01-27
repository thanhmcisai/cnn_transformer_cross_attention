import os
import time
import torch
import pandas as pd
from tqdm import tqdm

from thop import profile

from src.utils.io import load_configs, get_models
from src.models.models import build_model


def get_input_size(model_name):
    """Return input size depending on model"""
    if model_name == "coatnet_0_rw_224":
        return (1, 3, 224, 224)
    else:
        return (1, 3, 256, 256)


def benchmark_model(model_name, cfg, device, num_classes=100):
    input_size = get_input_size(model_name)

    dummy_input = torch.randn(input_size).to(device)

    model = build_model(model_name, cfg, num_classes).to(device)
    model.eval()

    # --------------------
    # Params & FLOPs
    # --------------------
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2  # 1 MAC ≈ 2 FLOPs

    params_m = params / 1e6
    flops_g = flops / 1e9

    # --------------------
    # Inference Time
    # --------------------
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        ender.record()

        torch.cuda.synchronize()
        avg_time_ms = starter.elapsed_time(ender) / 100
    else:
        start = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy_input)
        end = time.time()
        avg_time_ms = (end - start) * 1000 / 50

    return round(params_m, 2), round(flops_g, 2), round(avg_time_ms, 2)


def main():
    cfg = load_configs()
    DEVICE = cfg["device"]

    os.makedirs("results/analysis", exist_ok=True)
    save_path = "results/analysis/complexity_benchmark.csv"

    results = []

    print("\n" + "=" * 70)
    print("COMPUTATIONAL COMPLEXITY BENCHMARK")
    print("=" * 70)
    print(f"{'Model':<30} | {'Params (M)':<12} | {'GFLOPs':<10} | {'Time (ms)':<10}")
    print("-" * 70)

    for model_name in get_models(cfg):
        try:
            params_m, flops_g, time_ms = benchmark_model(
                model_name, cfg, DEVICE, num_classes=100
            )

            results.append({
                "Model": model_name,
                "Params (M)": params_m,
                "GFLOPs": flops_g,
                "Inference Time (ms)": time_ms
            })

            print(f"{model_name:<30} | {params_m:<12} | {flops_g:<10} | {time_ms:<10}")

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

    print("\n✅ Saved complexity benchmark to:", save_path)


if __name__ == "__main__":
    main()
