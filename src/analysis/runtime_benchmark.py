import argparse
import os
import time

import pandas as pd
import torch
from thop import profile

from src.models.models import build_model
from src.utils.io import get_model_train_cfg, get_models, load_configs


def input_size_for(cfg, model_name):
    model_cfg = get_model_train_cfg(cfg, model_name)
    size = int(model_cfg.get("resize_to", model_cfg.get("img_size", 256)))
    return size


def count_complexity(model, dummy):
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    return params / 1e6, (macs * 2) / 1e9


@torch.no_grad()
def measure_torch_latency(model, dummy, device, warmup, timed):
    model.eval()
    for _ in range(warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed):
            _ = model(dummy)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / timed
    start = time.perf_counter()
    for _ in range(timed):
        _ = model(dummy)
    return (time.perf_counter() - start) * 1000.0 / timed


def measure_onnx_latency(model, dummy, onnx_path, warmup, timed):
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception:
        return None
    torch.onnx.export(
        model.cpu(),
        dummy.cpu(),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    arr = dummy.cpu().numpy().astype("float32")
    for _ in range(warmup):
        session.run(None, {"input": arr})
    start = time.perf_counter()
    for _ in range(timed):
        session.run(None, {"input": arr})
    return (time.perf_counter() - start) * 1000.0 / timed


def parse_args():
    parser = argparse.ArgumentParser(description="Runtime, throughput, parameter, and FLOP benchmark.")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timed", type=int, default=100)
    parser.add_argument("--onnx", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    paths = cfg["experiments_cfg"].get("paths", {})
    out_dir = os.path.join(paths.get("results_dir", "results"), "runtime")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg["device"])
    rows = []

    for model_name in args.models or get_models(cfg):
        try:
            img_size = input_size_for(cfg, model_name)
            dummy = torch.randn(args.batch_size, 3, img_size, img_size, device=device)
            model = build_model(model_name, cfg, args.num_classes, pretrained=False).to(device).eval()
            params_m, gflops = count_complexity(model, dummy)
            torch_ms = measure_torch_latency(model, dummy, device, args.warmup, args.timed)
            onnx_ms = None
            if args.onnx:
                onnx_path = os.path.join(out_dir, f"{model_name}.onnx")
                onnx_ms = measure_onnx_latency(model, dummy, onnx_path, args.warmup, args.timed)
            rows.append({
                "model": model_name,
                "input_size": img_size,
                "batch_size": args.batch_size,
                "params_M": round(params_m, 3),
                "gflops": round(gflops, 3),
                "torch_latency_ms": round(torch_ms, 3),
                "torch_throughput_img_s": round(args.batch_size * 1000.0 / torch_ms, 2),
                "onnx_cpu_latency_ms": None if onnx_ms is None else round(onnx_ms, 3),
            })
            print(f"{model_name}: {torch_ms:.2f} ms, {params_m:.2f}M params, {gflops:.2f} GFLOPs")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[skip] {model_name}: {exc}")

    out_csv = os.path.join(out_dir, "runtime_benchmark.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
