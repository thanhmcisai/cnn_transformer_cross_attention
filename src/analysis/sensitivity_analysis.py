import argparse
import os
import subprocess
import sys

import pandas as pd

from src.utils.io import load_configs


DEFAULT_MODELS = ["WiT_depth1", "WiT_depth2", "WiT", "WiT_depth6", "WiT_global_query"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run WiT architecture sensitivity experiments.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="Only extract sensitivity rows from results_raw.csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs()
    paths = cfg["experiments_cfg"].get("paths", {})
    results_dir = paths.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    if not args.no_train:
        cmd = [sys.executable, "-m", "src.train.train_multiseed", "--models", *args.models]
        if args.datasets:
            cmd.extend(["--datasets", *args.datasets])
        if args.seeds:
            cmd.extend(["--seeds", *[str(s) for s in args.seeds]])
        if args.force:
            cmd.append("--force")
        subprocess.check_call(cmd)

    raw_csv = os.path.join(results_dir, "results_raw.csv")
    out_csv = os.path.join(results_dir, "sensitivity_results.csv")
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"{raw_csv} not found. Run training first or omit --no-train.")
    df = pd.read_csv(raw_csv)
    sens = df[df["model"].isin(args.models)].copy()
    sens.to_csv(out_csv, index=False)

    summary_csv = os.path.join(results_dir, "sensitivity_summary.csv")
    if not sens.empty:
        summary = (
            sens.groupby(["dataset", "model"])
            .agg(
                test_acc_mean=("test_acc", "mean"),
                test_acc_std=("test_acc", "std"),
                test_f1_mean=("test_f1", "mean"),
                test_f1_std=("test_f1", "std"),
                n_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        summary.to_csv(summary_csv, index=False)
    print(f"Saved {out_csv}")
    print(f"Saved {summary_csv}")


if __name__ == "__main__":
    main()
