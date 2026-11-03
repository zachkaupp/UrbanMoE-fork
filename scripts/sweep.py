"""Train main.py for each config and seed, then write the test metrics averaged over seeds to a CSV.

    uv run python scripts/sweep.py
    uv run python scripts/sweep.py --seeds 132 0 1 --output results.csv

Run from the repo root. Each run logs to logs/sweep/<config>/<seed>.log, and runs whose log already has test
metrics are skipped, so rerunning after a crash only trains what is missing. All runs share one checkpoint
directory, since each checkpoint is ~1.7 GB and only the test metrics are kept.
"""
import argparse
import os
import re
import subprocess
import sys

import pandas as pd

COMMON = ["--dataset", "NewYork", "--no-use_poi"]
NO_REGION = ["--no-use_region_embedding"]

CONFIGS = {
    "image_text_region": ["--modalities", "image", "text"],
    "image_text": ["--modalities", "image", "text", *NO_REGION],
    "image_text_aef": ["--modalities", "image", "text", "alphaearth", *NO_REGION],
    "image_text_cc": ["--modalities", "image", "text", "contextual_count", *NO_REGION],
    "aef_text": ["--modalities", "alphaearth", "text", *NO_REGION],
    "image_cc": ["--modalities", "image", "contextual_count", *NO_REGION],
    "aef_cc": ["--modalities", "alphaearth", "contextual_count", *NO_REGION],
}


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[132, 0, 1])
    parser.add_argument("--configs", type=str, nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--output", type=str, default="sweep_results.csv")
    return parser.parse_args()


def test_metrics(log_path):
    """The last "Test metrics" line of a log as {metric: value}, or None if there is none."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        lines = [line for line in f if "Test metrics:" in line]
    if not lines:
        return None
    pairs = re.findall(r"'(\w+)': (?:np\.float64\()?([-0-9.e+]+)", lines[-1])
    return {name: float(value) for name, value in pairs}


def main():
    args = create_args()
    rows = []
    for name in args.configs:
        log_dir = os.path.join("logs", "sweep", name)
        for seed in args.seeds:
            log_path = os.path.join(log_dir, f"{seed}.log")
            if test_metrics(log_path) is None:
                command = [sys.executable, "main.py", *COMMON, *CONFIGS[name], "--seed", str(seed),
                           "--logging_dir", log_dir, "--checkpoint_dir", "checkpoint/sweep"]
                print(" ".join(command), flush=True)
                subprocess.run(command, check=True)
            rows.append({"config": name, "seed": seed, **test_metrics(log_path)})

    runs = pd.DataFrame(rows)
    runs = runs[[c for c in runs.columns if not c.startswith("mape")]]
    summary = runs.drop(columns="seed").groupby("config", sort=False).agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "seeds", runs.groupby("config", sort=False).size())
    summary.insert(1, "args", [" ".join(CONFIGS[name]) for name in summary.index])
    summary.round(4).to_csv(args.output)
    print(summary.filter(like="r2_").round(3).to_string())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
