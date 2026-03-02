#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.repro.common import ensure_file_exists, run_python, write_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Table 1 count-prediction summary.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument("--results-dir", type=Path, default=REPO_DIR / "results" / "model_comparison")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metapath", type=str, default="CbGpPW")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 2000 if args.smoke else 10000
    script_args = [args.metapath, "--all", "--n_samples", str(n_samples), "--random_state", str(args.random_seed)]
    run_python(REPO_DIR / "test_src" / "test_nonlinear_mean_models.py", script_args)

    source_file = REPO_DIR / "results" / "nonlinear_mean_models" / f"{args.metapath}_comparison.csv"
    ensure_file_exists(source_file)

    df = pd.read_csv(source_file)
    summary = (
        df.groupby("model_type", as_index=False)
        .agg(
            r_mean=("r_mean", "mean"),
            z_mean=("z_mean", "mean"),
            z_std=("z_std", "mean"),
            z_outliers=("z_outliers", "mean"),
        )
        .sort_values("r_mean", ascending=False)
    )

    output_file = args.results_dir / "table1_count_prediction.csv"
    summary.to_csv(output_file, index=False)
    ensure_file_exists(output_file)

    write_run_summary(
        args.results_dir / "repro_table1_count_prediction_run_summary.json",
        args,
        extra={"source_file": source_file, "output_file": output_file},
    )


if __name__ == "__main__":
    main()
