#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.repro.common import copy_file, ensure_file_exists, run_python, write_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Figure 16 z-score and QQ diagnostics.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "model_comparison" / "qq_and_zscore",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metapath", type=str, default="CbGpPW")
    parser.add_argument("--model", type=str, default="rf")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 2000 if args.smoke else 10000
    script_args = [
        args.metapath,
        "--model",
        args.model,
        "--n_samples",
        str(n_samples),
        "--random_state",
        str(args.random_seed),
    ]
    run_python(REPO_DIR / "test_src" / "test_nonlinear_mean_models.py", script_args)

    src_dir = REPO_DIR / "results" / "nonlinear_mean_models"
    src_fig = src_dir / f"distribution_diagnostics_{args.model}.png"
    src_csv = src_dir / f"{args.metapath}_{args.model}.csv"

    dst_fig = args.results_dir / src_fig.name
    dst_csv = args.results_dir / src_csv.name

    ensure_file_exists(src_csv)
    copy_file(src_csv, dst_csv)
    if not args.skip_plots:
        ensure_file_exists(src_fig)
        copy_file(src_fig, dst_fig)

    write_run_summary(
        args.results_dir / "repro_fig16_zscore_qq_run_summary.json",
        args,
        extra={
            "source_dir": src_dir,
            "output_figure": dst_fig,
            "output_csv": dst_csv,
            "skip_plots": args.skip_plots,
        },
    )


if __name__ == "__main__":
    main()
