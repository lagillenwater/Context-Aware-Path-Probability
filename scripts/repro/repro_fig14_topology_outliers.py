#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.repro.common import ensure_dir_exists, ensure_file_exists, run_python, write_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Figure 14 topology outlier analysis.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "topology_specific_outliers",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    script_args = [
        "--data-dir", str(args.data_dir),
        "--results-dir", str(args.results_dir),
        "--random-state", str(args.random_seed),
    ]
    if args.smoke:
        script_args.append("--smoke")
    if args.skip_plots:
        script_args.append("--skip-plots")

    run_python(REPO_DIR / "test_src" / "test_topology_specific_outliers.py", script_args)

    summary_file = args.results_dir / "topology_specific_outlier_summary.tsv"
    ensure_dir_exists(args.results_dir)
    ensure_file_exists(summary_file)

    write_run_summary(
        args.results_dir / "repro_fig14_topology_outliers_run_summary.json",
        args,
        extra={"summary_file": summary_file},
    )


if __name__ == "__main__":
    main()
