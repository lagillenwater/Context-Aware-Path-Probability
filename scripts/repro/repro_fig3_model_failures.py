#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.repro.common import ensure_file_exists, run_python, write_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Figure 3 model-failure analysis.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument("--results-dir", type=Path, default=REPO_DIR / "results" / "model_failures")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = args.results_dir / "model_failure_analysis.png"

    script_args = [
        "--data-dir", str(args.data_dir),
        "--results-dir", str(args.results_dir),
        "--random-state", str(args.random_seed),
    ]
    if args.smoke:
        script_args.extend(["--smoke", "--n-samples", "2000"])
    if args.skip_plots:
        script_args.append("--skip-plots")

    run_python(REPO_DIR / "test_src" / "visualize_model_failures.py", script_args)
    if not args.skip_plots:
        ensure_file_exists(output_file)

    write_run_summary(
        args.results_dir / "repro_fig3_model_failures_run_summary.json",
        args,
        extra={"output_file": output_file, "skip_plots": args.skip_plots},
    )


if __name__ == "__main__":
    main()
