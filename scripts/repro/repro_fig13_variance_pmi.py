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
    parser = argparse.ArgumentParser(description="Reproduce Figure 13 variance-vs-PMI analysis.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument("--results-dir", type=Path, default=REPO_DIR / "results" / "variance_pmi")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    run_python(REPO_DIR / "test_src" / "analyze_variance_pmi_relationship.py")

    src_dir = REPO_DIR / "results" / "variance_pmi_analysis"
    src_fig = src_dir / "variance_pmi_analysis.png"
    src_csv = src_dir / "variance_pmi_correlations.csv"

    dst_fig = args.results_dir / "variance_pmi_analysis.png"
    dst_csv = args.results_dir / "variance_pmi_correlations.csv"

    ensure_file_exists(src_csv)
    copy_file(src_csv, dst_csv)

    if not args.skip_plots:
        ensure_file_exists(src_fig)
        copy_file(src_fig, dst_fig)

    write_run_summary(
        args.results_dir / "repro_fig13_variance_pmi_run_summary.json",
        args,
        extra={
            "source_dir": src_dir,
            "output_file": dst_fig,
            "correlation_file": dst_csv,
            "skip_plots": args.skip_plots,
        },
    )


if __name__ == "__main__":
    main()
