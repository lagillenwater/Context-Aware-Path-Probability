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
    parser = argparse.ArgumentParser(description="Reproduce Figure 2 path-count heatmap.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "path_count_visualization",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = args.results_dir / "CbGpPWpG_path_count_heatmap.png"

    script_args = [
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--output-filename",
        output_file.name,
    ]
    if args.smoke:
        script_args.append("--smoke")

    run_python(REPO_DIR / "test_src" / "create_path_count_heatmap.py", script_args)
    ensure_file_exists(output_file)

    write_run_summary(
        args.results_dir / "repro_fig2_pathcount_heatmap_run_summary.json",
        args,
        extra={"output_file": output_file},
    )


if __name__ == "__main__":
    main()
