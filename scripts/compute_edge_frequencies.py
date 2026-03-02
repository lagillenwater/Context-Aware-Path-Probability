"""Compute empirical edge frequencies by (source_degree, target_degree).

This script replaces notebook-driven edge-frequency computation with a direct CLI.
It preserves the notebook 3 calculation pattern:
  frequency = edge_count_over_all_perms / total_possible_pairs_over_all_perms
for each observed (source_degree, target_degree) combination.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


DEFAULT_EDGE_FILES = [
    "AdG.sparse.npz",
    "AeG.sparse.npz",
    "AuG.sparse.npz",
    "CbG.sparse.npz",
    "CcSE.sparse.npz",
    "CdG.sparse.npz",
    "CpD.sparse.npz",
    "CrC.sparse.npz",
    "CtD.sparse.npz",
    "CuG.sparse.npz",
    "DaG.sparse.npz",
    "DdG.sparse.npz",
    "DlA.sparse.npz",
    "DpS.sparse.npz",
    "DrD.sparse.npz",
    "DuG.sparse.npz",
    "GcG.sparse.npz",
    "GiG.sparse.npz",
    "GpBP.sparse.npz",
    "GpCC.sparse.npz",
    "GpMF.sparse.npz",
    "GpPW.sparse.npz",
    "Gr>G.sparse.npz",
    "PCiC.sparse.npz",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compute empirical edge-frequency CSVs from downloaded Hetionet "
            "permutations."
        )
    )
    parser.add_argument(
        "--edge-file",
        action="append",
        default=[],
        help=(
            "Edge matrix filename to process (e.g., AeG.sparse.npz). "
            "Repeat to pass multiple values. Default: all canonical edge files."
        ),
    )
    parser.add_argument(
        "--perm-dir",
        type=Path,
        default=repo / "data" / "downloads" / "hetionet-permutations" / "permutations",
        help="Directory containing downloaded *.hetmat permutation folders.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo / "results" / "empirical_edge_frequencies",
        help="Output directory for edge_frequency_by_degree_*.csv files.",
    )
    parser.add_argument(
        "--max-perms",
        type=int,
        default=None,
        help="Optional cap on number of permutation folders to process.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N permutations.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.max_perms is not None and args.max_perms <= 0:
        raise ValueError("--max-perms must be > 0")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")
    for edge_file in args.edge_file:
        if not edge_file.endswith(".sparse.npz"):
            raise ValueError(
                f"--edge-file value '{edge_file}' must end with '.sparse.npz'"
            )


def get_perm_folders(perm_dir: Path, max_perms: int | None) -> list[Path]:
    if not perm_dir.exists():
        raise FileNotFoundError(f"Permutation directory not found: {perm_dir}")
    folders = sorted(p for p in perm_dir.iterdir() if p.is_dir() and p.name.endswith(".hetmat"))
    if max_perms is not None:
        folders = folders[:max_perms]
    return folders


def compute_frequency_df(
    edge_file: str,
    perm_folders: list[Path],
    progress_every: int,
) -> tuple[pd.DataFrame, int]:
    freq: dict[tuple[int, int], int] = defaultdict(int)
    total_counts: dict[tuple[int, int], int] = defaultdict(int)
    n_perms = 0

    for perm_idx, perm_path in enumerate(perm_folders):
        edge_path = perm_path / "edges" / edge_file
        if progress_every > 0 and perm_idx % progress_every == 0:
            print(
                f"Processing permutation {perm_idx + 1}/{len(perm_folders)}: "
                f"{perm_path.name}"
            )
        if not edge_path.exists():
            continue

        adj = sp.load_npz(edge_path)
        src_degrees = np.asarray(adj.sum(axis=1)).ravel().astype(int)
        tgt_degrees = np.asarray(adj.sum(axis=0)).ravel().astype(int)

        src_nodes, tgt_nodes = adj.nonzero()
        for s, t in zip(src_nodes, tgt_nodes):
            key = (int(src_degrees[s]), int(tgt_degrees[t]))
            freq[key] += 1

        src_degree_counts = np.bincount(src_degrees)
        tgt_degree_counts = np.bincount(tgt_degrees)

        for src_deg, src_count in enumerate(src_degree_counts):
            if src_count == 0:
                continue
            for tgt_deg, tgt_count in enumerate(tgt_degree_counts):
                if tgt_count == 0:
                    continue
                total_counts[(src_deg, tgt_deg)] += int(src_count * tgt_count)

        n_perms += 1

    freq_matrix = {k: freq[k] / total_counts[k] for k in freq}
    freq_df = pd.DataFrame(
        {
            "source_degree": [k[0] for k in freq_matrix.keys()],
            "target_degree": [k[1] for k in freq_matrix.keys()],
            "frequency": [v for v in freq_matrix.values()],
        }
    )
    return freq_df, n_perms


def main() -> int:
    args = parse_args()
    edge_files = args.edge_file if args.edge_file else DEFAULT_EDGE_FILES
    perm_folders = get_perm_folders(args.perm_dir, args.max_perms)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Permutation directory: {args.perm_dir}")
    print(f"Found {len(perm_folders)} permutation folders.")
    print(f"Results directory: {args.results_dir}")
    print(f"Edge files to process: {len(edge_files)}")

    for edge_file in edge_files:
        edge_type = edge_file.replace(".sparse.npz", "")
        print("\n" + "=" * 80)
        print(f"Computing empirical frequencies for {edge_type}")
        print("=" * 80)
        freq_df, n_perms = compute_frequency_df(
            edge_file=edge_file,
            perm_folders=perm_folders,
            progress_every=args.progress_every,
        )
        out_path = args.results_dir / f"edge_frequency_by_degree_{edge_type}.csv"
        freq_df.to_csv(out_path, index=False)
        print(f"Processed {n_perms} permutations successfully.")
        print(f"Rows written: {len(freq_df)}")
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
