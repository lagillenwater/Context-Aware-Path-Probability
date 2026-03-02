#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.repro.common import ensure_file_exists, write_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Figure 4 permutation similarity heatmap.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "permuations_similarlity",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--edge-type", type=str, default="AeG")
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def list_available_perms(data_dir: Path) -> list[int]:
    perm_dir = data_dir / "permutations"
    if not perm_dir.exists():
        return []
    ids: list[int] = []
    for child in perm_dir.glob("*.hetmat"):
        try:
            ids.append(int(child.stem))
        except ValueError:
            continue
    return sorted(set(ids))


def load_edge_matrix(data_dir: Path, perm_id: int, edge_type: str) -> sp.csr_matrix:
    path = data_dir / "permutations" / f"{perm_id:03d}.hetmat" / "edges" / f"{edge_type}.sparse.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return sp.load_npz(path).astype(bool).tocsr()


def jaccard(a: sp.csr_matrix, b: sp.csr_matrix) -> float:
    inter = a.multiply(b).nnz
    union = a.nnz + b.nnz - inter
    if union == 0:
        return 0.0
    return float(inter / union)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    n_perms = 6 if args.smoke else args.n_permutations
    available = list_available_perms(args.data_dir)
    if len(available) < 2:
        raise RuntimeError("Need at least 2 local permutations for similarity analysis.")

    selected = available[: min(n_perms, len(available))]
    matrices: dict[int, sp.csr_matrix] = {}
    for perm_id in selected:
        try:
            matrices[perm_id] = load_edge_matrix(args.data_dir, perm_id, args.edge_type)
        except FileNotFoundError:
            continue

    perm_ids = sorted(matrices)
    if len(perm_ids) < 2:
        raise RuntimeError(f"No usable permutations found for edge type {args.edge_type}.")

    sim = np.zeros((len(perm_ids), len(perm_ids)), dtype=float)
    for i, pi in enumerate(perm_ids):
        for j, pj in enumerate(perm_ids):
            sim[i, j] = jaccard(matrices[pi], matrices[pj])

    csv_path = args.results_dir / f"{args.edge_type}_permutation_similarity.csv"
    np.savetxt(csv_path, sim, delimiter=",", fmt="%.6f")

    fig_path = args.results_dir / f"{args.edge_type}_permutation_similarity.png"
    if not args.skip_plots:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sim, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"Permutation Similarity ({args.edge_type}, Jaccard)")
        ax.set_xlabel("Permutation ID")
        ax.set_ylabel("Permutation ID")
        ax.set_xticks(range(len(perm_ids)))
        ax.set_yticks(range(len(perm_ids)))
        ax.set_xticklabels([f"{p:03d}" for p in perm_ids], rotation=90)
        ax.set_yticklabels([f"{p:03d}" for p in perm_ids])
        fig.colorbar(im, ax=ax, label="Jaccard similarity")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        ensure_file_exists(fig_path)

    write_run_summary(
        args.results_dir / "repro_fig4_permutation_similarity_run_summary.json",
        args,
        extra={
            "output_file": fig_path,
            "csv_file": csv_path,
            "n_loaded_permutations": len(perm_ids),
            "permutation_ids": perm_ids,
        },
    )


if __name__ == "__main__":
    main()
