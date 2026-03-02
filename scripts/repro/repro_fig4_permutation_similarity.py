#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.repro.common import ensure_file_exists, write_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 4 permutation similarity (degree preservation + degree correlation)."
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "permuations_similarlity",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--edge-type", type=str, default="AeG")
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--max-scatter-points", type=int, default=5000)
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


def load_perm_edge_matrix(data_dir: Path, perm_id: int, edge_type: str) -> sp.csr_matrix:
    path = data_dir / "permutations" / f"{perm_id:03d}.hetmat" / "edges" / f"{edge_type}.sparse.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return sp.load_npz(path).tocsr()


def load_reference_edge_matrix(data_dir: Path, edge_type: str, available_perms: list[int]) -> tuple[sp.csr_matrix, str]:
    original_path = data_dir / "edges" / f"{edge_type}.sparse.npz"
    if original_path.exists():
        return sp.load_npz(original_path).tocsr(), "Hetionet"

    if 0 in available_perms:
        return load_perm_edge_matrix(data_dir, 0, edge_type), "Perm 000"

    if available_perms:
        fallback = available_perms[0]
        return load_perm_edge_matrix(data_dir, fallback, edge_type), f"Perm {fallback:03d}"

    raise RuntimeError("No reference or permutation edge matrix available.")


def load_for_perm_or_reference(
    data_dir: Path,
    perm_id: int,
    edge_type: str,
    available_perms: list[int],
) -> sp.csr_matrix:
    """Load matrix for a permutation ID, with fallback for perm 000."""
    if perm_id in available_perms:
        return load_perm_edge_matrix(data_dir, perm_id, edge_type)
    if perm_id == 0:
        ref_matrix, _ = load_reference_edge_matrix(data_dir, edge_type, available_perms)
        return ref_matrix
    raise FileNotFoundError(f"Permutation {perm_id:03d} missing for edge type {edge_type}")


def compute_degree_correlation(matrix: sp.csr_matrix) -> float:
    source_degrees = np.asarray(matrix.sum(axis=1)).ravel()
    target_degrees = np.asarray(matrix.sum(axis=0)).ravel()
    edges_i, edges_j = matrix.nonzero()

    if edges_i.size < 2:
        return float("nan")

    source_at_edges = source_degrees[edges_i]
    target_at_edges = target_degrees[edges_j]

    if np.std(source_at_edges) == 0 or np.std(target_at_edges) == 0:
        return float("nan")

    return float(np.corrcoef(source_at_edges, target_at_edges)[0, 1])


def assign_group(perm_id: int) -> str:
    if perm_id == 0:
        return "Hetionet"
    if 1 <= perm_id <= 20:
        return "Training (001-020)"
    if 21 <= perm_id <= 30:
        return "Validation (021-030)"
    return "Remaining (031+)"


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    n_perms = 6 if args.smoke else args.n_permutations
    available = list_available_perms(args.data_dir)
    if not available:
        raise RuntimeError("No local permutations found under data/permutations.")

    nonzero_available = [perm for perm in available if perm > 0]
    selected = nonzero_available[: min(n_perms, len(nonzero_available))]
    if len(selected) < 1:
        # If only perm 0 is available, still allow running with it.
        selected = available[:1]

    ref_matrix, ref_label = load_reference_edge_matrix(args.data_dir, args.edge_type, available)
    ref_source_deg = np.asarray(ref_matrix.sum(axis=1)).ravel()
    # Prefer permutation 000 as the reference correlation to match notebook semantics.
    if 0 in available:
        ref_corr = compute_degree_correlation(load_perm_edge_matrix(args.data_dir, 0, args.edge_type))
    else:
        ref_corr = compute_degree_correlation(ref_matrix)

    analysis_perm_ids = selected.copy()
    if 0 not in analysis_perm_ids:
        analysis_perm_ids = [0, *analysis_perm_ids]

    records: list[dict[str, object]] = []
    sample_perm = selected[0]
    sample_source_deg = None

    for perm_id in analysis_perm_ids:
        matrix = load_for_perm_or_reference(args.data_dir, perm_id, args.edge_type, available)
        src_deg = np.asarray(matrix.sum(axis=1)).ravel()
        corr = compute_degree_correlation(matrix)

        if perm_id == sample_perm:
            sample_source_deg = src_deg

        records.append(
            {
                "perm_id": perm_id,
                "group": assign_group(perm_id),
                "degree_correlation": corr,
                "mean_source_degree": float(np.mean(src_deg)),
                "mean_target_degree": float(np.mean(np.asarray(matrix.sum(axis=0)).ravel())),
            }
        )

    degree_df = pd.DataFrame(records).sort_values("perm_id").reset_index(drop=True)

    corr_csv = args.results_dir / f"{args.edge_type}_degree_correlation.csv"
    degree_df.to_csv(corr_csv, index=False)

    preservation_csv = args.results_dir / f"{args.edge_type}_source_degree_preservation.csv"
    if sample_source_deg is None:
        sample_source_deg = ref_source_deg
    pd.DataFrame(
        {
            "reference_source_degree": ref_source_deg,
            f"perm_{sample_perm:03d}_source_degree": sample_source_deg,
        }
    ).to_csv(preservation_csv, index=False)

    fig_path = args.results_dir / f"{args.edge_type}_permutation_similarity.png"
    if not args.skip_plots:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left panel: degree preservation scatter
        ax1 = axes[0]
        n_points = len(ref_source_deg)
        if n_points > args.max_scatter_points:
            rng = np.random.default_rng(args.random_seed)
            idx = rng.choice(n_points, size=args.max_scatter_points, replace=False)
            x_vals = ref_source_deg[idx]
            y_vals = sample_source_deg[idx]
        else:
            x_vals = ref_source_deg
            y_vals = sample_source_deg

        ax1.scatter(x_vals, y_vals, alpha=0.6, s=12, label=f"Perm {sample_perm:03d}")
        max_deg = max(float(np.max(x_vals)), float(np.max(y_vals)))
        ax1.plot([0, max_deg], [0, max_deg], "r--", linewidth=2, label="Perfect preservation (y=x)")
        ax1.set_xlabel(f"{ref_label} Source Degree")
        ax1.set_ylabel(f"Perm {sample_perm:03d} Source Degree")
        ax1.set_title("Degree Preservation: Source Nodes", fontweight="bold")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Right panel: degree correlation across permutations
        ax2 = axes[1]
        color_map = {
            "Hetionet": "red",
            "Training (001-020)": "blue",
            "Validation (021-030)": "green",
            "Remaining (031+)": "orange",
        }
        colors = [color_map.get(g, "gray") for g in degree_df["group"]]

        ax2.scatter(
            degree_df["perm_id"],
            degree_df["degree_correlation"],
            c=colors,
            alpha=0.75,
            s=35,
            edgecolors="black",
            linewidth=0.4,
        )
        ax2.plot(degree_df["perm_id"], degree_df["degree_correlation"], "k-", alpha=0.3, linewidth=1)

        if np.isfinite(ref_corr):
            ax2.axhline(ref_corr, color="red", linestyle="--", linewidth=2, alpha=0.7,
                        label=f"{ref_label} (r={ref_corr:.3f})")

        ax2.set_xlabel("Permutation ID")
        ax2.set_ylabel("Degree Correlation")
        ax2.set_title("Degree Correlation Across Permutations", fontweight="bold")
        ax2.grid(alpha=0.3)
        ax2.legend()

        fig.suptitle(f"Permutation Similarity Analysis: {args.edge_type}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        ensure_file_exists(fig_path)
        print(f"Saved figure: {fig_path}")

    write_run_summary(
        args.results_dir / "repro_fig4_permutation_similarity_run_summary.json",
        args,
        extra={
            "output_file": fig_path,
            "degree_correlation_csv": corr_csv,
            "source_degree_preservation_csv": preservation_csv,
            "n_loaded_permutations": int(len(degree_df)),
            "permutation_ids": [int(p) for p in degree_df["perm_id"].tolist()],
            "analysis_permutation_ids": [int(p) for p in analysis_perm_ids],
            "reference_label": ref_label,
            "reference_degree_correlation": ref_corr,
        },
    )

    print(f"Saved degree-correlation CSV: {corr_csv}")
    print(f"Saved source-degree CSV: {preservation_csv}")
    if args.skip_plots:
        print("Skipped figure output because --skip-plots was set.")
    print(
        "Saved run summary: "
        f"{args.results_dir / 'repro_fig4_permutation_similarity_run_summary.json'}"
    )


if __name__ == "__main__":
    main()
