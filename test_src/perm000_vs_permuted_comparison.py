#!/usr/bin/env python3
"""
Perm000 vs permuted comparison.

Tests correlation between a reference permutation (default 000) and the
average of other permutations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
XDG_CACHE_DIR = REPO_DIR / ".cache"
MPL_CACHE_DIR = XDG_CACHE_DIR / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(REPO_DIR / "src"))

from permutation_validation import (  # noqa: E402
    analyze_residuals,
    extract_pathway_bins_from_permutations,
    extract_pathway_bins_from_single_permutation,
    test_original_vs_permutation_correlation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare perm000 (or chosen reference) against averaged permutations.",
    )
    parser.add_argument("--metapath", type=str, default="CbGpPW")
    parser.add_argument("--edge1-type", type=str, default="CbG")
    parser.add_argument("--edge2-type", type=str, default="GpPW")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--reference-perm", type=int, default=0)
    parser.add_argument(
        "--perm-ids",
        type=int,
        nargs="+",
        default=None,
        help="Permutation IDs to average against reference (defaults to 1..20 filtered by availability).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Repository data directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "perm000_validation",
        help="Output directory.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def list_available_permutation_ids(data_dir: Path) -> list[int]:
    perm_dir = data_dir / "permutations"
    if not perm_dir.exists():
        return []
    perm_ids = []
    for child in perm_dir.glob("*.hetmat"):
        try:
            perm_ids.append(int(child.stem))
        except ValueError:
            continue
    return sorted(set(perm_ids))


def resolve_comparison_perm_ids(
    available: list[int],
    reference_perm: int,
    requested: list[int] | None,
    smoke: bool,
) -> list[int]:
    if requested:
        missing = [perm for perm in requested if perm not in available]
        if missing:
            raise FileNotFoundError(f"Requested permutations missing: {sorted(set(missing))}")
        perm_ids = [perm for perm in requested if perm != reference_perm]
    else:
        default_ids = list(range(1, 21))
        perm_ids = [perm for perm in default_ids if perm in available and perm != reference_perm]
        if not perm_ids:
            perm_ids = [perm for perm in available if perm != reference_perm]

    if smoke:
        perm_ids = perm_ids[: min(3, len(perm_ids))]

    if not perm_ids:
        raise ValueError("No comparison permutations available after filtering.")
    return perm_ids


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    available = list_available_permutation_ids(data_dir)
    if args.reference_perm not in available:
        raise FileNotFoundError(
            f"Reference permutation {args.reference_perm:03d} not found in {data_dir / 'permutations'}"
        )
    perm_ids = resolve_comparison_perm_ids(
        available=available,
        reference_perm=args.reference_perm,
        requested=args.perm_ids,
        smoke=args.smoke,
    )

    print("=" * 80)
    print("Perm000 vs Permuted Comparison")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Reference permutation: {args.reference_perm:03d}")
    print(f"Comparison permutations: {perm_ids}")
    print(f"Results directory: {results_dir}")
    print()

    print("Step 1: Extracting pathway bins from reference permutation...")
    ref_bins = extract_pathway_bins_from_single_permutation(
        args.edge1_type,
        args.edge2_type,
        args.reference_perm,
        data_dir,
        args.n_bins,
    )
    print(f"  Extracted {len(ref_bins)} bins from reference permutation")
    print(f"  Mean pathway count: {ref_bins['pathway_count'].mean():.4f}")
    print(f"  Std pathway count: {ref_bins['pathway_count'].std():.4f}")
    print()

    print("Step 2: Extracting average pathway bins from comparison permutations...")
    perm_avg_bins = extract_pathway_bins_from_permutations(
        args.edge1_type,
        args.edge2_type,
        perm_ids,
        data_dir,
        args.n_bins,
    )
    print(f"  Averaged over {len(perm_ids)} permutations")
    print(f"  Extracted {len(perm_avg_bins)} bins")
    print(f"  Mean pathway count: {perm_avg_bins['mean_pathway_count'].mean():.4f}")
    print(f"  Std across bins: {perm_avg_bins['mean_pathway_count'].std():.4f}")
    print()

    print("Step 3: Computing correlation...")
    correlation_results = test_original_vs_permutation_correlation(ref_bins, perm_avg_bins)
    print(f"  Correlation (r): {correlation_results['correlation']:.4f}")
    print(f"  P-value: {correlation_results['p_value']:.2e}")
    print(f"  RMSE: {correlation_results['rmse']:.4f}")
    print(f"  MAE: {correlation_results['mae']:.4f}")
    print(f"  Hypothesis validated (r > 0.85): {correlation_results['hypothesis_valid']}")
    print()

    print("Step 4: Analyzing residuals...")
    residual_results = analyze_residuals(ref_bins, perm_avg_bins)
    stats = residual_results["statistics"]
    print(f"  Mean residual: {stats['mean_residual']:.4f}")
    print(f"  Std residual: {stats['std_residual']:.4f}")
    print(f"  Mean absolute residual: {stats['mean_abs_residual']:.4f}")
    print(f"  Median absolute residual: {stats['median_abs_residual']:.4f}")
    print(f"  Max absolute residual: {stats['max_abs_residual']:.4f}")
    print(f"  95th percentile absolute residual: {stats['q95_abs_residual']:.4f}")
    print()

    print("Step 5: Saving results...")
    summary = {
        "metapath": args.metapath,
        "edge1_type": args.edge1_type,
        "edge2_type": args.edge2_type,
        "reference_perm": int(args.reference_perm),
        "comparison_perms": [int(perm_id) for perm_id in perm_ids],
        "n_bins": int(args.n_bins),
        "correlation": float(correlation_results["correlation"]),
        "p_value": float(correlation_results["p_value"]),
        "rmse": float(correlation_results["rmse"]),
        "mae": float(correlation_results["mae"]),
        "hypothesis_valid": bool(correlation_results["hypothesis_valid"]),
        "residual_stats": {k: float(v) for k, v in stats.items()},
    }

    summary_file = results_dir / f"{args.metapath}_perm000_validation_summary.json"
    with open(summary_file, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  Summary saved: {summary_file}")

    merged_data = correlation_results["merged_data"]
    merged_file = results_dir / f"{args.metapath}_perm000_vs_perms.csv"
    merged_data.to_csv(merged_file, index=False)
    print(f"  Merged data saved: {merged_file}")

    residual_data = residual_results["merged_data"]
    residual_file = results_dir / f"{args.metapath}_residual_analysis.csv"
    residual_data.to_csv(residual_file, index=False)
    print(f"  Residual analysis saved: {residual_file}")

    if not args.skip_plots:
        print("Step 6: Creating visualization...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        ref_vals = merged_data["pathway_count"].values
        avg_vals = merged_data["mean_pathway_count"].values
        ax1.scatter(ref_vals, avg_vals, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)
        min_val = min(ref_vals.min(), avg_vals.min())
        max_val = max(ref_vals.max(), avg_vals.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect agreement")
        ax1.set_xlabel(f"Perm {args.reference_perm:03d} Pathway Count", fontsize=11)
        ax1.set_ylabel("Average Comparison Pathway Count", fontsize=11)
        ax1.set_title(
            f"{args.metapath}: Ref vs Avg\n"
            f"r = {correlation_results['correlation']:.4f}, RMSE = {correlation_results['rmse']:.3f}",
            fontsize=12,
            fontweight="bold",
        )
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2 = axes[1]
        residuals = residual_data["residual"].values
        ax2.scatter(avg_vals, residuals, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)
        ax2.axhline(0, color="r", linestyle="--", linewidth=2)
        ax2.set_xlabel("Average Comparison Pathway Count", fontsize=11)
        ax2.set_ylabel("Residual (Reference - Avg)", fontsize=11)
        ax2.set_title(
            f"Residual Analysis\n"
            f"MAE = {correlation_results['mae']:.3f}, "
            f"Max Error = {stats['max_abs_residual']:.3f}",
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plot_file = results_dir / f"{args.metapath}_perm000_vs_perms.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Visualization saved: {plot_file}")
    print()

    print("=" * 80)
    print("PHASE 0b RESULTS")
    print("=" * 80)
    print(f"Correlation: r = {correlation_results['correlation']:.4f}")
    print(f"Hypothesis validated (r > 0.85): {correlation_results['hypothesis_valid']}")
    if correlation_results["hypothesis_valid"]:
        print("SUCCESS: Reference permutation is highly correlated with permutation average.")
    else:
        print("FAILED: Reference permutation correlation with permutation average is weak.")
    print("=" * 80)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
