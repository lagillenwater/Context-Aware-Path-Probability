"""
Test whether high count outliers are topology-specific across permutations.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
XDG_CACHE_DIR = REPO_DIR / ".cache"
MPL_CACHE_DIR = XDG_CACHE_DIR / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(REPO_DIR))

from test_src.validate_mean_variance_prediction import (  # noqa: E402
    compute_pathway_counts,
    extract_degree_features,
    load_permuted_edge_matrices,
    sample_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze topology-specific high-count outliers across permutations.",
    )
    parser.add_argument("--edge1-type", type=str, default="CbG")
    parser.add_argument("--edge2-type", type=str, default="GpPW")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Repository data directory containing permutations/.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "topology_specific_outliers",
        help="Output directory for figures and summaries.",
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--train-perms",
        type=int,
        nargs="+",
        default=None,
        help="Explicit training permutation IDs.",
    )
    parser.add_argument(
        "--test-perms",
        type=int,
        nargs="+",
        default=None,
        help="Explicit test permutation IDs.",
    )
    parser.add_argument("--train-count", type=int, default=5)
    parser.add_argument("--test-count", type=int, default=6)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smaller sample sizes for quick validation.",
    )
    return parser.parse_args()


def list_available_permutation_ids(perm_dir: Path) -> list[int]:
    if not perm_dir.exists():
        return []
    perm_ids = []
    for child in perm_dir.glob("*.hetmat"):
        try:
            perm_ids.append(int(child.stem))
        except ValueError:
            continue
    return sorted(set(perm_ids))


def resolve_permutation_split(
    available: list[int],
    train_ids: list[int] | None,
    test_ids: list[int] | None,
    train_count: int,
    test_count: int,
) -> tuple[list[int], list[int]]:
    if len(available) < 2:
        raise ValueError("Need at least 2 permutations for train/test split.")

    if train_ids or test_ids:
        if not (train_ids and test_ids):
            raise ValueError("Provide both --train-perms and --test-perms, or neither.")
        missing = [perm for perm in [*train_ids, *test_ids] if perm not in available]
        if missing:
            raise FileNotFoundError(f"Requested permutations missing: {sorted(set(missing))}")
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            raise ValueError(f"Train/test permutation overlap is not allowed: {sorted(overlap)}")
        return train_ids, test_ids

    canonical_train = [0, 1, 2, 3, 4]
    canonical_test = [15, 16, 17, 18, 19, 20]
    if all(perm in available for perm in canonical_train + canonical_test):
        return canonical_train, canonical_test

    train_count = max(1, min(train_count, len(available) - 1))
    test_count = max(1, min(test_count, len(available) - train_count))
    train_split = available[:train_count]
    test_split = available[-test_count:]

    if set(train_split) & set(test_split):
        test_split = available[train_count : train_count + test_count]
    if not test_split:
        raise ValueError("Unable to construct a valid test split.")

    return train_split, test_split


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def analyze_topology_specific_outliers(
    edge1_type: str,
    edge2_type: str,
    data_dir: Path,
    output_dir: Path,
    train_perms: list[int],
    test_perms: list[int],
    n_samples: int,
    random_state: int,
    skip_plots: bool,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Topology-Specific Outlier Analysis")
    print("=" * 70)
    print(f"Metapath edges: {edge1_type} + {edge2_type}")
    print(f"Train permutations: {train_perms}")
    print(f"Test permutations: {test_perms}")
    print(f"n_samples: {n_samples}")

    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(
        edge1_type,
        edge2_type,
        train_perms[0],
        data_dir,
    )
    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples=n_samples, random_state=random_state)
    x_features = extract_degree_features(pairs, edge1_perm0, edge2_perm0)
    print(f"Sampled {len(pairs)} pairs")

    counts_train = []
    print("\nComputing counts for training permutations...")
    for perm in train_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_train.append(counts)
        print(f"  Perm {perm:03d}: mean={counts.mean():.2f}, max={counts.max():.0f}")
    counts_train = np.column_stack(counts_train)
    mean_train = counts_train.mean(axis=1)

    counts_test = []
    print("\nComputing counts for test permutations...")
    for perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)
        print(f"  Perm {perm:03d}: mean={counts.mean():.2f}, max={counts.max():.0f}")
    counts_test = np.column_stack(counts_test)

    threshold = float(np.percentile(mean_train, 99))
    high_in_mean = mean_train > threshold
    high_in_perms = counts_train > threshold
    any_high_in_train = high_in_perms.any(axis=1)
    topology_specific = any_high_in_train & ~high_in_mean
    never_high = ~any_high_in_train
    high_mean_pairs = np.where(high_in_mean)[0]

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"99th percentile threshold (mean train): {threshold:.2f}")
    print(f"Pairs high in mean: {high_in_mean.sum()} ({100 * high_in_mean.mean():.2f}%)")
    print(f"Pairs high in >=1 train perm: {any_high_in_train.sum()} ({100 * any_high_in_train.mean():.2f}%)")
    print(
        f"Topology-specific outliers: {topology_specific.sum()} "
        f"({100 * topology_specific.mean():.2f}%)"
    )

    consistency_counts = []
    for pair_idx in high_mean_pairs:
        consistency_counts.append((counts_train[pair_idx, :] > threshold).sum())
    consistency_counts = np.asarray(consistency_counts)

    test_consistency = []
    for pair_idx in high_mean_pairs:
        test_consistency.append((counts_test[pair_idx, :] > threshold).sum())
    test_consistency = np.asarray(test_consistency)

    var_train = counts_train.var(axis=1)
    deg_product = x_features[:, 2]

    print("\nVariance across training permutations:")
    print(f"  overall: {safe_mean(var_train):.3f}")
    print(f"  consistent-high: {safe_mean(var_train[high_in_mean]):.3f}")
    print(f"  topology-specific: {safe_mean(var_train[topology_specific]):.3f}")
    print(f"  never-high: {safe_mean(var_train[never_high]):.3f}")

    if high_mean_pairs.size > 0:
        mean_fraction = safe_mean(test_consistency / max(1, len(test_perms)))
        print(f"Mean fraction of test perms with high count for high-mean pairs: {mean_fraction:.3f}")

    results = {
        "mean_train": mean_train,
        "counts_train": counts_train,
        "counts_test": counts_test,
        "high_in_mean": high_in_mean,
        "topology_specific": topology_specific,
        "deg_product": deg_product,
        "var_train": var_train,
        "consistency_counts": consistency_counts,
        "train_perms": np.asarray(train_perms),
        "test_perms": np.asarray(test_perms),
    }

    summary_rows = pd_summary_rows(results)
    np.savetxt(
        output_dir / "topology_specific_outlier_summary.tsv",
        summary_rows,
        delimiter="\t",
        fmt="%s",
    )

    if not skip_plots:
        create_plots(results, output_dir)

    return results


def pd_summary_rows(results: dict[str, np.ndarray]) -> np.ndarray:
    mean_train = results["mean_train"]
    high_in_mean = results["high_in_mean"]
    topology_specific = results["topology_specific"]
    var_train = results["var_train"]
    deg_product = results["deg_product"]

    rows = [
        ("metric", "value"),
        ("pairs_total", str(len(mean_train))),
        ("high_in_mean_count", str(int(high_in_mean.sum()))),
        ("high_in_mean_fraction", f"{float(high_in_mean.mean()):.6f}"),
        ("topology_specific_count", str(int(topology_specific.sum()))),
        ("topology_specific_fraction", f"{float(topology_specific.mean()):.6f}"),
        ("mean_variance_overall", f"{safe_mean(var_train):.6f}"),
        ("mean_variance_high_in_mean", f"{safe_mean(var_train[high_in_mean]):.6f}"),
        ("mean_variance_topology_specific", f"{safe_mean(var_train[topology_specific]):.6f}"),
        ("mean_deg_product_high_in_mean", f"{safe_mean(deg_product[high_in_mean]):.6f}"),
        ("mean_deg_product_topology_specific", f"{safe_mean(deg_product[topology_specific]):.6f}"),
    ]
    return np.asarray(rows, dtype=str)


def create_plots(results: dict[str, np.ndarray], output_dir: Path) -> None:
    mean_train = results["mean_train"]
    counts_train = results["counts_train"]
    high_in_mean = results["high_in_mean"]
    topology_specific = results["topology_specific"]
    deg_product = results["deg_product"]
    var_train = results["var_train"]
    consistency_counts = results["consistency_counts"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.hist(mean_train, bins=50, alpha=0.6, label="Mean train counts", density=True)
    ax.hist(counts_train[:, 0], bins=50, alpha=0.5, label="Train perm 0", density=True)
    ax.set_xlabel("Pathway Count")
    ax.set_ylabel("Density")
    ax.set_title("Mean vs Single-Permutation Count Distribution")
    ax.legend()
    ax.set_yscale("log")

    ax = axes[0, 1]
    ax.scatter(mean_train + 1, var_train + 1, alpha=0.1, s=2)
    ax.set_xlabel("Mean Count (+1)")
    ax.set_ylabel("Variance Across Train Perms (+1)")
    ax.set_title("Mean-Variance Relationship")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[1, 0]
    ax.scatter(
        deg_product[~high_in_mean] + 1,
        var_train[~high_in_mean] + 1,
        alpha=0.1,
        s=2,
        label="Normal",
        c="blue",
    )
    ax.scatter(
        deg_product[high_in_mean] + 1,
        var_train[high_in_mean] + 1,
        alpha=0.6,
        s=10,
        label="Consistent high",
        c="red",
    )
    ax.scatter(
        deg_product[topology_specific] + 1,
        var_train[topology_specific] + 1,
        alpha=0.6,
        s=10,
        label="Topology-specific",
        c="orange",
    )
    ax.set_xlabel("Degree Product (+1)")
    ax.set_ylabel("Variance Across Train Perms (+1)")
    ax.set_title("Variance by Degree and Outlier Type")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1, 1]
    if consistency_counts.size == 0:
        ax.text(0.5, 0.5, "No high-mean pairs", ha="center", va="center")
        ax.axis("off")
    else:
        bins = np.arange(consistency_counts.max() + 2) - 0.5
        ax.hist(consistency_counts, bins=bins, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Number of train perms with high count")
        ax.set_ylabel("Number of pairs")
        ax.set_title("Consistency of High Counts Across Train Perms")
        ax.set_xticks(range(int(consistency_counts.max()) + 1))

    plt.tight_layout()
    output_file = output_dir / "topology_specific_outliers.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_samples = min(args.n_samples, 2_000)

    available = list_available_permutation_ids(args.data_dir / "permutations")
    train_perms, test_perms = resolve_permutation_split(
        available=available,
        train_ids=args.train_perms,
        test_ids=args.test_perms,
        train_count=args.train_count,
        test_count=args.test_count,
    )

    results = analyze_topology_specific_outliers(
        edge1_type=args.edge1_type,
        edge2_type=args.edge2_type,
        data_dir=args.data_dir.resolve(),
        output_dir=args.results_dir.resolve(),
        train_perms=train_perms,
        test_perms=test_perms,
        n_samples=args.n_samples,
        random_state=args.random_state,
        skip_plots=args.skip_plots,
    )

    topology_frac = float(results["topology_specific"].mean())
    high_frac = float(results["high_in_mean"].mean())
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"Fraction high-in-mean pairs: {high_frac:.4f}")
    print(f"Fraction topology-specific pairs: {topology_frac:.4f}")
    if topology_frac > 0.01:
        print("Topology-specific outliers are substantial.")
    else:
        print("Topology-specific outliers are limited in this run.")


if __name__ == "__main__":
    main()
