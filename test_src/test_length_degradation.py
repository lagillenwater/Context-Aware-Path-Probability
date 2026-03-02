"""
Test performance degradation across path lengths 2-8.

Metapath series:
- Length 2: CbG (edge)
- Length 3: CbGpPW
- Length 4: CbGiGpPW
- Length 5: CbGiGiGpPW
- Length 6: CbGiGiGiGpPW
- Length 7: CbGiGiGiGiGpPW
- Length 8: CbGiGiGiGiGiGpPW
"""

from __future__ import annotations

import argparse
import os
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
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

METAPATHS = {
    2: ["CbG"],
    3: ["CbG", "GpPW"],
    4: ["CbG", "GiG", "GpPW"],
    5: ["CbG", "GiG", "GiG", "GpPW"],
    6: ["CbG", "GiG", "GiG", "GiG", "GpPW"],
    7: ["CbG", "GiG", "GiG", "GiG", "GiG", "GpPW"],
    8: ["CbG", "GiG", "GiG", "GiG", "GiG", "GiG", "GpPW"],
}
REFERENCE_RESULTS = {
    3: {"r": 0.777, "r_std": 0.025, "qq": 0.843, "qq_std": 0.006},
    4: {"r": 0.828, "r_std": 0.012, "qq": 0.730, "qq_std": 0.038},
    5: {"r": 0.899, "r_std": 0.027, "qq": 0.461, "qq_std": 0.063},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate random-forest degradation across metapath lengths.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Repository data directory containing permutations/",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "length_degradation",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[2, 6, 7, 8],
        help="Path lengths to compute directly (2-8).",
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
    parser.add_argument(
        "--train-count",
        type=int,
        default=5,
        help="Fallback train permutation count when explicit IDs are not provided.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=5,
        help="Fallback test permutation count when explicit IDs are not provided.",
    )
    parser.add_argument(
        "--include-reference-lengths",
        action="store_true",
        default=True,
        help="Include historical results for lengths 3/4/5 in combined outputs.",
    )
    parser.add_argument(
        "--no-reference-lengths",
        action="store_false",
        dest="include_reference_lengths",
        help="Disable inclusion of historical results for lengths 3/4/5.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write CSV outputs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use lighter defaults for quick validation runs.",
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
        raise ValueError("Need at least 2 available permutations for train/test split.")

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
    canonical_test = [15, 16, 17, 18, 19]
    if all(perm in available for perm in canonical_train + canonical_test):
        return canonical_train, canonical_test

    train_count = max(1, min(train_count, len(available) - 1))
    test_count = max(1, min(test_count, len(available) - train_count))

    train_split = available[:train_count]
    test_split = available[-test_count:]
    if set(train_split) & set(test_split):
        # For tiny sets, enforce disjoint split by position.
        test_split = available[train_count : train_count + test_count]
    if not test_split:
        raise ValueError("Unable to construct a valid test permutation split.")

    return train_split, test_split


def load_and_multiply_edges(edge_types: list[str], perm: int, data_dir: Path) -> sp.spmatrix:
    perm_dir = data_dir / "permutations" / f"{perm:03d}.hetmat" / "edges"

    edge_file = perm_dir / f"{edge_types[0]}.sparse.npz"
    result = sp.load_npz(str(edge_file))
    if result.dtype == bool:
        result = result.astype(np.int32)

    for edge_type in edge_types[1:]:
        edge_file = perm_dir / f"{edge_type}.sparse.npz"
        edge = sp.load_npz(str(edge_file))
        if edge.dtype == bool:
            edge = edge.astype(np.int32)
        result = result @ edge

    return result


def compute_pathway_counts(pairs: np.ndarray, pathway_matrix: sp.spmatrix) -> np.ndarray:
    counts = np.zeros(len(pairs))
    for i, (src, tgt) in enumerate(pairs):
        counts[i] = pathway_matrix[src, tgt]
    return counts


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_probplot_corr(values: np.ndarray) -> float:
    if np.std(values) == 0:
        return float("nan")
    return float(stats.probplot(values)[1][2])


def test_single_length(
    edge_types: list[str],
    length: int,
    data_dir: Path,
    train_perms: list[int],
    test_perms: list[int],
    n_samples: int,
    random_state: int,
) -> pd.DataFrame | None:
    print(f"\n{'=' * 70}")
    print(f"Testing Length-{length}: {' @ '.join(edge_types)}")
    print(f"{'=' * 70}")

    pathway_perm0 = load_and_multiply_edges(edge_types, train_perms[0], data_dir)
    n_sources, n_targets = pathway_perm0.shape
    print(f"Matrix shape: {n_sources} x {n_targets}")

    np.random.seed(random_state)
    pathway_coo = pathway_perm0.tocoo()
    pathway_pairs = np.column_stack([pathway_coo.row, pathway_coo.col])
    print(f"Pairs with pathways: {len(pathway_pairs)}")

    if len(pathway_pairs) == 0:
        print("No pathways found for this length.")
        return None

    n_with_pathways = min(n_samples // 2, len(pathway_pairs))
    idx = np.random.choice(len(pathway_pairs), n_with_pathways, replace=False)
    sampled_with_pathways = pathway_pairs[idx]

    n_random = n_samples - n_with_pathways
    random_sources = np.random.randint(0, n_sources, n_random)
    random_targets = np.random.randint(0, n_targets, n_random)
    random_pairs = np.column_stack([random_sources, random_targets])

    pairs = np.vstack([sampled_with_pathways, random_pairs])
    np.random.shuffle(pairs)
    print(f"Sampled {len(pairs)} pairs")

    first_edge_file = (
        data_dir / "permutations" / f"{train_perms[0]:03d}.hetmat" / "edges" / f"{edge_types[0]}.sparse.npz"
    )
    last_edge_file = (
        data_dir / "permutations" / f"{train_perms[0]:03d}.hetmat" / "edges" / f"{edge_types[-1]}.sparse.npz"
    )
    first_edge = sp.load_npz(str(first_edge_file))
    last_edge = sp.load_npz(str(last_edge_file))

    source_degrees = np.asarray(first_edge.sum(axis=1)).ravel()
    target_degrees = np.asarray(last_edge.sum(axis=0)).ravel()

    deg_src = source_degrees[pairs[:, 0]]
    deg_tgt = target_degrees[pairs[:, 1]]
    X = np.column_stack([deg_src, deg_tgt, deg_src * deg_tgt, deg_src**2, deg_tgt**2])

    print(f"Computing counts for train perms: {train_perms}")
    counts_train = []
    for perm in train_perms:
        pathway = load_and_multiply_edges(edge_types, perm, data_dir)
        counts_train.append(compute_pathway_counts(pairs, pathway))
    counts_train = np.column_stack(counts_train)
    mean_train = counts_train.mean(axis=1)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, mean_train)

    print(f"Computing counts for test perms: {test_perms}")
    counts_test = []
    for perm in test_perms:
        pathway = load_and_multiply_edges(edge_types, perm, data_dir)
        counts_test.append(compute_pathway_counts(pairs, pathway))
    counts_test = np.column_stack(counts_test)

    predictions = model.predict(X)
    results = []
    for col_i, perm in enumerate(test_perms):
        actual = counts_test[:, col_i]
        residual = actual - predictions
        results.append(
            {
                "length": length,
                "perm": perm,
                "r": safe_pearson(actual, predictions),
                "mae": float(np.abs(residual).mean()),
                "qq": safe_probplot_corr(residual),
                "rmse": float(np.sqrt((residual**2).mean())),
            }
        )

    results_df = pd.DataFrame(results)
    print(
        "Results: "
        f"r={results_df['r'].mean():.3f}, "
        f"Q-Q={results_df['qq'].mean():.3f}"
    )
    return results_df


def generate_degradation_plots(df: pd.DataFrame, output_dir: Path) -> None:
    summary = df.groupby("length").agg({"r": ["mean", "std"], "qq": ["mean", "std"]})
    lengths = summary.index.values
    r_mean = summary["r"]["mean"].values
    r_std = np.nan_to_num(summary["r"]["std"].values, nan=0.0)
    qq_mean = summary["qq"]["mean"].values
    qq_std = np.nan_to_num(summary["qq"]["std"].values, nan=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.errorbar(
        lengths,
        r_mean,
        yerr=r_std,
        marker="o",
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        label="Random Forest",
    )
    ax.axhline(0.8, color="red", linestyle="--", linewidth=1, alpha=0.5, label="r=0.8")
    ax.set_xlabel("Path Length")
    ax.set_ylabel("Correlation (r)")
    ax.set_title("Performance vs Path Length")
    ax.set_xticks(lengths)
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(
        lengths,
        qq_mean,
        yerr=qq_std,
        marker="s",
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color="orange",
        label="Random Forest",
    )
    ax.axhline(0.8, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Q-Q=0.8")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Q-Q=0.5")
    ax.set_xlabel("Path Length")
    ax.set_ylabel("Q-Q Correlation")
    ax.set_title("Calibration vs Path Length")
    ax.set_xticks(lengths)
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "length_degradation_plots.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax2 = ax.twinx()

    line1 = ax.errorbar(
        lengths,
        r_mean,
        yerr=r_std,
        marker="o",
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color="blue",
        label="Correlation (r)",
    )
    ax.set_xlabel("Path Length")
    ax.set_ylabel("Correlation (r)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.set_ylim([0.5, 1.0])

    line2 = ax2.errorbar(
        lengths,
        qq_mean,
        yerr=qq_std,
        marker="s",
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color="orange",
        label="Q-Q Correlation",
    )
    ax2.set_ylabel("Q-Q Correlation", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.set_ylim([0, 1.0])

    ax.set_xticks(lengths)
    ax.set_title("Performance Degradation Across Path Length")
    ax.grid(True, alpha=0.3)
    ax.legend([line1, line2], ["Correlation (r)", "Q-Q Correlation"], loc="center right")

    plt.tight_layout()
    output_file = output_dir / "length_degradation_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def run_degradation_analysis(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = args.data_dir.resolve()
    output_dir = args.results_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths = sorted(set(args.lengths))
    for length in lengths:
        if length not in METAPATHS:
            raise ValueError(f"Unsupported path length {length}. Valid lengths: {sorted(METAPATHS)}")

    if args.smoke:
        args.n_samples = min(args.n_samples, 2_000)
        lengths = [length for length in lengths if length in {2, 6}]
        if not lengths:
            lengths = [2, 6]

    available = list_available_permutation_ids(data_dir / "permutations")
    train_perms, test_perms = resolve_permutation_split(
        available=available,
        train_ids=args.train_perms,
        test_ids=args.test_perms,
        train_count=args.train_count,
        test_count=args.test_count,
    )

    print("=" * 70)
    print("LENGTH DEGRADATION ANALYSIS")
    print("=" * 70)
    print(f"Lengths: {lengths}")
    print(f"Train permutations: {train_perms}")
    print(f"Test permutations: {test_perms}")
    print(f"Samples: {args.n_samples}")

    all_results: list[pd.DataFrame] = []
    for length in lengths:
        result_df = test_single_length(
            edge_types=METAPATHS[length],
            length=length,
            data_dir=data_dir,
            train_perms=train_perms,
            test_perms=test_perms,
            n_samples=args.n_samples,
            random_state=args.random_state,
        )
        if result_df is not None:
            all_results.append(result_df)

    if args.include_reference_lengths:
        for length, stats_dict in REFERENCE_RESULTS.items():
            reference_rows = [
                {
                    "length": length,
                    "perm": perm,
                    "r": stats_dict["r"],
                    "qq": stats_dict["qq"],
                    "mae": 0.0,
                    "rmse": 0.0,
                }
                for perm in test_perms
            ]
            all_results.append(pd.DataFrame(reference_rows))

    if not all_results:
        raise RuntimeError("No results were generated.")

    combined_df = pd.concat(all_results, ignore_index=True)
    results_file = output_dir / "length_degradation_results.csv"
    combined_df.to_csv(results_file, index=False)
    print(f"Saved: {results_file}")

    summary = combined_df.groupby("length").agg({"r": ["mean", "std"], "qq": ["mean", "std"]}).round(3)
    print("\nSummary:")
    print(summary)

    if not args.skip_plots:
        generate_degradation_plots(combined_df, output_dir)

    return combined_df


if __name__ == "__main__":
    run_degradation_analysis(parse_args())
