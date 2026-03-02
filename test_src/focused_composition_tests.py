#!/usr/bin/env python3
"""
Focused composition tests.

Learns to weight composition using endpoint degrees and sparsity terms.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict
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
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.append(str(REPO_DIR / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused composition tests.",
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
        default=REPO_DIR / "results" / "hierarchical_prediction",
        help="Output directory.",
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--empirical-perms",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="Permutations used to estimate empirical GpPW edge frequency.",
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


def resolve_empirical_perms(available: list[int], requested: list[int]) -> list[int]:
    if not available:
        raise FileNotFoundError("No local permutations found in data/permutations.")
    chosen = [perm for perm in requested if perm in available]
    if chosen:
        missing = [perm for perm in requested if perm not in available]
        if missing:
            print(f"Skipping unavailable requested perms: {missing}")
        return chosen

    fallback = [perm for perm in available if perm > 0]
    if not fallback:
        fallback = available
    return fallback[: min(4, len(fallback))]


def load_edge_matrix(data_dir: Path, edge_abbrev: str, perm_num: int | str = "original") -> sp.spmatrix:
    if perm_num == "original":
        edge_file = data_dir / "edges" / f"{edge_abbrev}.sparse.npz"
    else:
        edge_file = data_dir / "permutations" / f"{int(perm_num):03d}.hetmat" / "edges" / f"{edge_abbrev}.sparse.npz"
    if not edge_file.exists():
        raise FileNotFoundError(f"Missing edge file: {edge_file}")
    return sp.load_npz(str(edge_file)).astype(np.int32)


def sample_pairs_stratified(
    pathway_matrix: sp.spmatrix,
    n_samples: int,
    random_state: int,
) -> list[tuple[int, int]]:
    np.random.seed(random_state)
    sources_nonzero, targets_nonzero = pathway_matrix.nonzero()
    n_nonzero = len(sources_nonzero)
    n_nonzero_sample = min(int(n_samples * 0.5), n_nonzero)

    sampled_sources: list[int]
    sampled_targets: list[int]
    if n_nonzero_sample > 0:
        idx_nonzero = np.random.choice(n_nonzero, n_nonzero_sample, replace=False)
        sampled_sources = list(sources_nonzero[idx_nonzero])
        sampled_targets = list(targets_nonzero[idx_nonzero])
    else:
        sampled_sources = []
        sampled_targets = []

    n_random = n_samples - len(sampled_sources)
    random_sources = np.random.randint(0, pathway_matrix.shape[0], n_random)
    random_targets = np.random.randint(0, pathway_matrix.shape[1], n_random)

    sampled_sources.extend(random_sources.tolist())
    sampled_targets.extend(random_targets.tolist())
    return list(zip(sampled_sources, sampled_targets))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def compute_empirical_frequency(
    data_dir: Path,
    empirical_perms: list[int],
    gene_degrees: np.ndarray,
    pathway_degrees: np.ndarray,
) -> dict[tuple[int, int], float]:
    edge_counts_by_degree: dict[tuple[int, int], float] = defaultdict(float)
    total_counts_by_degree: dict[tuple[int, int], float] = defaultdict(float)

    gene_unique, gene_counts = np.unique(gene_degrees.astype(int), return_counts=True)
    pathway_unique, pathway_counts = np.unique(pathway_degrees.astype(int), return_counts=True)
    for gene_deg, gene_count in zip(gene_unique, gene_counts):
        for pathway_deg, pathway_count in zip(pathway_unique, pathway_counts):
            total_counts_by_degree[(int(gene_deg), int(pathway_deg))] = float(
                gene_count * pathway_count * len(empirical_perms)
            )

    for perm_num in empirical_perms:
        gp_pw_perm = load_edge_matrix(data_dir, "GpPW", perm_num=perm_num).tocoo()
        deg_gene = gene_degrees[gp_pw_perm.row].astype(int)
        deg_pathway = pathway_degrees[gp_pw_perm.col].astype(int)
        degree_pairs = np.column_stack((deg_gene, deg_pathway))
        unique_pairs, counts = np.unique(degree_pairs, axis=0, return_counts=True)
        for pair, count in zip(unique_pairs, counts):
            edge_counts_by_degree[(int(pair[0]), int(pair[1]))] += float(count)

    empirical_freq = {}
    for deg_pair, total in total_counts_by_degree.items():
        empirical_freq[deg_pair] = edge_counts_by_degree[deg_pair] / total if total > 0 else 0.0
    return empirical_freq


def main(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_samples = min(args.n_samples, 2_000)

    available_perms = list_available_permutation_ids(data_dir)
    empirical_perms = resolve_empirical_perms(available_perms, args.empirical_perms)

    print("=" * 80)
    print("EXPERIMENT 2G: FOCUSED COMPOSITION MODEL")
    print("=" * 80)
    print(f"n_samples: {args.n_samples}")
    print(f"empirical_perms: {empirical_perms}")

    cbg_0 = load_edge_matrix(data_dir, "CbG", perm_num=0)
    gig_0 = load_edge_matrix(data_dir, "GiG", perm_num=0)
    gppw_0 = load_edge_matrix(data_dir, "GpPW", perm_num=0)

    cb_gig_0 = cbg_0 @ gig_0
    cb_gig_gppw_0 = cb_gig_0 @ gppw_0

    compound_degrees = np.array(cbg_0.sum(axis=1)).flatten()
    gene_degrees = np.array(gig_0.sum(axis=1)).flatten()
    pathway_degrees = np.array(gppw_0.sum(axis=0)).flatten()

    pairs = sample_pairs_stratified(cb_gig_gppw_0, n_samples=args.n_samples, random_state=args.random_state)
    empirical_freq = compute_empirical_frequency(data_dir, empirical_perms, gene_degrees, pathway_degrees)

    cb_gig_0_lil = cb_gig_0.tolil()
    cb_gig_gppw_0_lil = cb_gig_gppw_0.tolil()
    gppw_0_csr = gppw_0.tocsr()

    feature_rows = []
    targets = []
    for compound_idx, pathway_idx in pairs:
        deg_c = compound_degrees[compound_idx]
        deg_pw = pathway_degrees[pathway_idx]
        genes_to_pathway = gppw_0_csr[:, pathway_idx].nonzero()[0]

        composition_sum = 0.0
        n_intermediates = 0
        for gene_idx in genes_to_pathway:
            cb_gig_count = cb_gig_0_lil[compound_idx, gene_idx]
            if cb_gig_count == 0:
                continue
            n_intermediates += 1
            deg_gene = gene_degrees[gene_idx]
            p_edge = empirical_freq.get((int(deg_gene), int(deg_pw)), 0.0)
            composition_sum += float(cb_gig_count) * p_edge

        feature_rows.append(
            np.array(
                [
                    deg_c,
                    deg_pw,
                    deg_c * deg_pw,
                    deg_c**2,
                    deg_pw**2,
                    n_intermediates,
                    composition_sum,
                    n_intermediates * composition_sum,
                ],
                dtype=np.float64,
            )
        )
        targets.append(float(cb_gig_gppw_0_lil[compound_idx, pathway_idx]))

    x_features = np.array(feature_rows)
    y_targets = np.array(targets)
    feature_names = [
        "deg_C",
        "deg_PW",
        "deg_C*deg_PW",
        "deg_C^2",
        "deg_PW^2",
        "n_intermediates",
        "composition_sum",
        "n_inter*comp",
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_targets, test_size=0.2, random_state=args.random_state
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    r_train = safe_pearson(y_train_pred, y_train)
    r_test = safe_pearson(y_test_pred, y_test)

    composition_sum_test = x_test[:, 6]
    r_exp2e = safe_pearson(composition_sum_test, y_test)
    improvement = r_test - r_exp2e if np.isfinite(r_exp2e) else float("nan")

    mae_train = float(mean_absolute_error(y_train, y_train_pred))
    mae_test = float(mean_absolute_error(y_test, y_test_pred))

    print(f"\nTrain r: {r_train:.4f}")
    print(f"Test r: {r_test:.4f}")
    print(f"Baseline (composition only) r: {r_exp2e:.4f}")
    print(f"Improvement: {improvement:.4f}")
    print(f"Train MAE: {mae_train:.4f}")
    print(f"Test MAE: {mae_test:.4f}")

    results_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_test_pred,
            "composition_sum": composition_sum_test,
        }
    )
    results_file = results_dir / "experiment2g_results.csv"
    results_df.to_csv(results_file, index=False)

    feature_importance = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    ).sort_values("coefficient", key=lambda x: np.abs(x), ascending=False)
    importance_file = results_dir / "experiment2g_feature_importance.csv"
    feature_importance.to_csv(importance_file, index=False)

    print(f"Saved: {results_file}")
    print(f"Saved: {importance_file}")

    if not args.skip_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(y_test, y_test_pred, alpha=0.3, s=10)
        lim = float(max(np.max(y_test), np.max(y_test_pred), 1.0))
        ax.plot([0, lim], [0, lim], "r--", label="Perfect prediction")
        ax.set_xlabel("True Count")
        ax.set_ylabel("Predicted Count")
        ax.set_title(f"Exp 2G prediction (r={r_test:.3f})")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.scatter(composition_sum_test, y_test, alpha=0.3, s=10, label="True")
        ax.scatter(composition_sum_test, y_test_pred, alpha=0.3, s=10, label="Pred")
        ax.set_xlabel("composition_sum")
        ax.set_ylabel("Count")
        ax.set_title("Composition vs true/predicted")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_file = results_dir / "experiment2g_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_file}")

    print("\nEXPERIMENT 2G COMPLETE")


if __name__ == "__main__":
    main(parse_args())
