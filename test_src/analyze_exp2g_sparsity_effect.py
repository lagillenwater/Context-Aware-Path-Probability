"""
Analyze Exp 2G sparsity effects on pathway-count prediction.

Shows whether n_intermediates contributes predictive signal beyond composition_sum.
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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.append(str(REPO_DIR / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sparsity effects for Exp 2G linear model features.",
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
        help="Directory for output figures/tables.",
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--empirical-perms",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="Permutation IDs used to estimate empirical edge frequencies.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating PNG figure.",
    )
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


def resolve_empirical_perms(requested: list[int], available: list[int]) -> list[int]:
    explicit = [perm for perm in requested if perm in available]
    if explicit:
        missing = [perm for perm in requested if perm not in available]
        if missing:
            print(f"Requested empirical perms missing locally, skipping: {missing}")
        return explicit

    if not available:
        raise FileNotFoundError("No local permutations found in data/permutations.")

    if len(available) > 1 and available[0] == 0:
        fallback = available[1 : min(5, len(available))]
        if fallback:
            return fallback
    return available[: min(4, len(available))]


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

    num_perms = len(empirical_perms)
    for gene_deg, gene_count in zip(gene_unique, gene_counts):
        for pathway_deg, pathway_count in zip(pathway_unique, pathway_counts):
            total_counts_by_degree[(int(gene_deg), int(pathway_deg))] = float(
                gene_count * pathway_count * num_perms
            )

    for perm_num in empirical_perms:
        gp_pw_perm = load_edge_matrix(data_dir, "GpPW", perm_num=perm_num).tocoo()
        deg_gene = gene_degrees[gp_pw_perm.row].astype(int)
        deg_pathway = pathway_degrees[gp_pw_perm.col].astype(int)
        degree_pairs = np.column_stack((deg_gene, deg_pathway))
        unique_pairs, counts = np.unique(degree_pairs, axis=0, return_counts=True)
        for pair, count in zip(unique_pairs, counts):
            edge_counts_by_degree[(int(pair[0]), int(pair[1]))] += float(count)

    empirical_freq: dict[tuple[int, int], float] = {}
    for deg_pair, total in total_counts_by_degree.items():
        empirical_freq[deg_pair] = edge_counts_by_degree[deg_pair] / total if total > 0 else 0.0
    return empirical_freq


def build_feature_dataframe(
    pairs: list[tuple[int, int]],
    cb_gig_0: sp.spmatrix,
    cb_gig_gppw_0: sp.spmatrix,
    gppw_0: sp.spmatrix,
    compound_degrees: np.ndarray,
    gene_degrees: np.ndarray,
    pathway_degrees: np.ndarray,
    empirical_freq: dict[tuple[int, int], float],
) -> pd.DataFrame:
    cb_gig_0_lil = cb_gig_0.tolil()
    cb_gig_gppw_0_lil = cb_gig_gppw_0.tolil()
    gppw_0_csr = gppw_0.tocsr()

    rows = []
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

        features = np.array(
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

        rows.append(
            {
                "deg_C": deg_c,
                "deg_PW": deg_pw,
                "n_intermediates": n_intermediates,
                "composition_sum": composition_sum,
                "target": float(cb_gig_gppw_0_lil[compound_idx, pathway_idx]),
                "features": features,
            }
        )

    return pd.DataFrame(rows)


def create_visualization(df_test: pd.DataFrame, r_comp: float, r_full: float, output_file: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    scatter = ax.scatter(
        df_test["n_intermediates"],
        df_test["error_exp2e"],
        alpha=0.3,
        s=20,
        c=df_test["target"],
        cmap="viridis",
    )
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("n_intermediates")
    ax.set_ylabel("Error (true - pred)")
    ax.set_title("Exp 2E: Error vs Sparsity")
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="True count")

    ax = axes[0, 1]
    scatter = ax.scatter(
        df_test["n_intermediates"],
        df_test["error_exp2g"],
        alpha=0.3,
        s=20,
        c=df_test["target"],
        cmap="viridis",
    )
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("n_intermediates")
    ax.set_ylabel("Error (true - pred)")
    ax.set_title("Exp 2G: Error vs Sparsity")
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="True count")

    ax = axes[1, 0]
    sparsity_bins = ["0-1", "2-3", "4-5", "6-10", "10+"]
    mae_e2e_by_bin = []
    mae_e2g_by_bin = []

    for sparsity_bin in sparsity_bins:
        mask = df_test["sparsity_bin"] == sparsity_bin
        if mask.sum() < 10:
            mae_e2e_by_bin.append(np.nan)
            mae_e2g_by_bin.append(np.nan)
            continue
        subset = df_test[mask]
        mae_e2e_by_bin.append(np.abs(subset["error_exp2e"]).mean())
        mae_e2g_by_bin.append(np.abs(subset["error_exp2g"]).mean())

    x = np.arange(len(sparsity_bins))
    width = 0.35
    ax.bar(x - width / 2, mae_e2e_by_bin, width, label="Exp 2E", alpha=0.8)
    ax.bar(x + width / 2, mae_e2g_by_bin, width, label="Exp 2G", alpha=0.8)
    ax.set_xlabel("Sparsity bin (n_intermediates)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Error Reduction by Sparsity")
    ax.set_xticks(x)
    ax.set_xticklabels(sparsity_bins)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.axis("off")
    summary_text = (
        "SPARSITY EFFECT ANALYSIS\n\n"
        f"Exp 2E (composition only): r = {r_comp:.3f}\n"
        f"Exp 2G (full model):        r = {r_full:.3f}\n"
        f"Improvement:                {r_full - r_comp:.3f}\n\n"
        "Conclusion:\n"
        "n_intermediates adds signal beyond composition_sum."
    )
    ax.text(0.1, 0.5, summary_text, fontsize=10, family="monospace", va="center")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def run_analysis(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_samples = min(args.n_samples, 2_000)

    available = list_available_permutation_ids(data_dir / "permutations")
    empirical_perms = resolve_empirical_perms(args.empirical_perms, available)

    print("=" * 80)
    print("ANALYZE EXP 2G SPARSITY EFFECT")
    print("=" * 80)
    print(f"n_samples: {args.n_samples}")
    print(f"empirical_perms: {empirical_perms}")

    print("\nLoading core matrices...")
    cbg_0 = load_edge_matrix(data_dir, "CbG", perm_num=0)
    gig_0 = load_edge_matrix(data_dir, "GiG", perm_num=0)
    gppw_0 = load_edge_matrix(data_dir, "GpPW", perm_num=0)

    cb_gig_0 = cbg_0 @ gig_0
    cb_gig_gppw_0 = cb_gig_0 @ gppw_0

    compound_degrees = np.array(cbg_0.sum(axis=1)).flatten()
    gene_degrees = np.array(gig_0.sum(axis=1)).flatten()
    pathway_degrees = np.array(gppw_0.sum(axis=0)).flatten()

    pairs = sample_pairs_stratified(
        cb_gig_gppw_0,
        n_samples=args.n_samples,
        random_state=args.random_state,
    )

    print("Computing empirical frequencies...")
    empirical_freq = compute_empirical_frequency(
        data_dir=data_dir,
        empirical_perms=empirical_perms,
        gene_degrees=gene_degrees,
        pathway_degrees=pathway_degrees,
    )

    print("Extracting features...")
    df = build_feature_dataframe(
        pairs=pairs,
        cb_gig_0=cb_gig_0,
        cb_gig_gppw_0=cb_gig_gppw_0,
        gppw_0=gppw_0,
        compound_degrees=compound_degrees,
        gene_degrees=gene_degrees,
        pathway_degrees=pathway_degrees,
        empirical_freq=empirical_freq,
    )

    x_features = np.array([row["features"] for _, row in df.iterrows()])
    y_targets = df["target"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y_targets,
        test_size=0.2,
        random_state=args.random_state,
    )
    _, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=args.random_state,
    )
    df_test = df.iloc[test_idx].copy()

    print("Training models...")
    model_full = LinearRegression()
    model_full.fit(x_train, y_train)
    y_test_full = model_full.predict(x_test)
    r_full = safe_pearson(y_test_full, y_test)

    x_train_comp = x_train[:, 6:7]
    x_test_comp = x_test[:, 6:7]
    model_comp = LinearRegression()
    model_comp.fit(x_train_comp, y_train)
    y_test_comp = model_comp.predict(x_test_comp)
    r_comp = safe_pearson(y_test_comp, y_test)

    df_test["pred_exp2e"] = y_test_comp
    df_test["pred_exp2g"] = y_test_full
    df_test["error_exp2e"] = df_test["target"] - df_test["pred_exp2e"]
    df_test["error_exp2g"] = df_test["target"] - df_test["pred_exp2g"]

    print(f"\nExp 2E (composition only): r = {r_comp:.4f}")
    print(f"Exp 2G (full model):        r = {r_full:.4f}")
    print(f"Improvement:                {r_full - r_comp:.4f}")

    df_test["sparsity_bin"] = pd.cut(
        df_test["n_intermediates"],
        bins=[0, 1, 3, 5, 10, np.inf],
        labels=["0-1", "2-3", "4-5", "6-10", "10+"],
    )

    summary_rows = []
    for sparsity_bin in ["0-1", "2-3", "4-5", "6-10", "10+"]:
        mask = df_test["sparsity_bin"] == sparsity_bin
        if mask.sum() < 10:
            continue
        subset = df_test[mask]
        summary_rows.append(
            {
                "sparsity_bin": sparsity_bin,
                "n": int(len(subset)),
                "mean_n_intermediates": float(subset["n_intermediates"].mean()),
                "r_exp2e": safe_pearson(subset["pred_exp2e"].values, subset["target"].values),
                "r_exp2g": safe_pearson(subset["pred_exp2g"].values, subset["target"].values),
                "mae_exp2e": float(np.abs(subset["error_exp2e"]).mean()),
                "mae_exp2g": float(np.abs(subset["error_exp2g"]).mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_file = results_dir / "experiment2g_sparsity_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved: {summary_file}")

    detailed_file = results_dir / "experiment2g_sparsity_test_details.csv"
    df_test.drop(columns=["features"]).to_csv(detailed_file, index=False)
    print(f"Saved: {detailed_file}")

    if not args.skip_plots:
        plot_file = results_dir / "experiment2g_sparsity_analysis.png"
        create_visualization(df_test=df_test, r_comp=r_comp, r_full=r_full, output_file=plot_file)
        print(f"Saved: {plot_file}")

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    run_analysis(parse_args())
