"""Degree-conditioned compositionality analysis.

Runs corrected Option A compositional analysis for one 2-hop metapath:
1) Hetionet (perm 000 by default)
2) Null permutations (001-020 by default)

Outputs CSV/JSON summaries and optional plots under results/compositionality/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, pearsonr, ttest_ind
from tqdm import tqdm


REPO_DIR = Path(__file__).resolve().parents[1]

METAPATHS: dict[str, tuple[str, str]] = {
    "CbGpPW": ("CbG", "GpPW"),
    "CtDaG": ("CtD", "DaG"),
    "CrCbG": ("CrC", "CbG"),
    "CbGiG": ("CbG", "GiG"),
    "CpDaG": ("CpD", "DaG"),
    "CbGpBP": ("CbG", "GpBP"),
    "CbGpCC": ("CbG", "GpCC"),
}

DEGREE_BINS = [0, 5, 20, 100, np.inf]
DEGREE_LABELS = ["Very Low (0-5)", "Low (5-20)", "Medium (20-100)", "High (>100)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run degree-conditioned compositionality analysis using the corrected "
            "Option A compositional formula."
        )
    )
    parser.add_argument(
        "--metapath",
        default="CbGpPW",
        help=f"Metapath key (default: CbGpPW). Options: {', '.join(METAPATHS)}",
    )
    parser.add_argument("--edge1-type", default=None, help="Override metapath edge1 type.")
    parser.add_argument("--edge2-type", default=None, help="Override metapath edge2 type.")
    parser.add_argument(
        "--perm-start",
        type=int,
        default=1,
        help="First null permutation id (inclusive). Default: 1.",
    )
    parser.add_argument(
        "--perm-end",
        type=int,
        default=20,
        help="Last null permutation id (inclusive). Default: 20.",
    )
    parser.add_argument(
        "--hetionet-id",
        type=int,
        default=0,
        help="Hetionet permutation id. Default: 0.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Repository data directory containing permutations/.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "compositionality",
        help="Output directory for analysis artifacts.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on rows per permutation result for smoke tests.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for subsampling when --max-pairs is set.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating PNG plot outputs.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.perm_start > args.perm_end:
        raise ValueError("--perm-start must be <= --perm-end")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be > 0")
    if args.edge1_type is None and args.edge2_type is None and args.metapath not in METAPATHS:
        raise ValueError(
            f"Unknown metapath '{args.metapath}'. "
            f"Provide --edge1-type/--edge2-type or use one of: {', '.join(METAPATHS)}"
        )
    if (args.edge1_type is None) ^ (args.edge2_type is None):
        raise ValueError("Provide both --edge1-type and --edge2-type together.")


def resolve_edge_types(args: argparse.Namespace) -> tuple[str, str]:
    if args.edge1_type and args.edge2_type:
        return args.edge1_type, args.edge2_type
    return METAPATHS[args.metapath]


def load_edge_matrix(data_dir: Path, edge_type: str, perm_id: int) -> sp.csr_matrix:
    edge_file = data_dir / "permutations" / f"{perm_id:03d}.hetmat" / "edges" / f"{edge_type}.sparse.npz"
    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")
    return sp.load_npz(edge_file).tocsr()


def analytical_prior(u: float, v: float, m: float) -> float:
    uv = u * v
    denominator = np.sqrt(uv ** 2 + (m - u - v + 1) ** 2)
    return float(uv / denominator) if denominator > 0 else 0.0


def bin_degrees(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ordered_categories = pd.CategoricalDtype(categories=DEGREE_LABELS, ordered=True)
    out["compound_degree_bin"] = pd.cut(out["compound_degree"], bins=DEGREE_BINS, labels=DEGREE_LABELS).astype(
        ordered_categories
    )
    out["pathway_degree_bin"] = pd.cut(out["pathway_degree"], bins=DEGREE_BINS, labels=DEGREE_LABELS).astype(
        ordered_categories
    )
    return out


def limit_rows(df: pd.DataFrame, max_rows: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def compute_metapath_compositionality(edge1_matrix: sp.csr_matrix, edge2_matrix: sp.csr_matrix, perm_id: int) -> pd.DataFrame:
    if edge1_matrix.shape[1] != edge2_matrix.shape[0]:
        raise ValueError(f"Gene dimension mismatch: {edge1_matrix.shape} vs {edge2_matrix.shape}")

    compound_degrees = np.asarray(edge1_matrix.sum(axis=1)).ravel()
    pathway_degrees = np.asarray(edge2_matrix.sum(axis=0)).ravel()
    compound_nonzero = np.where(compound_degrees > 0)[0]
    pathway_nonzero = np.where(pathway_degrees > 0)[0]

    edge1_aligned = edge1_matrix[compound_nonzero, :]
    edge2_aligned = edge2_matrix[:, pathway_nonzero]

    gene_degrees_in = np.asarray(edge1_aligned.sum(axis=0)).ravel()
    gene_degrees_out = np.asarray(edge2_aligned.sum(axis=1)).ravel()

    metapath_matrix = (edge1_aligned @ edge2_aligned).tocsr()
    row_idx, col_idx = metapath_matrix.nonzero()

    # Precompute sparse adjacency neighborhoods.
    compound_genes = [set(edge1_aligned.getrow(i).nonzero()[1].tolist()) for i in range(edge1_aligned.shape[0])]
    pathway_genes = [set(edge2_aligned.getcol(j).nonzero()[0].tolist()) for j in range(edge2_aligned.shape[1])]

    # Edge priors.
    edge1_priors: dict[tuple[int, int], float] = {}
    edge2_priors: dict[tuple[int, int], float] = {}

    m1 = float(edge1_aligned.nnz)
    src1 = np.asarray(edge1_aligned.sum(axis=1)).ravel()
    tgt1 = np.asarray(edge1_aligned.sum(axis=0)).ravel()
    r1, c1 = edge1_aligned.nonzero()
    for i, j in zip(r1, c1):
        u, v = src1[i], tgt1[j]
        if u > 0 and v > 0:
            edge1_priors[(int(i), int(j))] = analytical_prior(float(u), float(v), m1)

    m2 = float(edge2_aligned.nnz)
    src2 = np.asarray(edge2_aligned.sum(axis=1)).ravel()
    tgt2 = np.asarray(edge2_aligned.sum(axis=0)).ravel()
    r2, c2 = edge2_aligned.nonzero()
    for i, j in zip(r2, c2):
        u, v = src2[i], tgt2[j]
        if u > 0 and v > 0:
            edge2_priors[(int(i), int(j))] = analytical_prior(float(u), float(v), m2)

    rows: list[dict[str, float | int]] = []
    for i, j in zip(row_idx, col_idx):
        c_genes = compound_genes[int(i)]
        p_genes = pathway_genes[int(j)]
        shared_genes = c_genes & p_genes
        if not shared_genes:
            continue

        n_possible = len(c_genes)
        p_observed = len(shared_genes) / n_possible if n_possible > 0 else 0.0
        if p_observed <= 0:
            continue

        failure_prob = 1.0
        pathway_degree_product_total = 1.0
        pathway_degree_product_joint = 1.0

        for gene in shared_genes:
            p_edge1 = edge1_priors.get((int(i), int(gene)), 0.0)
            p_edge2 = edge2_priors.get((int(gene), int(j)), 0.0)
            individual_prob = p_edge1 * p_edge2
            failure_prob *= (1.0 - individual_prob)

            gene_in = gene_degrees_in[int(gene)]
            gene_out = gene_degrees_out[int(gene)]
            pathway_degree_product_total *= (gene_in + gene_out)
            pathway_degree_product_joint *= (gene_in * gene_out)

        p_comp = max(0.0, min(1.0, 1.0 - failure_prob))
        if p_comp <= 0:
            continue

        pmi = float(np.log2(p_observed / p_comp))

        orig_compound_idx = int(compound_nonzero[int(i)])
        orig_pathway_idx = int(pathway_nonzero[int(j)])

        rows.append(
            {
                "perm_id": int(perm_id),
                "compound_idx": orig_compound_idx,
                "pathway_idx": orig_pathway_idx,
                "compound_degree": int(compound_degrees[orig_compound_idx]),
                "pathway_degree": int(pathway_degrees[orig_pathway_idx]),
                "pathway_degree_product_total": float(pathway_degree_product_total),
                "pathway_degree_product_joint": float(pathway_degree_product_joint),
                "observed_freq": float(p_observed),
                "compositional_prob": float(p_comp),
                "pmi": pmi,
                "residual": float(p_observed - p_comp),
            }
        )

    return pd.DataFrame(rows)


def degree_bin_comparison(hetionet_df: pd.DataFrame, perm_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for degree_col in ["compound_degree_bin", "pathway_degree_bin"]:
        for label in DEGREE_LABELS:
            het_subset = hetionet_df.loc[hetionet_df[degree_col] == label, "pmi"].dropna()
            null_subset = perm_df.loc[perm_df[degree_col] == label, "pmi"].dropna()
            if len(het_subset) == 0 or len(null_subset) == 0:
                continue
            try:
                _, pvalue = mannwhitneyu(het_subset, null_subset, alternative="two-sided")
            except Exception:
                pvalue = np.nan
            rows.append(
                {
                    "degree_axis": degree_col,
                    "degree_bin": label,
                    "hetionet_n": int(len(het_subset)),
                    "null_n": int(len(null_subset)),
                    "hetionet_mean_pmi": float(het_subset.mean()),
                    "null_mean_pmi": float(null_subset.mean()),
                    "mean_diff": float(het_subset.mean() - null_subset.mean()),
                    "mannwhitney_p": float(pvalue),
                }
            )
    return pd.DataFrame(rows)


def safe_pair_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    try:
        return float(pearsonr(x, y)[0])
    except Exception:
        return float("nan")


def maybe_make_plots(
    *,
    metapath_name: str,
    results_dir: Path,
    hetionet_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    null_corrs: list[float],
    hetionet_corr: float,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    het_pmi = hetionet_df["pmi"].to_numpy()
    null_pmi = perm_df["pmi"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    ax = axes[0, 0]
    ax.hist(het_pmi, bins=50, alpha=0.7, label="Hetionet", color="red", density=True, edgecolor="black")
    ax.hist(null_pmi, bins=50, alpha=0.7, label="Null", color="blue", density=True, edgecolor="black")
    ax.axvline(float(np.mean(het_pmi)), color="darkred", linestyle="--", linewidth=2)
    ax.axvline(float(np.mean(null_pmi)), color="darkblue", linestyle="--", linewidth=2)
    ax.set_xlabel("PMI")
    ax.set_ylabel("Density")
    ax.set_title("PMI Distribution: Hetionet vs Null")
    ax.legend()
    ax.grid(alpha=0.3)

    x = np.arange(len(DEGREE_LABELS))
    width = 0.35
    ax = axes[0, 1]
    het_grouped = hetionet_df.groupby("compound_degree_bin")["pmi"].mean()
    null_grouped = perm_df.groupby("compound_degree_bin")["pmi"].mean()
    ax.bar(x - width / 2, het_grouped.reindex(DEGREE_LABELS, fill_value=0), width, label="Hetionet", color="red", alpha=0.7)
    ax.bar(x + width / 2, null_grouped.reindex(DEGREE_LABELS, fill_value=0), width, label="Null", color="blue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(DEGREE_LABELS, rotation=45, ha="right")
    ax.set_title("Mean PMI by Compound Degree")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    het_grouped = hetionet_df.groupby("pathway_degree_bin")["pmi"].mean()
    null_grouped = perm_df.groupby("pathway_degree_bin")["pmi"].mean()
    ax.bar(x - width / 2, het_grouped.reindex(DEGREE_LABELS, fill_value=0), width, label="Hetionet", color="red", alpha=0.7)
    ax.bar(x + width / 2, null_grouped.reindex(DEGREE_LABELS, fill_value=0), width, label="Null", color="blue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(DEGREE_LABELS, rotation=45, ha="right")
    ax.set_title("Mean PMI by Pathway Degree")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if null_corrs:
        ax.hist(null_corrs, bins=20, alpha=0.7, color="blue", edgecolor="black", label="Null")
    ax.axvline(hetionet_corr, color="red", linestyle="--", linewidth=3, label=f"Hetionet r={hetionet_corr:.3f}")
    ax.set_xlabel("Correlation (Observed vs Compositional)")
    ax.set_title("Compositional Model Fit: Hetionet vs Null")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_file = results_dir / f"metapath_{metapath_name}_degree_conditioned_analysis.png"
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    het_pivot = hetionet_df.pivot_table(values="pmi", index="pathway_degree_bin", columns="compound_degree_bin", aggfunc="mean")
    het_pivot = het_pivot.reindex(index=DEGREE_LABELS[::-1], columns=DEGREE_LABELS)
    sns.heatmap(het_pivot, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax, cbar_kws={"label": "Mean PMI"})
    ax.set_title("Hetionet: Mean PMI by Degree Bins")
    ax.set_xlabel("Compound Degree")
    ax.set_ylabel("Pathway Degree")

    ax = axes[1]
    null_pivot = perm_df.pivot_table(values="pmi", index="pathway_degree_bin", columns="compound_degree_bin", aggfunc="mean")
    null_pivot = null_pivot.reindex(index=DEGREE_LABELS[::-1], columns=DEGREE_LABELS)
    sns.heatmap(null_pivot, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax, cbar_kws={"label": "Mean PMI"})
    ax.set_title("Null: Mean PMI by Degree Bins")
    ax.set_xlabel("Compound Degree")
    ax.set_ylabel("Pathway Degree")

    plt.tight_layout()
    heatmap_file = results_dir / f"metapath_{metapath_name}_pmi_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    edge1_type, edge2_type = resolve_edge_types(args)

    rng = np.random.default_rng(args.random_seed)
    perm_ids = list(range(args.perm_start, args.perm_end + 1))
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Degree-Conditioned Compositionality (Option A)")
    print("=" * 80)
    print(f"Metapath: {args.metapath} ({edge1_type} -> {edge2_type})")
    print(f"Hetionet perm id: {args.hetionet_id:03d}")
    print(f"Null permutations: {args.perm_start:03d}-{args.perm_end:03d}")
    print(f"Data dir: {args.data_dir}")
    print(f"Results dir: {args.results_dir}")
    if args.max_pairs is not None:
        print(f"Per-permutation max rows: {args.max_pairs}")

    print("\nAnalyzing Hetionet...")
    edge1_het = load_edge_matrix(args.data_dir, edge1_type, args.hetionet_id)
    edge2_het = load_edge_matrix(args.data_dir, edge2_type, args.hetionet_id)
    hetionet_df = compute_metapath_compositionality(edge1_het, edge2_het, args.hetionet_id)
    hetionet_df = limit_rows(hetionet_df, args.max_pairs, rng)
    if hetionet_df.empty:
        raise RuntimeError("Hetionet analysis produced no rows. Check edge types and input data.")
    hetionet_df = bin_degrees(hetionet_df)
    print(f"  Hetionet rows: {len(hetionet_df):,}")

    print("\nAnalyzing null permutations...")
    all_perm_results: list[pd.DataFrame] = []
    for perm_id in tqdm(perm_ids, desc="Permutations"):
        edge1 = load_edge_matrix(args.data_dir, edge1_type, perm_id)
        edge2 = load_edge_matrix(args.data_dir, edge2_type, perm_id)
        perm_df = compute_metapath_compositionality(edge1, edge2, perm_id)
        perm_df = limit_rows(perm_df, args.max_pairs, rng)
        perm_df = bin_degrees(perm_df)
        all_perm_results.append(perm_df)

    perm_df = pd.concat(all_perm_results, ignore_index=True)
    print(f"  Null rows total: {len(perm_df):,}")

    het_pmi = hetionet_df["pmi"].dropna().to_numpy()
    null_pmi = perm_df["pmi"].dropna().to_numpy()

    u_stat, u_pval = mannwhitneyu(het_pmi, null_pmi, alternative="two-sided")
    t_stat, t_pval = ttest_ind(het_pmi, null_pmi)
    het_corr = safe_pair_pearson(
        hetionet_df["observed_freq"].to_numpy(),
        hetionet_df["compositional_prob"].to_numpy(),
    )
    null_corrs: list[float] = []
    for perm_id in perm_ids:
        subset = perm_df[perm_df["perm_id"] == perm_id]
        if len(subset) < 2:
            continue
        corr = safe_pair_pearson(subset["observed_freq"].to_numpy(), subset["compositional_prob"].to_numpy())
        if not np.isnan(corr):
            null_corrs.append(corr)

    perm_summary = (
        perm_df.groupby("perm_id")
        .agg(pmi_mean=("pmi", "mean"), pmi_median=("pmi", "median"), pmi_std=("pmi", "std"), n_rows=("pmi", "size"))
        .reset_index()
    )
    degree_bin_df = degree_bin_comparison(hetionet_df, perm_df)

    summary = {
        "metapath": args.metapath,
        "edge_types": [edge1_type, edge2_type],
        "hetionet_rows": int(len(hetionet_df)),
        "null_rows": int(len(perm_df)),
        "hetionet_mean_pmi": float(np.mean(het_pmi)),
        "hetionet_median_pmi": float(np.median(het_pmi)),
        "hetionet_std_pmi": float(np.std(het_pmi)),
        "null_mean_pmi": float(np.mean(null_pmi)),
        "null_median_pmi": float(np.median(null_pmi)),
        "null_std_pmi": float(np.std(null_pmi)),
        "difference_mean_pmi": float(np.mean(het_pmi) - np.mean(null_pmi)),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(u_pval),
        "t_statistic": float(t_stat),
        "t_test_p": float(t_pval),
        "hetionet_correlation": float(het_corr),
        "null_mean_correlation": float(np.mean(null_corrs)) if null_corrs else np.nan,
        "null_std_correlation": float(np.std(null_corrs)) if null_corrs else np.nan,
        "perm_range": [args.perm_start, args.perm_end],
        "max_pairs": args.max_pairs,
    }

    hetionet_out = args.results_dir / f"metapath_{args.metapath}_hetionet_degree_results_option_a.csv"
    null_out = args.results_dir / f"metapath_{args.metapath}_null_degree_results_option_a.csv"
    summary_csv = args.results_dir / f"metapath_{args.metapath}_degree_conditioned_summary_option_a.csv"
    summary_json = args.results_dir / f"metapath_{args.metapath}_degree_conditioned_summary_option_a.json"
    perm_summary_out = args.results_dir / f"metapath_{args.metapath}_permutation_summary_option_a.csv"
    degree_bin_out = args.results_dir / f"metapath_{args.metapath}_degree_bin_comparison_option_a.csv"

    hetionet_df.to_csv(hetionet_out, index=False)
    perm_df.to_csv(null_out, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2))
    perm_summary.to_csv(perm_summary_out, index=False)
    degree_bin_df.to_csv(degree_bin_out, index=False)

    if not args.skip_plots:
        maybe_make_plots(
            metapath_name=args.metapath,
            results_dir=args.results_dir,
            hetionet_df=hetionet_df,
            perm_df=perm_df,
            null_corrs=null_corrs,
            hetionet_corr=het_corr,
        )

    print("\nSummary metrics:")
    print(f"  Hetionet mean PMI: {summary['hetionet_mean_pmi']:.4f}")
    print(f"  Null mean PMI: {summary['null_mean_pmi']:.4f}")
    print(f"  Mean difference: {summary['difference_mean_pmi']:.4f}")
    print(f"  Mann-Whitney p: {summary['mann_whitney_p']:.3e}")
    print(f"  Hetionet correlation: {summary['hetionet_correlation']:.4f}")
    print(f"  Null mean correlation: {summary['null_mean_correlation']:.4f}")

    print("\nSaved outputs:")
    print(f"  - {hetionet_out}")
    print(f"  - {null_out}")
    print(f"  - {summary_csv}")
    print(f"  - {summary_json}")
    print(f"  - {perm_summary_out}")
    print(f"  - {degree_bin_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
