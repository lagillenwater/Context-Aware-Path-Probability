"""Degree-aware compositional model analysis.

Compares:
1) Naive compositional model (global m in analytical prior)
2) Continuous degree-aware compositional model (local effective m by degree window)

For one 2-hop metapath on Hetionet and a range of null permutations.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

DEGREE_BINS_VIZ = [0, 2, 4, 8, 16, 32, 64, 128, np.inf]
DEGREE_LABELS_VIZ = ["1-2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", ">128"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze naive vs continuous degree-aware compositional predictions "
            "for a 2-hop metapath."
        )
    )
    parser.add_argument(
        "--metapath",
        default="CbGpPW",
        help=f"Metapath key (default: CbGpPW). Options: {', '.join(METAPATHS)}",
    )
    parser.add_argument("--edge1-type", default=None, help="Override metapath edge1 type.")
    parser.add_argument("--edge2-type", default=None, help="Override metapath edge2 type.")
    parser.add_argument("--hetionet-id", type=int, default=0, help="Hetionet permutation id. Default: 0.")
    parser.add_argument("--perm-start", type=int, default=1, help="First null permutation id (inclusive). Default: 1.")
    parser.add_argument("--perm-end", type=int, default=5, help="Last null permutation id (inclusive). Default: 5.")
    parser.add_argument(
        "--degree-window-pct",
        type=float,
        default=0.2,
        help="Relative degree window for effective m (default: 0.2).",
    )
    parser.add_argument(
        "--min-window",
        type=int,
        default=2,
        help="Minimum absolute degree window width (default: 2).",
    )
    parser.add_argument(
        "--min-edge-count",
        type=int,
        default=10,
        help="Minimum edge count before falling back to wider/global m (default: 10).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on comparison pairs for smoke testing.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for optional pair subsampling.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Repository data directory containing permutations/.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "compositionality" / "degree_aware",
        help="Output directory for notebook-12 migration artifacts.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip PNG plot outputs.")
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.perm_start > args.perm_end:
        raise ValueError("--perm-start must be <= --perm-end")
    if args.degree_window_pct <= 0:
        raise ValueError("--degree-window-pct must be > 0")
    if args.min_window < 1:
        raise ValueError("--min-window must be >= 1")
    if args.min_edge_count < 1:
        raise ValueError("--min-edge-count must be >= 1")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be > 0")
    if (args.edge1_type is None) ^ (args.edge2_type is None):
        raise ValueError("Provide both --edge1-type and --edge2-type together.")
    if args.edge1_type is None and args.metapath not in METAPATHS:
        raise ValueError(
            f"Unknown metapath '{args.metapath}'. "
            f"Provide --edge1-type/--edge2-type or one of: {', '.join(METAPATHS)}"
        )


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
    denominator = np.sqrt(uv**2 + (m - u - v + 1) ** 2)
    return float(uv / denominator) if denominator > 0 else 0.0


def compute_degree_window(degree: int, *, window_pct: float, min_window: int) -> tuple[int, int]:
    window_size = max(min_window, int(degree * window_pct))
    lower = max(1, degree - window_size)
    upper = degree + window_size
    return lower, upper


def build_degree_pair_index(edge_matrix: sp.csr_matrix) -> dict[tuple[int, int], int]:
    source_degrees = np.asarray(edge_matrix.sum(axis=1)).ravel().astype(int)
    target_degrees = np.asarray(edge_matrix.sum(axis=0)).ravel().astype(int)
    degree_pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    rows, cols = edge_matrix.nonzero()
    for i, j in zip(rows, cols):
        degree_pair_counts[(int(source_degrees[i]), int(target_degrees[j]))] += 1
    return dict(degree_pair_counts)


def get_effective_m(
    u: int,
    v: int,
    *,
    degree_pair_index: dict[tuple[int, int], int],
    window_pct: float,
    min_window: int,
    min_edge_count: int,
) -> float:
    u_lower, u_upper = compute_degree_window(u, window_pct=window_pct, min_window=min_window)
    v_lower, v_upper = compute_degree_window(v, window_pct=window_pct, min_window=min_window)

    edge_count = 0
    for (deg_u, deg_v), count in degree_pair_index.items():
        if u_lower <= deg_u <= u_upper and v_lower <= deg_v <= v_upper:
            edge_count += count

    if edge_count < min_edge_count:
        u_lower, u_upper = compute_degree_window(u, window_pct=window_pct * 2.0, min_window=min_window)
        v_lower, v_upper = compute_degree_window(v, window_pct=window_pct * 2.0, min_window=min_window)
        edge_count = 0
        for (deg_u, deg_v), count in degree_pair_index.items():
            if u_lower <= deg_u <= u_upper and v_lower <= deg_v <= v_upper:
                edge_count += count

    if edge_count < min_edge_count:
        edge_count = sum(degree_pair_index.values())

    return float(edge_count)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    try:
        return float(pearsonr(x, y)[0])
    except Exception:
        return float("nan")


def compute_observed_and_pairs(edge1: sp.csr_matrix, edge2: sp.csr_matrix) -> tuple[dict[tuple[int, int], float], list[tuple[int, int, set[int]]]]:
    if edge1.shape[1] != edge2.shape[0]:
        raise ValueError(f"Matrix shape mismatch: {edge1.shape} then {edge2.shape}")

    compound_degrees = np.asarray(edge1.sum(axis=1)).ravel()
    pathway_degrees = np.asarray(edge2.sum(axis=0)).ravel()
    compound_nonzero = np.where(compound_degrees > 0)[0]
    pathway_nonzero = np.where(pathway_degrees > 0)[0]

    edge1_filt = edge1[compound_nonzero, :]
    edge2_filt = edge2[:, pathway_nonzero]
    metapath_matrix = (edge1_filt @ edge2_filt).tocsr()
    rows, cols = metapath_matrix.nonzero()

    compound_genes = [set(edge1_filt.getrow(i).nonzero()[1].tolist()) for i in range(edge1_filt.shape[0])]
    pathway_genes = [set(edge2_filt.getcol(j).nonzero()[0].tolist()) for j in range(edge2_filt.shape[1])]

    observed: dict[tuple[int, int], float] = {}
    pair_gene_sets: list[tuple[int, int, set[int]]] = []

    for i, j in zip(rows, cols):
        shared_genes = compound_genes[int(i)] & pathway_genes[int(j)]
        if not shared_genes:
            continue
        n_possible = len(compound_genes[int(i)])
        if n_possible <= 0:
            continue
        orig_i = int(compound_nonzero[int(i)])
        orig_j = int(pathway_nonzero[int(j)])
        observed[(orig_i, orig_j)] = float(len(shared_genes) / n_possible)
        pair_gene_sets.append((orig_i, orig_j, shared_genes))

    return observed, pair_gene_sets


def compute_priors_naive(
    edge1: sp.csr_matrix,
    edge2: sp.csr_matrix,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    compound_degrees = np.asarray(edge1.sum(axis=1)).ravel()
    pathway_degrees = np.asarray(edge2.sum(axis=0)).ravel()
    gene_degrees_1 = np.asarray(edge1.sum(axis=0)).ravel()
    gene_degrees_2 = np.asarray(edge2.sum(axis=1)).ravel()

    m1 = float(edge1.nnz)
    m2 = float(edge2.nnz)

    edge1_priors: dict[tuple[int, int], float] = {}
    r1, c1 = edge1.nonzero()
    for i, j in zip(r1, c1):
        u, v = compound_degrees[i], gene_degrees_1[j]
        if u > 0 and v > 0:
            edge1_priors[(int(i), int(j))] = analytical_prior(float(u), float(v), m1)

    edge2_priors: dict[tuple[int, int], float] = {}
    r2, c2 = edge2.nonzero()
    for i, j in zip(r2, c2):
        u, v = gene_degrees_2[i], pathway_degrees[j]
        if u > 0 and v > 0:
            edge2_priors[(int(i), int(j))] = analytical_prior(float(u), float(v), m2)

    return edge1_priors, edge2_priors


def compute_priors_degree_aware(
    edge1: sp.csr_matrix,
    edge2: sp.csr_matrix,
    *,
    window_pct: float,
    min_window: int,
    min_edge_count: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[str, dict[tuple[int, int], float]]]:
    compound_degrees = np.asarray(edge1.sum(axis=1)).ravel().astype(int)
    pathway_degrees = np.asarray(edge2.sum(axis=0)).ravel().astype(int)
    gene_degrees_1 = np.asarray(edge1.sum(axis=0)).ravel().astype(int)
    gene_degrees_2 = np.asarray(edge2.sum(axis=1)).ravel().astype(int)

    edge1_degree_index = build_degree_pair_index(edge1)
    edge2_degree_index = build_degree_pair_index(edge2)

    edge1_priors: dict[tuple[int, int], float] = {}
    edge1_effective_m: dict[tuple[int, int], float] = {}
    r1, c1 = edge1.nonzero()
    for i, j in zip(r1, c1):
        u = int(compound_degrees[i])
        v = int(gene_degrees_1[j])
        if u > 0 and v > 0:
            m_eff = get_effective_m(
                u,
                v,
                degree_pair_index=edge1_degree_index,
                window_pct=window_pct,
                min_window=min_window,
                min_edge_count=min_edge_count,
            )
            edge1_priors[(int(i), int(j))] = analytical_prior(float(u), float(v), m_eff)
            edge1_effective_m[(u, v)] = m_eff

    edge2_priors: dict[tuple[int, int], float] = {}
    edge2_effective_m: dict[tuple[int, int], float] = {}
    r2, c2 = edge2.nonzero()
    for i, j in zip(r2, c2):
        u = int(gene_degrees_2[i])
        v = int(pathway_degrees[j])
        if u > 0 and v > 0:
            m_eff = get_effective_m(
                u,
                v,
                degree_pair_index=edge2_degree_index,
                window_pct=window_pct,
                min_window=min_window,
                min_edge_count=min_edge_count,
            )
            edge2_priors[(int(i), int(j))] = analytical_prior(float(u), float(v), m_eff)
            edge2_effective_m[(u, v)] = m_eff

    diagnostics = {
        "edge1_effective_m": edge1_effective_m,
        "edge2_effective_m": edge2_effective_m,
    }
    return edge1_priors, edge2_priors, diagnostics


def compute_pair_predictions(
    pair_gene_sets: list[tuple[int, int, set[int]]],
    *,
    edge1_priors: dict[tuple[int, int], float],
    edge2_priors: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    predictions: dict[tuple[int, int], float] = {}
    for src, dst, shared_genes in pair_gene_sets:
        total_prob = 0.0
        for gene in shared_genes:
            total_prob += edge1_priors.get((src, gene), 0.0) * edge2_priors.get((gene, dst), 0.0)
        if total_prob > 0:
            predictions[(src, dst)] = float(total_prob)
    return predictions


def compute_pmi(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    mask = (observed > 0) & (predicted > 0)
    return np.log2(observed[mask] / predicted[mask])


def maybe_limit_common_pairs(
    common_pairs: list[tuple[int, int]],
    *,
    max_pairs: int | None,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if max_pairs is None or len(common_pairs) <= max_pairs:
        return common_pairs
    idx = rng.choice(len(common_pairs), size=max_pairs, replace=False)
    return [common_pairs[int(i)] for i in idx]


def maybe_make_plots(
    *,
    results_dir: Path,
    metapath_name: str,
    results_df: pd.DataFrame,
    null_df: pd.DataFrame,
    naive_corr: float,
    degree_aware_corr: float,
    pmi_naive: np.ndarray,
    pmi_degree_aware: np.ndarray,
    het_improvement: float,
    null_improvement: float,
    diagnostics: dict[str, dict[tuple[int, int], float]],
    edge1_nnz: int,
    edge2_nnz: int,
    edge1_type: str,
    edge2_type: str,
) -> None:
    import matplotlib.pyplot as plt

    # Effective m diagnostics
    edge1_m_values = diagnostics.get("edge1_effective_m", {})
    edge2_m_values = diagnostics.get("edge2_effective_m", {})
    if edge1_m_values and edge2_m_values:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        edge1_m_by_degree: dict[int, list[float]] = defaultdict(list)
        for (u, _v), m in edge1_m_values.items():
            edge1_m_by_degree[u].append(m)
        degrees = sorted(edge1_m_by_degree.keys())
        mean_m = [float(np.mean(edge1_m_by_degree[d])) for d in degrees]
        ax.scatter(degrees, mean_m, alpha=0.6, s=30, edgecolors="black", linewidths=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(edge1_nnz, color="red", linestyle="--", alpha=0.6, label=f"global m={edge1_nnz}")
        ax.set_title(f"Effective m vs Degree: {edge1_type}")
        ax.set_xlabel("Source Degree")
        ax.set_ylabel("Mean Effective m")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        edge2_m_by_degree: dict[int, list[float]] = defaultdict(list)
        for (u, _v), m in edge2_m_values.items():
            edge2_m_by_degree[u].append(m)
        degrees = sorted(edge2_m_by_degree.keys())
        mean_m = [float(np.mean(edge2_m_by_degree[d])) for d in degrees]
        ax.scatter(degrees, mean_m, alpha=0.6, s=30, edgecolors="black", linewidths=0.4, color="green")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(edge2_nnz, color="red", linestyle="--", alpha=0.6, label=f"global m={edge2_nnz}")
        ax.set_title(f"Effective m vs Degree: {edge2_type}")
        ax.set_xlabel("Source Degree")
        ax.set_ylabel("Mean Effective m")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(results_dir / f"{metapath_name}_effective_m_diagnostics.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Main analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    y_true = results_df["observed"].to_numpy()
    y_naive = results_df["naive_pred"].to_numpy()
    y_da = results_df["degree_aware_pred"].to_numpy()

    ax = axes[0, 0]
    ax.scatter(y_true, y_naive, alpha=0.4, s=10, edgecolors="none")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.8, linewidth=2)
    ax.set_title(f"Naive Model (r={naive_corr:.3f})")
    ax.set_xlabel("Observed Frequency")
    ax.set_ylabel("Naive Prediction")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(y_true, y_da, alpha=0.4, s=10, edgecolors="none", color="green")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.8, linewidth=2)
    ax.set_title(f"Degree-Aware Model (r={degree_aware_corr:.3f})")
    ax.set_xlabel("Observed Frequency")
    ax.set_ylabel("Degree-Aware Prediction")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.hist(pmi_naive, bins=50, alpha=0.6, label="Naive", edgecolor="black")
    ax.hist(pmi_degree_aware, bins=50, alpha=0.6, label="Degree-Aware", edgecolor="black", color="green")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="PMI=0")
    ax.set_title("PMI Distribution")
    ax.set_xlabel("PMI")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    corr_rows = []
    for bin_label in DEGREE_LABELS_VIZ:
        subset = results_df[results_df["compound_bin"] == bin_label]
        if len(subset) <= 10:
            continue
        corr_rows.append(
            {
                "bin": bin_label,
                "naive": safe_pearson(subset["observed"].to_numpy(), subset["naive_pred"].to_numpy()),
                "degree_aware": safe_pearson(subset["observed"].to_numpy(), subset["degree_aware_pred"].to_numpy()),
                "n": int(len(subset)),
            }
        )
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        x = np.arange(len(corr_df))
        width = 0.35
        ax.bar(x - width / 2, corr_df["naive"], width, label="Naive", alpha=0.7)
        ax.bar(x + width / 2, corr_df["degree_aware"], width, label="Degree-Aware", alpha=0.7, color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(corr_df["bin"], rotation=45, ha="right")
        ax.set_title("Correlation by Compound Degree Bin")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.hist(null_df["improvement"], bins=15, alpha=0.7, edgecolor="black", color="blue")
    ax.axvline(het_improvement, color="red", linestyle="--", linewidth=2, label=f"Hetionet={het_improvement:+.3f}")
    ax.axvline(null_improvement, color="blue", linestyle="--", linewidth=2, label=f"Null mean={null_improvement:+.3f}")
    ax.set_title("Improvement Distribution")
    ax.set_xlabel("Degree-Aware - Naive Correlation")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    sample_df = results_df.sample(min(5000, len(results_df)), random_state=42) if len(results_df) > 0 else results_df
    improvement = sample_df["degree_aware_pred"] - sample_df["naive_pred"]
    sc = ax.scatter(
        sample_df["compound_degree"],
        sample_df["pathway_degree"],
        c=improvement,
        cmap="RdYlGn",
        alpha=0.6,
        s=10,
        edgecolors="none",
        vmin=-0.1,
        vmax=0.1,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Improvement by Degree Pair")
    ax.set_xlabel("Compound Degree")
    ax.set_ylabel("Pathway Degree")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.savefig(results_dir / f"{metapath_name}_degree_aware_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_single_network(
    *,
    edge1: sp.csr_matrix,
    edge2: sp.csr_matrix,
    window_pct: float,
    min_window: int,
    min_edge_count: int,
    max_pairs: int | None,
    rng: np.random.Generator,
    return_diagnostics: bool = False,
) -> tuple[dict[str, float | np.ndarray | pd.DataFrame], dict[str, dict[tuple[int, int], float]] | None]:
    observed, pair_gene_sets = compute_observed_and_pairs(edge1, edge2)
    edge1_priors_naive, edge2_priors_naive = compute_priors_naive(edge1, edge2)
    edge1_priors_da, edge2_priors_da, diagnostics = compute_priors_degree_aware(
        edge1,
        edge2,
        window_pct=window_pct,
        min_window=min_window,
        min_edge_count=min_edge_count,
    )

    naive_pred = compute_pair_predictions(pair_gene_sets, edge1_priors=edge1_priors_naive, edge2_priors=edge2_priors_naive)
    da_pred = compute_pair_predictions(pair_gene_sets, edge1_priors=edge1_priors_da, edge2_priors=edge2_priors_da)

    common_pairs = list(set(observed) & set(naive_pred) & set(da_pred))
    common_pairs = maybe_limit_common_pairs(common_pairs, max_pairs=max_pairs, rng=rng)
    if len(common_pairs) < 2:
        raise RuntimeError("Not enough common pairs for correlation metrics.")

    y_true = np.array([observed[p] for p in common_pairs], dtype=float)
    y_naive = np.array([naive_pred[p] for p in common_pairs], dtype=float)
    y_da = np.array([da_pred[p] for p in common_pairs], dtype=float)

    naive_corr = safe_pearson(y_true, y_naive)
    da_corr = safe_pearson(y_true, y_da)
    naive_rmse = float(np.sqrt(mean_squared_error(y_true, y_naive)))
    da_rmse = float(np.sqrt(mean_squared_error(y_true, y_da)))
    naive_mae = float(mean_absolute_error(y_true, y_naive))
    da_mae = float(mean_absolute_error(y_true, y_da))
    pmi_naive = compute_pmi(y_true, y_naive)
    pmi_da = compute_pmi(y_true, y_da)

    compound_degrees = np.asarray(edge1.sum(axis=1)).ravel()
    pathway_degrees = np.asarray(edge2.sum(axis=0)).ravel()
    results_df = pd.DataFrame(
        {
            "compound_idx": [p[0] for p in common_pairs],
            "pathway_idx": [p[1] for p in common_pairs],
            "observed": y_true,
            "naive_pred": y_naive,
            "degree_aware_pred": y_da,
        }
    )
    results_df["compound_degree"] = results_df["compound_idx"].map(lambda i: int(compound_degrees[int(i)]))
    results_df["pathway_degree"] = results_df["pathway_idx"].map(lambda i: int(pathway_degrees[int(i)]))
    results_df["pmi_naive"] = np.log2(results_df["observed"] / results_df["naive_pred"])
    results_df["pmi_degree_aware"] = np.log2(results_df["observed"] / results_df["degree_aware_pred"])
    results_df["compound_bin"] = pd.cut(results_df["compound_degree"], bins=DEGREE_BINS_VIZ, labels=DEGREE_LABELS_VIZ)
    results_df["pathway_bin"] = pd.cut(results_df["pathway_degree"], bins=DEGREE_BINS_VIZ, labels=DEGREE_LABELS_VIZ)

    metrics = {
        "naive_corr": float(naive_corr),
        "degree_aware_corr": float(da_corr),
        "improvement": float(da_corr - naive_corr),
        "naive_rmse": naive_rmse,
        "degree_aware_rmse": da_rmse,
        "naive_mae": naive_mae,
        "degree_aware_mae": da_mae,
        "n_pairs": int(len(common_pairs)),
        "pmi_naive_mean": float(np.mean(pmi_naive)),
        "pmi_degree_aware_mean": float(np.mean(pmi_da)),
        "results_df": results_df,
        "pmi_naive": pmi_naive,
        "pmi_degree_aware": pmi_da,
    }
    return metrics, diagnostics if return_diagnostics else None


def main() -> int:
    args = parse_args()
    edge1_type, edge2_type = resolve_edge_types(args)
    rng = np.random.default_rng(args.random_seed)
    perm_ids = list(range(args.perm_start, args.perm_end + 1))

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Degree-Aware Compositional Model (Notebook 12 Migration)")
    print("=" * 80)
    print(f"Metapath: {args.metapath} ({edge1_type} -> {edge2_type})")
    print(f"Hetionet id: {args.hetionet_id:03d}")
    print(f"Null permutations: {args.perm_start:03d}-{args.perm_end:03d}")
    print(
        f"Degree window: ±{args.degree_window_pct * 100:.0f}% (min {args.min_window}), "
        f"min edge count: {args.min_edge_count}"
    )
    if args.max_pairs is not None:
        print(f"Pair cap: {args.max_pairs}")

    print("\nAnalyzing Hetionet...")
    edge1_het = load_edge_matrix(args.data_dir, edge1_type, args.hetionet_id)
    edge2_het = load_edge_matrix(args.data_dir, edge2_type, args.hetionet_id)
    het_metrics, diagnostics = analyze_single_network(
        edge1=edge1_het,
        edge2=edge2_het,
        window_pct=args.degree_window_pct,
        min_window=args.min_window,
        min_edge_count=args.min_edge_count,
        max_pairs=args.max_pairs,
        rng=rng,
        return_diagnostics=True,
    )
    results_df = het_metrics["results_df"]  # type: ignore[assignment]
    pmi_naive = het_metrics["pmi_naive"]  # type: ignore[assignment]
    pmi_degree_aware = het_metrics["pmi_degree_aware"]  # type: ignore[assignment]

    print(f"  pairs: {het_metrics['n_pairs']:,}")
    print(f"  naive r: {het_metrics['naive_corr']:.4f}")
    print(f"  degree-aware r: {het_metrics['degree_aware_corr']:.4f}")
    print(f"  improvement: {het_metrics['improvement']:+.4f}")

    print("\nAnalyzing null networks...")
    null_rows: list[dict[str, float | int]] = []
    for perm_id in tqdm(perm_ids, desc="Permutations"):
        edge1_perm = load_edge_matrix(args.data_dir, edge1_type, perm_id)
        edge2_perm = load_edge_matrix(args.data_dir, edge2_type, perm_id)
        try:
            m, _ = analyze_single_network(
                edge1=edge1_perm,
                edge2=edge2_perm,
                window_pct=args.degree_window_pct,
                min_window=args.min_window,
                min_edge_count=args.min_edge_count,
                max_pairs=args.max_pairs,
                rng=rng,
                return_diagnostics=False,
            )
        except RuntimeError:
            continue
        null_rows.append(
            {
                "perm_id": int(perm_id),
                "naive_corr": float(m["naive_corr"]),
                "degree_aware_corr": float(m["degree_aware_corr"]),
                "improvement": float(m["improvement"]),
                "n_pairs": int(m["n_pairs"]),
            }
        )

    if not null_rows:
        raise RuntimeError("No null permutation results were produced.")

    null_df = pd.DataFrame(null_rows)
    het_improvement = float(het_metrics["improvement"])
    null_improvement = float(null_df["improvement"].mean())

    summary = {
        "metapath": args.metapath,
        "edge_types": [edge1_type, edge2_type],
        "hetionet_naive_corr": float(het_metrics["naive_corr"]),
        "hetionet_degree_aware_corr": float(het_metrics["degree_aware_corr"]),
        "hetionet_improvement": het_improvement,
        "hetionet_naive_rmse": float(het_metrics["naive_rmse"]),
        "hetionet_degree_aware_rmse": float(het_metrics["degree_aware_rmse"]),
        "hetionet_naive_mae": float(het_metrics["naive_mae"]),
        "hetionet_degree_aware_mae": float(het_metrics["degree_aware_mae"]),
        "hetionet_naive_pmi_mean": float(het_metrics["pmi_naive_mean"]),
        "hetionet_degree_aware_pmi_mean": float(het_metrics["pmi_degree_aware_mean"]),
        "null_naive_corr_mean": float(null_df["naive_corr"].mean()),
        "null_degree_aware_corr_mean": float(null_df["degree_aware_corr"].mean()),
        "null_improvement_mean": null_improvement,
        "improvement_difference": float(het_improvement - null_improvement),
        "n_hetionet_pairs": int(het_metrics["n_pairs"]),
        "n_null_networks": int(len(null_df)),
        "perm_range": [args.perm_start, args.perm_end],
        "degree_window_pct": float(args.degree_window_pct),
        "min_window": int(args.min_window),
        "min_edge_count": int(args.min_edge_count),
        "max_pairs": args.max_pairs,
    }

    hetionet_file = args.results_dir / f"{args.metapath}_hetionet_results.csv"
    null_file = args.results_dir / f"{args.metapath}_null_results.csv"
    summary_csv = args.results_dir / f"{args.metapath}_summary.csv"
    summary_json = args.results_dir / f"{args.metapath}_summary.json"
    diagnostics_csv = args.results_dir / f"{args.metapath}_effective_m_diagnostics.csv"

    results_df.to_csv(hetionet_file, index=False)
    null_df.to_csv(null_file, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2))

    if diagnostics is not None:
        drows: list[dict[str, float | int | str]] = []
        for model_edge, mapping in diagnostics.items():
            for (u, v), m_eff in mapping.items():
                drows.append(
                    {
                        "edge_model": model_edge,
                        "source_degree": int(u),
                        "target_degree": int(v),
                        "effective_m": float(m_eff),
                    }
                )
        pd.DataFrame(drows).to_csv(diagnostics_csv, index=False)

    if not args.skip_plots:
        maybe_make_plots(
            results_dir=args.results_dir,
            metapath_name=args.metapath,
            results_df=results_df,
            null_df=null_df,
            naive_corr=float(het_metrics["naive_corr"]),
            degree_aware_corr=float(het_metrics["degree_aware_corr"]),
            pmi_naive=pmi_naive,  # type: ignore[arg-type]
            pmi_degree_aware=pmi_degree_aware,  # type: ignore[arg-type]
            het_improvement=het_improvement,
            null_improvement=null_improvement,
            diagnostics=diagnostics or {"edge1_effective_m": {}, "edge2_effective_m": {}},
            edge1_nnz=edge1_het.nnz,
            edge2_nnz=edge2_het.nnz,
            edge1_type=edge1_type,
            edge2_type=edge2_type,
        )

    print("\nSummary:")
    print(f"  hetionet naive r: {summary['hetionet_naive_corr']:.4f}")
    print(f"  hetionet degree-aware r: {summary['hetionet_degree_aware_corr']:.4f}")
    print(f"  hetionet improvement: {summary['hetionet_improvement']:+.4f}")
    print(f"  null mean improvement: {summary['null_improvement_mean']:+.4f}")
    print(f"  difference: {summary['improvement_difference']:+.4f}")
    print("\nSaved outputs:")
    print(f"  - {hetionet_file}")
    print(f"  - {null_file}")
    print(f"  - {summary_csv}")
    print(f"  - {summary_json}")
    print(f"  - {diagnostics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
