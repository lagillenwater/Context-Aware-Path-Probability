"""Run positive-anomaly detection with DWPC comparison (notebook 18h migration)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, pearsonr, spearmanr

REPO_DIR = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(REPO_DIR / ".cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(REPO_DIR / ".cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.intermediate_signatures import assign_to_bins, create_degree_bins
from src.node_labels import load_node_labels
from src.pathway_model_io import load_degree_sig_nn, predict_degree_sig_nn

DEFAULT_METAPATHS: dict[str, tuple[str, str]] = {
    "CbGpPW": ("CbG", "GpPW"),
    "CtDaG": ("CtD", "DaG"),
    "CbGaD": ("CbG", "GaD"),
    "CrCbG": ("CrC", "CbG"),
    "CbGiG": ("CbG", "GiG"),
    "CpDaG": ("CpD", "DaG"),
    "CbGpBP": ("CbG", "GpBP"),
    "CbGpCC": ("CbG", "GpCC"),
}


@dataclass(frozen=True)
class MetapathSpec:
    metapath: str
    edge1_type: str
    edge2_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect positive anomalies (enrichment) using Degree Sig NN expected counts "
            "and permutation-derived variance estimates. Script-first replacement for notebook 18h."
        )
    )
    parser.add_argument(
        "--metapath",
        action="append",
        default=[],
        help=(
            "Metapath key to process (repeatable). "
            f"Supported defaults: {', '.join(DEFAULT_METAPATHS.keys())}"
        ),
    )
    parser.add_argument(
        "--all-metapaths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Process all default metapaths.",
    )
    parser.add_argument("--edge1-type", default=None, help="Override edge1 type for custom metapath run.")
    parser.add_argument("--edge2-type", default=None, help="Override edge2 type for custom metapath run.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Data directory containing edges/ and nodes/.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "trained_models",
        help="Directory containing <metapath>_Degree_Sig_NN.pt checkpoints.",
    )
    parser.add_argument(
        "--variance-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "variance_analysis",
        help="Directory containing <metapath>_variance_estimates.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "anomaly_detection",
        help="Output directory for anomaly-detection artifacts.",
    )
    parser.add_argument("--significance-threshold", type=float, default=0.01, help="One-tailed p-value threshold.")
    parser.add_argument("--min-pathway-count", type=float, default=1.0, help="Minimum observed path count to include.")
    parser.add_argument("--n-degree-bins", type=int, default=10, help="Degree bin count.")
    parser.add_argument("--n-inter-bins", type=int, default=10, help="Intermediate signature bin count.")
    parser.add_argument("--dwpc-damping", type=float, default=0.4, help="DWPC damping exponent.")
    parser.add_argument("--predict-batch-size", type=int, default=4096, help="Batch size for model predictions.")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on candidate pathway pairs for smoke testing.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Torch device used for prediction.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed (recorded in metadata).")
    parser.add_argument(
        "--skip-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip plot generation (CSV/JSON outputs still written).",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue to remaining metapaths if one fails.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.significance_threshold <= 0 or args.significance_threshold >= 1:
        raise ValueError("--significance-threshold must be in (0, 1)")
    if args.min_pathway_count < 0:
        raise ValueError("--min-pathway-count must be >= 0")
    if args.n_degree_bins <= 0:
        raise ValueError("--n-degree-bins must be > 0")
    if args.n_inter_bins <= 0:
        raise ValueError("--n-inter-bins must be > 0")
    if args.dwpc_damping < 0:
        raise ValueError("--dwpc-damping must be >= 0")
    if args.predict_batch_size <= 0:
        raise ValueError("--predict-batch-size must be > 0")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be > 0 when provided")
    if (args.edge1_type is None) ^ (args.edge2_type is None):
        raise ValueError("Provide both --edge1-type and --edge2-type together.")
    if args.edge1_type is not None and args.all_metapaths:
        raise ValueError("--edge1-type/--edge2-type cannot be combined with --all-metapaths.")
    if args.edge1_type is not None and len(args.metapath) > 1:
        raise ValueError("--edge1-type/--edge2-type supports one metapath label at a time.")
    unknown = [name for name in args.metapath if name not in DEFAULT_METAPATHS]
    if unknown and args.edge1_type is None:
        raise ValueError(
            f"Unknown --metapath values: {', '.join(unknown)}. "
            f"Supported: {', '.join(DEFAULT_METAPATHS.keys())}"
        )


def resolve_metapaths(args: argparse.Namespace) -> list[MetapathSpec]:
    if args.edge1_type is not None:
        label = args.metapath[0] if args.metapath else "custom"
        return [MetapathSpec(metapath=label, edge1_type=args.edge1_type, edge2_type=args.edge2_type)]

    if args.all_metapaths:
        names = list(DEFAULT_METAPATHS.keys())
    elif args.metapath:
        names = args.metapath
    else:
        names = ["CbGpPW"]

    return [
        MetapathSpec(metapath=name, edge1_type=DEFAULT_METAPATHS[name][0], edge2_type=DEFAULT_METAPATHS[name][1])
        for name in names
    ]


def select_device(device_arg: str) -> str:
    if device_arg == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
    return device_arg


def load_edge_matrix(data_dir: Path, edge_type: str) -> sp.spmatrix:
    primary = data_dir / "edges" / f"{edge_type}.sparse.npz"
    fallback = data_dir / "permutations" / "000.hetmat" / "edges" / f"{edge_type}.sparse.npz"
    if primary.exists():
        return sp.load_npz(str(primary))
    if fallback.exists():
        return sp.load_npz(str(fallback))
    raise FileNotFoundError(f"Edge file not found for {edge_type}: checked {primary} and {fallback}")


def infer_node_types(data_dir: Path, edge1_type: str, edge2_type: str) -> tuple[str | None, str | None]:
    metagraph_file = data_dir / "metagraph.json"
    nodes_dir = data_dir / "nodes"
    if not metagraph_file.exists() or not nodes_dir.exists():
        return None, None

    try:
        payload = json.loads(metagraph_file.read_text())
        kind_to_abbrev = payload.get("kind_to_abbrev", {})
    except Exception:
        return None, None

    node_types = {p.stem for p in nodes_dir.glob("*.tsv")}
    node_abbrev_to_type = {
        str(abbrev): str(kind)
        for kind, abbrev in kind_to_abbrev.items()
        if kind in node_types and isinstance(abbrev, str)
    }
    if not node_abbrev_to_type:
        return None, None

    abbrevs = sorted(node_abbrev_to_type.keys(), key=len, reverse=True)

    source_type = None
    for abbrev in abbrevs:
        if edge1_type.startswith(abbrev):
            source_type = node_abbrev_to_type[abbrev]
            break

    target_type = None
    for abbrev in abbrevs:
        if edge2_type.endswith(abbrev):
            target_type = node_abbrev_to_type[abbrev]
            break

    return source_type, target_type


def safe_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    try:
        pearson_r, pearson_p = pearsonr(x, y)
    except Exception:
        pearson_r, pearson_p = float("nan"), float("nan")
    try:
        spearman_r, spearman_p = spearmanr(x, y)
    except Exception:
        spearman_r, spearman_p = float("nan"), float("nan")
    return float(pearson_r), float(pearson_p), float(spearman_r), float(spearman_p)


def compute_source_signatures(
    edge1_matrix: sp.spmatrix,
    intermediate_in_degrees: np.ndarray,
    intermediate_out_degrees: np.ndarray,
    source_indices: np.ndarray,
    n_inter_bins: int,
) -> dict[int, np.ndarray]:
    signatures: dict[int, np.ndarray] = {}
    edge1_csr = edge1_matrix.tocsr()

    for src_idx in np.unique(source_indices):
        intermediates = edge1_csr.getrow(int(src_idx)).indices
        if len(intermediates) == 0:
            signatures[int(src_idx)] = np.zeros(n_inter_bins * n_inter_bins, dtype=np.float32)
            continue

        in_values = intermediate_in_degrees[intermediates]
        out_values = intermediate_out_degrees[intermediates]

        in_edges = np.percentile(in_values, np.linspace(0, 100, n_inter_bins + 1))
        out_edges = np.percentile(out_values, np.linspace(0, 100, n_inter_bins + 1))

        in_bins = np.digitize(in_values, in_edges) - 1
        out_bins = np.digitize(out_values, out_edges) - 1

        in_bins = np.clip(in_bins, 0, n_inter_bins - 1)
        out_bins = np.clip(out_bins, 0, n_inter_bins - 1)

        hist = np.zeros((n_inter_bins, n_inter_bins), dtype=np.float32)
        for ib, ob in zip(in_bins, out_bins):
            hist[int(ib), int(ob)] += 1.0
        total = float(hist.sum())
        if total > 0:
            hist /= total
        signatures[int(src_idx)] = hist.ravel()

    return signatures


def compute_dwpc_matrix(edge1_matrix: sp.spmatrix, edge2_matrix: sp.spmatrix, damping: float) -> sp.spmatrix:
    edge1_csr = edge1_matrix.tocsr().astype(np.float64)
    edge2_csr = edge2_matrix.tocsr().astype(np.float64)

    edge1_degrees = np.asarray(edge1_csr.sum(axis=1)).ravel().astype(float)
    edge2_degrees = np.asarray(edge2_csr.sum(axis=1)).ravel().astype(float)

    w1 = np.zeros_like(edge1_degrees)
    mask1 = edge1_degrees > 0
    w1[mask1] = np.power(edge1_degrees[mask1], -damping)

    w2 = np.zeros_like(edge2_degrees)
    mask2 = edge2_degrees > 0
    w2[mask2] = np.power(edge2_degrees[mask2], -damping)

    edge1_csr.data *= np.repeat(w1, np.diff(edge1_csr.indptr))
    edge2_csr.data *= np.repeat(w2, np.diff(edge2_csr.indptr))

    return edge1_csr @ edge2_csr


def plot_dwpc_comparison(
    anomaly_df: pd.DataFrame,
    *,
    significance_threshold: float,
    high_dwpc_threshold: float,
    high_z_threshold: float,
    overlap: int,
    pearson_r: float,
    output_file: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    colors = ["red" if s else "orange" for s in anomaly_df["significant"]]
    axes[0, 0].scatter(anomaly_df["dwpc_score"], anomaly_df["z_score"], c=colors, alpha=0.5, s=20, edgecolors="none")
    axes[0, 0].set_xlabel("DWPC Score")
    axes[0, 0].set_ylabel("Anomaly Z-score (Enrichment)")
    axes[0, 0].set_title(f"DWPC vs Anomaly Score (Positive Only)\n(r = {pearson_r:.4f})")
    axes[0, 0].grid(True, alpha=0.3)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label=f"Significant (p<{significance_threshold})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=8, label="Non-significant"),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="best", fontsize=9)

    high_dwpc = anomaly_df["dwpc_score"] >= high_dwpc_threshold
    high_z = anomaly_df["z_score"] >= high_z_threshold
    axes[0, 1].scatter(anomaly_df.loc[~high_dwpc & ~high_z, "dwpc_score"], anomaly_df.loc[~high_dwpc & ~high_z, "z_score"], c="lightgray", alpha=0.3, s=10, label="Lower/Lower")
    axes[0, 1].scatter(anomaly_df.loc[high_dwpc & ~high_z, "dwpc_score"], anomaly_df.loc[high_dwpc & ~high_z, "z_score"], c="blue", alpha=0.6, s=20, label="High DWPC/Lower Z")
    axes[0, 1].scatter(anomaly_df.loc[~high_dwpc & high_z, "dwpc_score"], anomaly_df.loc[~high_dwpc & high_z, "z_score"], c="orange", alpha=0.6, s=20, label="Lower DWPC/High Z (Novel)")
    axes[0, 1].scatter(anomaly_df.loc[high_dwpc & high_z, "dwpc_score"], anomaly_df.loc[high_dwpc & high_z, "z_score"], c="green", alpha=0.6, s=20, label="High DWPC/High Z")
    axes[0, 1].axhline(high_z_threshold, color="red", linestyle="--", lw=1)
    axes[0, 1].axvline(high_dwpc_threshold, color="red", linestyle="--", lw=1)
    axes[0, 1].set_xlabel("DWPC Score")
    axes[0, 1].set_ylabel("Anomaly Z-score (Enrichment)")
    axes[0, 1].set_title("Quadrant Analysis (Enriched Pairs)")
    axes[0, 1].legend(loc="best", fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    top_n = min(100, len(anomaly_df))
    top_dwpc = anomaly_df.nlargest(top_n, "dwpc_score").copy()
    top_dwpc["rank_dwpc"] = np.arange(1, len(top_dwpc) + 1)
    top_z = anomaly_df.nlargest(top_n, "z_score").copy()
    top_z["rank_z"] = np.arange(1, len(top_z) + 1)
    rank_comparison = top_dwpc.merge(top_z, on=["source_idx", "target_idx"], how="outer", suffixes=("_dwpc", "_z"))
    axes[1, 0].scatter(rank_comparison["rank_dwpc"], rank_comparison["rank_z"], alpha=0.6, s=30, edgecolors="k", linewidth=0.5)
    axes[1, 0].plot([1, top_n], [1, top_n], "r--", lw=2, label="Perfect agreement")
    axes[1, 0].set_xlabel("DWPC Rank")
    axes[1, 0].set_ylabel("Z-score Rank")
    axes[1, 0].set_title(f"Rank Comparison (Top {top_n} Enriched)\nOverlap: {overlap}/{top_n}")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(anomaly_df["dwpc_score"], bins=50, alpha=0.5, label="DWPC", color="blue", edgecolor="black")
    ax2 = axes[1, 1].twinx()
    ax2.hist(anomaly_df["z_score"], bins=50, alpha=0.5, label="Z-score", color="red", edgecolor="black")
    axes[1, 1].set_xlabel("DWPC Score", color="blue")
    axes[1, 1].set_ylabel("Frequency (DWPC)", color="blue")
    ax2.set_ylabel("Frequency (Z-score)", color="red")
    axes[1, 1].set_title("Score Distributions (Positive Anomalies)")
    axes[1, 1].tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")
    axes[1, 1].legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_volcano(anomaly_df: pd.DataFrame, *, significance_threshold: float, metapath: str, output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["red" if s else "orange" for s in anomaly_df["significant"]]
    minus_log10_p = -np.log10(anomaly_df["p_value"].clip(lower=1e-100))

    ax.scatter(anomaly_df["z_score"], minus_log10_p, c=colors, alpha=0.6, s=30, edgecolors="k", linewidth=0.3)
    sig_line = -np.log10(significance_threshold)
    ax.axhline(sig_line, color="blue", linestyle="--", lw=2, label=f"p = {significance_threshold}")
    ax.set_xlabel("Anomaly Z-score (Enrichment)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(f"Volcano Plot: {metapath} (Positive Anomalies Only)\n{int(anomaly_df['significant'].sum()):,} significant enrichments")
    legend_elements = [
        Patch(facecolor="red", edgecolor="k", label="Significant enrichment"),
        Patch(facecolor="orange", edgecolor="k", label="Non-significant enrichment"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(
    anomaly_df: pd.DataFrame,
    *,
    significance_threshold: float,
    bonferroni_threshold: float,
    output_file: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].hist(anomaly_df["z_score"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Anomaly Z-score (Enrichment)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title(f"Z-score Distribution\nmean={anomaly_df['z_score'].mean():.3f}, std={anomaly_df['z_score'].std():.3f}")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(anomaly_df["p_value"], bins=50, color="orange", edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(significance_threshold, color="red", linestyle="--", lw=2, label=f"p={significance_threshold}")
    axes[0, 1].set_xlabel("P-value (one-tailed)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("P-value Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    colors_scatter = ["red" if s else "gray" for s in anomaly_df["significant"]]
    axes[1, 0].scatter(anomaly_df["expected_count"], anomaly_df["actual_count"], c=colors_scatter, alpha=0.5, s=20, edgecolors="none")
    max_val = float(max(anomaly_df["actual_count"].max(), anomaly_df["expected_count"].max()))
    axes[1, 0].plot([0, max_val], [0, max_val], "k--", lw=2, label="actual = expected")
    axes[1, 0].set_xlabel("Expected Count")
    axes[1, 0].set_ylabel("Actual Count")
    axes[1, 0].set_title("Actual vs Expected Pathway Counts")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    sig_counts = {
        f"p < {significance_threshold}": int(anomaly_df["significant"].sum()),
        f"p < {bonferroni_threshold:.2e}": int(anomaly_df["significant_bonferroni"].sum()),
        "All enriched": int(len(anomaly_df)),
    }
    axes[1, 1].bar(range(len(sig_counts)), list(sig_counts.values()), color=["red", "darkred", "gray"], edgecolor="black", alpha=0.7)
    axes[1, 1].set_xticks(range(len(sig_counts)))
    axes[1, 1].set_xticklabels(list(sig_counts.keys()), rotation=15, ha="right")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Significance Levels")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_one_metapath(spec: MetapathSpec, args: argparse.Namespace, *, device: str) -> dict[str, Any]:
    model_file = args.model_dir / f"{spec.metapath}_Degree_Sig_NN.pt"
    variance_file = args.variance_dir / f"{spec.metapath}_variance_estimates.csv"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_file}. "
            "Run pathway-train-degree-signature-nn first."
        )
    if not variance_file.exists():
        raise FileNotFoundError(
            f"Variance estimates not found: {variance_file}. "
            "Run pathway-variance-estimation first."
        )

    model, model_metadata = load_degree_sig_nn(model_file, device=device)
    variance_df = pd.read_csv(variance_file)
    expected_input_dim = int(getattr(model, "input_dim", 0))
    configured_input_dim = 2 + (args.n_inter_bins * args.n_inter_bins)
    runtime_inter_bins = args.n_inter_bins
    if expected_input_dim and expected_input_dim != configured_input_dim:
        inferred_side = int(round((expected_input_dim - 2) ** 0.5))
        if inferred_side > 0 and (2 + inferred_side * inferred_side) == expected_input_dim:
            runtime_inter_bins = inferred_side
            print(
                "Model/input mismatch detected; "
                f"using n_inter_bins={runtime_inter_bins} from checkpoint input_dim={expected_input_dim}."
            )
        else:
            raise ValueError(
                "Model input dimension does not match configured n_inter_bins and cannot be inferred: "
                f"expected_input_dim={expected_input_dim}, configured_input_dim={configured_input_dim}"
            )

    edge1_matrix = load_edge_matrix(args.data_dir, spec.edge1_type)
    edge2_matrix = load_edge_matrix(args.data_dir, spec.edge2_type)

    if edge1_matrix.dtype == bool or edge1_matrix.dtype == np.bool_:
        edge1_matrix = edge1_matrix.astype(np.int32)
    if edge2_matrix.dtype == bool or edge2_matrix.dtype == np.bool_:
        edge2_matrix = edge2_matrix.astype(np.int32)

    pathway_matrix = edge1_matrix @ edge2_matrix

    source_degrees = np.asarray(edge1_matrix.sum(axis=1)).ravel().astype(float)
    target_degrees = np.asarray(edge2_matrix.sum(axis=1)).ravel().astype(float)
    source_bins = create_degree_bins(source_degrees, args.n_degree_bins)
    target_bins = create_degree_bins(target_degrees, args.n_degree_bins)
    source_bin_assignments = assign_to_bins(source_degrees, source_bins)
    target_bin_assignments = assign_to_bins(target_degrees, target_bins)

    pathway_coo = pathway_matrix.tocoo()
    rows = pathway_coo.row.astype(int)
    cols = pathway_coo.col.astype(int)
    actual_counts = pathway_coo.data.astype(float)

    keep = actual_counts >= args.min_pathway_count
    rows = rows[keep]
    cols = cols[keep]
    actual_counts = actual_counts[keep]

    if args.max_pairs is not None and len(rows) > args.max_pairs:
        rows = rows[: args.max_pairs]
        cols = cols[: args.max_pairs]
        actual_counts = actual_counts[: args.max_pairs]

    if len(rows) == 0:
        raise RuntimeError("No pathway pairs passed filters for anomaly detection.")

    src_bins = source_bin_assignments[rows].astype(int)
    tgt_bins = target_bin_assignments[cols].astype(int)

    intermediate_in_degrees = np.asarray(edge1_matrix.sum(axis=0)).ravel().astype(float)
    intermediate_out_degrees = np.asarray(edge2_matrix.sum(axis=1)).ravel().astype(float)
    source_signatures = compute_source_signatures(
        edge1_matrix,
        intermediate_in_degrees,
        intermediate_out_degrees,
        rows,
        runtime_inter_bins,
    )

    feature_dim = runtime_inter_bins * runtime_inter_bins
    X = np.zeros((len(rows), 2 + feature_dim), dtype=np.float32)
    X[:, 0] = src_bins
    X[:, 1] = tgt_bins
    for i, src_idx in enumerate(rows):
        X[i, 2:] = source_signatures[int(src_idx)]

    expected_counts = predict_degree_sig_nn(
        model,
        X,
        device=device,
        batch_size=args.predict_batch_size,
    ).astype(float)

    positive_mask = actual_counts > expected_counts
    rows_pos = rows[positive_mask]
    cols_pos = cols[positive_mask]
    src_bins_pos = src_bins[positive_mask]
    tgt_bins_pos = tgt_bins[positive_mask]
    actual_pos = actual_counts[positive_mask]
    expected_pos = expected_counts[positive_mask]

    if len(rows_pos) == 0:
        raise RuntimeError("No positive anomalies found (actual_count > expected_count).")

    variance_map = {
        (int(r.source_bin), int(r.target_bin)): float(r.std_count_across_perms)
        for r in variance_df.itertuples(index=False)
    }
    fallback_std = float(variance_df["std_count_across_perms"].mean())

    expected_std = np.array(
        [variance_map.get((int(sb), int(tb)), fallback_std) for sb, tb in zip(src_bins_pos, tgt_bins_pos)],
        dtype=float,
    )
    expected_std_safe = np.where(expected_std > 0, expected_std, fallback_std if fallback_std > 0 else 1.0)

    z_scores = (actual_pos - expected_pos) / expected_std_safe
    p_values = 1.0 - norm.cdf(z_scores)

    anomaly_df = pd.DataFrame(
        {
            "source_idx": rows_pos.astype(int),
            "target_idx": cols_pos.astype(int),
            "source_degree": source_degrees[rows_pos],
            "target_degree": target_degrees[cols_pos],
            "source_bin": src_bins_pos.astype(int),
            "target_bin": tgt_bins_pos.astype(int),
            "actual_count": actual_pos,
            "expected_count": expected_pos,
            "expected_std": expected_std,
            "z_score": z_scores,
            "p_value": p_values,
        }
    )

    n_tests = len(anomaly_df)
    bonferroni_threshold = args.significance_threshold / n_tests if n_tests > 0 else float("inf")
    anomaly_df["significant"] = anomaly_df["p_value"] < args.significance_threshold
    anomaly_df["significant_bonferroni"] = anomaly_df["p_value"] < bonferroni_threshold

    source_type, target_type = infer_node_types(args.data_dir, spec.edge1_type, spec.edge2_type)
    if source_type and target_type:
        try:
            source_labels = load_node_labels(args.data_dir, source_type)
            target_labels = load_node_labels(args.data_dir, target_type)
            anomaly_df["source_name"] = anomaly_df["source_idx"].apply(
                lambda idx: source_labels[int(idx)] if int(idx) < len(source_labels) else f"Unknown_{idx}"
            )
            anomaly_df["target_name"] = anomaly_df["target_idx"].apply(
                lambda idx: target_labels[int(idx)] if int(idx) < len(target_labels) else f"Unknown_{idx}"
            )
        except Exception:
            anomaly_df["source_name"] = "Source_" + anomaly_df["source_idx"].astype(str)
            anomaly_df["target_name"] = "Target_" + anomaly_df["target_idx"].astype(str)
    else:
        anomaly_df["source_name"] = "Source_" + anomaly_df["source_idx"].astype(str)
        anomaly_df["target_name"] = "Target_" + anomaly_df["target_idx"].astype(str)

    dwpc_matrix = compute_dwpc_matrix(edge1_matrix, edge2_matrix, args.dwpc_damping).tocsr()
    dwpc_scores = np.asarray(dwpc_matrix[rows_pos, cols_pos]).reshape(-1)
    anomaly_df["dwpc_score"] = dwpc_scores

    pearson_r, pearson_p, spearman_r, spearman_p = safe_correlations(
        anomaly_df["z_score"].to_numpy(),
        anomaly_df["dwpc_score"].to_numpy(),
    )

    high_dwpc_threshold = float(anomaly_df["dwpc_score"].quantile(0.75))
    high_z_threshold = float(anomaly_df["z_score"].quantile(0.75))

    quadrants = {
        "high_dwpc_high_z": int(((anomaly_df["dwpc_score"] >= high_dwpc_threshold) & (anomaly_df["z_score"] >= high_z_threshold)).sum()),
        "high_dwpc_low_z": int(((anomaly_df["dwpc_score"] >= high_dwpc_threshold) & (anomaly_df["z_score"] < high_z_threshold)).sum()),
        "low_dwpc_high_z": int(((anomaly_df["dwpc_score"] < high_dwpc_threshold) & (anomaly_df["z_score"] >= high_z_threshold)).sum()),
        "low_dwpc_low_z": int(((anomaly_df["dwpc_score"] < high_dwpc_threshold) & (anomaly_df["z_score"] < high_z_threshold)).sum()),
    }

    top_n = min(100, len(anomaly_df))
    top_dwpc_indices = set(anomaly_df.nlargest(top_n, "dwpc_score").index)
    top_z_indices = set(anomaly_df.nlargest(top_n, "z_score").index)
    overlap = int(len(top_dwpc_indices & top_z_indices))

    novel_discoveries = anomaly_df[
        (anomaly_df["dwpc_score"] < high_dwpc_threshold)
        & (anomaly_df["z_score"] >= high_z_threshold)
        & (anomaly_df["significant"])
    ].sort_values("z_score", ascending=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_file = args.output_dir / f"{spec.metapath}_all_anomalies.csv"
    significant_file = args.output_dir / f"{spec.metapath}_significant_anomalies.csv"
    novel_file = args.output_dir / f"{spec.metapath}_novel_discoveries.csv"

    anomaly_df.to_csv(all_file, index=False)
    anomaly_df[anomaly_df["significant"]].to_csv(significant_file, index=False)
    novel_discoveries.to_csv(novel_file, index=False)

    threshold_files: list[str] = []
    for p_thresh in [0.005, 0.001, 0.0001]:
        subset = anomaly_df[anomaly_df["p_value"] < p_thresh]
        thresh_str = str(p_thresh).replace(".", "")
        file_path = args.output_dir / f"{spec.metapath}_anomalies_p{thresh_str}.csv"
        subset.to_csv(file_path, index=False)
        threshold_files.append(str(file_path))

    dwpc_plot_file: Path | None = None
    volcano_file: Path | None = None
    dist_file: Path | None = None
    if not args.skip_plots:
        dwpc_plot_file = args.output_dir / f"{spec.metapath}_dwpc_comparison.png"
        volcano_file = args.output_dir / f"{spec.metapath}_volcano_plot.png"
        dist_file = args.output_dir / f"{spec.metapath}_anomaly_distributions.png"

        plot_dwpc_comparison(
            anomaly_df,
            significance_threshold=args.significance_threshold,
            high_dwpc_threshold=high_dwpc_threshold,
            high_z_threshold=high_z_threshold,
            overlap=overlap,
            pearson_r=pearson_r,
            output_file=dwpc_plot_file,
        )
        plot_volcano(
            anomaly_df,
            significance_threshold=args.significance_threshold,
            metapath=spec.metapath,
            output_file=volcano_file,
        )
        plot_distributions(
            anomaly_df,
            significance_threshold=args.significance_threshold,
            bonferroni_threshold=bonferroni_threshold,
            output_file=dist_file,
        )

    summary = {
        "metapath": spec.metapath,
        "edge1_type": spec.edge1_type,
        "edge2_type": spec.edge2_type,
        "n_source_nodes": int(edge1_matrix.shape[0]),
        "n_target_nodes": int(edge2_matrix.shape[1]),
        "n_intermediate_nodes": int(edge1_matrix.shape[1]),
        "total_pairs_after_filters": int(len(rows)),
        "total_enriched_pairs_analyzed": int(len(anomaly_df)),
        "n_significant": int(anomaly_df["significant"].sum()),
        "n_significant_bonferroni": int(anomaly_df["significant_bonferroni"].sum()),
        "n_novel_discoveries": int(len(novel_discoveries)),
        "significance_threshold": float(args.significance_threshold),
        "bonferroni_threshold": float(bonferroni_threshold),
        "dwpc_damping": float(args.dwpc_damping),
        "analysis_type": "positive_anomalies_only",
        "dwpc_vs_zscore_correlation": {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        },
        "quadrant_analysis": quadrants,
        "rank_overlap_top100": overlap,
        "z_score_stats": {
            "mean": float(anomaly_df["z_score"].mean()),
            "std": float(anomaly_df["z_score"].std()),
            "min": float(anomaly_df["z_score"].min()),
            "max": float(anomaly_df["z_score"].max()),
            "q25": float(anomaly_df["z_score"].quantile(0.25)),
            "median": float(anomaly_df["z_score"].median()),
            "q75": float(anomaly_df["z_score"].quantile(0.75)),
        },
        "dwpc_score_stats": {
            "mean": float(anomaly_df["dwpc_score"].mean()),
            "std": float(anomaly_df["dwpc_score"].std()),
            "min": float(anomaly_df["dwpc_score"].min()),
            "max": float(anomaly_df["dwpc_score"].max()),
            "q25": float(anomaly_df["dwpc_score"].quantile(0.25)),
            "median": float(anomaly_df["dwpc_score"].median()),
            "q75": float(anomaly_df["dwpc_score"].quantile(0.75)),
        },
        "model_file": str(model_file),
        "variance_file": str(variance_file),
        "model_metadata": model_metadata,
    }
    summary_file = args.output_dir / f"{spec.metapath}_anomaly_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    return {
        "metapath": spec.metapath,
        "all_file": str(all_file),
        "significant_file": str(significant_file),
        "novel_file": str(novel_file),
        "summary_file": str(summary_file),
        "threshold_files": threshold_files,
        "dwpc_plot_file": str(dwpc_plot_file) if dwpc_plot_file else None,
        "volcano_file": str(volcano_file) if volcano_file else None,
        "dist_file": str(dist_file) if dist_file else None,
        "n_positive": int(len(anomaly_df)),
        "n_significant": int(anomaly_df["significant"].sum()),
    }


def args_to_json(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, list):
            out[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_seed)
    device = select_device(args.device)
    specs = resolve_metapaths(args)

    print(f"Data dir: {args.data_dir}")
    print(f"Model dir: {args.model_dir}")
    print(f"Variance dir: {args.variance_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Metapaths: {', '.join(spec.metapath for spec in specs)}")

    successful: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for spec in specs:
        print("=" * 80)
        print(f"ANOMALY DETECTION: {spec.metapath} ({spec.edge1_type} -> {spec.edge2_type})")
        print("=" * 80)
        try:
            result = run_one_metapath(spec, args, device=device)
            successful.append(result)
            print(
                f"Complete {spec.metapath}: positive={result['n_positive']}, "
                f"significant={result['n_significant']}"
            )
            print(f"Saved all anomalies: {result['all_file']}")
            print(f"Saved summary: {result['summary_file']}")
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            failed.append({"metapath": spec.metapath, "error": message})
            print(f"FAILED {spec.metapath}: {message}")
            if not args.continue_on_error:
                raise

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_file = args.output_dir / "pathway_anomaly_detection_run_summary.json"
    run_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": args_to_json(args),
        "device": device,
        "successful_metapaths": [x["metapath"] for x in successful],
        "failed_metapaths": failed,
        "n_successful": len(successful),
        "n_failed": len(failed),
    }
    run_summary_file.write_text(json.dumps(run_summary, indent=2))
    print(f"Run summary: {run_summary_file}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
