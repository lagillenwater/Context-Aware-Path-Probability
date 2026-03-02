"""Aggregate model-comparison outputs across edge types."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_DIR / ".cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from degree_analysis import identify_small_graphs, run_degree_analysis_pipeline  # noqa: E402


DEFAULT_EDGE_TYPES = [
    "AdG",
    "AeG",
    "AuG",
    "CbG",
    "CcSE",
    "CdG",
    "CpD",
    "CrC",
    "CtD",
    "CuG",
    "DaG",
    "DdG",
    "DlA",
    "DpS",
    "DrD",
    "DuG",
    "GcG",
    "GiG",
    "GpBP",
    "GpCC",
    "GpMF",
    "GpPW",
    "Gr>G",
    "PCiC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize notebook-04-style model comparison outputs across edge types. "
            "This replaces notebook 05 with a script-first pipeline."
        )
    )
    parser.add_argument(
        "--edge-type",
        action="append",
        default=[],
        help="Edge type to include (repeatable). Default: auto-discover from results dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "model_comparison",
        help="Directory containing <edge_type>_results folders.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=REPO_DIR / "results" / "model_comparison_summary_with_degree",
        help="Output directory for aggregated summary files.",
    )
    parser.add_argument(
        "--degree-analysis-dir",
        type=Path,
        default=REPO_DIR / "results" / "degree_analysis_enhanced",
        help="Directory for per-edge degree analysis outputs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Data directory for edge matrix lookups.",
    )
    parser.add_argument(
        "--small-graph-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only small graphs selected via identify_small_graphs.",
    )
    parser.add_argument(
        "--max-edges-small",
        type=int,
        default=10_000,
        help="Maximum edges for small-graph selection.",
    )
    parser.add_argument(
        "--run-degree-analysis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run degree_analysis pipeline for each edge type before aggregation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue when one edge type fails during optional degree analysis.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip summary plot generation.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.max_edges_small <= 0:
        raise ValueError("--max-edges-small must be > 0")


def discover_edge_types(results_dir: Path) -> list[str]:
    discovered: list[str] = []
    if results_dir.exists():
        for d in sorted(results_dir.glob("*_results")):
            name = d.name.replace("_results", "")
            if name:
                discovered.append(name)
    return discovered


def resolve_edge_types(args: argparse.Namespace) -> list[str]:
    if args.edge_type:
        return args.edge_type

    if args.small_graph_mode:
        small = identify_small_graphs(args.data_dir, max_edges=args.max_edges_small)
        return [row["edge_type"] for row in small]

    discovered = discover_edge_types(args.results_dir)
    return discovered if discovered else DEFAULT_EDGE_TYPES


def load_graph_characteristics(data_dir: Path, edge_types: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for edge_type in edge_types:
        edge_file = data_dir / "permutations" / "000.hetmat" / "edges" / f"{edge_type}.sparse.npz"
        if not edge_file.exists():
            edge_file = data_dir / "edges" / f"{edge_type}.sparse.npz"
        if not edge_file.exists():
            continue

        mat = sp.load_npz(str(edge_file))
        src_deg = np.asarray(mat.sum(axis=1)).ravel()
        tgt_deg = np.asarray(mat.sum(axis=0)).ravel()
        rows.append(
            {
                "edge_type": edge_type,
                "n_edges": int(mat.nnz),
                "n_source_nodes": int(mat.shape[0]),
                "n_target_nodes": int(mat.shape[1]),
                "density": float(mat.nnz / (mat.shape[0] * mat.shape[1])),
                "mean_source_degree": float(src_deg[src_deg > 0].mean()) if np.any(src_deg > 0) else 0.0,
                "mean_target_degree": float(tgt_deg[tgt_deg > 0].mean()) if np.any(tgt_deg > 0) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def load_edge_results(results_dir: Path, edge_type: str) -> dict[str, pd.DataFrame]:
    edge_results = results_dir / f"{edge_type}_results"
    if not edge_results.exists():
        return {}

    payload: dict[str, pd.DataFrame] = {}

    file_map = {
        "model_comparison": "model_comparison.csv",
        "analytical_comparison": "models_vs_analytical_comparison.csv",
        "empirical_comparison": "test_vs_empirical_comparison.csv",
        "analytical_vs_empirical": "analytical_vs_empirical_comparison.csv",
    }

    for key, filename in file_map.items():
        fp = edge_results / filename
        if fp.exists():
            try:
                payload[key] = pd.read_csv(fp)
            except Exception:
                continue

    return payload


def extract_edge_and_model_from_metrics_path(path: Path) -> tuple[str | None, str | None]:
    stem = path.stem
    if stem.endswith("_degree_metrics"):
        stem = stem[: -len("_degree_metrics")]

    for edge_type in DEFAULT_EDGE_TYPES:
        prefix = f"{edge_type}_"
        if stem.startswith(prefix):
            model_part = stem[len(prefix) :]
            return edge_type, model_part.replace("_", " ")

    parent = path.parent.name
    for edge_type in DEFAULT_EDGE_TYPES:
        prefix = f"{edge_type}_"
        if parent.startswith(prefix):
            model_part = parent[len(prefix) :]
            return edge_type, model_part.replace("_", " ")

    return None, None


def aggregate_degree_metrics(degree_analysis_dir: Path) -> pd.DataFrame:
    metrics_files = sorted(degree_analysis_dir.glob("**/*_degree_metrics.csv"))
    rows: list[pd.DataFrame] = []

    for fp in metrics_files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        edge_type, model = extract_edge_and_model_from_metrics_path(fp)
        if "edge_type" not in df.columns:
            df["edge_type"] = edge_type
        if "model" not in df.columns:
            df["model"] = model

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def maybe_make_plots(
    *,
    binary_df: pd.DataFrame,
    empirical_df: pd.DataFrame,
    graph_df: pd.DataFrame,
    summary_dir: Path,
) -> None:
    sns.set_style("whitegrid")

    if not binary_df.empty and "Correlation" in binary_df.columns:
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(data=binary_df, x="Model", y="Correlation", ax=ax)
        ax.set_title("Binary Outcome Correlation by Model")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fig.savefig(summary_dir / "binary_correlation_by_model.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if not empirical_df.empty and "Correlation vs Empirical" in empirical_df.columns:
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(data=empirical_df, x="Model", y="Correlation vs Empirical", ax=ax)
        ax.set_title("Empirical Frequency Correlation by Model")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fig.savefig(summary_dir / "empirical_correlation_by_model.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if not empirical_df.empty and not graph_df.empty and "Correlation vs Empirical" in empirical_df.columns:
        merged = empirical_df.merge(graph_df[["edge_type", "density"]], on="edge_type", how="left")
        merged["density_category"] = pd.cut(
            merged["density"],
            bins=[0, 0.01, 0.03, 0.05, 1.0],
            labels=["Very Sparse (<1%)", "Sparse (1-3%)", "Medium (3-5%)", "Dense (>5%)"],
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=merged.dropna(subset=["density_category"]),
            x="density_category",
            y="Correlation vs Empirical",
            hue="Model",
            ax=ax,
        )
        ax.set_title("Empirical Correlation by Density Category")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        fig.savefig(summary_dir / "correlation_boxplots_by_density.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.degree_analysis_dir.mkdir(parents=True, exist_ok=True)

    edge_types = resolve_edge_types(args)
    print(f"Edge types to summarize: {edge_types}")

    all_model_rows: list[pd.DataFrame] = []
    all_analytical_rows: list[pd.DataFrame] = []
    all_empirical_rows: list[pd.DataFrame] = []
    all_analytical_empirical_rows: list[pd.DataFrame] = []

    loaded_edge_types: list[str] = []
    for edge_type in edge_types:
        result = load_edge_results(args.results_dir, edge_type)
        if not result:
            continue

        loaded_edge_types.append(edge_type)

        if "model_comparison" in result:
            df = result["model_comparison"].copy()
            df["edge_type"] = edge_type
            all_model_rows.append(df)

        if "analytical_comparison" in result:
            df = result["analytical_comparison"].copy()
            df["edge_type"] = edge_type
            all_analytical_rows.append(df)

        if "empirical_comparison" in result:
            df = result["empirical_comparison"].copy()
            df["edge_type"] = edge_type
            all_empirical_rows.append(df)

        if "analytical_vs_empirical" in result:
            df = result["analytical_vs_empirical"].copy()
            df["edge_type"] = edge_type
            all_analytical_empirical_rows.append(df)

    binary_df = pd.concat(all_model_rows, ignore_index=True) if all_model_rows else pd.DataFrame()
    analytical_df = pd.concat(all_analytical_rows, ignore_index=True) if all_analytical_rows else pd.DataFrame()
    empirical_df = pd.concat(all_empirical_rows, ignore_index=True) if all_empirical_rows else pd.DataFrame()
    analytical_empirical_df = (
        pd.concat(all_analytical_empirical_rows, ignore_index=True)
        if all_analytical_empirical_rows
        else pd.DataFrame()
    )

    if not binary_df.empty:
        binary_df.to_csv(args.summary_dir / "model_comparison_all_edges.csv", index=False)
    if not analytical_df.empty:
        analytical_df.to_csv(args.summary_dir / "analytical_comparison_all_edges.csv", index=False)
    if not empirical_df.empty:
        empirical_df.to_csv(args.summary_dir / "empirical_comparison_all_edges.csv", index=False)
    if not analytical_empirical_df.empty:
        analytical_empirical_df.to_csv(
            args.summary_dir / "analytical_vs_empirical_all_edges.csv", index=False
        )

    summary_table = binary_df.copy()
    if not summary_table.empty and not empirical_df.empty:
        merge_cols = ["edge_type", "Model"]
        empirical_subset_cols = [c for c in ["edge_type", "Model", "Correlation vs Empirical", "MAE vs Empirical", "RMSE vs Empirical"] if c in empirical_df.columns]
        if set(merge_cols).issubset(empirical_df.columns):
            summary_table = summary_table.merge(
                empirical_df[empirical_subset_cols],
                on=merge_cols,
                how="left",
            )

    if not summary_table.empty:
        summary_table.to_csv(args.summary_dir / "model_performance_summary.csv", index=False)

    graph_df = load_graph_characteristics(args.data_dir, loaded_edge_types)
    if not graph_df.empty:
        graph_df.to_csv(args.summary_dir / "graph_characteristics.csv", index=False)

    degree_run_errors: dict[str, str] = {}
    if args.run_degree_analysis:
        for edge_type in loaded_edge_types:
            try:
                run_degree_analysis_pipeline(
                    edge_type=edge_type,
                    data_dir=args.data_dir,
                    results_dir=args.results_dir,
                    output_dir=args.degree_analysis_dir,
                    small_graph_mode=args.small_graph_mode,
                )
            except Exception as exc:
                degree_run_errors[edge_type] = str(exc)
                if not args.continue_on_error:
                    raise

    degree_metrics_df = aggregate_degree_metrics(args.degree_analysis_dir)
    if not degree_metrics_df.empty:
        degree_metrics_df.to_csv(args.summary_dir / "aggregate_degree_metrics.csv", index=False)

    if not args.skip_plots:
        maybe_make_plots(
            binary_df=binary_df,
            empirical_df=empirical_df,
            graph_df=graph_df,
            summary_dir=args.summary_dir,
        )

    summary_payload = {
        "loaded_edge_types": loaded_edge_types,
        "n_loaded_edge_types": len(loaded_edge_types),
        "n_model_comparison_rows": int(len(binary_df)),
        "n_analytical_rows": int(len(analytical_df)),
        "n_empirical_rows": int(len(empirical_df)),
        "n_analytical_empirical_rows": int(len(analytical_empirical_df)),
        "degree_analysis_run_requested": bool(args.run_degree_analysis),
        "degree_analysis_errors": degree_run_errors,
        "n_degree_metrics_rows": int(len(degree_metrics_df)),
    }
    (args.summary_dir / "degree_analysis_summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )

    print("\nModel testing summary complete.")
    print(f"Summary output directory: {args.summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
