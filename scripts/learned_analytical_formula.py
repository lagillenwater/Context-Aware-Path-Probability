"""Run learned analytical formula workflow without papermill (A2 migration of notebook 8).

This script supports the notebook 8 baseline flow and optional degree-analysis
extensions from the notebook 8-with-degree workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from learned_analytical import LearnedAnalyticalFormula  # noqa: E402


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
            "Learn an analytical edge-probability formula from permutations and "
            "compare against empirical/analytical baselines (script-first notebook 8 migration)."
        )
    )
    parser.add_argument(
        "--edge-type",
        action="append",
        default=[],
        help="Edge type to run (repeatable). Default: all canonical edge types.",
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
        default=REPO_DIR / "results",
        help="Repository results directory containing empirical_edge_frequencies/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_DIR / "results" / "learned_formula",
        help="Output root for learned-formula artifacts.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        nargs="+",
        default=[2, 3, 5, 7, 10, 15, 20, 30, 40, 50],
        help="Permutation counts to test for minimum-N convergence.",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.0001,
        help="Convergence threshold for minimum-N search.",
    )
    parser.add_argument(
        "--target-metric",
        choices=["correlation", "mae", "rmse", "r2"],
        default="correlation",
        help="Metric used to assess convergence/target attainment.",
    )
    parser.add_argument(
        "--min-metric-value",
        type=float,
        default=0.9999,
        help="Target performance threshold (interpretation depends on metric).",
    )
    parser.add_argument(
        "--improved",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use improved train/test split methodology "
            "(find_minimum_permutations_improved). Default: false for notebook-8 parity."
        ),
    )
    parser.add_argument(
        "--formula-type",
        choices=["original", "extended", "polynomial"],
        default="original",
        help="Learned formula family.",
    )
    parser.add_argument("--n-random-starts", type=int, default=10)
    parser.add_argument("--regularization-lambda", type=float, default=0.001)
    parser.add_argument("--l1-lambda", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument(
        "--degree-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run residual and degree-error analysis outputs.",
    )
    parser.add_argument(
        "--small-graph-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use degree-analysis settings tuned for small graphs.",
    )
    parser.add_argument(
        "--run-parameter-importance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run parameter sensitivity analysis and save importance plot.",
    )
    parser.add_argument(
        "--run-cross-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run k-fold cross-validation on the learned formula.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--predict-all-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate all-pairs learned predictions CSVs.",
    )
    parser.add_argument(
        "--skip-comparison-plot",
        action="store_true",
        help="Skip empirical vs learned/analytical scatter comparison plot.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue with remaining edge types if one fails.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.n_random_starts <= 0:
        raise ValueError("--n-random-starts must be > 0")
    if args.regularization_lambda < 0:
        raise ValueError("--regularization-lambda must be >= 0")
    if args.l1_lambda < 0:
        raise ValueError("--l1-lambda must be >= 0")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be > 0")
    if args.ensemble_size <= 0:
        raise ValueError("--ensemble-size must be > 0")
    if args.cv_folds <= 1:
        raise ValueError("--cv-folds must be > 1")
    if args.convergence_threshold < 0:
        raise ValueError("--convergence-threshold must be >= 0")
    if not args.n_candidates:
        raise ValueError("--n-candidates cannot be empty")
    if any(n <= 0 for n in args.n_candidates):
        raise ValueError("all --n-candidates values must be > 0")


def copy_internal_convergence_plots(*, results_dir: Path, edge_type: str, output_dir: Path) -> None:
    internal_dir = results_dir / "learned_analytical"
    if not internal_dir.exists():
        return

    for name in [
        f"{edge_type}_convergence_curve.png",
        f"{edge_type}_improved_convergence_curve.png",
    ]:
        src = internal_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def build_convergence_df(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in results.get("convergence_curve", []):
        if "val_metrics" in row:
            rows.append(
                {
                    "N": row["N"],
                    "train_mae": row["train_metrics"].get("mae"),
                    "train_rmse": row["train_metrics"].get("rmse"),
                    "train_correlation": row["train_metrics"].get("correlation"),
                    "validation_mae": row["val_metrics"].get("mae"),
                    "validation_rmse": row["val_metrics"].get("rmse"),
                    "validation_correlation": row["val_metrics"].get("correlation"),
                }
            )
        else:
            rows.append(
                {
                    "N": row.get("N"),
                    "N_test": row.get("N_test"),
                    "train_mae": row["train_metrics"].get("mae"),
                    "train_rmse": row["train_metrics"].get("rmse"),
                    "train_correlation": row["train_metrics"].get("correlation"),
                    "test_mae": row["test_metrics"].get("mae"),
                    "test_rmse": row["test_metrics"].get("rmse"),
                    "test_correlation": row["test_metrics"].get("correlation"),
                }
            )
    return pd.DataFrame(rows)


def save_comparison_plot(
    *,
    edge_type: str,
    output_dir: Path,
    predictions_df: pd.DataFrame,
    empirical_df: pd.DataFrame,
) -> dict[str, float]:
    merge_df = empirical_df.merge(
        predictions_df,
        on=["source_degree", "target_degree"],
        how="inner",
    )
    if merge_df.empty:
        return {"matched_degree_pairs": 0, "learned_corr": np.nan, "analytical_corr": np.nan}

    learned_corr = float(np.corrcoef(merge_df["frequency"], merge_df["learned_probability"])[0, 1])
    analytical_corr = float(np.corrcoef(merge_df["frequency"], merge_df["analytical_probability"])[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(merge_df["frequency"], merge_df["learned_probability"], alpha=0.5, s=20)
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=2)
    axes[0].set_xlabel("Empirical Frequency (200 perms)")
    axes[0].set_ylabel("Predicted Probability")
    axes[0].set_title(f"Learned Formula vs Empirical\nr = {learned_corr:.4f}")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(merge_df["frequency"], merge_df["analytical_probability"], alpha=0.5, s=20, color="orange")
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=2)
    axes[1].set_xlabel("Empirical Frequency (200 perms)")
    axes[1].set_ylabel("Predicted Probability")
    axes[1].set_title(f"Current Analytical vs Empirical\nr = {analytical_corr:.4f}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"{edge_type}_learned_vs_analytical_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "matched_degree_pairs": int(len(merge_df)),
        "learned_corr": learned_corr,
        "analytical_corr": analytical_corr,
    }


def run_single_edge(edge_type: str, args: argparse.Namespace) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"LEARNED ANALYTICAL MIGRATION RUN: {edge_type}")
    print("=" * 80)

    output_dir = args.output_root / f"{edge_type}_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    learner = LearnedAnalyticalFormula(
        n_random_starts=args.n_random_starts,
        regularization_lambda=args.regularization_lambda,
        l1_lambda=args.l1_lambda,
        formula_type=args.formula_type,
        bootstrap_samples=args.bootstrap_samples,
        ensemble_size=args.ensemble_size,
    )

    if args.improved:
        results = learner.find_minimum_permutations_improved(
            graph_name=edge_type,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            N_candidates=args.n_candidates,
            convergence_threshold=args.convergence_threshold,
            target_metric=args.target_metric,
            min_metric_value=args.min_metric_value,
        )
    else:
        results = learner.find_minimum_permutations(
            graph_name=edge_type,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            N_candidates=args.n_candidates,
            convergence_threshold=args.convergence_threshold,
            target_metric=args.target_metric,
            min_metric_value=args.min_metric_value,
            degree_stratified=args.degree_analysis,
            small_graph_mode=args.small_graph_mode,
        )

    learner.save_results(results, output_dir)
    copy_internal_convergence_plots(results_dir=args.results_dir, edge_type=edge_type, output_dir=output_dir)

    convergence_df = build_convergence_df(results)
    if not convergence_df.empty:
        convergence_df.to_csv(output_dir / f"{edge_type}_convergence_data.csv", index=False)

    predictions_df: pd.DataFrame | None = None
    if args.predict_all_edges:
        predictions_df = learner.predict_all_edges(edge_type, args.data_dir)
        pred_csv = output_dir / f"{edge_type}_learned_predictions.csv"
        pred_gz = output_dir / f"{edge_type}_learned_predictions.csv.gz"
        predictions_df.to_csv(pred_csv, index=False)
        predictions_df.to_csv(pred_gz, index=False, compression="gzip")

    empirical_file = args.results_dir / "empirical_edge_frequencies" / f"edge_frequency_by_degree_{edge_type}.csv"
    comparison_stats = {"matched_degree_pairs": 0, "learned_corr": np.nan, "analytical_corr": np.nan}
    if predictions_df is not None and empirical_file.exists() and not args.skip_comparison_plot:
        empirical_df = pd.read_csv(empirical_file)
        if "frequency" not in empirical_df.columns and "empirical_frequency" in empirical_df.columns:
            empirical_df = empirical_df.rename(columns={"empirical_frequency": "frequency"})
        comparison_stats = save_comparison_plot(
            edge_type=edge_type,
            output_dir=output_dir,
            predictions_df=predictions_df,
            empirical_df=empirical_df,
        )

    if args.degree_analysis:
        if not empirical_file.exists():
            raise FileNotFoundError(f"Empirical frequency file not found: {empirical_file}")

        empirical_200 = learner._load_200_perm_empirical(edge_type, args.results_dir)
        graph_stats = results["graph_stats"]
        residuals_df, _degree_error_metrics = learner.analyze_residuals(
            empirical_200=empirical_200,
            m=graph_stats["m"],
            density=graph_stats["density"],
            results_dir=output_dir,
            graph_name=edge_type,
            small_graph_mode=args.small_graph_mode,
        )
        residuals_df.to_csv(output_dir / f"{edge_type}_enhanced_residuals.csv", index=False)

        if args.run_parameter_importance:
            sens_df = learner.analyze_parameter_importance(
                empirical_200=empirical_200,
                m=graph_stats["m"],
                density=graph_stats["density"],
                graph_name=edge_type,
                results_dir=output_dir,
            )
            sens_df.to_csv(output_dir / f"{edge_type}_parameter_sensitivity.csv", index=False)

        if args.run_cross_validation:
            cv_results = learner.cross_validate_model(
                empirical_data=empirical_200,
                m=graph_stats["m"],
                density=graph_stats["density"],
                k_folds=args.cv_folds,
                use_stratified=True,
            )
            with open(output_dir / f"{edge_type}_cross_validation_results.json", "w") as f:
                json.dump(cv_results, f, indent=2, default=float)

    if "final_metrics" in results:
        final_metrics = results["final_metrics"]
    else:
        final_metrics = results.get("final_test_metrics", {})

    summary = {
        "edge_type": edge_type,
        "output_dir": str(output_dir),
        "N_min": int(results["N_min"]),
        "graph_stats": results.get("graph_stats", {}),
        "target_metric": args.target_metric,
        "min_metric_value": args.min_metric_value,
        "formula_type": args.formula_type,
        "improved_method": bool(args.improved),
        "final_metrics": final_metrics,
        "baseline_metrics": results.get("baseline_metrics", {}),
        "comparison_plot": comparison_stats,
        "degree_analysis": bool(args.degree_analysis),
        "predict_all_edges": bool(args.predict_all_edges),
    }
    with open(output_dir / f"{edge_type}_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"Completed learned analytical run for {edge_type}")
    print(f"Output: {output_dir}")
    return summary


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    edge_types = args.edge_type if args.edge_type else DEFAULT_EDGE_TYPES

    all_summaries: list[dict[str, Any]] = []
    failures: dict[str, str] = {}

    for edge_type in edge_types:
        try:
            summary = run_single_edge(edge_type, args)
            all_summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            failures[edge_type] = str(exc)
            print(f"ERROR [{edge_type}]: {exc}")
            if not args.continue_on_error:
                raise

    manifest = {
        "edge_types_requested": edge_types,
        "n_success": len(all_summaries),
        "n_failures": len(failures),
        "failures": failures,
        "runs": all_summaries,
    }
    with open(args.output_root / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    print("\n" + "=" * 80)
    print("LEARNED ANALYTICAL MIGRATION SUMMARY")
    print("=" * 80)
    print(f"Success: {len(all_summaries)}")
    print(f"Failures: {len(failures)}")
    print(f"Manifest: {args.output_root / 'run_manifest.json'}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
