"""model comparison workflow.

Outputs are written under:
  results/model_comparison/<EDGE_TYPE>_results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from model_comparison import (  # noqa: E402
    ModelCollection,
    create_degree_grid,
    prepare_edge_features_and_labels,
)
from model_evaluation import ModelEvaluator, get_best_models  # noqa: E402
from model_training import (  # noqa: E402
    ModelTrainer,
    compare_raw_logits,
    predict_with_model,
)
from model_visualization import (  # noqa: E402
    ModelVisualizer,
    create_comparison_table_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run model comparison for one edge type without papermill. "
            "This replaces notebook 04 for script-first execution."
        )
    )
    parser.add_argument("--edge-type", default="CtD", help="Edge type label (e.g., CtD).")
    parser.add_argument(
        "--edge-file",
        default=None,
        help="Edge file name (default: <edge-type>.sparse.npz).",
    )
    parser.add_argument(
        "--edge-dir",
        type=Path,
        default=REPO_DIR / "data" / "permutations" / "000.hetmat" / "edges",
        help="Directory containing edge sparse matrices.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_DIR / "results" / "model_comparison",
        help="Base output directory for edge-type result folders.",
    )
    parser.add_argument(
        "--empirical-dir",
        type=Path,
        default=REPO_DIR / "results" / "empirical_edge_frequencies",
        help="Directory containing edge_frequency_by_degree_<edge_type>.csv files.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.01,
        help="Negative sampling ratio passed to prepare_edge_features_and_labels.",
    )
    parser.add_argument(
        "--adaptive-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive sampling (default: true).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation (CSV outputs still produced).",
    )
    parser.add_argument(
        "--generate-all-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate <edge_type>_all_model_predictions.csv outputs (default: true).",
    )
    parser.add_argument(
        "--max-all-pairs",
        type=int,
        default=2_000_000,
        help=(
            "Skip full all-pairs prediction export when n_source*n_target exceeds this limit. "
            "Default: 2,000,000."
        ),
    )
    parser.add_argument(
        "--all-predictions-chunk-size",
        type=int,
        default=200_000,
        help="Chunk size for all-pairs model prediction export.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.edge_type:
        raise ValueError("--edge-type cannot be empty")
    if args.sample_ratio <= 0:
        raise ValueError("--sample-ratio must be > 0")
    if not (0 < args.test_size < 1):
        raise ValueError("--test-size must be in (0, 1)")
    if not (0 < args.val_size < 1):
        raise ValueError("--val-size must be in (0, 1)")
    if args.max_all_pairs <= 0:
        raise ValueError("--max-all-pairs must be > 0")
    if args.all_predictions_chunk_size <= 0:
        raise ValueError("--all-predictions-chunk-size must be > 0")


def model_slug(model_name: str) -> str:
    return model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()


def predict_in_chunks(
    *,
    model: Any,
    model_name: str,
    scaler: Any,
    features: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n = len(features)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_pred = predict_with_model(model, features[start:end], model_name, scaler)
        chunks.append(np.asarray(chunk_pred, dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def save_data_distribution_plot(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    random_state: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].hist(features[:, 0], bins=50, alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Source Degree")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Source Degrees")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(features[:, 1], bins=50, alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Target Degree")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Target Degrees")
    axes[0, 1].grid(True, alpha=0.3)

    positive_mask = labels == 1
    negative_mask = labels == 0
    n_sample = min(10_000, len(features))
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(features), size=n_sample, replace=False)

    sample_pos = sample_idx[positive_mask[sample_idx]]
    sample_neg = sample_idx[negative_mask[sample_idx]]

    axes[1, 0].scatter(
        features[sample_neg, 0],
        features[sample_neg, 1],
        alpha=0.3,
        s=1,
        label="No Edge",
        color="red",
    )
    axes[1, 0].scatter(
        features[sample_pos, 0],
        features[sample_pos, 1],
        alpha=0.3,
        s=1,
        label="Edge Exists",
        color="blue",
    )
    axes[1, 0].set_xlabel("Source Degree")
    axes[1, 0].set_ylabel("Target Degree")
    axes[1, 0].set_title("Degree Relationships (Sample)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    label_counts = pd.Series(labels).value_counts().sort_index()
    axes[1, 1].bar(["No Edge", "Edge Exists"], label_counts.values, color=["red", "blue"], alpha=0.7)
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Edge Distribution")
    axes[1, 1].grid(True, alpha=0.3)

    for i, count in enumerate(label_counts.values):
        axes[1, 1].text(i, count + len(features) * 0.01, str(count), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_test_vs_empirical_scatter(
    *,
    comparison: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    n_models = len(comparison)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for i, (model_name, row) in enumerate(comparison.items()):
        empirical = row["matched_empirical"]
        predictions = row["matched_predictions"]
        correlation = row["correlation_vs_empirical"]

        axes[i].scatter(empirical, predictions, alpha=0.6, s=20)
        axes[i].plot([0, 1], [0, 1], "r--", alpha=0.8)
        axes[i].set_xlabel("Empirical Frequency")
        axes[i].set_ylabel("Model Prediction")
        axes[i].set_title(f"{model_name}\nr = {correlation:.3f}")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_all_pair_predictions(
    *,
    edge_type: str,
    edge_file: str,
    edge_path: Path,
    results_dir: Path,
    training_results: dict[str, Any],
    evaluator: ModelEvaluator,
    max_all_pairs: int,
    chunk_size: int,
) -> dict[str, Any]:
    edge_matrix = sp.load_npz(str(edge_path)).tocsr()
    n_sources, n_targets = edge_matrix.shape
    total_pairs = int(n_sources * n_targets)

    metadata: dict[str, Any] = {
        "edge_type": edge_type,
        "edge_file": edge_file,
        "source_nodes": int(n_sources),
        "target_nodes": int(n_targets),
        "total_pairs": total_pairs,
        "max_all_pairs": int(max_all_pairs),
        "generated": False,
    }

    if total_pairs > max_all_pairs:
        metadata["skip_reason"] = (
            f"total_pairs={total_pairs:,} exceeds max_all_pairs={max_all_pairs:,}"
        )
        return metadata

    source_idx = np.repeat(np.arange(n_sources, dtype=np.int32), n_targets)
    target_idx = np.tile(np.arange(n_targets, dtype=np.int32), n_sources)

    source_degrees = np.asarray(edge_matrix.sum(axis=1)).ravel().astype(np.float32)
    target_degrees = np.asarray(edge_matrix.sum(axis=0)).ravel().astype(np.float32)

    source_deg_all = source_degrees[source_idx]
    target_deg_all = target_degrees[target_idx]
    all_features = np.column_stack([source_deg_all, target_deg_all]).astype(np.float32)

    # Sparse advanced indexing returns a matrix; convert to flat bool array.
    edge_exists = np.asarray(edge_matrix[source_idx, target_idx]).ravel().astype(bool)

    predictions_df = pd.DataFrame(
        {
            "source_index": source_idx,
            "target_index": target_idx,
            "source_degree": source_deg_all,
            "target_degree": target_deg_all,
            "degree_product": source_deg_all * target_deg_all,
            "edge_exists": edge_exists,
        }
    )

    total_edges_m = edge_matrix.nnz
    predictions_df["analytical_approximation"] = evaluator.analytical_approximation(
        source_deg_all,
        target_deg_all,
        total_edges_m,
    ).astype(np.float32)

    prediction_times: dict[str, float] = {}
    for model_name, model_result in training_results.items():
        if model_name == "data_splits":
            continue

        model = model_result["model"]
        scaler = model_result["training_result"].get("scaler")
        start = time.time()
        preds = predict_in_chunks(
            model=model,
            model_name=model_name,
            scaler=scaler,
            features=all_features,
            chunk_size=chunk_size,
        )
        elapsed = time.time() - start
        prediction_times[model_name] = float(elapsed)

        predictions_df[f"{model_slug(model_name)}_prediction"] = preds

    full_csv = results_dir / f"{edge_type}_all_model_predictions.csv"
    full_gz = results_dir / f"{edge_type}_all_model_predictions.csv.gz"
    predictions_df.to_csv(full_csv, index=False)
    predictions_df.to_csv(full_gz, index=False, compression="gzip")

    agg: dict[str, list[str]] = {
        "edge_exists": ["count", "sum", "mean"],
        "analytical_approximation": ["mean", "std"],
    }
    for model_name, model_result in training_results.items():
        if model_name == "data_splits":
            continue
        col = f"{model_slug(model_name)}_prediction"
        if col in predictions_df.columns:
            agg[col] = ["mean", "std"]

    degree_summary = predictions_df.groupby(["source_degree", "target_degree"]).agg(agg).round(6)
    degree_summary.columns = ["_".join(c).strip("_") for c in degree_summary.columns.values]
    degree_summary = degree_summary.reset_index()

    degree_summary_file = results_dir / f"{edge_type}_predictions_by_degree.csv"
    degree_summary.to_csv(degree_summary_file, index=False)

    metadata.update(
        {
            "generated": True,
            "existing_edges": int(predictions_df["edge_exists"].sum()),
            "edge_density": float(predictions_df["edge_exists"].mean()),
            "models": [m for m in training_results if m != "data_splits"],
            "prediction_times_seconds": prediction_times,
            "file_sizes_mb": {
                "full_csv": round(full_csv.stat().st_size / (1024 * 1024), 2),
                "compressed_csv": round(full_gz.stat().st_size / (1024 * 1024), 2),
            },
        }
    )
    return metadata


def main() -> int:
    args = parse_args()

    edge_file = args.edge_file if args.edge_file else f"{args.edge_type}.sparse.npz"
    edge_path = args.edge_dir / edge_file
    if not edge_path.exists():
        raise FileNotFoundError(f"Edge matrix not found: {edge_path}")

    results_dir = args.results_root / f"{args.edge_type}_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Edge type: {args.edge_type}")
    print(f"Edge file: {edge_path}")
    print(f"Results directory: {results_dir}")

    features, labels = prepare_edge_features_and_labels(
        str(edge_path),
        sample_ratio=args.sample_ratio,
        adaptive_sampling=args.adaptive_sampling,
        enhanced_features=False,
    )

    if not args.skip_plots:
        save_data_distribution_plot(
            features=features,
            labels=labels,
            out_path=results_dir / "data_distribution.png",
            random_state=args.random_state,
        )

    evaluator = ModelEvaluator()
    visualizer = ModelVisualizer()

    empirical_freq_file = args.empirical_dir / f"edge_frequency_by_degree_{args.edge_type}.csv"
    validation_results: dict[str, Any] | None = None
    if empirical_freq_file.exists():
        empirical_df = evaluator.load_empirical_frequencies(str(empirical_freq_file))
        validation_results = evaluator.validate_analytical_approximation_vs_empirical(
            empirical_df["source_degree"].values,
            empirical_df["target_degree"].values,
            empirical_df["empirical_frequency"].values,
            evaluator.get_total_edges_from_file(str(edge_path)),
        )

        if not args.skip_plots:
            visualizer.plot_analytical_validation(
                validation_results,
                save_path=results_dir / "analytical_validation.png",
            )
            source_bins_ana, target_bins_ana, _ = create_degree_grid(
                empirical_df["source_degree"].values,
                empirical_df["target_degree"].values,
                n_bins=50,
            )
            visualizer.create_analytical_heatmap(
                source_bins_ana,
                target_bins_ana,
                validation_results["total_edges_m"],
                save_path=results_dir / "analytical_heatmap.png",
            )

    model_collection = ModelCollection(random_state=args.random_state)
    models = model_collection.create_models(
        use_class_weights=True,
        input_dim=features.shape[1],
        edge_file_path=str(edge_path),
    )

    trainer = ModelTrainer(random_state=args.random_state)
    training_results = trainer.train_all_models(
        models,
        features,
        labels,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    x_test = training_results["data_splits"]["X_test"]
    y_test = training_results["data_splits"]["y_test"]
    evaluation_results = evaluator.evaluate_all_models(training_results, x_test, y_test)

    # Raw logit comparison outputs
    models_dict: dict[str, Any] = {}
    scalers_dict: dict[str, Any] = {}
    for model_name, model_result in training_results.items():
        if model_name == "data_splits":
            continue
        models_dict[model_name] = model_result["model"]
        scalers_dict[model_name] = model_result["training_result"].get("scaler")

    raw_logit_results = compare_raw_logits(models_dict, x_test, y_test, scalers_dict)
    raw_rows: list[dict[str, Any]] = []
    prob_vs_raw_rows: list[dict[str, Any]] = []
    for model_name in models_dict:
        raw_row = raw_logit_results.get(model_name, {})
        if "error" not in raw_row:
            raw_rows.append(
                {
                    "Model": model_name,
                    "Raw Correlation": raw_row["raw_correlation"],
                    "Raw AUC": raw_row["raw_auc"],
                    "Logit Mean": raw_row["logit_mean"],
                    "Logit Std": raw_row["logit_std"],
                    "Logit Range": raw_row["logit_range"],
                }
            )

        prob_corr = evaluation_results[model_name]["regression"]["correlation"]
        raw_corr = raw_row.get("raw_correlation", np.nan)
        improvement = raw_corr - prob_corr if not np.isnan(raw_corr) else np.nan
        rel_improvement = (
            (improvement / prob_corr * 100)
            if np.isfinite(prob_corr) and prob_corr != 0 and np.isfinite(improvement)
            else np.nan
        )
        prob_vs_raw_rows.append(
            {
                "Model": model_name,
                "Probability Correlation": prob_corr,
                "Raw Logit Correlation": raw_corr,
                "Improvement": improvement,
                "Relative Improvement %": rel_improvement,
            }
        )

    if raw_rows:
        raw_df = pd.DataFrame(raw_rows).sort_values("Raw Correlation", ascending=False)
        raw_df.to_csv(results_dir / "raw_logit_comparison.csv", index=False)

    prob_vs_raw_df = pd.DataFrame(prob_vs_raw_rows).sort_values(
        "Raw Logit Correlation", ascending=False
    )
    prob_vs_raw_df.to_csv(results_dir / "probability_vs_raw_logit_comparison.csv", index=False)

    comparison_df = evaluator.create_comparison_dataframe(evaluation_results)
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)

    best_models = get_best_models(evaluation_results)
    (results_dir / "best_models.json").write_text(json.dumps(best_models, indent=2))

    if not args.skip_plots:
        create_comparison_table_plot(comparison_df, save_path=results_dir / "comparison_table.png")
        visualizer.plot_roc_curves(evaluation_results, save_path=results_dir / "roc_curves.png")
        visualizer.plot_precision_recall_curves(
            evaluation_results,
            save_path=results_dir / "precision_recall_curves.png",
        )
        visualizer.plot_performance_comparison(
            evaluation_results,
            save_path=results_dir / "performance_comparison.png",
        )

        source_bins, target_bins, grid_features = create_degree_grid(
            features[:, 0],
            features[:, 1],
            n_bins=50,
        )
        visualizer.create_all_prediction_heatmaps(
            training_results,
            source_bins,
            target_bins,
            save_dir=str(results_dir),
            grid_features=grid_features,
        )
        visualizer.create_combined_heatmap_grid(
            training_results,
            source_bins,
            target_bins,
            save_path=results_dir / "combined_heatmaps.png",
        )
        visualizer.plot_training_history(
            training_results,
            save_path=results_dir / "training_history.png",
        )

    analytical_comparison = evaluator.compare_models_vs_analytical_approximation(
        evaluation_results,
        training_results,
        x_test,
        str(edge_path),
    )
    analytical_df = evaluator.create_analytical_comparison_dataframe(analytical_comparison)
    analytical_df.to_csv(results_dir / "models_vs_analytical_comparison.csv", index=False)

    if not args.skip_plots:
        visualizer.plot_models_vs_analytical_comparison(
            analytical_comparison,
            save_path=results_dir / "models_vs_analytical_scatter.png",
        )

    test_empirical_df: pd.DataFrame | None = None
    if empirical_freq_file.exists():
        test_empirical_comparison = evaluator.compare_test_predictions_with_empirical(
            evaluation_results,
            training_results,
            x_test,
            str(empirical_freq_file),
        )
        if test_empirical_comparison:
            test_empirical_df = evaluator.create_test_empirical_comparison_dataframe(
                test_empirical_comparison
            )
            test_empirical_df.to_csv(
                results_dir / "test_vs_empirical_comparison.csv",
                index=False,
            )
            if not args.skip_plots:
                save_test_vs_empirical_scatter(
                    comparison=test_empirical_comparison,
                    out_path=results_dir / "test_vs_empirical_scatter.png",
                )

    export_metadata = {
        "edge_type": args.edge_type,
        "edge_file": edge_file,
        "all_predictions_requested": bool(args.generate_all_predictions),
    }
    if args.generate_all_predictions:
        export_metadata.update(
            export_all_pair_predictions(
                edge_type=args.edge_type,
                edge_file=edge_file,
                edge_path=edge_path,
                results_dir=results_dir,
                training_results=training_results,
                evaluator=evaluator,
                max_all_pairs=args.max_all_pairs,
                chunk_size=args.all_predictions_chunk_size,
            )
        )
    (results_dir / f"{args.edge_type}_predictions_metadata.json").write_text(
        json.dumps(export_metadata, indent=2)
    )

    summary = {
        "edge_type": args.edge_type,
        "edge_file": edge_file,
        "results_dir": str(results_dir),
        "empirical_freq_file_exists": empirical_freq_file.exists(),
        "n_training_samples": int(len(training_results["data_splits"]["X_train_full"])),
        "n_test_samples": int(len(x_test)),
        "models": [m for m in training_results if m != "data_splits"],
        "best_models": best_models,
        "generated_core_files": [
            "model_comparison.csv",
            "models_vs_analytical_comparison.csv",
            "test_vs_empirical_comparison.csv",
            "raw_logit_comparison.csv",
            "probability_vs_raw_logit_comparison.csv",
        ],
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nModel comparison run complete.")
    print(f"Output directory: {results_dir}")
    if test_empirical_df is None:
        print("Empirical comparison CSV not generated (missing empirical frequency file).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
