"""Train and validate null-edge models.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR / "src") not in sys.path:
    sys.path.append(str(REPO_DIR / "src"))

from model_comparison import prepare_edge_features_and_labels  # noqa: E402


ALL_EDGE_TYPES = [
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
            "Train null models (polynomial ridge + random forest) using "
            "permutation edge matrices under data/permutations."
        )
    )
    parser.add_argument(
        "--edge-type",
        action="append",
        default=[],
        help=(
            "Edge type to process (repeatable, e.g. --edge-type CbG). "
            "Default: all 24 edge types."
        ),
    )
    parser.add_argument(
        "--permutations-dir",
        type=Path,
        default=REPO_DIR / "data" / "permutations",
        help="Directory containing ###.hetmat permutation folders.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "null_models",
        help="Output directory for trained models and validation CSVs.",
    )
    parser.add_argument(
        "--empirical-dir",
        type=Path,
        default=REPO_DIR / "results" / "empirical_edge_frequencies",
        help="Directory containing edge_frequency_by_degree_*.csv files.",
    )
    parser.add_argument(
        "--training-perm-start",
        type=int,
        default=1,
        help="First permutation id for training (inclusive). Default: 1.",
    )
    parser.add_argument(
        "--training-perm-end",
        type=int,
        default=20,
        help="Last permutation id for training (inclusive). Default: 20.",
    )
    parser.add_argument(
        "--validation-perm-start",
        type=int,
        default=21,
        help="First permutation id for held-out validation (inclusive). Default: 21.",
    )
    parser.add_argument(
        "--validation-perm-end",
        type=int,
        default=30,
        help="Last permutation id for held-out validation (inclusive). Default: 30.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.01,
        help="Base negative sampling ratio passed to prepare_edge_features_and_labels.",
    )
    parser.add_argument(
        "--adaptive-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable adaptive sampling. Default: enabled.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip held-out binary-label validation on validation permutations.",
    )
    parser.add_argument(
        "--skip-empirical-validation",
        action="store_true",
        help="Skip validation against empirical frequency CSVs.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip writing validation_performance.png.",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=100,
        help="RandomForestRegressor n_estimators. Default: 100.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=10,
        help="RandomForestRegressor max_depth. Default: 10.",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=-1,
        help="RandomForestRegressor n_jobs. Default: -1.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.training_perm_start > args.training_perm_end:
        raise ValueError("training-perm-start must be <= training-perm-end")
    if args.validation_perm_start > args.validation_perm_end:
        raise ValueError("validation-perm-start must be <= validation-perm-end")
    if args.sample_ratio <= 0:
        raise ValueError("sample-ratio must be > 0")
    if args.rf_estimators <= 0:
        raise ValueError("rf-estimators must be > 0")
    if args.rf_max_depth <= 0:
        raise ValueError("rf-max-depth must be > 0")


def load_permutation_samples(
    *,
    edge_type: str,
    perm_ids: list[int],
    permutations_dir: Path,
    sample_ratio: float,
    adaptive_sampling: bool,
) -> tuple[np.ndarray, np.ndarray]:
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    print(f"\nLoading samples for {edge_type} from {len(perm_ids)} permutations...")
    for perm_id in perm_ids:
        edge_file = (
            permutations_dir
            / f"{perm_id:03d}.hetmat"
            / "edges"
            / f"{edge_type}.sparse.npz"
        )
        if not edge_file.exists():
            print(f"  Warning: missing {edge_file}")
            continue

        features, labels = prepare_edge_features_and_labels(
            str(edge_file),
            sample_ratio=sample_ratio,
            adaptive_sampling=adaptive_sampling,
            enhanced_features=False,
        )
        all_features.append(features)
        all_labels.append(labels)
        print(
            f"  Perm {perm_id:03d}: {len(features):,} samples "
            f"({int(labels.sum()):,} edges, {int((labels == 0).sum()):,} non-edges)"
        )

    if not all_features:
        return np.array([]).reshape(0, 2), np.array([])

    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    print(
        f"Combined {edge_type}: {len(combined_features):,} samples "
        f"({int(combined_labels.sum()):,} edges, {int((combined_labels == 0).sum()):,} non-edges)"
    )
    return combined_features, combined_labels


def analyze_errors_by_degree(
    predictions: np.ndarray,
    actuals: np.ndarray,
    source_degrees: np.ndarray,
    target_degrees: np.ndarray,
) -> pd.DataFrame:
    degree_bins = [0, 1, 2, 5, 10, 20, 50, 100, 500, np.inf]
    degree_labels = ["0", "1", "2-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500+"]
    results = []

    source_bins = pd.cut(source_degrees, bins=degree_bins, labels=degree_labels)
    target_bins = pd.cut(target_degrees, bins=degree_bins, labels=degree_labels)

    for bin_label in degree_labels:
        mask = source_bins == bin_label
        if int(mask.sum()) >= 10:
            results.append(
                _error_row(
                    stratification="source_degree",
                    bin_label=bin_label,
                    predictions=predictions[mask],
                    actuals=actuals[mask],
                    mean_value=float(source_degrees[mask].mean()),
                    mean_key="mean_degree",
                )
            )

    for bin_label in degree_labels:
        mask = target_bins == bin_label
        if int(mask.sum()) >= 10:
            results.append(
                _error_row(
                    stratification="target_degree",
                    bin_label=bin_label,
                    predictions=predictions[mask],
                    actuals=actuals[mask],
                    mean_value=float(target_degrees[mask].mean()),
                    mean_key="mean_degree",
                )
            )

    degree_products = source_degrees * target_degrees
    product_bins = pd.cut(
        degree_products,
        bins=[0, 1, 10, 100, 1000, 10000, 100000, np.inf],
        labels=["0-1", "2-10", "11-100", "101-1K", "1K-10K", "10K-100K", "100K+"],
    )

    for bin_label in product_bins.unique():
        if pd.notna(bin_label):
            mask = product_bins == bin_label
            if int(mask.sum()) >= 10:
                results.append(
                    _error_row(
                        stratification="degree_product",
                        bin_label=str(bin_label),
                        predictions=predictions[mask],
                        actuals=actuals[mask],
                        mean_value=float(degree_products[mask].mean()),
                        mean_key="mean_product",
                    )
                )

    return pd.DataFrame(results)


def _error_row(
    *,
    stratification: str,
    bin_label: str,
    predictions: np.ndarray,
    actuals: np.ndarray,
    mean_value: float,
    mean_key: str,
) -> dict[str, float | int | str]:
    corr = np.nan
    if len(predictions) > 1:
        with np.errstate(invalid="ignore"):
            corr = float(np.corrcoef(predictions, actuals)[0, 1])

    row: dict[str, float | int | str] = {
        "stratification": stratification,
        "bin": bin_label,
        "n_samples": int(len(predictions)),
        "mean_prediction": float(predictions.mean()),
        "mean_actual": float(actuals.mean()),
        "mae": float(np.abs(predictions - actuals).mean()),
        "rmse": float(np.sqrt(((predictions - actuals) ** 2).mean())),
        "correlation": corr,
    }
    row[mean_key] = mean_value
    return row


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return np.nan
    try:
        corr, _ = pearsonr(y_true, y_pred)
        return float(corr)
    except Exception:
        return np.nan


def analytical_prior(u: np.ndarray, v: np.ndarray, m: int) -> np.ndarray:
    numerator = u * v
    denominator = np.sqrt((u * v) ** 2 + (m - u - v + 1) ** 2)
    return np.where(denominator > 0, numerator / denominator, 0.0)


def train_models_for_edge_type(
    *,
    edge_type: str,
    args: argparse.Namespace,
    training_perm_ids: list[int],
) -> dict[str, str | int] | None:
    print("\n" + "=" * 70)
    print(f"Training null models for {edge_type}")
    print("=" * 70)

    x_train, y_train = load_permutation_samples(
        edge_type=edge_type,
        perm_ids=training_perm_ids,
        permutations_dir=args.permutations_dir,
        sample_ratio=args.sample_ratio,
        adaptive_sampling=args.adaptive_sampling,
    )
    if len(x_train) == 0:
        print(f"  No training data found for {edge_type}; skipping.")
        return None

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = poly_features.fit_transform(x_train)

    poly_model = Ridge(alpha=1.0, random_state=42)
    poly_model.fit(x_train_poly, y_train)

    rf_model = RandomForestRegressor(
        n_estimators=args.rf_estimators,
        max_depth=args.rf_max_depth,
        random_state=42,
        n_jobs=args.rf_n_jobs,
    )
    rf_model.fit(x_train, y_train)

    poly_model_file = args.results_dir / f"{edge_type}_poly_null.pkl"
    poly_features_file = args.results_dir / f"{edge_type}_poly_features.pkl"
    rf_model_file = args.results_dir / f"{edge_type}_rf_null.pkl"

    joblib.dump(poly_model, poly_model_file)
    joblib.dump(poly_features, poly_features_file)
    joblib.dump(rf_model, rf_model_file)

    print(f"Saved models for {edge_type}:")
    print(f"  - {poly_model_file.name}")
    print(f"  - {poly_features_file.name}")
    print(f"  - {rf_model_file.name}")

    return {
        "edge_type": edge_type,
        "n_training_samples": int(len(x_train)),
        "n_positive_samples": int(y_train.sum()),
        "n_negative_samples": int((y_train == 0).sum()),
        "poly_model_file": str(poly_model_file),
        "rf_model_file": str(rf_model_file),
    }


def validate_models_on_permutations(
    *,
    edge_type: str,
    args: argparse.Namespace,
    validation_perm_ids: list[int],
) -> tuple[dict[str, float | int | str] | None, pd.DataFrame | None]:
    poly_model_file = args.results_dir / f"{edge_type}_poly_null.pkl"
    poly_features_file = args.results_dir / f"{edge_type}_poly_features.pkl"
    rf_model_file = args.results_dir / f"{edge_type}_rf_null.pkl"
    if not (poly_model_file.exists() and poly_features_file.exists() and rf_model_file.exists()):
        print(f"  Models missing for {edge_type}; skipping held-out validation.")
        return None, None

    x_val, y_val = load_permutation_samples(
        edge_type=edge_type,
        perm_ids=validation_perm_ids,
        permutations_dir=args.permutations_dir,
        sample_ratio=args.sample_ratio,
        adaptive_sampling=args.adaptive_sampling,
    )
    if len(x_val) == 0:
        print(f"  No validation data found for {edge_type}; skipping held-out validation.")
        return None, None

    poly_model = joblib.load(poly_model_file)
    poly_features = joblib.load(poly_features_file)
    rf_model = joblib.load(rf_model_file)

    y_pred_poly = np.clip(poly_model.predict(poly_features.transform(x_val)), 0, 1)
    y_pred_rf = np.clip(rf_model.predict(x_val), 0, 1)

    poly_corr = safe_pearsonr(y_val, y_pred_poly)
    rf_corr = safe_pearsonr(y_val, y_pred_rf)
    poly_mae = float(np.abs(y_val - y_pred_poly).mean())
    rf_mae = float(np.abs(y_val - y_pred_rf).mean())
    poly_rmse = float(np.sqrt(((y_val - y_pred_poly) ** 2).mean()))
    rf_rmse = float(np.sqrt(((y_val - y_pred_rf) ** 2).mean()))

    print(
        f"  {edge_type} held-out: "
        f"Poly r={poly_corr:.4f}, RF r={rf_corr:.4f}, "
        f"Poly RMSE={poly_rmse:.4f}, RF RMSE={rf_rmse:.4f}"
    )

    error_df = analyze_errors_by_degree(y_pred_rf, y_val, x_val[:, 0], x_val[:, 1])
    error_df["edge_type"] = edge_type

    return (
        {
            "edge_type": edge_type,
            "n_validation_samples": int(len(x_val)),
            "n_val_positive": int(y_val.sum()),
            "n_val_negative": int((y_val == 0).sum()),
            "poly_correlation": poly_corr,
            "poly_mae": poly_mae,
            "poly_rmse": poly_rmse,
            "rf_correlation": rf_corr,
            "rf_mae": rf_mae,
            "rf_rmse": rf_rmse,
        },
        error_df,
    )


def validate_models_on_empirical(
    *,
    edge_type: str,
    args: argparse.Namespace,
) -> dict[str, float | int | str] | None:
    poly_model_file = args.results_dir / f"{edge_type}_poly_null.pkl"
    poly_features_file = args.results_dir / f"{edge_type}_poly_features.pkl"
    rf_model_file = args.results_dir / f"{edge_type}_rf_null.pkl"
    empirical_file = args.empirical_dir / f"edge_frequency_by_degree_{edge_type}.csv"

    if not (poly_model_file.exists() and poly_features_file.exists() and rf_model_file.exists()):
        print(f"  Models missing for {edge_type}; skipping empirical validation.")
        return None
    if not empirical_file.exists():
        print(f"  Empirical frequency file missing for {edge_type}; skipping.")
        return None

    empirical_df = pd.read_csv(empirical_file)
    if empirical_df.empty:
        print(f"  Empirical frequency file empty for {edge_type}; skipping.")
        return None

    poly_model = joblib.load(poly_model_file)
    poly_features = joblib.load(poly_features_file)
    rf_model = joblib.load(rf_model_file)

    x_emp = empirical_df[["source_degree", "target_degree"]].to_numpy()
    y_emp = empirical_df["frequency"].to_numpy()

    y_poly = np.clip(poly_model.predict(poly_features.transform(x_emp)), 0, 1)
    y_rf = np.clip(rf_model.predict(x_emp), 0, 1)

    poly_corr = safe_pearsonr(y_emp, y_poly)
    rf_corr = safe_pearsonr(y_emp, y_rf)
    poly_mae = float(np.abs(y_emp - y_poly).mean())
    rf_mae = float(np.abs(y_emp - y_rf).mean())

    edge_file = args.permutations_dir / "001.hetmat" / "edges" / f"{edge_type}.sparse.npz"
    if edge_file.exists():
        total_edges = int(sp.load_npz(str(edge_file)).nnz)
    else:
        total_edges = 1000
    y_analytical = analytical_prior(x_emp[:, 0], x_emp[:, 1], total_edges)
    analytical_corr = safe_pearsonr(y_emp, y_analytical)

    print(
        f"  {edge_type} empirical: Poly r={poly_corr:.4f}, RF r={rf_corr:.4f}, "
        f"Analytical r={analytical_corr:.4f}"
    )

    return {
        "edge_type": edge_type,
        "n_degree_combinations": int(len(empirical_df)),
        "total_edges": total_edges,
        "poly_corr_vs_empirical": poly_corr,
        "poly_mae_vs_empirical": poly_mae,
        "rf_corr_vs_empirical": rf_corr,
        "rf_mae_vs_empirical": rf_mae,
        "analytical_corr_vs_empirical": analytical_corr,
    }


def write_validation_plot(validation_df: pd.DataFrame, output_path: Path) -> None:
    if validation_df.empty:
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(validation_df))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width / 2, validation_df["poly_correlation"], width, label="Polynomial", alpha=0.7)
    ax.bar(x + width / 2, validation_df["rf_correlation"], width, label="Random Forest", alpha=0.7)
    ax.set_xlabel("Edge Type")
    ax.set_ylabel("Correlation")
    ax.set_title("Validation Correlation by Edge Type")
    ax.set_xticks(x)
    ax.set_xticklabels(validation_df["edge_type"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - width / 2, validation_df["poly_rmse"], width, label="Polynomial", alpha=0.7)
    ax.bar(x + width / 2, validation_df["rf_rmse"], width, label="Random Forest", alpha=0.7)
    ax.set_xlabel("Edge Type")
    ax.set_ylabel("RMSE")
    ax.set_title("Validation RMSE by Edge Type")
    ax.set_xticks(x)
    ax.set_xticklabels(validation_df["edge_type"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.scatter(validation_df["poly_correlation"], validation_df["rf_correlation"], s=80, alpha=0.7)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.4)
    ax.set_xlabel("Polynomial Correlation")
    ax.set_ylabel("RF Correlation")
    ax.set_title("Model Comparison")
    ax.grid(alpha=0.3)
    for _, row in validation_df.iterrows():
        ax.annotate(str(row["edge_type"]), (row["poly_correlation"], row["rf_correlation"]), fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    edge_types = args.edge_type if args.edge_type else ALL_EDGE_TYPES
    training_perm_ids = list(range(args.training_perm_start, args.training_perm_end + 1))
    validation_perm_ids = list(range(args.validation_perm_start, args.validation_perm_end + 1))

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Permutation directory: {args.permutations_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Empirical frequency directory: {args.empirical_dir}")
    print(f"Edge types: {len(edge_types)}")
    print(f"Training permutations: {training_perm_ids[0]}-{training_perm_ids[-1]}")
    print(f"Validation permutations: {validation_perm_ids[0]}-{validation_perm_ids[-1]}")

    training_rows: list[dict[str, str | int]] = []
    for edge_type in edge_types:
        row = train_models_for_edge_type(edge_type=edge_type, args=args, training_perm_ids=training_perm_ids)
        if row is not None:
            training_rows.append(row)

    training_df = pd.DataFrame(training_rows)
    training_file = args.results_dir / "training_summary.csv"
    training_df.to_csv(training_file, index=False)
    print(f"\nSaved training summary: {training_file}")

    validation_rows: list[dict[str, float | int | str]] = []
    if not args.skip_validation:
        for edge_type in edge_types:
            row, error_df = validate_models_on_permutations(
                edge_type=edge_type,
                args=args,
                validation_perm_ids=validation_perm_ids,
            )
            if row is None:
                continue
            validation_rows.append(row)
            error_file = args.results_dir / f"{edge_type}_error_analysis.csv"
            error_df.to_csv(error_file, index=False)
    validation_df = pd.DataFrame(validation_rows)
    validation_file = args.results_dir / "validation_results.csv"
    validation_df.to_csv(validation_file, index=False)
    print(f"Saved held-out validation results: {validation_file}")

    if not args.skip_plot and not validation_df.empty:
        plot_file = args.results_dir / "validation_performance.png"
        write_validation_plot(validation_df, plot_file)
        print(f"Saved held-out validation plot: {plot_file}")

    empirical_rows: list[dict[str, float | int | str]] = []
    if not args.skip_empirical_validation:
        for edge_type in edge_types:
            row = validate_models_on_empirical(edge_type=edge_type, args=args)
            if row is not None:
                empirical_rows.append(row)
    empirical_df = pd.DataFrame(empirical_rows)
    empirical_file = args.results_dir / "empirical_validation_results.csv"
    empirical_df.to_csv(empirical_file, index=False)
    if not args.skip_empirical_validation:
        print(f"Saved empirical validation results: {empirical_file}")

    print("\n" + "=" * 70)
    print("NULL MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Trained edge types: {len(training_df)}")
    if not validation_df.empty:
        print(f"Held-out validation edge types: {len(validation_df)}")
        print(f"Mean RF held-out correlation: {validation_df['rf_correlation'].mean():.4f}")
    else:
        print("Held-out validation: skipped or no rows.")
    if not empirical_df.empty:
        print(f"Empirical validation edge types: {len(empirical_df)}")
        print(
            f"Mean RF empirical correlation: "
            f"{empirical_df['rf_corr_vs_empirical'].mean():.4f}"
        )
    else:
        print("Empirical validation: skipped or no rows.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
