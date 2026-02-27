"""Compute compositional null probabilities without papermill.

This script ports notebook 14 (optimized compositional null) into a direct CLI.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import ks_2samp, pearsonr
from sklearn.metrics import r2_score


REPO_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 2-edge compositional null probabilities from trained null "
            "models and permutation edge matrices."
        )
    )
    parser.add_argument("--metapath", default="CbGpPW", help="Metapath label for output naming.")
    parser.add_argument("--edge1-type", default="CbG", help="First edge type in 2-edge metapath.")
    parser.add_argument("--edge2-type", default="GpPW", help="Second edge type in 2-edge metapath.")
    parser.add_argument(
        "--model-type",
        choices=["rf", "poly", "ensemble"],
        default="rf",
        help="Model type for compositional predictions.",
    )
    parser.add_argument(
        "--validation-perm-start",
        type=int,
        default=21,
        help="First permutation id for true-null extraction (inclusive). Default: 21.",
    )
    parser.add_argument(
        "--validation-perm-end",
        type=int,
        default=30,
        help="Last permutation id for true-null extraction (inclusive). Default: 30.",
    )
    parser.add_argument(
        "--permutations-dir",
        type=Path,
        default=REPO_DIR / "data" / "permutations",
        help="Directory containing ###.hetmat permutation folders.",
    )
    parser.add_argument(
        "--base-edges-dir",
        type=Path,
        default=REPO_DIR / "data" / "edges",
        help="Fallback edge directory used when permutation edge files are missing.",
    )
    parser.add_argument(
        "--null-models-dir",
        type=Path,
        default=REPO_DIR / "results" / "null_models",
        help="Directory containing *_rf_null.pkl and *_poly_null.pkl files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "compositional_null",
        help="Output directory for compositional-null artifacts.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of unique degree pairs to process per chunk. Default: 50000.",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable degree-pair caching. Default: enabled.",
    )
    parser.add_argument(
        "--save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable writing true-null checkpoint pickle. Default: enabled.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore any checkpoint and recompute true-null extraction.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of degree pairs for quick smoke tests.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip writing validation plot PNG.",
    )
    parser.add_argument(
        "--transpose-edge1",
        action="store_true",
        help="Transpose edge1 matrix before metapath multiplication.",
    )
    parser.add_argument(
        "--transpose-edge2",
        action="store_true",
        help="Transpose edge2 matrix before metapath multiplication.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.validation_perm_start > args.validation_perm_end:
        raise ValueError("validation-perm-start must be <= validation-perm-end")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be > 0")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("max-pairs must be > 0")


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 2:
        return np.nan, np.nan
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return np.nan, np.nan
    try:
        corr, p_val = pearsonr(y_true, y_pred)
        return float(corr), float(p_val)
    except Exception:
        return np.nan, np.nan


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    with np.errstate(invalid="ignore"):
        val = np.corrcoef(a, b)[0, 1]
    return float(val) if np.isfinite(val) else np.nan


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) < 1e-12:
        return np.nan
    try:
        val = float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan
    return val if np.isfinite(val) else np.nan


def clean_number(value: float) -> float | None:
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None


def resolve_edge_file(
    *,
    permutations_dir: Path,
    base_edges_dir: Path,
    perm_id: int,
    edge_type: str,
) -> Path | None:
    perm_edge = permutations_dir / f"{perm_id:03d}.hetmat" / "edges" / f"{edge_type}.sparse.npz"
    if perm_edge.exists():
        return perm_edge
    base_edge = base_edges_dir / f"{edge_type}.sparse.npz"
    if base_edge.exists():
        return base_edge
    return None


def load_edge_matrix(path: Path, transpose: bool) -> sp.csr_matrix:
    matrix = sp.load_npz(str(path)).tocsr()
    if transpose:
        matrix = matrix.transpose().tocsr()
    return matrix


def load_null_model(edge_type: str, model_type: str, null_models_dir: Path) -> dict[str, object]:
    models: dict[str, object] = {}

    if model_type in {"rf", "ensemble"}:
        rf_file = null_models_dir / f"{edge_type}_rf_null.pkl"
        if rf_file.exists():
            models["rf"] = joblib.load(rf_file)

    if model_type in {"poly", "ensemble"}:
        poly_file = null_models_dir / f"{edge_type}_poly_null.pkl"
        poly_features_file = null_models_dir / f"{edge_type}_poly_features.pkl"
        if poly_file.exists() and poly_features_file.exists():
            models["poly"] = joblib.load(poly_file)
            models["poly_features"] = joblib.load(poly_features_file)

    if model_type == "rf" and "rf" not in models:
        raise FileNotFoundError(f"Missing RF model for {edge_type} in {null_models_dir}")
    if model_type == "poly" and ("poly" not in models or "poly_features" not in models):
        raise FileNotFoundError(f"Missing poly model/features for {edge_type} in {null_models_dir}")
    if model_type == "ensemble" and ("rf" not in models or "poly" not in models or "poly_features" not in models):
        raise FileNotFoundError(f"Missing RF or poly model/components for {edge_type} in {null_models_dir}")

    return models


def get_intermediate_degree_frequency(
    matrix1: sp.csr_matrix,
    matrix2: sp.csr_matrix,
) -> dict[int, float]:
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError(
            "Matrix shape mismatch for metapath composition: "
            f"{matrix1.shape} then {matrix2.shape}."
        )

    incoming = np.asarray(matrix1.sum(axis=0)).ravel().astype(int)
    outgoing = np.asarray(matrix2.sum(axis=1)).ravel().astype(int)
    combined = incoming + outgoing
    combined = combined[combined > 0]
    if len(combined) == 0:
        return {1: 1.0}

    unique_deg, counts = np.unique(combined, return_counts=True)
    freqs = counts / counts.sum()
    return {int(deg): float(freq) for deg, freq in zip(unique_deg, freqs)}


def extract_metapath_frequencies_for_perm(
    *,
    perm_id: int,
    edge1_type: str,
    edge2_type: str,
    permutations_dir: Path,
    base_edges_dir: Path,
    transpose_edge1: bool,
    transpose_edge2: bool,
) -> pd.DataFrame | None:
    edge1_path = resolve_edge_file(
        permutations_dir=permutations_dir,
        base_edges_dir=base_edges_dir,
        perm_id=perm_id,
        edge_type=edge1_type,
    )
    edge2_path = resolve_edge_file(
        permutations_dir=permutations_dir,
        base_edges_dir=base_edges_dir,
        perm_id=perm_id,
        edge_type=edge2_type,
    )
    if edge1_path is None or edge2_path is None:
        return None

    matrix1 = load_edge_matrix(edge1_path, transpose_edge1)
    matrix2 = load_edge_matrix(edge2_path, transpose_edge2)
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError(
            "Matrix shape mismatch for permutation "
            f"{perm_id:03d}: {edge1_type} {matrix1.shape}, {edge2_type} {matrix2.shape}. "
            "Try --transpose-edge1 or --transpose-edge2."
        )

    metapath_matrix = (matrix1 @ matrix2).tocoo()
    if metapath_matrix.nnz == 0:
        return pd.DataFrame(
            columns=[
                "source_idx",
                "target_idx",
                "source_degree",
                "target_degree",
                "metapath_count",
                "perm_id",
            ]
        )

    source_degrees = np.asarray(matrix1.sum(axis=1)).ravel().astype(int)
    target_degrees = np.asarray(matrix2.sum(axis=0)).ravel().astype(int)

    return pd.DataFrame(
        {
            "source_idx": metapath_matrix.row,
            "target_idx": metapath_matrix.col,
            "source_degree": source_degrees[metapath_matrix.row],
            "target_degree": target_degrees[metapath_matrix.col],
            "metapath_count": metapath_matrix.data.astype(int),
            "perm_id": np.full(metapath_matrix.nnz, perm_id, dtype=int),
        }
    )


class OptimizedCompositionalCalculator:
    def __init__(
        self,
        edge1_models: dict[str, object],
        edge2_models: dict[str, object],
        intermediate_degree_freq: dict[int, float],
        *,
        model_type: str,
        use_cache: bool,
    ) -> None:
        self.edge1_models = edge1_models
        self.edge2_models = edge2_models
        self.model_type = model_type
        self.use_cache = use_cache
        self.cache: dict[tuple[int, int], float] = {}

        self.inter_degrees = np.array(list(intermediate_degree_freq.keys()), dtype=float)
        self.inter_freqs = np.array(
            [intermediate_degree_freq[int(d)] for d in self.inter_degrees],
            dtype=float,
        )

    def _batch_predict(self, degree_pairs: np.ndarray, models: dict[str, object]) -> np.ndarray:
        preds: list[np.ndarray] = []

        if self.model_type in {"rf", "ensemble"} and "rf" in models:
            preds.append(models["rf"].predict(degree_pairs))

        if self.model_type in {"poly", "ensemble"} and "poly" in models and "poly_features" in models:
            x_poly = models["poly_features"].transform(degree_pairs)
            preds.append(models["poly"].predict(x_poly))

        if not preds:
            return np.zeros(len(degree_pairs), dtype=float)
        if len(preds) == 1:
            pred = preds[0]
        else:
            pred = np.mean(np.vstack(preds), axis=0)
        return np.clip(pred, 0, 1)

    def _compute_single(self, source_deg: float, target_deg: float) -> float:
        source_inter_pairs = np.column_stack(
            [
                np.full(len(self.inter_degrees), source_deg, dtype=float),
                self.inter_degrees,
            ]
        )
        inter_target_pairs = np.column_stack(
            [
                self.inter_degrees,
                np.full(len(self.inter_degrees), target_deg, dtype=float),
            ]
        )

        p1 = self._batch_predict(source_inter_pairs, self.edge1_models)
        p2 = self._batch_predict(inter_target_pairs, self.edge2_models)
        return float(np.sum(p1 * p2 * self.inter_freqs))

    def compute_metapath_null_vectorized(
        self,
        source_degrees: np.ndarray,
        target_degrees: np.ndarray,
        *,
        chunk_size: int,
    ) -> tuple[np.ndarray, int]:
        pairs = np.column_stack([source_degrees.astype(int), target_degrees.astype(int)])
        unique_pairs, inverse_idx = np.unique(pairs, axis=0, return_inverse=True)
        unique_probs = np.zeros(len(unique_pairs), dtype=float)

        for start in range(0, len(unique_pairs), chunk_size):
            end = min(start + chunk_size, len(unique_pairs))
            chunk = unique_pairs[start:end]
            for local_i, (source_deg, target_deg) in enumerate(chunk):
                key = (int(source_deg), int(target_deg))
                if self.use_cache and key in self.cache:
                    prob = self.cache[key]
                else:
                    prob = self._compute_single(float(source_deg), float(target_deg))
                    if self.use_cache:
                        self.cache[key] = prob
                unique_probs[start + local_i] = prob

            progress = end / len(unique_pairs) * 100 if len(unique_pairs) > 0 else 100.0
            print(
                f"  Processed unique degree pairs: {end:,}/{len(unique_pairs):,} "
                f"({progress:.1f}%)"
            )

        all_probs = unique_probs[inverse_idx]
        return all_probs, len(unique_pairs)


def analyze_errors_by_degree(
    predictions: np.ndarray,
    actuals: np.ndarray,
    source_degrees: np.ndarray,
    target_degrees: np.ndarray,
) -> pd.DataFrame:
    degree_bins = [0, 1, 2, 5, 10, 20, 50, 100, 500, np.inf]
    degree_labels = ["0", "1", "2-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500+"]
    rows = []

    source_bins = pd.cut(source_degrees, bins=degree_bins, labels=degree_labels)
    target_bins = pd.cut(target_degrees, bins=degree_bins, labels=degree_labels)

    for stratification, bins in [("source_degree", source_bins), ("target_degree", target_bins)]:
        for bin_label in degree_labels:
            mask = bins == bin_label
            if int(mask.sum()) < 10:
                continue
            pred = predictions[mask]
            act = actuals[mask]
            rows.append(
                {
                    "stratification": stratification,
                    "bin": bin_label,
                    "n_samples": int(mask.sum()),
                    "mean_prediction": float(pred.mean()),
                    "mean_actual": float(act.mean()),
                    "mae": float(np.abs(pred - act).mean()),
                    "rmse": float(np.sqrt(((pred - act) ** 2).mean())),
                    "correlation": safe_corrcoef(pred, act),
                }
            )

    return pd.DataFrame(rows)


def maybe_write_plot(
    *,
    valid_data: pd.DataFrame,
    error_df: pd.DataFrame,
    corr: float,
    out_file: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.scatter(valid_data["true_null_prob"], valid_data["ml_null_prob"], alpha=0.4, s=16)
    min_x = float(valid_data["true_null_prob"].min())
    max_x = float(valid_data["true_null_prob"].max())
    ax.plot([min_x, max_x], [min_x, max_x], "r--", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True Null Probability")
    ax.set_ylabel("ML-Compositional Null Probability")
    ax.set_title(f"True vs ML Null (r={corr:.3f})")
    ax.grid(alpha=0.3)

    source_err = error_df[error_df["stratification"] == "source_degree"]
    ax = axes[0, 1]
    ax.bar(range(len(source_err)), source_err["mae"], alpha=0.7)
    ax.set_xticks(range(len(source_err)))
    ax.set_xticklabels(source_err["bin"], rotation=45, ha="right")
    ax.set_title("Error by Source Degree (MAE)")
    ax.grid(axis="y", alpha=0.3)

    target_err = error_df[error_df["stratification"] == "target_degree"]
    ax = axes[1, 0]
    ax.bar(range(len(target_err)), target_err["mae"], alpha=0.7, color="orange")
    ax.set_xticks(range(len(target_err)))
    ax.set_xticklabels(target_err["bin"], rotation=45, ha="right")
    ax.set_title("Error by Target Degree (MAE)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.hist(np.log10(valid_data["true_null_prob"] + 1e-12), bins=50, alpha=0.6, density=True, label="True")
    ax.hist(np.log10(valid_data["ml_null_prob"] + 1e-12), bins=50, alpha=0.6, density=True, label="ML")
    ax.set_title("Probability Distributions")
    ax.set_xlabel("log10(probability)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    edge1_models = load_null_model(args.edge1_type, args.model_type, args.null_models_dir)
    edge2_models = load_null_model(args.edge2_type, args.model_type, args.null_models_dir)

    print(f"Metapath: {args.metapath}")
    print(f"Edges: {args.edge1_type} -> {args.edge2_type}")
    print(f"Model type: {args.model_type}")
    print(
        "Validation permutations: "
        f"{args.validation_perm_start}-{args.validation_perm_end}"
    )
    print(f"Permutation directory: {args.permutations_dir}")
    print(f"Null models directory: {args.null_models_dir}")
    print(f"Results directory: {args.results_dir}")

    checkpoint_file = args.results_dir / f"{args.metapath}_true_null_checkpoint.pkl"

    true_null_agg: pd.DataFrame | None = None
    if checkpoint_file.exists() and not args.force_recompute:
        print(f"Loading true-null checkpoint: {checkpoint_file}")
        with checkpoint_file.open("rb") as f:
            loaded = pickle.load(f)
        true_null_agg = loaded if isinstance(loaded, pd.DataFrame) else pd.DataFrame(loaded)
        print(f"Loaded {len(true_null_agg):,} rows from checkpoint")

    if true_null_agg is None:
        all_true_null = []
        for perm_id in range(args.validation_perm_start, args.validation_perm_end + 1):
            print(f"Extracting permutation {perm_id:03d}...")
            df = extract_metapath_frequencies_for_perm(
                perm_id=perm_id,
                edge1_type=args.edge1_type,
                edge2_type=args.edge2_type,
                permutations_dir=args.permutations_dir,
                base_edges_dir=args.base_edges_dir,
                transpose_edge1=args.transpose_edge1,
                transpose_edge2=args.transpose_edge2,
            )
            if df is None:
                print(f"  Missing required edge files for permutation {perm_id:03d}; skipping.")
                continue
            if not df.empty:
                all_true_null.append(df)
                print(f"  Found {len(df):,} metapath pairs")
            else:
                print("  No metapath pairs for this permutation")

        if not all_true_null:
            raise RuntimeError("No metapath frequency data found in requested permutations.")

        true_null_df = pd.concat(all_true_null, ignore_index=True)
        true_null_agg = (
            true_null_df
            .groupby(["source_idx", "target_idx", "source_degree", "target_degree"], as_index=False)
            .agg({"metapath_count": "mean"})
        )
        total_paths = float(true_null_agg["metapath_count"].sum())
        if total_paths > 0:
            true_null_agg["true_null_prob"] = true_null_agg["metapath_count"] / total_paths
        else:
            true_null_agg["true_null_prob"] = 0.0

        if args.save_checkpoint:
            with checkpoint_file.open("wb") as f:
                pickle.dump(true_null_agg, f)
            print(f"Saved true-null checkpoint: {checkpoint_file}")

    if args.max_pairs is not None and len(true_null_agg) > args.max_pairs:
        true_null_agg = true_null_agg.head(args.max_pairs).copy()
        print(f"Limited to first {len(true_null_agg):,} pairs using --max-pairs")

    probe_df = extract_metapath_frequencies_for_perm(
        perm_id=args.validation_perm_start,
        edge1_type=args.edge1_type,
        edge2_type=args.edge2_type,
        permutations_dir=args.permutations_dir,
        base_edges_dir=args.base_edges_dir,
        transpose_edge1=args.transpose_edge1,
        transpose_edge2=args.transpose_edge2,
    )
    if probe_df is None:
        raise RuntimeError("Could not load edge matrices for intermediate degree distribution.")

    edge1_probe_path = resolve_edge_file(
        permutations_dir=args.permutations_dir,
        base_edges_dir=args.base_edges_dir,
        perm_id=args.validation_perm_start,
        edge_type=args.edge1_type,
    )
    edge2_probe_path = resolve_edge_file(
        permutations_dir=args.permutations_dir,
        base_edges_dir=args.base_edges_dir,
        perm_id=args.validation_perm_start,
        edge_type=args.edge2_type,
    )
    if edge1_probe_path is None or edge2_probe_path is None:
        raise RuntimeError("Could not resolve probe edge files for degree distribution.")

    matrix1_probe = load_edge_matrix(edge1_probe_path, args.transpose_edge1)
    matrix2_probe = load_edge_matrix(edge2_probe_path, args.transpose_edge2)
    inter_degree_freq = get_intermediate_degree_frequency(matrix1_probe, matrix2_probe)
    print(
        "Intermediate degree distribution: "
        f"{len(inter_degree_freq)} unique degrees"
    )

    calculator = OptimizedCompositionalCalculator(
        edge1_models=edge1_models,
        edge2_models=edge2_models,
        intermediate_degree_freq=inter_degree_freq,
        model_type=args.model_type,
        use_cache=args.use_cache,
    )

    print(f"Computing compositional null for {len(true_null_agg):,} pairs...")
    ml_null_probs, n_unique_degree_pairs = calculator.compute_metapath_null_vectorized(
        true_null_agg["source_degree"].to_numpy(),
        true_null_agg["target_degree"].to_numpy(),
        chunk_size=args.chunk_size,
    )
    true_null_agg = true_null_agg.copy()
    true_null_agg["ml_null_prob"] = ml_null_probs

    valid_mask = (true_null_agg["true_null_prob"] > 0) & (true_null_agg["ml_null_prob"] > 0)
    valid_data = true_null_agg[valid_mask].copy()
    if len(valid_data) < 2:
        raise RuntimeError("Not enough valid rows after filtering positive probabilities.")

    corr, p_val = safe_pearsonr(valid_data["true_null_prob"].to_numpy(), valid_data["ml_null_prob"].to_numpy())
    mae = float(np.abs(valid_data["true_null_prob"] - valid_data["ml_null_prob"]).mean())
    rmse = float(np.sqrt(((valid_data["true_null_prob"] - valid_data["ml_null_prob"]) ** 2).mean()))
    r2 = safe_r2(
        valid_data["true_null_prob"].to_numpy(),
        valid_data["ml_null_prob"].to_numpy(),
    )
    try:
        ks_stat, ks_p = ks_2samp(valid_data["true_null_prob"], valid_data["ml_null_prob"])
        ks_stat = float(ks_stat)
        ks_p = float(ks_p)
    except Exception:
        ks_stat, ks_p = np.nan, np.nan

    print(f"Validation pairs: {len(valid_data):,} / {len(true_null_agg):,}")
    print(f"Pearson r: {corr:.4f}")
    print(f"MAE: {mae:.6e}")
    print(f"RMSE: {rmse:.6e}")
    print(f"R2: {r2:.4f}")
    print(f"KS stat: {ks_stat:.4f}")

    error_df = analyze_errors_by_degree(
        valid_data["ml_null_prob"].to_numpy(),
        valid_data["true_null_prob"].to_numpy(),
        valid_data["source_degree"].to_numpy(),
        valid_data["target_degree"].to_numpy(),
    )

    out_validation_opt = args.results_dir / f"{args.metapath}_null_validation_optimized.csv"
    out_validation_plain = args.results_dir / f"{args.metapath}_null_validation.csv"
    out_error = args.results_dir / f"{args.metapath}_error_analysis.csv"
    out_summary_csv = args.results_dir / f"{args.metapath}_summary.csv"
    out_summary_json = args.results_dir / f"{args.metapath}_summary_optimized.json"

    true_null_agg.to_csv(out_validation_opt, index=False)
    valid_data.to_csv(out_validation_plain, index=False)
    error_df.to_csv(out_error, index=False)

    summary = {
        "metapath": args.metapath,
        "edge_types": [args.edge1_type, args.edge2_type],
        "model_type": args.model_type,
        "n_pairs_total": int(len(true_null_agg)),
        "n_pairs_valid": int(len(valid_data)),
        "pearson_r": clean_number(corr),
        "pearson_pvalue": clean_number(p_val),
        "mae": clean_number(mae),
        "rmse": clean_number(rmse),
        "r2": clean_number(r2),
        "ks_stat": clean_number(ks_stat),
        "ks_pvalue": clean_number(ks_p),
        "validation_perm_range": [args.validation_perm_start, args.validation_perm_end],
        "optimization_settings": {
            "chunk_size": args.chunk_size,
            "use_cache": args.use_cache,
            "cache_size": len(calculator.cache),
            "unique_degree_pairs": int(n_unique_degree_pairs),
        },
    }

    pd.DataFrame([summary]).to_csv(out_summary_csv, index=False)
    out_summary_json.write_text(json.dumps(summary, indent=2))

    if not args.skip_plot:
        plot_file = args.results_dir / f"{args.metapath}_validation.png"
        maybe_write_plot(valid_data=valid_data, error_df=error_df, corr=corr, out_file=plot_file)
        print(f"Saved: {plot_file}")

    print("Saved outputs:")
    print(f"  - {out_validation_opt}")
    print(f"  - {out_validation_plain}")
    print(f"  - {out_error}")
    print(f"  - {out_summary_csv}")
    print(f"  - {out_summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
