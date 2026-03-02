"""Build metapath null distributions.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import ks_2samp, pearsonr


REPO_DIR = Path(__file__).resolve().parents[1]


@dataclass
class MetapathSpec:
    name: str
    edge1: str
    edge2: str
    transpose_edge1: bool = False
    transpose_edge2: bool = False
    description: str = ""


DEFAULT_METAPATHS: list[MetapathSpec] = [
    MetapathSpec(
        name="CbGpPW",
        edge1="CbG",
        edge2="GpPW",
        description="Compound binds Gene participates in Pathway",
    ),
    MetapathSpec(
        name="CtDaG",
        edge1="CtD",
        edge2="DaG",
        description="Compound treats Disease associates with Gene",
    ),
    MetapathSpec(
        name="CrCbG",
        edge1="CrC",
        edge2="CbG",
        description="Compound resembles Compound binds Gene",
    ),
    MetapathSpec(
        name="CbGaD",
        edge1="CbG",
        edge2="DaG",
        transpose_edge2=True,
        description="Compound binds Gene associates with Disease",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute observed 2-edge metapath path probabilities and compare "
            "against compositional null predictions built from edge null models."
        )
    )
    parser.add_argument(
        "--metapath",
        action="append",
        default=[],
        help=(
            "Default metapath name to run (repeatable). "
            f"Available: {', '.join(mp.name for mp in DEFAULT_METAPATHS)}"
        ),
    )
    parser.add_argument(
        "--metapath-spec",
        action="append",
        default=[],
        help=(
            "Custom metapath spec in format "
            "NAME:EDGE1:EDGE2[:TRANSPOSE_EDGE1[:TRANSPOSE_EDGE2]]. "
            "Transpose flags accept true/false."
        ),
    )
    parser.add_argument(
        "--model-type",
        action="append",
        choices=["rf", "poly", "ensemble"],
        default=[],
        help="Model type to evaluate (repeatable). Default: rf and poly.",
    )
    parser.add_argument(
        "--observed-edges-dir",
        type=Path,
        default=REPO_DIR / "data" / "edges",
        help="Directory containing observed edge matrices (*.sparse.npz).",
    )
    parser.add_argument(
        "--null-models-dir",
        type=Path,
        default=REPO_DIR / "results" / "null_models",
        help="Directory containing trained null models from train-null-models.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "metapath_nulls",
        help="Output directory for metapath analysis CSV/plot artifacts.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on metapath pairs per metapath for quick smoke tests.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for unique degree-pair null predictions. Default: 10000.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip per-metapath and cross-model plots.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue processing remaining metapaths if one fails. Default: true.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("max-pairs must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be > 0")


def parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in {"true", "1", "yes", "y", "t"}:
        return True
    if lowered in {"false", "0", "no", "n", "f"}:
        return False
    raise ValueError(f"Invalid boolean value: {text}")


def parse_metapath_spec(spec_text: str) -> MetapathSpec:
    parts = spec_text.split(":")
    if len(parts) < 3 or len(parts) > 5:
        raise ValueError(
            "metapath-spec must be NAME:EDGE1:EDGE2[:TRANSPOSE_EDGE1[:TRANSPOSE_EDGE2]]"
        )
    name, edge1, edge2 = parts[0], parts[1], parts[2]
    t1 = parse_bool(parts[3]) if len(parts) >= 4 else False
    t2 = parse_bool(parts[4]) if len(parts) >= 5 else False
    return MetapathSpec(name=name, edge1=edge1, edge2=edge2, transpose_edge1=t1, transpose_edge2=t2)


def resolve_metapaths(args: argparse.Namespace) -> list[MetapathSpec]:
    default_map = {mp.name: mp for mp in DEFAULT_METAPATHS}
    selected: list[MetapathSpec] = []

    if args.metapath:
        for name in args.metapath:
            if name not in default_map:
                raise ValueError(
                    f"Unknown metapath '{name}'. "
                    f"Choose from: {', '.join(default_map)}"
                )
            selected.append(default_map[name])
    else:
        selected.extend(DEFAULT_METAPATHS)

    for spec_text in args.metapath_spec:
        selected.append(parse_metapath_spec(spec_text))

    if not selected:
        raise ValueError("No metapaths selected.")
    return selected


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


def safe_ks(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    try:
        stat, p_val = ks_2samp(a, b)
        return float(stat), float(p_val)
    except Exception:
        return np.nan, np.nan


def safe_number(value: float) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def load_edge_matrix(path: Path, transpose: bool) -> sp.csr_matrix:
    matrix = sp.load_npz(str(path)).tocsr()
    return matrix.transpose().tocsr() if transpose else matrix


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
        raise FileNotFoundError(f"Missing RF model for {edge_type}")
    if model_type == "poly" and ("poly" not in models or "poly_features" not in models):
        raise FileNotFoundError(f"Missing poly model/features for {edge_type}")
    if model_type == "ensemble" and ("rf" not in models or "poly" not in models or "poly_features" not in models):
        raise FileNotFoundError(f"Missing RF or poly components for {edge_type}")
    return models


class DegreePairPredictor:
    def __init__(
        self,
        edge1_models: dict[str, object],
        edge2_models: dict[str, object],
        intermediate_degree_freq: dict[int, float],
        model_type: str,
    ) -> None:
        self.edge1_models = edge1_models
        self.edge2_models = edge2_models
        self.model_type = model_type
        self.inter_degrees = np.array(list(intermediate_degree_freq.keys()), dtype=float)
        self.inter_freqs = np.array(
            [intermediate_degree_freq[int(d)] for d in self.inter_degrees],
            dtype=float,
        )

    def _predict_batch(self, x: np.ndarray, models: dict[str, object]) -> np.ndarray:
        preds: list[np.ndarray] = []
        if self.model_type in {"rf", "ensemble"} and "rf" in models:
            preds.append(models["rf"].predict(x))
        if self.model_type in {"poly", "ensemble"} and "poly" in models and "poly_features" in models:
            preds.append(models["poly"].predict(models["poly_features"].transform(x)))
        if not preds:
            return np.zeros(len(x), dtype=float)
        pred = preds[0] if len(preds) == 1 else np.mean(np.vstack(preds), axis=0)
        return np.clip(pred, 0, 1)

    def _predict_single(self, source_deg: float, target_deg: float) -> float:
        source_to_inter = np.column_stack(
            [np.full(len(self.inter_degrees), source_deg, dtype=float), self.inter_degrees]
        )
        inter_to_target = np.column_stack(
            [self.inter_degrees, np.full(len(self.inter_degrees), target_deg, dtype=float)]
        )
        p1 = self._predict_batch(source_to_inter, self.edge1_models)
        p2 = self._predict_batch(inter_to_target, self.edge2_models)
        return float(np.sum(p1 * p2 * self.inter_freqs))

    def predict(
        self,
        source_degrees: np.ndarray,
        target_degrees: np.ndarray,
        chunk_size: int,
    ) -> tuple[np.ndarray, int]:
        pairs = np.column_stack([source_degrees.astype(int), target_degrees.astype(int)])
        unique_pairs, inverse_idx = np.unique(pairs, axis=0, return_inverse=True)
        unique_pred = np.zeros(len(unique_pairs), dtype=float)

        for start in range(0, len(unique_pairs), chunk_size):
            end = min(start + chunk_size, len(unique_pairs))
            for i, (src_deg, tgt_deg) in enumerate(unique_pairs[start:end], start=start):
                unique_pred[i] = self._predict_single(float(src_deg), float(tgt_deg))
            print(f"  Degree pairs processed: {end:,}/{len(unique_pairs):,}")

        return unique_pred[inverse_idx], len(unique_pairs)


def compute_intermediate_degree_frequency(matrix1: sp.csr_matrix, matrix2: sp.csr_matrix) -> dict[int, float]:
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError(f"Edge matrix shape mismatch: {matrix1.shape} then {matrix2.shape}")
    incoming = np.asarray(matrix1.sum(axis=0)).ravel().astype(int)
    outgoing = np.asarray(matrix2.sum(axis=1)).ravel().astype(int)
    combined = incoming + outgoing
    combined = combined[combined > 0]
    if len(combined) == 0:
        return {1: 1.0}
    unique_deg, counts = np.unique(combined, return_counts=True)
    freqs = counts / counts.sum()
    return {int(d): float(f) for d, f in zip(unique_deg, freqs)}


def maybe_write_metapath_plot(df: pd.DataFrame, output_file: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(df["null_prediction"], df["path_probability"], alpha=0.25, s=8)
    max_val = float(max(df["null_prediction"].max(), df["path_probability"].max()))
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.6)
    ax.set_xlabel("Null Prediction")
    ax.set_ylabel("Observed Path Probability")
    ax.set_title("Null vs Observed")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(df["residual"], bins=60, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Observed - Null")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(df["source_degree"], df["residual"], alpha=0.2, s=6)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Source Degree")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Source Degree")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(df["target_degree"], df["residual"], alpha=0.2, s=6, color="orange")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Target Degree")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Target Degree")
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def maybe_write_comparison_plot(comparison_df: pd.DataFrame, output_file: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x = np.arange(len(comparison_df))
    width = 0.35
    ax.bar(x - width / 2, comparison_df["correlation_rf"], width, label="RF", alpha=0.8)
    ax.bar(x + width / 2, comparison_df["correlation_poly"], width, label="Poly", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["metapath"], rotation=45, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - width / 2, comparison_df["rmse_rf"], width, label="RF", alpha=0.8)
    ax.bar(x + width / 2, comparison_df["rmse_poly"], width, label="Poly", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["metapath"], rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_metapath(
    spec: MetapathSpec,
    model_type: str,
    args: argparse.Namespace,
) -> dict[str, float | int | str | None]:
    edge1_path = args.observed_edges_dir / f"{spec.edge1}.sparse.npz"
    edge2_path = args.observed_edges_dir / f"{spec.edge2}.sparse.npz"
    if not edge1_path.exists():
        raise FileNotFoundError(f"Missing edge matrix: {edge1_path}")
    if not edge2_path.exists():
        raise FileNotFoundError(f"Missing edge matrix: {edge2_path}")

    print(f"\n{'='*70}")
    print(f"{spec.name} ({spec.edge1} -> {spec.edge2}) | model={model_type}")
    if spec.description:
        print(spec.description)
    print(f"{'='*70}")

    matrix1 = load_edge_matrix(edge1_path, spec.transpose_edge1)
    matrix2 = load_edge_matrix(edge2_path, spec.transpose_edge2)
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError(
            f"Shape mismatch for {spec.name}: {matrix1.shape} then {matrix2.shape}. "
            "Adjust transpose flags."
        )

    metapath_matrix = (matrix1 @ matrix2).tocoo()
    source_degrees = np.asarray(matrix1.sum(axis=1)).ravel().astype(int)
    target_degrees = np.asarray(matrix2.sum(axis=0)).ravel().astype(int)

    df = pd.DataFrame(
        {
            "source_id": metapath_matrix.row,
            "target_id": metapath_matrix.col,
            "path_count": metapath_matrix.data.astype(float),
            "source_degree": source_degrees[metapath_matrix.row],
            "target_degree": target_degrees[metapath_matrix.col],
        }
    )
    print(f"Observed metapath pairs: {len(df):,}")

    if args.max_pairs is not None and len(df) > args.max_pairs:
        df = df.head(args.max_pairs).copy()
        print(f"Limited to {len(df):,} rows (--max-pairs)")

    total_paths = float(df["path_count"].sum())
    if total_paths <= 0:
        raise RuntimeError(f"No positive path counts for {spec.name}")
    df["path_probability"] = df["path_count"] / total_paths

    edge1_models = load_null_model(spec.edge1, model_type, args.null_models_dir)
    edge2_models = load_null_model(spec.edge2, model_type, args.null_models_dir)
    inter_freq = compute_intermediate_degree_frequency(matrix1, matrix2)
    predictor = DegreePairPredictor(edge1_models, edge2_models, inter_freq, model_type=model_type)

    null_pred, unique_pairs = predictor.predict(
        source_degrees=df["source_degree"].to_numpy(),
        target_degrees=df["target_degree"].to_numpy(),
        chunk_size=args.chunk_size,
    )
    df["null_prediction"] = null_pred
    df["residual"] = df["path_probability"] - df["null_prediction"]
    df["abs_residual"] = np.abs(df["residual"])

    corr, corr_p = safe_pearsonr(df["null_prediction"].to_numpy(), df["path_probability"].to_numpy())
    rmse = float(np.sqrt(np.mean((df["null_prediction"] - df["path_probability"]) ** 2)))
    mae = float(np.abs(df["null_prediction"] - df["path_probability"]).mean())
    ks_stat, ks_p = safe_ks(df["null_prediction"].to_numpy(), df["path_probability"].to_numpy())

    print(f"Correlation: {corr:.4f}")
    print(f"MAE: {mae:.6e}")
    print(f"RMSE: {rmse:.6e}")
    print(f"KS: {ks_stat:.4f}")

    analysis_file = args.results_dir / f"{spec.name}_{model_type}_analysis.csv"
    df.to_csv(analysis_file, index=False)
    print(f"Saved: {analysis_file}")

    if not args.skip_plot:
        plot_file = args.results_dir / f"{spec.name}_{model_type}_analysis.png"
        maybe_write_metapath_plot(df, plot_file, title=f"{spec.name} ({model_type})")
        print(f"Saved: {plot_file}")

    return {
        "metapath": spec.name,
        "edge1": spec.edge1,
        "edge2": spec.edge2,
        "model_type": model_type,
        "n_pairs": int(len(df)),
        "unique_degree_pairs": int(unique_pairs),
        "correlation": safe_number(corr),
        "correlation_pvalue": safe_number(corr_p),
        "mae": safe_number(mae),
        "rmse": safe_number(rmse),
        "ks_stat": safe_number(ks_stat),
        "ks_pvalue": safe_number(ks_p),
    }


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    metapaths = resolve_metapaths(args)
    model_types = args.model_type if args.model_type else ["rf", "poly"]

    print(f"Observed edge directory: {args.observed_edges_dir}")
    print(f"Null models directory: {args.null_models_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Metapaths selected: {len(metapaths)}")
    print(f"Model types: {', '.join(model_types)}")

    all_rows: list[dict[str, float | int | str | None]] = []

    for model_type in model_types:
        model_rows: list[dict[str, float | int | str | None]] = []
        for spec in metapaths:
            try:
                row = run_metapath(spec, model_type, args)
                model_rows.append(row)
            except Exception as exc:
                msg = f"Failed {spec.name} ({model_type}): {exc}"
                if args.continue_on_error:
                    print(f"WARNING: {msg}")
                    continue
                raise RuntimeError(msg) from exc

        summary_df = pd.DataFrame(model_rows)
        summary_file = args.results_dir / f"summary_{model_type}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary: {summary_file}")
        all_rows.extend(model_rows)

    if set(model_types) >= {"rf", "poly"}:
        all_df = pd.DataFrame(all_rows)
        rf_df = all_df[all_df["model_type"] == "rf"][
            ["metapath", "correlation", "rmse"]
        ].rename(columns={"correlation": "correlation_rf", "rmse": "rmse_rf"})
        poly_df = all_df[all_df["model_type"] == "poly"][
            ["metapath", "correlation", "rmse"]
        ].rename(columns={"correlation": "correlation_poly", "rmse": "rmse_poly"})
        comparison = rf_df.merge(poly_df, on="metapath", how="inner")
        comparison_file = args.results_dir / "model_comparison.csv"
        comparison.to_csv(comparison_file, index=False)
        print(f"Saved comparison: {comparison_file}")

        if not args.skip_plot and not comparison.empty:
            comparison_plot = args.results_dir / "model_comparison.png"
            maybe_write_comparison_plot(comparison, comparison_plot)
            print(f"Saved: {comparison_plot}")

    print("\nMetapath null generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
