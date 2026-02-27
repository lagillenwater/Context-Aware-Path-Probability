"""Train Negative Binomial GLM models for pathway NN data (notebook 18d migration)."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_DIR / ".cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

from src.benchmarking import ModelBenchmarker

DEFAULT_METAPATHS = [
    "CbGpPW",
    "CtDaG",
    "CbGaD",
    "CrCbG",
    "CbGiG",
    "CpDaG",
    "CbGpBP",
    "CbGpCC",
]


@dataclass
class NegativeBinomialGLMModel:
    """Notebook-18d style Negative Binomial GLM wrapper with fallback mode."""

    max_iter: int = 1000
    tol: float = 1e-6
    mode_: str = "unfit"
    feature_mean_: np.ndarray | None = None
    feature_std_: np.ndarray | None = None
    params_: np.ndarray | None = None
    alpha_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NegativeBinomialGLMModel":
        self.feature_mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.feature_std_ = np.where(std > 0, std, 1.0)
        X_scaled = (X - self.feature_mean_) / self.feature_std_

        y_clean = np.clip(y.astype(float), 0.0, None)
        # Slight offset avoids issues on all-zero folds.
        y_for_fit = y_clean + 1e-8

        try:
            import statsmodels.api as sm

            X_const = sm.add_constant(X_scaled, has_constant="add")
            model = sm.GLM(
                y_for_fit,
                X_const,
                family=sm.families.NegativeBinomial(),
            )
            result = model.fit(maxiter=self.max_iter, tol=self.tol, disp=0)
            self.params_ = np.asarray(result.params, dtype=float)
            self.alpha_ = float(getattr(model.family, "alpha", 1.0))
            self.mode_ = "negbin_glm"
            return self
        except Exception:
            # Deterministic fallback: linear model in log-count space.
            X_const = np.column_stack([np.ones(len(X_scaled)), X_scaled])
            target = np.log1p(y_clean)
            coef, *_ = np.linalg.lstsq(X_const, target, rcond=None)
            self.params_ = coef.astype(float)
            self.alpha_ = 0.0
            self.mode_ = "loglinear_fallback"
            return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params_ is None or self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError("Model has not been fit.")
        X_scaled = (X - self.feature_mean_) / self.feature_std_
        X_const = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        eta = X_const @ self.params_
        mu = np.exp(eta)
        # Guard against overflow or invalid values.
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=0.0)
        return mu

    def get_params(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "tol": self.tol,
            "mode": self.mode_,
            "alpha": self.alpha_,
        }

    def save(self, path: Path) -> None:
        payload = {
            "model_name": "NegativeBinomialGLMModel",
            "params": self.get_params(),
            "feature_mean": self.feature_mean_.tolist() if self.feature_mean_ is not None else None,
            "feature_std": self.feature_std_.tolist() if self.feature_std_ is not None else None,
            "coefficients": self.params_.tolist() if self.params_ is not None else None,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Negative Binomial GLM model(s) from pathway training data. "
            "Script-first replacement for notebook 18d."
        )
    )
    parser.add_argument(
        "--metapath",
        action="append",
        default=[],
        help=(
            "Metapath to train (repeatable). "
            f"Supported defaults: {', '.join(DEFAULT_METAPATHS)}"
        ),
    )
    parser.add_argument(
        "--all-metapaths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train all default metapaths.",
    )
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "training_data",
        help="Directory containing <metapath>_training_data.csv files.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "trained_models",
        help="Output directory for <metapath>_NegBin_GLM.pkl.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "benchmarks",
        help="Output directory for <metapath>_NegBin_GLM_benchmark.json.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Train split fraction (default: 0.8 for notebook parity).",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for GLM solver.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance for GLM solver.")
    parser.add_argument(
        "--record-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write benchmark JSON outputs.",
    )
    parser.add_argument("--n-cores", type=int, default=1, help="Cores used for benchmark accounting.")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue processing remaining metapaths if one fails.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not (0 < args.train_fraction < 1):
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be > 0")
    if args.tol <= 0:
        raise ValueError("--tol must be > 0")
    if args.n_cores <= 0:
        raise ValueError("--n-cores must be > 0")
    unknown = [m for m in args.metapath if m not in DEFAULT_METAPATHS]
    if unknown:
        raise ValueError(f"Unknown metapath(s): {', '.join(unknown)}")


def resolve_metapaths(args: argparse.Namespace) -> list[str]:
    if args.all_metapaths:
        return list(DEFAULT_METAPATHS)
    if args.metapath:
        return args.metapath
    return ["CbGpPW"]


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    if n < 4:
        raise ValueError("Need at least 4 rows for benchmarkable train/test split.")

    n_train = int(train_fraction * n)
    n_train = max(2, min(n - 2, n_train))
    indices = np.random.RandomState(random_seed).permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def run_one_metapath(metapath: str, args: argparse.Namespace) -> dict[str, Any]:
    data_file = args.training_data_dir / f"{metapath}_training_data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_file}")

    df = pd.read_csv(data_file)
    feature_cols = [c for c in df.columns if c.startswith("inter_sig_")]
    X = df[["source_bin", "target_bin"] + feature_cols].to_numpy()
    y = df["pathway_count_mean"].to_numpy()

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        random_seed=args.random_seed,
        train_fraction=args.train_fraction,
    )

    benchmarker = ModelBenchmarker(
        model_name="NegBin_GLM",
        metapath=metapath,
        n_training_samples=len(X_train),
        n_cores=args.n_cores,
    )
    model = NegativeBinomialGLMModel(max_iter=args.max_iter, tol=args.tol)

    with benchmarker.time_training():
        model.fit(X_train, y_train)

    with benchmarker.time_prediction():
        predictions = model.predict(X_test)

    benchmarker.record_validation(predictions, y_test)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_file = args.model_dir / f"{metapath}_NegBin_GLM.pkl"
    model.save(model_file)

    benchmark_file: Path | None = None
    benchmark_result: dict[str, Any] | None = None
    if args.record_benchmarks:
        args.benchmark_dir.mkdir(parents=True, exist_ok=True)
        result = benchmarker.finalize()
        benchmark_file = args.benchmark_dir / f"{metapath}_NegBin_GLM_benchmark.json"
        result.save_json(benchmark_file)
        benchmark_result = result.to_dict()

    return {
        "metapath": metapath,
        "training_data_file": str(data_file),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "model_file": str(model_file),
        "benchmark_file": str(benchmark_file) if benchmark_file else None,
        "model_params": model.get_params(),
        "benchmark": benchmark_result,
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
    metapaths = resolve_metapaths(args)

    print(f"Training data dir: {args.training_data_dir}")
    print(f"Model dir: {args.model_dir}")
    print(f"Benchmark dir: {args.benchmark_dir}")
    print(f"Metapaths: {', '.join(metapaths)}")

    successful: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for metapath in metapaths:
        print("=" * 80)
        print(f"TRAIN NEGBIN GLM: {metapath}")
        print("=" * 80)
        try:
            summary = run_one_metapath(metapath, args)
            successful.append(summary)
            print(f"Saved model: {summary['model_file']}")
            if summary["benchmark_file"] is not None:
                print(f"Saved benchmark: {summary['benchmark_file']}")
                bench = summary["benchmark"]
                print(
                    f"Validation: r={bench['validation_r']:.4f}, "
                    f"MAE={bench['validation_mae']:.4f}, "
                    f"RMSE={bench['validation_rmse']:.4f}"
                )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            failed.append({"metapath": metapath, "error": message})
            print(f"FAILED {metapath}: {message}")
            if not args.continue_on_error:
                raise

    run_summary_file = args.benchmark_dir / "pathway_train_negbin_glm_run_summary.json"
    run_summary_file.parent.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": args_to_json(args),
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
