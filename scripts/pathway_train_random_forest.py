"""Train random forest baseline models for pathway NN data (notebook 18e migration)."""

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
from sklearn.ensemble import RandomForestRegressor

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
class RandomForestBaselineModel:
    n_estimators: int = 100
    max_depth: int | None = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    n_jobs: int = 1

    def __post_init__(self) -> None:
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaselineModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_feature_importances(self) -> np.ndarray:
        return np.asarray(self._model.feature_importances_, dtype=float)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def save(self, path: Path) -> None:
        payload = {
            "model_name": "RandomForestBaselineModel",
            "params": self.get_params(),
            "estimator": self._model,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train random forest baseline model(s) from pathway training data. "
            "Script-first replacement for notebook 18e."
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
        help="Output directory for <metapath>_Random_Forest.pkl.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "benchmarks",
        help="Output directory for <metapath>_Random_Forest_benchmark.json.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Train split fraction (default: 0.8 for notebook parity).",
    )
    parser.add_argument("--n-estimators", type=int, default=100, help="Random forest n_estimators.")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Random forest max_depth (set <=0 for unlimited).",
    )
    parser.add_argument("--min-samples-split", type=int, default=2, help="Random forest min_samples_split.")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Random forest min_samples_leaf.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Random forest parallel jobs.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--record-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write benchmark JSON outputs.",
    )
    parser.add_argument("--n-cores", type=int, default=4, help="Cores used for benchmark accounting.")
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
    if args.n_estimators <= 0:
        raise ValueError("--n-estimators must be > 0")
    if args.min_samples_split <= 1:
        raise ValueError("--min-samples-split must be > 1")
    if args.min_samples_leaf <= 0:
        raise ValueError("--min-samples-leaf must be > 0")
    if args.n_cores <= 0:
        raise ValueError("--n-cores must be > 0")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be 0")
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


def record_validation_safe(benchmarker: ModelBenchmarker, predictions: np.ndarray, actuals: np.ndarray) -> None:
    try:
        benchmarker.record_validation(predictions, actuals)
    except Exception:
        benchmarker.validation_r = float("nan")
        benchmarker.validation_mae = float(np.mean(np.abs(predictions - actuals)))
        benchmarker.validation_rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))


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

    max_depth = args.max_depth if args.max_depth > 0 else None
    model = RandomForestBaselineModel(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_seed,
        n_jobs=args.n_jobs,
    )

    benchmarker = ModelBenchmarker(
        model_name="Random_Forest",
        metapath=metapath,
        n_training_samples=len(X_train),
        n_cores=args.n_cores,
    )

    with benchmarker.time_training():
        model.fit(X_train, y_train)

    with benchmarker.time_prediction():
        predictions = model.predict(X_test)

    record_validation_safe(benchmarker, predictions, y_test)

    importances = model.get_feature_importances()
    top_indices = np.argsort(importances)[-5:][::-1]
    top_features: list[dict[str, Any]] = []
    for idx in top_indices:
        if int(idx) < 2:
            feature_name = ["source_bin", "target_bin"][int(idx)]
        else:
            feature_name = f"inter_sig_{int(idx) - 2}"
        top_features.append({"feature": feature_name, "importance": float(importances[idx])})

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_file = args.model_dir / f"{metapath}_Random_Forest.pkl"
    model.save(model_file)

    benchmark_file: Path | None = None
    benchmark_result: dict[str, Any] | None = None
    if args.record_benchmarks:
        args.benchmark_dir.mkdir(parents=True, exist_ok=True)
        result = benchmarker.finalize()
        benchmark_file = args.benchmark_dir / f"{metapath}_Random_Forest_benchmark.json"
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
        "top_feature_importances": top_features,
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
        print(f"TRAIN RANDOM FOREST: {metapath}")
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
            print("Top features:")
            for row in summary["top_feature_importances"]:
                print(f"  {row['feature']}: {row['importance']:.4f}")
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            failed.append({"metapath": metapath, "error": message})
            print(f"FAILED {metapath}: {message}")
            if not args.continue_on_error:
                raise

    run_summary_file = args.benchmark_dir / "pathway_train_random_forest_run_summary.json"
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
