"""Train degree-signature neural network models for pathway NN data (notebook 18f migration)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_DIR = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(REPO_DIR / ".cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(REPO_DIR / ".cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

import matplotlib.pyplot as plt

from src.benchmarking import ModelBenchmarker
from src.pathway_model_io import predict_degree_sig_nn, save_degree_sig_nn
from src.pathway_models_v2 import DegreeSignatureNN

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train degree-signature neural network model(s) from pathway training data. "
            "Script-first replacement for notebook 18f."
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
        help="Output directory for <metapath>_Degree_Sig_NN.pt.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "benchmarks",
        help="Output directory for <metapath>_Degree_Sig_NN_benchmark.json.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "visualizations",
        help="Output directory for <metapath>_Degree_Sig_NN_validation.png.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "intermediate",
        help="Output directory for test .npy reproducibility artifacts.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Train split fraction (default: 0.9 for notebook parity).",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[128, 64, 32],
        help="Hidden layer dimensions (default: 128 64 32).",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--n-epochs", type=int, default=1000, help="Maximum training epochs.")
    parser.add_argument("--early-stopping-patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Min val-loss delta.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Torch device selection (default: cpu).",
    )
    parser.add_argument("--predict-batch-size", type=int, default=4096, help="Batch size for predictions.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--record-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write benchmark JSON outputs.",
    )
    parser.add_argument("--n-cores", type=int, default=1, help="Cores used for benchmark accounting.")
    parser.add_argument(
        "--skip-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip validation plot generation.",
    )
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
    if any(x <= 0 for x in args.hidden_dims):
        raise ValueError("--hidden-dims values must be > 0")
    if not (0 <= args.dropout < 1):
        raise ValueError("--dropout must be in [0, 1)")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.n_epochs <= 0:
        raise ValueError("--n-epochs must be > 0")
    if args.early_stopping_patience <= 0:
        raise ValueError("--early-stopping-patience must be > 0")
    if args.predict_batch_size <= 0:
        raise ValueError("--predict-batch-size must be > 0")
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return device_arg


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


def train_model(
    model: DegreeSignatureNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    n_epochs: int,
    patience: int,
    min_delta: float,
    device: str,
) -> dict[str, Any]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict[str, float]] = []

    model.to(device)
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_examples = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            batch_size_actual = len(xb)
            train_loss_sum += float(loss.item()) * batch_size_actual
            n_train_examples += batch_size_actual

        train_loss = train_loss_sum / max(1, n_train_examples)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).squeeze(-1)
            val_loss = float(criterion(val_preds, y_val_t).item())

        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    return {
        "epochs_ran": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
    }


def record_validation_safe(benchmarker: ModelBenchmarker, predictions: np.ndarray, actuals: np.ndarray) -> None:
    try:
        benchmarker.record_validation(predictions, actuals)
    except Exception:
        benchmarker.validation_r = float("nan")
        benchmarker.validation_mae = float(np.mean(np.abs(predictions - actuals)))
        benchmarker.validation_rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def maybe_write_plot(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metapath: str,
    output_file: Path,
    validation_r: float,
    validation_mae: float,
    validation_rmse: float,
) -> None:
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors="k", linewidth=0.5)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual Pathway Count (Mean)")
    axes[0].set_ylabel("Predicted Pathway Count")
    axes[0].set_title(f"Predicted vs Actual (Degree Bins)\nr = {validation_r:.4f}, n = {len(y_true)} bins")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors="k", linewidth=0.5)
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Pathway Count")
    axes[1].set_ylabel("Residual (Predicted - Actual)")
    axes[1].set_title(f"Residual Plot\nMAE = {validation_mae:.4f}")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(y_true, bins=20, alpha=0.5, label="Actual", color="blue", edgecolor="black")
    axes[2].hist(y_pred, bins=20, alpha=0.5, label="Predicted", color="red", edgecolor="black")
    axes[2].set_xlabel("Pathway Count")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title(f"Distribution Comparison (Degree Bins)\nRMSE = {validation_rmse:.4f}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_one_metapath(metapath: str, args: argparse.Namespace, *, device: str) -> dict[str, Any]:
    data_file = args.training_data_dir / f"{metapath}_training_data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_file}")

    df = pd.read_csv(data_file)
    feature_cols = [c for c in df.columns if c.startswith("inter_sig_")]
    all_feature_cols = ["source_bin", "target_bin"] + feature_cols

    X = df[all_feature_cols].to_numpy(dtype=np.float32)
    y = df["pathway_count_mean"].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        random_seed=args.random_seed,
        train_fraction=args.train_fraction,
    )

    model = DegreeSignatureNN(input_dim=X.shape[1], hidden_dims=args.hidden_dims, dropout=args.dropout)

    benchmarker = ModelBenchmarker(
        model_name="Degree_Sig_NN",
        metapath=metapath,
        n_training_samples=len(X_train),
        n_cores=args.n_cores,
        n_parameters=int(model.n_parameters),
    )

    with benchmarker.time_training():
        training_history = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            device=device,
        )

    with benchmarker.time_prediction():
        predictions = predict_degree_sig_nn(
            model,
            X_test,
            device=device,
            batch_size=args.predict_batch_size,
        )

    record_validation_safe(benchmarker, predictions, y_test)

    r2_denom = float(np.sum((y_test - float(np.mean(y_test))) ** 2))
    if r2_denom > 0:
        r2_score = float(1 - np.sum((predictions - y_test) ** 2) / r2_denom)
    else:
        r2_score = float("nan")

    args.intermediate_dir.mkdir(parents=True, exist_ok=True)
    test_predictions_file = args.intermediate_dir / f"{metapath}_test_predictions.npy"
    test_actuals_file = args.intermediate_dir / f"{metapath}_test_actuals.npy"
    test_features_file = args.intermediate_dir / f"{metapath}_test_features.npy"
    np.save(test_predictions_file, predictions)
    np.save(test_actuals_file, y_test)
    np.save(test_features_file, X_test)

    plot_file: Path | None = None
    if not args.skip_plots:
        plot_file = args.visualization_dir / f"{metapath}_Degree_Sig_NN_validation.png"
        maybe_write_plot(
            y_true=y_test,
            y_pred=predictions,
            metapath=metapath,
            output_file=plot_file,
            validation_r=float(benchmarker.validation_r),
            validation_mae=float(benchmarker.validation_mae),
            validation_rmse=float(benchmarker.validation_rmse),
        )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_file = args.model_dir / f"{metapath}_Degree_Sig_NN.pt"
    save_degree_sig_nn(
        model_file,
        model,
        feature_columns=all_feature_cols,
        training_config={
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "train_fraction": args.train_fraction,
            "random_seed": args.random_seed,
            "device": device,
        },
        training_history=training_history,
        metadata={
            "metapath": metapath,
            "n_rows": int(len(df)),
            "n_features": int(X.shape[1]),
        },
    )

    benchmark_file: Path | None = None
    benchmark_result: dict[str, Any] | None = None
    if args.record_benchmarks:
        args.benchmark_dir.mkdir(parents=True, exist_ok=True)
        result = benchmarker.finalize()
        benchmark_file = args.benchmark_dir / f"{metapath}_Degree_Sig_NN_benchmark.json"
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
        "visualization_file": str(plot_file) if plot_file else None,
        "intermediate_files": {
            "test_predictions": str(test_predictions_file),
            "test_actuals": str(test_actuals_file),
            "test_features": str(test_features_file),
        },
        "training_history": {
            "epochs_ran": int(training_history["epochs_ran"]),
            "best_epoch": int(training_history["best_epoch"]),
            "best_val_loss": float(training_history["best_val_loss"]),
        },
        "r2_score": r2_score,
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
    set_seed(args.random_seed)
    device = select_device(args.device)
    metapaths = resolve_metapaths(args)

    print(f"Training data dir: {args.training_data_dir}")
    print(f"Model dir: {args.model_dir}")
    print(f"Benchmark dir: {args.benchmark_dir}")
    print(f"Visualization dir: {args.visualization_dir}")
    print(f"Intermediate dir: {args.intermediate_dir}")
    print(f"Device: {device}")
    print(f"Metapaths: {', '.join(metapaths)}")

    successful: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for metapath in metapaths:
        print("=" * 80)
        print(f"TRAIN DEGREE SIGNATURE NN: {metapath}")
        print("=" * 80)
        try:
            summary = run_one_metapath(metapath, args, device=device)
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
            print(
                "Training details: "
                f"epochs={summary['training_history']['epochs_ran']} "
                f"(best={summary['training_history']['best_epoch']}), "
                f"R2={summary['r2_score']:.4f}"
            )
            if summary["visualization_file"]:
                print(f"Saved visualization: {summary['visualization_file']}")
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            failed.append({"metapath": metapath, "error": message})
            print(f"FAILED {metapath}: {message}")
            if not args.continue_on_error:
                raise

    run_summary_file = args.benchmark_dir / "pathway_train_degree_signature_nn_run_summary.json"
    run_summary_file.parent.mkdir(parents=True, exist_ok=True)
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
