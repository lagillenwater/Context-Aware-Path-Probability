"""NN architecture exploration

"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_DIR = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_DIR / "src"))

from model_comparison import SimpleNN, filter_zero_degree_nodes, prepare_edge_features_and_labels
from model_evaluation import evaluate_model
from simple_models import SingleLayerNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run notebook-21 neural-network architecture/optimizer/loss exploration "
            "as a script-first pipeline."
        )
    )
    parser.add_argument("--edge-type", default="CbG", help="Edge type to analyze (default: CbG).")
    parser.add_argument("--perm-id", type=int, default=0, help="Permutation id for edge matrix (default: 0).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Repository data directory containing permutations/.",
    )
    parser.add_argument("--sample-ratio", type=float, default=0.01, help="Negative sampling ratio for feature prep.")
    parser.add_argument(
        "--adaptive-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adaptive sampling in feature preparation (default: true).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on total samples after feature prep.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Train/test split fraction (default: 0.2).")
    parser.add_argument(
        "--tests",
        default="1,2,3,4,5,6",
        help="Comma-separated tests to run from {1,2,3,4,5,6}. Default: all.",
    )
    parser.add_argument("--max-epochs-linear", type=int, default=30, help="Max epochs for single-layer models.")
    parser.add_argument("--max-epochs-deep", type=int, default=30, help="Max epochs for SimpleNN models.")
    parser.add_argument("--patience-linear", type=int, default=10, help="Early stopping patience for single-layer models.")
    parser.add_argument("--patience-deep", type=int, default=5, help="Early stopping patience for SimpleNN models.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for SimpleNN models.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--save-pkl", action=argparse.BooleanOptionalAction, default=True, help="Save notebook-compatible PKL outputs.")
    parser.add_argument(
        "--save-model-objects",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include sklearn/torch model objects in PKL payloads (default: false).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "nn_optimizer_comparison",
        help="Directory for notebook-21 output artifacts.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_ratio <= 0 or args.sample_ratio > 1:
        raise ValueError("--sample-ratio must be in (0, 1].")
    if args.test_size <= 0 or args.test_size >= 1:
        raise ValueError("--test-size must be in (0, 1).")
    if args.max_samples is not None and args.max_samples < 100:
        raise ValueError("--max-samples must be >= 100 when provided.")
    if args.max_epochs_linear < 1 or args.max_epochs_deep < 1:
        raise ValueError("Epoch counts must be >= 1.")
    if args.patience_linear < 1 or args.patience_deep < 1:
        raise ValueError("Patience values must be >= 1.")
    if args.batch_size < 32:
        raise ValueError("--batch-size must be >= 32.")
    if not args.data_dir.exists():
        raise ValueError(f"--data-dir does not exist: {args.data_dir}")
    requested = parse_tests(args.tests)
    if not requested:
        raise ValueError("No tests requested.")


def parse_tests(text: str) -> list[int]:
    out: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1 or value > 6:
            raise ValueError(f"Invalid test id: {value}. Allowed: 1-6.")
        out.append(value)
    return sorted(set(out))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def maybe_subsample(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_target = max(1, int(max_samples * (len(pos_idx) / len(y))))
    neg_target = max_samples - pos_target
    pos_keep = rng.choice(pos_idx, size=min(pos_target, len(pos_idx)), replace=False)
    neg_keep = rng.choice(neg_idx, size=min(neg_target, len(neg_idx)), replace=False)
    keep = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(keep)
    return X[keep], y[keep]


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    edge_file = args.data_dir / "permutations" / f"{args.perm_id:03d}.hetmat" / "edges" / f"{args.edge_type}.sparse.npz"
    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    edge_matrix = sp.load_npz(str(edge_file))
    filtered_matrix, _, _ = filter_zero_degree_nodes(edge_matrix)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"filtered_{args.edge_type}.sparse.npz"
        sp.save_npz(str(tmp_path), filtered_matrix)
        X, y = prepare_edge_features_and_labels(
            str(tmp_path),
            sample_ratio=args.sample_ratio,
            adaptive_sampling=args.adaptive_sampling,
            enhanced_features=False,
        )

    X, y = maybe_subsample(X, y, args.max_samples, args.random_seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    context = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_tensor": torch.FloatTensor(X_train),
        "X_test_tensor": torch.FloatTensor(X_test),
        "X_train_scaled_tensor": torch.FloatTensor(X_train_scaled),
        "X_test_scaled_tensor": torch.FloatTensor(X_test_scaled),
        "y_train_tensor": torch.FloatTensor(y_train),
        "y_test_tensor": torch.FloatTensor(y_test),
        "scaler": scaler,
    }
    return context


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if float(np.std(y_true)) == 0 or float(np.std(y_pred)) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def train_single_layer(
    *,
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
    optimizer_name: str,
    loss_mode: str,
    pos_weight: torch.Tensor | None,
    sample_weight_ratio: float | None,
    max_epochs: int,
    patience: int,
    random_seed: int,
) -> dict[str, Any]:
    torch.manual_seed(random_seed)
    model = SingleLayerNN()
    min_delta = 1e-6

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=max_epochs,
            steps_per_epoch=1,
            pct_start=0.3,
            anneal_strategy="cos",
        )
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=20)
        scheduler = None
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if loss_mode == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_mode == "mse":
        criterion = nn.MSELoss()
    elif loss_mode == "weighted_mse":
        criterion = None
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")

    if loss_mode == "weighted_mse":
        if sample_weight_ratio is None:
            raise ValueError("sample_weight_ratio is required for weighted_mse")
        sample_weights = torch.where(y_train_tensor == 1, sample_weight_ratio, 1.0)
    else:
        sample_weights = None

    train_losses: list[float] = []
    test_losses: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    converged_epoch = max_epochs

    start = time.time()

    for epoch in range(max_epochs):
        model.train()
        if optimizer_name == "lbfgs":

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                outputs = model(X_train_tensor).squeeze()
                if loss_mode == "bce":
                    loss = criterion(outputs, y_train_tensor)  # type: ignore[arg-type]
                elif loss_mode == "mse":
                    loss = criterion(torch.sigmoid(outputs), y_train_tensor)  # type: ignore[arg-type]
                else:
                    preds = torch.sigmoid(outputs)
                    loss = (sample_weights * (preds - y_train_tensor) ** 2).mean()  # type: ignore[operator]
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            train_loss_value = float(loss.item())
        else:
            optimizer.zero_grad()
            outputs = model(X_train_tensor).squeeze()
            if loss_mode == "bce":
                loss = criterion(outputs, y_train_tensor)  # type: ignore[arg-type]
            elif loss_mode == "mse":
                loss = criterion(torch.sigmoid(outputs), y_train_tensor)  # type: ignore[arg-type]
            else:
                preds = torch.sigmoid(outputs)
                loss = (sample_weights * (preds - y_train_tensor) ** 2).mean()  # type: ignore[operator]
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_loss_value = float(loss.item())

        train_losses.append(train_loss_value)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor).squeeze()
            if loss_mode == "bce":
                test_loss = criterion(test_outputs, y_test_tensor)  # type: ignore[arg-type]
            elif loss_mode == "mse":
                test_loss = criterion(torch.sigmoid(test_outputs), y_test_tensor)  # type: ignore[arg-type]
            else:
                test_loss = ((torch.sigmoid(test_outputs) - y_test_tensor) ** 2).mean()
            test_loss_value = float(test_loss.item())
            test_losses.append(test_loss_value)

        if test_loss_value < best_loss - min_delta:
            best_loss = test_loss_value
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            converged_epoch = epoch + 1
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - start
    model.eval()
    with torch.no_grad():
        pred_train_logits = model(X_train_tensor).squeeze()
        pred_test_logits = model(X_test_tensor).squeeze()
        pred_train = torch.sigmoid(pred_train_logits).cpu().numpy().flatten()
        pred_test = torch.sigmoid(pred_test_logits).cpu().numpy().flatten()

    return {
        "model": model,
        "train_pred": pred_train,
        "test_pred": pred_test,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_time": float(elapsed),
        "converged_epoch": int(converged_epoch),
    }


def train_simplenn(
    *,
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
    batch_size: int,
    max_epochs: int,
    patience: int,
    pos_weight: torch.Tensor | None,
    loss_mode: str,
    sample_weight_ratio: float | None,
    random_seed: int,
) -> dict[str, Any]:
    torch.manual_seed(random_seed)
    use_class_weights = loss_mode == "bce"
    model = SimpleNN(
        input_dim=2,
        hidden_dims=(128, 64, 32),
        dropout_rate=0.3,
        use_class_weights=use_class_weights,
    )

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=max_epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    if loss_mode == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_mode == "mse":
        criterion = nn.MSELoss()
    elif loss_mode == "weighted_mse":
        criterion = None
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")

    if loss_mode == "weighted_mse":
        if sample_weight_ratio is None:
            raise ValueError("sample_weight_ratio is required for weighted_mse")

    train_losses: list[float] = []
    test_losses: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0

    start = time.time()
    for _epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            if loss_mode == "bce":
                loss = criterion(outputs, batch_y)  # type: ignore[arg-type]
            elif loss_mode == "mse":
                loss = criterion(outputs, batch_y)  # type: ignore[arg-type]
            else:
                batch_weights = torch.where(batch_y == 1, sample_weight_ratio, 1.0)  # type: ignore[arg-type]
                loss = (batch_weights * (outputs - batch_y) ** 2).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())
        train_losses.append(epoch_loss / max(1, len(train_loader)))

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor).squeeze()
            if loss_mode == "bce":
                test_loss = criterion(test_outputs, y_test_tensor)  # type: ignore[arg-type]
                pred_test = torch.sigmoid(test_outputs)
            elif loss_mode == "mse":
                test_loss = criterion(test_outputs, y_test_tensor)  # type: ignore[arg-type]
                pred_test = test_outputs
            else:
                test_loss = ((test_outputs - y_test_tensor) ** 2).mean()
                pred_test = test_outputs
            test_loss_val = float(test_loss.item())
            test_losses.append(test_loss_val)

        if test_loss_val < best_loss:
            best_loss = test_loss_val
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - start
    model.eval()
    with torch.no_grad():
        pred_train_raw = model(X_train_tensor).squeeze()
        pred_test_raw = model(X_test_tensor).squeeze()
        if loss_mode == "bce":
            pred_train = torch.sigmoid(pred_train_raw).cpu().numpy().flatten()
            pred_test = torch.sigmoid(pred_test_raw).cpu().numpy().flatten()
        else:
            pred_train = pred_train_raw.cpu().numpy().flatten()
            pred_test = pred_test_raw.cpu().numpy().flatten()
    pred_train = np.clip(pred_train, 0, 1)
    pred_test = np.clip(pred_test, 0, 1)

    return {
        "model": model,
        "train_pred": pred_train,
        "test_pred": pred_test,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_time": float(elapsed),
    }


def train_logreg(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, class_weight: str | None, random_seed: int) -> dict[str, Any]:
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=random_seed,
        max_iter=1000,
    )
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    pred_train = model.predict_proba(X_train)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]
    return {
        "model": model,
        "train_pred": pred_train,
        "test_pred": pred_test,
        "train_time": float(elapsed),
    }


def train_ridge(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, sample_weights: np.ndarray | None, random_seed: int) -> dict[str, Any]:
    model = Ridge(alpha=1.0, random_state=random_seed)
    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.time() - start
    pred_train = np.clip(model.predict(X_train), 0, 1)
    pred_test = np.clip(model.predict(X_test), 0, 1)
    return {
        "model": model,
        "train_pred": pred_train,
        "test_pred": pred_test,
        "train_time": float(elapsed),
    }


def evaluation_row(
    test_id: int,
    model_label: str,
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float,
    loss_mode: str,
    weighted: bool,
    scaled: bool,
) -> dict[str, Any]:
    architecture = f"Test {test_id} - {model_label}"
    return {
        "Test": int(test_id),
        "Architecture": architecture,
        "Model": model_label,
        "Final AUC": float(metrics["auc"]),
        "Average Precision": float(metrics["average_precision"]),
        "Correlation": safe_corr(y_true, y_pred),
        "Train Time (s)": float(train_time),
        "Loss Mode": loss_mode,
        "Class Weighted": bool(weighted),
        "Scaled Features": bool(scaled),
    }


def maybe_save_pickle(path: Path, payload: dict[str, Any], save_model_objects: bool) -> None:
    if not save_model_objects and "model" in payload:
        payload = {k: v for k, v in payload.items() if k != "model"}
    with path.open("wb") as f:
        pickle.dump(payload, f)


def plot_loss(train_losses: list[float], test_losses: list[float], out_file: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.plot(test_losses, label="Test Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr_roc(y_test: np.ndarray, models_data: list[tuple[str, np.ndarray, dict[str, Any]]], pr_out: Path, roc_out: Path, title_prefix: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for name, preds, metrics in models_data:
        precision, recall, _ = precision_recall_curve(y_test, preds)
        ax.plot(recall, precision, linewidth=2.2, label=f"{name} (AP={metrics['average_precision']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix}: Precision-Recall")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(pr_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for name, preds, metrics in models_data:
        fpr, tpr, _ = roc_curve(y_test, preds)
        ax.plot(fpr, tpr, linewidth=2.2, label=f"{name} (AUC={metrics['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_prefix}: ROC")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_global_diagnostics(
    *,
    architecture_df: pd.DataFrame,
    prediction_store: dict[str, np.ndarray],
    y_test: np.ndarray,
    X_test_raw: np.ndarray,
    results_dir: Path,
) -> None:
    ranked = architecture_df.sort_values("Final AUC", ascending=False).reset_index(drop=True)
    top_names = ranked["Architecture"].head(4).tolist()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for name in top_names:
        preds = prediction_store.get(name)
        if preds is None:
            continue
        frac_pos, mean_pred = calibration_curve(y_test, preds, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Fraction")
    ax.set_title("Sanity Check Calibration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "sanity_check_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if len(ranked) == 0:
        return
    best_name = ranked.loc[0, "Architecture"]
    best_preds = prediction_store.get(best_name)
    if best_preds is None:
        return

    degree_product = X_test_raw[:, 0] * X_test_raw[:, 1]
    residual = np.abs(y_test - best_preds)
    try:
        bins = np.quantile(degree_product, np.linspace(0, 1, 11))
        bins = np.unique(bins)
        if len(bins) < 3:
            raise ValueError("Insufficient unique quantile bins")
        bin_ids = np.digitize(degree_product, bins[1:-1], right=True)
        x_vals, y_vals = [], []
        for b in range(bin_ids.min(), bin_ids.max() + 1):
            m = bin_ids == b
            if np.sum(m) < 5:
                continue
            x_vals.append(float(np.median(degree_product[m])))
            y_vals.append(float(np.mean(residual[m])))
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(x_vals, y_vals, marker="o", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Degree Product (source_degree × target_degree)")
        ax.set_ylabel("Mean |Residual|")
        ax.set_title(f"Path Dependency Decay ({best_name})")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "path_dependency_decay.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def write_markdown_summary(architecture_df: pd.DataFrame, results_dir: Path, args: argparse.Namespace) -> None:
    if architecture_df.empty:
        text = "# NN Architecture Analysis\n\nNo results were produced.\n"
        (results_dir / "nn_architecture_analysis_complete.md").write_text(text)
        return

    ranked = architecture_df.sort_values("Final AUC", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    top10_table = ranked.head(10)[
        ["Architecture", "Final AUC", "Average Precision", "Correlation", "Train Time (s)"]
    ].to_string(index=False)

    lines = [
        "# NN Architecture Analysis Complete",
        "",
        "## Run Config",
        f"- edge_type: `{args.edge_type}`",
        f"- perm_id: `{args.perm_id}`",
        f"- tests: `{args.tests}`",
        f"- sample_ratio: `{args.sample_ratio}`",
        f"- max_samples: `{args.max_samples}`",
        f"- max_epochs_linear: `{args.max_epochs_linear}`",
        f"- max_epochs_deep: `{args.max_epochs_deep}`",
        "",
        "## Best Architecture",
        f"- Architecture: `{best['Architecture']}`",
        f"- Final AUC: `{best['Final AUC']:.4f}`",
        f"- Average Precision: `{best['Average Precision']:.4f}`",
        f"- Correlation: `{best['Correlation']:.4f}`",
        "",
        "## Top 10",
        "",
        "```text",
        top10_table,
        "```",
        "",
    ]
    (results_dir / "nn_architecture_analysis_complete.md").write_text("\n".join(lines))


def run_test_bundle(
    *,
    context: dict[str, Any],
    args: argparse.Namespace,
    test_id: int,
    output_rows: list[dict[str, Any]],
    prediction_store: dict[str, np.ndarray],
) -> None:
    y_test = context["y_test"]
    y_train = context["y_train"]
    pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())], dtype=torch.float32)
    sample_weight_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    if test_id == 1:
        adam = train_single_layer(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="adam",
            loss_mode="bce",
            pos_weight=None,
            sample_weight_ratio=None,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        logreg = train_logreg(context["X_train"], y_train, context["X_test"], class_weight=None, random_seed=args.random_seed)

        metrics_adam = evaluate_model(y_test, adam["test_pred"], "Single Layer NN (Adam)")
        metrics_logreg = evaluate_model(y_test, logreg["test_pred"], "Logistic Regression")

        if args.save_pkl:
            maybe_save_pickle(args.results_dir / "single_layer_nn_adam.pkl", {**adam, "metrics": metrics_adam}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "logistic_regression.pkl", {**logreg, "metrics": metrics_logreg}, args.save_model_objects)

        if not args.skip_plots:
            plot_loss(adam["train_losses"], adam["test_losses"], args.results_dir / "single_layer_nn_adam_loss.png", "Single Layer NN (Adam) Training History")

        rows = [
            evaluation_row(
                test_id=1,
                model_label="Single Layer NN (Adam)",
                metrics=metrics_adam,
                y_true=y_test,
                y_pred=adam["test_pred"],
                train_time=adam["train_time"],
                loss_mode="bce",
                weighted=False,
                scaled=False,
            ),
            evaluation_row(
                test_id=1,
                model_label="Logistic Regression",
                metrics=metrics_logreg,
                y_true=y_test,
                y_pred=logreg["test_pred"],
                train_time=logreg["train_time"],
                loss_mode="bce",
                weighted=False,
                scaled=False,
            ),
        ]
        output_rows.extend(rows)
        for row, preds in zip(rows, [adam["test_pred"], logreg["test_pred"]]):
            prediction_store[row["Architecture"]] = preds
        pd.DataFrame(rows).to_csv(args.results_dir / "test1_model_comparison.csv", index=False)
        return

    if test_id == 2:
        adam = train_single_layer(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="adam",
            loss_mode="bce",
            pos_weight=None,
            sample_weight_ratio=None,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        lbfgs = train_single_layer(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="lbfgs",
            loss_mode="bce",
            pos_weight=None,
            sample_weight_ratio=None,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        logreg = train_logreg(context["X_train"], y_train, context["X_test"], class_weight=None, random_seed=args.random_seed)

        metrics_adam = evaluate_model(y_test, adam["test_pred"], "Single Layer NN (Adam)")
        metrics_lbfgs = evaluate_model(y_test, lbfgs["test_pred"], "Single Layer NN (L-BFGS)")
        metrics_logreg = evaluate_model(y_test, logreg["test_pred"], "Logistic Regression")

        if args.save_pkl:
            maybe_save_pickle(args.results_dir / "single_layer_nn_lbfgs.pkl", {**lbfgs, "metrics": metrics_lbfgs}, args.save_model_objects)
            comparison_summary = {
                "models": ["Single Layer NN (Adam)", "Single Layer NN (L-BFGS)", "Logistic Regression"],
                "true_labels": y_test,
                "test_predictions": {
                    "adam": adam["test_pred"],
                    "lbfgs": lbfgs["test_pred"],
                    "logreg": logreg["test_pred"],
                },
            }
            maybe_save_pickle(args.results_dir / "comparison_summary.pkl", comparison_summary, args.save_model_objects)

        if not args.skip_plots:
            plot_loss(lbfgs["train_losses"], lbfgs["test_losses"], args.results_dir / "single_layer_nn_lbfgs_loss.png", "Single Layer NN (L-BFGS) Training History")
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes[0].plot(adam["train_losses"], label="Train")
            axes[0].plot(adam["test_losses"], label="Test")
            axes[0].set_title("Single Layer NN (Adam)")
            axes[0].grid(alpha=0.3)
            axes[0].legend()
            axes[1].plot(lbfgs["train_losses"], label="Train")
            axes[1].plot(lbfgs["test_losses"], label="Test")
            axes[1].set_title("Single Layer NN (L-BFGS)")
            axes[1].grid(alpha=0.3)
            axes[1].legend()
            plt.tight_layout()
            plt.savefig(args.results_dir / "optimizer_comparison_losses.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            models_data = [
                ("Single Layer NN (Adam)", adam["test_pred"], metrics_adam),
                ("Single Layer NN (L-BFGS)", lbfgs["test_pred"], metrics_lbfgs),
                ("Logistic Regression", logreg["test_pred"], metrics_logreg),
            ]
            plot_pr_roc(
                y_test,
                models_data,
                args.results_dir / "precision_recall_comparison.png",
                args.results_dir / "roc_comparison.png",
                "Test 2",
            )

        rows = [
            evaluation_row(2, "Single Layer NN (Adam)", metrics_adam, y_test, adam["test_pred"], adam["train_time"], "bce", False, False),
            evaluation_row(2, "Single Layer NN (L-BFGS)", metrics_lbfgs, y_test, lbfgs["test_pred"], lbfgs["train_time"], "bce", False, False),
            evaluation_row(2, "Logistic Regression", metrics_logreg, y_test, logreg["test_pred"], logreg["train_time"], "bce", False, False),
        ]
        output_rows.extend(rows)
        for row, preds in zip(rows, [adam["test_pred"], lbfgs["test_pred"], logreg["test_pred"]]):
            prediction_store[row["Architecture"]] = preds
        comparison_df = pd.DataFrame(rows)
        comparison_df.to_csv(args.results_dir / "model_comparison.csv", index=False)
        comparison_df.to_csv(args.results_dir / "test2_model_comparison.csv", index=False)
        return

    if test_id == 3:
        adam = train_single_layer(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="adam",
            loss_mode="bce",
            pos_weight=pos_weight,
            sample_weight_ratio=None,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        lbfgs = train_single_layer(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="lbfgs",
            loss_mode="bce",
            pos_weight=pos_weight,
            sample_weight_ratio=None,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        logreg = train_logreg(context["X_train"], y_train, context["X_test"], class_weight="balanced", random_seed=args.random_seed)
        simplenn = train_simplenn(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            batch_size=args.batch_size,
            max_epochs=args.max_epochs_deep,
            patience=args.patience_deep,
            pos_weight=pos_weight,
            loss_mode="bce",
            sample_weight_ratio=None,
            random_seed=args.random_seed,
        )

        metrics_adam = evaluate_model(y_test, adam["test_pred"], "Single Layer NN (Adam, Weighted)")
        metrics_lbfgs = evaluate_model(y_test, lbfgs["test_pred"], "Single Layer NN (L-BFGS, Weighted)")
        metrics_logreg = evaluate_model(y_test, logreg["test_pred"], "Logistic Regression (Weighted)")
        metrics_simplenn = evaluate_model(y_test, simplenn["test_pred"], "SimpleNN (Notebook 04)")

        if args.save_pkl:
            maybe_save_pickle(args.results_dir / "test3_single_layer_nn_adam_weighted.pkl", {**adam, "metrics": metrics_adam}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test3_single_layer_nn_lbfgs_weighted.pkl", {**lbfgs, "metrics": metrics_lbfgs}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test3_logreg_weighted.pkl", {**logreg, "metrics": metrics_logreg}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test3_simple_nn.pkl", {**simplenn, "metrics": metrics_simplenn}, args.save_model_objects)

        if not args.skip_plots:
            models_data = [
                ("Adam NN (Weighted)", adam["test_pred"], metrics_adam),
                ("L-BFGS NN (Weighted)", lbfgs["test_pred"], metrics_lbfgs),
                ("LogReg (Weighted)", logreg["test_pred"], metrics_logreg),
                ("SimpleNN", simplenn["test_pred"], metrics_simplenn),
            ]
            plot_pr_roc(
                y_test,
                models_data,
                args.results_dir / "test3_precision_recall.png",
                args.results_dir / "test3_roc_curves.png",
                "Test 3",
            )

        rows = [
            evaluation_row(3, "Single Layer NN (Adam, Weighted)", metrics_adam, y_test, adam["test_pred"], adam["train_time"], "bce", True, False),
            evaluation_row(3, "Single Layer NN (L-BFGS, Weighted)", metrics_lbfgs, y_test, lbfgs["test_pred"], lbfgs["train_time"], "bce", True, False),
            evaluation_row(3, "Logistic Regression (Weighted)", metrics_logreg, y_test, logreg["test_pred"], logreg["train_time"], "bce", True, False),
            evaluation_row(3, "SimpleNN (Notebook 04)", metrics_simplenn, y_test, simplenn["test_pred"], simplenn["train_time"], "bce", True, False),
        ]
        output_rows.extend(rows)
        for row, preds in zip(rows, [adam["test_pred"], lbfgs["test_pred"], logreg["test_pred"], simplenn["test_pred"]]):
            prediction_store[row["Architecture"]] = preds
        pd.DataFrame(rows).to_csv(args.results_dir / "test3_model_comparison.csv", index=False)
        return

    if test_id == 4:
        adam = train_single_layer(
            X_train_tensor=context["X_train_scaled_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_scaled_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="adam",
            loss_mode="bce",
            pos_weight=pos_weight,
            sample_weight_ratio=None,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        lbfgs = train_single_layer(
            X_train_tensor=context["X_train_scaled_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_scaled_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="lbfgs",
            loss_mode="bce",
            pos_weight=pos_weight,
            sample_weight_ratio=None,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        logreg = train_logreg(context["X_train_scaled"], y_train, context["X_test_scaled"], class_weight="balanced", random_seed=args.random_seed)
        simplenn = train_simplenn(
            X_train_tensor=context["X_train_scaled_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_scaled_tensor"],
            y_test_tensor=context["y_test_tensor"],
            batch_size=args.batch_size,
            max_epochs=args.max_epochs_deep,
            patience=args.patience_deep,
            pos_weight=pos_weight,
            loss_mode="bce",
            sample_weight_ratio=None,
            random_seed=args.random_seed,
        )

        metrics_adam = evaluate_model(y_test, adam["test_pred"], "Single Layer NN (Adam, Weighted, Scaled)")
        metrics_lbfgs = evaluate_model(y_test, lbfgs["test_pred"], "Single Layer NN (L-BFGS, Weighted, Scaled)")
        metrics_logreg = evaluate_model(y_test, logreg["test_pred"], "Logistic Regression (Weighted, Scaled)")
        metrics_simplenn = evaluate_model(y_test, simplenn["test_pred"], "SimpleNN (Weighted, Scaled)")

        if args.save_pkl:
            maybe_save_pickle(args.results_dir / "test4_single_layer_nn_adam_weighted_scaled.pkl", {**adam, "metrics": metrics_adam, "scaler": context["scaler"]}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test4_single_layer_nn_lbfgs_weighted_scaled.pkl", {**lbfgs, "metrics": metrics_lbfgs, "scaler": context["scaler"]}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test4_logreg_weighted_scaled.pkl", {**logreg, "metrics": metrics_logreg, "scaler": context["scaler"]}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test4_simple_nn_weighted_scaled.pkl", {**simplenn, "metrics": metrics_simplenn, "scaler": context["scaler"]}, args.save_model_objects)

        if not args.skip_plots:
            models_data = [
                ("Adam NN (Scaled)", adam["test_pred"], metrics_adam),
                ("L-BFGS NN (Scaled)", lbfgs["test_pred"], metrics_lbfgs),
                ("LogReg (Scaled)", logreg["test_pred"], metrics_logreg),
                ("SimpleNN (Scaled)", simplenn["test_pred"], metrics_simplenn),
            ]
            plot_pr_roc(
                y_test,
                models_data,
                args.results_dir / "test4_precision_recall.png",
                args.results_dir / "test4_roc_curves.png",
                "Test 4",
            )

        rows = [
            evaluation_row(4, "Single Layer NN (Adam, Weighted, Scaled)", metrics_adam, y_test, adam["test_pred"], adam["train_time"], "bce", True, True),
            evaluation_row(4, "Single Layer NN (L-BFGS, Weighted, Scaled)", metrics_lbfgs, y_test, lbfgs["test_pred"], lbfgs["train_time"], "bce", True, True),
            evaluation_row(4, "Logistic Regression (Weighted, Scaled)", metrics_logreg, y_test, logreg["test_pred"], logreg["train_time"], "bce", True, True),
            evaluation_row(4, "SimpleNN (Weighted, Scaled)", metrics_simplenn, y_test, simplenn["test_pred"], simplenn["train_time"], "bce", True, True),
        ]
        output_rows.extend(rows)
        for row, preds in zip(rows, [adam["test_pred"], lbfgs["test_pred"], logreg["test_pred"], simplenn["test_pred"]]):
            prediction_store[row["Architecture"]] = preds
        pd.DataFrame(rows).to_csv(args.results_dir / "test4_model_comparison.csv", index=False)
        return

    if test_id == 5:
        adam = train_single_layer(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="adam",
            loss_mode="mse",
            pos_weight=None,
            sample_weight_ratio=None,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        lbfgs = train_single_layer(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="lbfgs",
            loss_mode="mse",
            pos_weight=None,
            sample_weight_ratio=None,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        simplenn = train_simplenn(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            batch_size=args.batch_size,
            max_epochs=args.max_epochs_deep,
            patience=args.patience_deep,
            pos_weight=None,
            loss_mode="mse",
            sample_weight_ratio=None,
            random_seed=args.random_seed,
        )
        ridge = train_ridge(context["X_train"], y_train, context["X_test"], sample_weights=None, random_seed=args.random_seed)

        metrics_adam = evaluate_model(y_test, adam["test_pred"], "Single Layer NN (Adam, MSE)")
        metrics_lbfgs = evaluate_model(y_test, lbfgs["test_pred"], "Single Layer NN (L-BFGS, MSE)")
        metrics_simplenn = evaluate_model(y_test, simplenn["test_pred"], "SimpleNN (MSE)")
        metrics_ridge = evaluate_model(y_test, ridge["test_pred"], "Ridge (MSE)")

        if args.save_pkl:
            maybe_save_pickle(args.results_dir / "test5_single_layer_nn_adam_mse.pkl", {**adam, "metrics": metrics_adam}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test5_single_layer_nn_lbfgs_mse.pkl", {**lbfgs, "metrics": metrics_lbfgs}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test5_simple_nn_mse.pkl", {**simplenn, "metrics": metrics_simplenn}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test5_ridge.pkl", {**ridge, "metrics": metrics_ridge}, args.save_model_objects)

        rows = [
            evaluation_row(5, "Single Layer NN (Adam, MSE)", metrics_adam, y_test, adam["test_pred"], adam["train_time"], "mse", False, False),
            evaluation_row(5, "Single Layer NN (L-BFGS, MSE)", metrics_lbfgs, y_test, lbfgs["test_pred"], lbfgs["train_time"], "mse", False, False),
            evaluation_row(5, "SimpleNN (MSE)", metrics_simplenn, y_test, simplenn["test_pred"], simplenn["train_time"], "mse", False, False),
            evaluation_row(5, "Ridge (MSE)", metrics_ridge, y_test, ridge["test_pred"], ridge["train_time"], "mse", False, False),
        ]
        output_rows.extend(rows)
        for row, preds in zip(rows, [adam["test_pred"], lbfgs["test_pred"], simplenn["test_pred"], ridge["test_pred"]]):
            prediction_store[row["Architecture"]] = preds
        pd.DataFrame(rows).to_csv(args.results_dir / "test5_model_comparison.csv", index=False)
        return

    if test_id == 6:
        sample_weights_np = np.where(y_train == 1, sample_weight_ratio, 1.0)
        adam = train_single_layer(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="adam",
            loss_mode="weighted_mse",
            pos_weight=None,
            sample_weight_ratio=sample_weight_ratio,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        lbfgs = train_single_layer(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            optimizer_name="lbfgs",
            loss_mode="weighted_mse",
            pos_weight=None,
            sample_weight_ratio=sample_weight_ratio,
            max_epochs=args.max_epochs_linear,
            patience=args.patience_linear,
            random_seed=args.random_seed,
        )
        simplenn = train_simplenn(
            X_train_tensor=context["X_train_tensor"],
            y_train_tensor=context["y_train_tensor"],
            X_test_tensor=context["X_test_tensor"],
            y_test_tensor=context["y_test_tensor"],
            batch_size=args.batch_size,
            max_epochs=args.max_epochs_deep,
            patience=args.patience_deep,
            pos_weight=None,
            loss_mode="weighted_mse",
            sample_weight_ratio=sample_weight_ratio,
            random_seed=args.random_seed,
        )
        ridge = train_ridge(context["X_train"], y_train, context["X_test"], sample_weights=sample_weights_np, random_seed=args.random_seed)

        metrics_adam = evaluate_model(y_test, adam["test_pred"], "Single Layer NN (Adam, Weighted MSE)")
        metrics_lbfgs = evaluate_model(y_test, lbfgs["test_pred"], "Single Layer NN (L-BFGS, Weighted MSE)")
        metrics_simplenn = evaluate_model(y_test, simplenn["test_pred"], "SimpleNN (Weighted MSE)")
        metrics_ridge = evaluate_model(y_test, ridge["test_pred"], "Ridge (Weighted MSE)")

        if args.save_pkl:
            maybe_save_pickle(args.results_dir / "test6_single_layer_nn_adam_weighted_mse.pkl", {**adam, "metrics": metrics_adam}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test6_single_layer_nn_lbfgs_weighted_mse.pkl", {**lbfgs, "metrics": metrics_lbfgs}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test6_simple_nn_weighted_mse.pkl", {**simplenn, "metrics": metrics_simplenn}, args.save_model_objects)
            maybe_save_pickle(args.results_dir / "test6_ridge_weighted.pkl", {**ridge, "metrics": metrics_ridge}, args.save_model_objects)

        rows = [
            evaluation_row(6, "Single Layer NN (Adam, Weighted MSE)", metrics_adam, y_test, adam["test_pred"], adam["train_time"], "weighted_mse", True, False),
            evaluation_row(6, "Single Layer NN (L-BFGS, Weighted MSE)", metrics_lbfgs, y_test, lbfgs["test_pred"], lbfgs["train_time"], "weighted_mse", True, False),
            evaluation_row(6, "SimpleNN (Weighted MSE)", metrics_simplenn, y_test, simplenn["test_pred"], simplenn["train_time"], "weighted_mse", True, False),
            evaluation_row(6, "Ridge (Weighted MSE)", metrics_ridge, y_test, ridge["test_pred"], ridge["train_time"], "weighted_mse", True, False),
        ]
        output_rows.extend(rows)
        for row, preds in zip(rows, [adam["test_pred"], lbfgs["test_pred"], simplenn["test_pred"], ridge["test_pred"]]):
            prediction_store[row["Architecture"]] = preds
        pd.DataFrame(rows).to_csv(args.results_dir / "test6_model_comparison.csv", index=False)
        return

    raise ValueError(f"Unsupported test id: {test_id}")


def main() -> int:
    args = parse_args()
    set_global_seed(args.random_seed)
    requested_tests = parse_tests(args.tests)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NN Architecture Exploration (Notebook 21 Migration)")
    print("=" * 80)
    print(f"edge_type={args.edge_type} perm_id={args.perm_id}")
    print(f"tests={requested_tests}")
    print(f"sample_ratio={args.sample_ratio} max_samples={args.max_samples}")
    print(f"results_dir={args.results_dir}")

    context = prepare_dataset(args)
    print(
        f"Prepared dataset: train={len(context['y_train']):,} test={len(context['y_test']):,} "
        f"positive_ratio={float(np.mean(context['y_train'])):.3f}"
    )

    output_rows: list[dict[str, Any]] = []
    prediction_store: dict[str, np.ndarray] = {}

    for test_id in requested_tests:
        print(f"\nRunning Test {test_id}...")
        run_test_bundle(
            context=context,
            args=args,
            test_id=test_id,
            output_rows=output_rows,
            prediction_store=prediction_store,
        )

    architecture_df = pd.DataFrame(output_rows)
    if architecture_df.empty:
        raise RuntimeError("No results were produced.")

    architecture_df = architecture_df.sort_values("Final AUC", ascending=False).reset_index(drop=True)
    architecture_csv = args.results_dir / "architecture_comparison.csv"
    architecture_df.to_csv(architecture_csv, index=False)

    if not args.skip_plots:
        generate_global_diagnostics(
            architecture_df=architecture_df,
            prediction_store=prediction_store,
            y_test=context["y_test"],
            X_test_raw=context["X_test"],
            results_dir=args.results_dir,
        )

    write_markdown_summary(architecture_df, args.results_dir, args)

    run_meta = {
        "edge_type": args.edge_type,
        "perm_id": args.perm_id,
        "tests": requested_tests,
        "n_train": int(len(context["y_train"])),
        "n_test": int(len(context["y_test"])),
        "best_architecture": architecture_df.loc[0, "Architecture"],
        "best_auc": float(architecture_df.loc[0, "Final AUC"]),
    }
    (args.results_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    print("\nTop 5 architectures:")
    print(architecture_df[["Architecture", "Final AUC", "Average Precision", "Correlation"]].head(5).to_string(index=False))
    print("\nSaved outputs:")
    print(f"  - {architecture_csv}")
    print(f"  - {args.results_dir / 'nn_architecture_analysis_complete.md'}")
    print(f"  - {args.results_dir / 'run_metadata.json'}")
    if not args.skip_plots:
        print(f"  - {args.results_dir / 'sanity_check_calibration.png'}")
        print(f"  - {args.results_dir / 'path_dependency_decay.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
