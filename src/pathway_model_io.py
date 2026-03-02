"""Utilities to save/load and run Degree Signature NN checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.pathway_models_v2 import DegreeSignatureNN


def _extract_linear_weights(state_dict: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    linear_weights: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        if not key.startswith("network.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        linear_weights.append((idx, value))
    linear_weights.sort(key=lambda x: x[0])
    return [weight for _, weight in linear_weights]


def _infer_architecture(state_dict: dict[str, torch.Tensor]) -> tuple[int, list[int]]:
    linear_weights = _extract_linear_weights(state_dict)
    if not linear_weights:
        raise ValueError("Could not infer architecture from state_dict.")
    input_dim = int(linear_weights[0].shape[1])
    hidden_dims = [int(weight.shape[0]) for weight in linear_weights[:-1]]
    return input_dim, hidden_dims


def save_degree_sig_nn(
    path: Path,
    model: DegreeSignatureNN,
    *,
    feature_columns: list[str],
    training_config: dict[str, Any] | None = None,
    training_history: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_name": "Degree_Sig_NN",
        "model_class": "src.pathway_models_v2.DegreeSignatureNN",
        "input_dim": int(model.input_dim),
        "hidden_dims": [int(x) for x in model.hidden_dims],
        "dropout": float(model.dropout_rate),
        "state_dict": model.state_dict(),
        "feature_columns": feature_columns,
        "training_config": training_config or {},
        "training_history": training_history or {},
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_degree_sig_nn(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[DegreeSignatureNN, dict[str, Any]]:
    payload = torch.load(path, map_location=device)

    metadata: dict[str, Any] = {}
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        input_dim = int(payload.get("input_dim", 0))
        hidden_dims = [int(x) for x in payload.get("hidden_dims", [])]
        dropout = float(payload.get("dropout", 0.1))
        metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    elif isinstance(payload, dict):
        # Backward-compatibility: file contains raw state_dict.
        state_dict = payload
        input_dim = 0
        hidden_dims = []
        dropout = 0.1
    else:
        raise ValueError(f"Unsupported checkpoint format at {path}")

    if not input_dim or not hidden_dims:
        inferred_input_dim, inferred_hidden_dims = _infer_architecture(state_dict)
        input_dim = input_dim or inferred_input_dim
        hidden_dims = hidden_dims or inferred_hidden_dims

    model = DegreeSignatureNN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata


def predict_degree_sig_nn(
    model: DegreeSignatureNN,
    X: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            xb = torch.from_numpy(batch.astype(np.float32)).to(device)
            yb = model(xb).squeeze(-1)
            outputs.append(yb.cpu().numpy())

    if not outputs:
        return np.array([], dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32)
