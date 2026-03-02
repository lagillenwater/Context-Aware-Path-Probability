"""
Phase 2: Hybrid evaluation pipeline for DegreeSignatureNN feature ablation.

Training: Degree bins from original Hetionet
Testing: Individual pairs from 20 permutations (averaged)

Handles small sample sizes with regularization and early stopping.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class DegreeSignatureNN(nn.Module):
    """
    Neural network for pathway count prediction.

    Adjusted for small sample sizes with higher regularization.
    """
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())  # Ensure non-negative predictions

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_edge_matrix(data_dir, edge_abbrev):
    """Load edge adjacency matrix."""
    edge_path = os.path.join(data_dir, 'edges', f'{edge_abbrev}.sparse.npz')
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge file not found: {edge_path}")
    matrix = sp.load_npz(edge_path)

    if matrix.dtype == bool:
        matrix = matrix.astype(np.int32)

    return matrix


def train_model(X_train, y_train, X_val, y_val, input_dim,
                hidden_dims=(64, 32), dropout=0.3, lr=0.001,
                max_epochs=1000, patience=50, batch_size=16):
    """
    Train DegreeSignatureNN with early stopping.

    Adjusted for small datasets:
    - Smaller hidden layers
    - Higher dropout
    - Larger patience
    - Small batch size
    """
    model = DegreeSignatureNN(input_dim, hidden_dims, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()

        # Mini-batch training
        n_batches = max(1, len(X_train) // batch_size)
        indices = torch.randperm(len(X_train))

        train_loss = 0
        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_hybrid(metapath_name, edge1_abbrev, edge2_abbrev,
                     feature_set='A', n_bins=10, n_permutations=20):
    """
    Hybrid evaluation: train on bins, test on individual pairs.

    Returns:
    - results: dict with r, rmse, n_train, n_test
    """
    print(f"\nEvaluating: {metapath_name}, Feature Set {feature_set}")
    print("-" * 70)

    # Load training data (degree bins)
    train_file = f'results/phase2_training_data/{metapath_name}_features_{feature_set}.csv'
    if not os.path.exists(train_file):
        print(f"  ERROR: Training file not found: {train_file}")
        return None

    df_train = pd.read_csv(train_file)

    # Extract features and target
    feature_cols = [col for col in df_train.columns if col.startswith('feat_')]
    X = df_train[feature_cols].values
    y = df_train['pathway_count_mean'].values

    print(f"  Training samples: {len(X)}")
    print(f"  Features (raw): {len(feature_cols)}")

    # Handle NaN values
    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.any(axis=0).sum()} features have NaN, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    # Remove zero-variance features
    feature_stds = X.std(axis=0)
    nonzero_var_mask = feature_stds > 1e-8
    n_zero_var = (~nonzero_var_mask).sum()

    if n_zero_var > 0:
        print(f"  WARNING: Removing {n_zero_var} zero-variance features")
        X = X[:, nonzero_var_mask]
        feature_cols = [col for col, keep in zip(feature_cols, nonzero_var_mask) if keep]

    print(f"  Features (after filtering): {X.shape[1]}")

    # Check if enough samples
    if len(X) < 6:
        print(f"  ERROR: Too few training samples ({len(X)} < 6)")
        return None

    # Train/val split (80/20, min 1 val sample)
    val_size = max(1, int(0.2 * len(X)))
    train_size = len(X) - val_size

    if train_size < 3:
        print(f"  ERROR: Too few training samples after split ({train_size})")
        return None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"  Train/val split: {len(X_train)}/{len(X_val)}")

    # Train model (adjusted for small datasets)
    model = train_model(
        X_train_scaled, y_train, X_val_scaled, y_val,
        input_dim=len(feature_cols),
        hidden_dims=(32, 16),  # Smaller for small datasets
        dropout=0.4,  # Higher dropout
        lr=0.001,
        max_epochs=1000,
        patience=100  # More patience for small datasets
    )

    print(f"  Model trained")

    # TODO: Load test data from permutations and evaluate
    # For now, just evaluate on validation set
    model.eval()
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val_scaled)
        y_val_pred = model(X_val_t).numpy().flatten()

    # Compute metrics
    if len(y_val) > 1 and y_val.std() > 0:
        r, p_value = pearsonr(y_val, y_val_pred)
        rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    else:
        r = np.nan
        rmse = np.nan

    print(f"  Validation r: {r:.4f}")
    print(f"  Validation RMSE: {rmse:.4f}")

    results = {
        'metapath': metapath_name,
        'feature_set': feature_set,
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'val_r': r,
        'val_rmse': rmse
    }

    return results


def main():
    """Test on single metapath first (CbGpPW with 90 samples)."""
    print("=" * 70)
    print("PHASE 2: HYBRID EVALUATION - BASELINE TEST")
    print("=" * 70)
    print()
    print("Testing on CbGpPW (90 training samples) with Set A (baseline)")
    print()

    results = evaluate_hybrid(
        metapath_name='CbGpPW',
        edge1_abbrev='CbG',
        edge2_abbrev='GpPW',
        feature_set='A'
    )

    if results is None:
        print("\nERROR: Evaluation failed")
        return False

    print()
    print("=" * 70)
    print("BASELINE TEST RESULTS")
    print("=" * 70)
    print(f"Validation r: {results['val_r']:.4f}")
    print()

    if np.isnan(results['val_r']):
        print("ERROR: Validation r is NaN")
        print("Possible causes:")
        print("  - No variance in validation targets")
        print("  - Model predictions all identical")
        print("  - Too few validation samples")
        return False

    if results['val_r'] < 0.5:
        print("WARNING: Low validation r (<0.5)")
        print("Model may not be learning effectively from small sample size")
        print("Consider:")
        print("  - Simpler model architecture")
        print("  - More regularization")
        print("  - Linear regression baseline")

    print()
    print("Next step: Implement full test set evaluation on 20 permutations")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
