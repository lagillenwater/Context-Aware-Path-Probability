"""
Phase 2: Full test set evaluation with residual analysis and visualizations.

Training: Degree bins from original Hetionet
Testing: Individual pairs from 20 permutations (averaged)

Includes:
- Full permutation-based testing
- Residual analysis
- Performance visualizations
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.enhanced_features import extract_enhanced_features


class DegreeSignatureNN(nn.Module):
    """Neural network for pathway count prediction."""
    def __init__(self, input_dim, hidden_dims=(32, 16), dropout=0.4):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_edge_matrix(path, edge_abbrev):
    """Load edge adjacency matrix."""
    edge_path = os.path.join(path, 'edges', f'{edge_abbrev}.sparse.npz')
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge file not found: {edge_path}")
    matrix = sp.load_npz(edge_path)

    if matrix.dtype == bool:
        matrix = matrix.astype(np.int32)

    return matrix


def train_model(X_train, y_train, X_val, y_val, input_dim,
                hidden_dims=(32, 16), dropout=0.4, lr=0.001,
                max_epochs=1000, patience=100, batch_size=16):
    """Train DegreeSignatureNN with early stopping."""
    model = DegreeSignatureNN(input_dim, hidden_dims, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()

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

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def load_test_data_from_permutations(perm_dir, edge1_abbrev, edge2_abbrev,
                                       perm_range, subsample_ratio=0.01,
                                       max_pairs=100000):
    """
    Load test data from permutations and compute average pathway counts.

    Subsampling for efficiency (test on 1% of pairs or max 100k).

    Returns:
    - source_nodes: array of source node indices
    - target_nodes: array of target node indices
    - avg_pathway_counts: average across permutations
    - edge1_ref: reference edge1 for degree lookup
    - edge2_ref: reference edge2 for degree lookup
    """
    print(f"  Loading test data from {len(perm_range)} permutations...")

    # Load reference edges (original Hetionet) for degrees
    edge1_ref = load_edge_matrix('data', edge1_abbrev)
    edge2_ref = load_edge_matrix('data', edge2_abbrev)

    # Determine which pairs to test (subsample for efficiency)
    n_source = edge1_ref.shape[0]
    n_target = edge2_ref.shape[1]
    total_pairs = n_source * n_target

    n_test_pairs = min(max_pairs, int(total_pairs * subsample_ratio))

    print(f"  Total possible pairs: {total_pairs:,}")
    print(f"  Testing on: {n_test_pairs:,} pairs ({100*n_test_pairs/total_pairs:.2f}%)")

    # Random subsample of pairs
    np.random.seed(42)
    test_indices = np.random.choice(total_pairs, size=n_test_pairs, replace=False)
    source_nodes = test_indices // n_target
    target_nodes = test_indices % n_target

    # Collect pathway counts across permutations
    pathway_counts_list = []

    for perm_id in perm_range:
        perm_path = os.path.join(perm_dir, f'{perm_id:03d}.hetmat')

        edge1_perm = load_edge_matrix(perm_path, edge1_abbrev)
        edge2_perm = load_edge_matrix(perm_path, edge2_abbrev)

        pathway_matrix = edge1_perm.dot(edge2_perm)
        if sp.issparse(pathway_matrix):
            pathway_matrix = pathway_matrix.toarray()

        # Extract counts for selected pairs
        counts = pathway_matrix[source_nodes, target_nodes]
        pathway_counts_list.append(counts)

        if (perm_id + 1) % 5 == 0:
            print(f"    Loaded permutation {perm_id:03d}")

    # Average across permutations
    pathway_counts_array = np.array(pathway_counts_list)
    avg_pathway_counts = pathway_counts_array.mean(axis=0)

    print(f"  Average pathway count: {avg_pathway_counts.mean():.4f} ± {avg_pathway_counts.std():.4f}")
    print(f"  Range: [{avg_pathway_counts.min():.4f}, {avg_pathway_counts.max():.4f}]")

    return source_nodes, target_nodes, avg_pathway_counts, edge1_ref, edge2_ref


def create_visualizations(y_true, y_pred, output_dir, metapath_name, feature_set):
    """Create residual analysis and performance visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style('whitegrid')

    # Figure 1: Predicted vs Actual
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1a: Scatter plot
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Pathway Count')
    ax.set_ylabel('Predicted Pathway Count')
    ax.set_title(f'Predicted vs Actual\n{metapath_name}, Feature Set {feature_set}')
    ax.legend()

    # 1b: Residual plot
    ax = axes[0, 1]
    residuals = y_pred - y_true
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Pathway Count')
    ax.set_ylabel('Residual (Predicted - Actual)')
    ax.set_title('Residual Plot')

    # 1c: Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}')

    # 1d: Q-Q plot
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metapath_name}_{feature_set}_diagnostics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Performance metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 2a: Binned performance
    ax = axes[0]
    n_bins_viz = 10
    bin_edges = np.percentile(y_true, np.linspace(0, 100, n_bins_viz + 1))
    bin_indices = np.digitize(y_true, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins_viz - 1)

    bin_r_values = []
    bin_centers = []
    for i in range(n_bins_viz):
        mask = bin_indices == i
        if mask.sum() > 5:
            bin_r, _ = pearsonr(y_true[mask], y_pred[mask])
            bin_r_values.append(bin_r)
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)

    ax.plot(bin_centers, bin_r_values, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0.95, color='g', linestyle='--', label='Target r=0.95')
    ax.set_xlabel('Actual Pathway Count (binned)')
    ax.set_ylabel('Pearson r')
    ax.set_title('Performance by Pathway Count Range')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2b: Error distribution by magnitude
    ax = axes[1]
    abs_errors = np.abs(residuals)
    for i in range(n_bins_viz):
        mask = bin_indices == i
        if mask.sum() > 0:
            ax.boxplot(abs_errors[mask], positions=[i], widths=0.6)

    ax.set_xlabel('Pathway Count Bin')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Distribution by Count Range')
    ax.grid(True, alpha=0.3)

    # 2c: Cumulative error
    ax = axes[2]
    sorted_abs_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
    ax.plot(sorted_abs_errors, cumulative, linewidth=2)
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Error Distribution')
    ax.grid(True, alpha=0.3)

    # Mark percentiles
    for percentile in [50, 90, 95, 99]:
        val = np.percentile(abs_errors, percentile)
        ax.axvline(x=val, color='r', linestyle='--', alpha=0.5)
        ax.text(val, 0.5, f'{percentile}th', rotation=90, va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metapath_name}_{feature_set}_performance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Visualizations saved to {output_dir}/")


def evaluate_full(metapath_name, edge1_abbrev, edge2_abbrev,
                  feature_set='A', n_bins=10, n_permutations=20,
                  subsample_ratio=0.01, max_pairs=100000):
    """
    Full evaluation: train on bins, test on individual pairs from permutations.
    """
    print(f"\n{'='*70}")
    print(f"FULL EVALUATION: {metapath_name}, Feature Set {feature_set}")
    print(f"{'='*70}\n")

    # ===== TRAINING =====
    print("STEP 1: Training on degree bins")
    print("-" * 70)

    train_file = f'results/phase2_training_data/{metapath_name}_features_{feature_set}.csv'
    df_train = pd.read_csv(train_file)

    # Extract and filter features
    feature_cols = [col for col in df_train.columns if col.startswith('feat_')]
    X = df_train[feature_cols].values
    y = df_train['pathway_count_mean'].values

    print(f"  Training samples: {len(X)}")
    print(f"  Features (raw): {len(feature_cols)}")

    # Handle NaN and zero-variance
    X = np.nan_to_num(X, nan=0.0)
    feature_stds = X.std(axis=0)
    nonzero_var_mask = feature_stds > 1e-8
    n_zero_var = (~nonzero_var_mask).sum()

    if n_zero_var > 0:
        print(f"  Removing {n_zero_var} zero-variance features")
        X = X[:, nonzero_var_mask]
        feature_cols_filtered = [col for col, keep in zip(feature_cols, nonzero_var_mask) if keep]
    else:
        feature_cols_filtered = feature_cols

    print(f"  Features (filtered): {X.shape[1]}")

    # Train/val split
    val_size = max(1, int(0.2 * len(X)))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"  Train/val split: {len(X_train)}/{len(X_val)}")

    # Train
    model = train_model(
        X_train_scaled, y_train, X_val_scaled, y_val,
        input_dim=X.shape[1],
        hidden_dims=(32, 16),
        dropout=0.4,
        lr=0.001,
        max_epochs=1000,
        patience=100
    )

    model.eval()
    with torch.no_grad():
        y_val_pred = model(torch.FloatTensor(X_val_scaled)).numpy().flatten()

    val_r, _ = pearsonr(y_val, y_val_pred)
    print(f"  Validation r: {val_r:.4f}\n")

    # ===== TESTING =====
    print("STEP 2: Testing on individual pairs from permutations")
    print("-" * 70)

    source_nodes, target_nodes, avg_pathway_counts, edge1_ref, edge2_ref = \
        load_test_data_from_permutations(
            'data/permutations', edge1_abbrev, edge2_abbrev,
            range(n_permutations), subsample_ratio, max_pairs
        )

    # Extract features for test pairs
    print(f"  Extracting features for {len(source_nodes):,} test pairs...")

    # Process in batches to avoid memory issues
    batch_size_feat = 10000
    X_test_list = []

    for i in range(0, len(source_nodes), batch_size_feat):
        batch_src = source_nodes[i:i+batch_size_feat]
        batch_tgt = target_nodes[i:i+batch_size_feat]

        X_batch = extract_enhanced_features(
            batch_src, batch_tgt,
            edge1_ref, edge2_ref,
            n_bins=n_bins,
            feature_set=feature_set
        )

        X_test_list.append(X_batch)

        if (i + batch_size_feat) % 50000 == 0:
            print(f"    Processed {i + batch_size_feat:,} pairs...")

    X_test = np.vstack(X_test_list)
    print(f"  Feature extraction complete")

    # Filter same features as training
    X_test = np.nan_to_num(X_test, nan=0.0)
    X_test = X_test[:, nonzero_var_mask]
    X_test_scaled = scaler.transform(X_test)

    # Predict
    print(f"  Making predictions...")
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_scaled)
        y_test_pred = model(X_test_t).numpy().flatten()

    # ===== EVALUATION =====
    print("\nSTEP 3: Evaluation")
    print("-" * 70)

    test_r, test_p = pearsonr(avg_pathway_counts, y_test_pred)
    test_rmse = np.sqrt(np.mean((avg_pathway_counts - y_test_pred) ** 2))
    test_mae = np.mean(np.abs(avg_pathway_counts - y_test_pred))

    print(f"  Pearson r: {test_r:.4f} (p={test_p:.2e})")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print()

    # ===== VISUALIZATIONS =====
    print("STEP 4: Creating visualizations")
    print("-" * 70)

    output_dir = f'results/phase2_visualizations/{metapath_name}'
    create_visualizations(avg_pathway_counts, y_test_pred, output_dir,
                          metapath_name, feature_set)

    # ===== SUMMARY =====
    results = {
        'metapath': metapath_name,
        'feature_set': feature_set,
        'n_features_raw': len(feature_cols),
        'n_features_filtered': X.shape[1],
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(avg_pathway_counts),
        'val_r': val_r,
        'test_r': test_r,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'success': test_r >= 0.95
    }

    return results


def main():
    """Test on CbGpPW with full pipeline."""
    print("\n")
    print("=" * 70)
    print("PHASE 2: FULL TEST SET EVALUATION")
    print("=" * 70)
    print()

    results = evaluate_full(
        metapath_name='CbGpPW',
        edge1_abbrev='CbG',
        edge2_abbrev='GpPW',
        feature_set='A',
        n_permutations=20,
        subsample_ratio=0.01,  # Test on 1% of pairs
        max_pairs=100000  # Cap at 100k pairs
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Metapath: {results['metapath']}")
    print(f"Feature Set: {results['feature_set']}")
    print(f"Features: {results['n_features_filtered']} (filtered from {results['n_features_raw']})")
    print(f"Training samples: {results['n_train']}")
    print(f"Test pairs: {results['n_test']:,}")
    print()
    print(f"Validation r: {results['val_r']:.4f}")
    print(f"Test r: {results['test_r']:.4f}")
    print(f"Test RMSE: {results['test_rmse']:.4f}")
    print(f"Test MAE: {results['test_mae']:.4f}")
    print()

    if results['success']:
        print("SUCCESS: Test r >= 0.95")
    else:
        print(f"Gap to target: {0.95 - results['test_r']:.4f}")
        print("Next: Test additional feature sets (B-F)")

    return results['success']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
