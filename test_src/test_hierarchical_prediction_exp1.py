"""
Hierarchical Path Prediction - Experiment 1

Test whether we can predict length-2 pathway counts from length-1 edge counts
using node-level aggregated features and linear regression.

This is a sanity check that should succeed with r > 0.99.

Date: 2025-11-04
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import time
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("HIERARCHICAL PATH PREDICTION - EXPERIMENT 1")
print("Predict Length-2 from Length-1 (Sanity Check)")
print("="*80)


def load_edge_matrix(edge_abbrev):
    """Load edge matrix, handling bidirectional edges."""
    edge_file = data_dir / 'edges' / f'{edge_abbrev}.sparse.npz'

    if edge_file.exists():
        return sp.load_npz(str(edge_file))

    reverse_map = {
        'GaD': 'DaG', 'GbC': 'CbG', 'GeA': 'AeG',
        'DaG': 'GaD', 'CbG': 'GbC', 'AeG': 'GeA'
    }

    if edge_abbrev in reverse_map:
        reverse_file = data_dir / 'edges' / f'{reverse_map[edge_abbrev]}.sparse.npz'
        if reverse_file.exists():
            print(f"  Using reverse edge {reverse_map[edge_abbrev]} transposed")
            return sp.load_npz(str(reverse_file)).T

    raise FileNotFoundError(f"Could not find edge file for {edge_abbrev}")


def sample_node_pairs(path_count_matrix, n_samples=5000, target_ratio=0.5,
                      random_state=42):
    """
    Sample node pairs for training/testing.

    Args:
        path_count_matrix: Sparse matrix of path counts
        n_samples: Number of pairs to sample
        target_ratio: Target fraction with nonzero paths
        random_state: Random seed

    Returns:
        source_indices, target_indices, path_counts
    """
    np.random.seed(random_state)
    n_source, n_target = path_count_matrix.shape

    nonzero_sources, nonzero_targets = path_count_matrix.nonzero()
    n_nonzero = len(nonzero_sources)

    if n_nonzero == 0:
        raise ValueError("No non-zero paths found")

    n_nonzero_sample = min(int(n_samples * target_ratio), n_nonzero)
    nonzero_idx = np.random.choice(n_nonzero, n_nonzero_sample, replace=False)
    sampled_sources_nz = nonzero_sources[nonzero_idx]
    sampled_targets_nz = nonzero_targets[nonzero_idx]

    n_random_sample = n_samples - n_nonzero_sample
    random_sources = np.random.randint(0, n_source, n_random_sample)
    random_targets = np.random.randint(0, n_target, n_random_sample)

    all_sources = np.concatenate([sampled_sources_nz, random_sources])
    all_targets = np.concatenate([sampled_targets_nz, random_targets])

    # Convert sparse matrix to lil format for efficient single element access
    path_count_lil = path_count_matrix.tolil()
    path_counts = np.array([
        path_count_lil[s, t] for s, t in zip(all_sources, all_targets)
    ], dtype=float).flatten()

    return all_sources, all_targets, path_counts


print("\nMetapath: CbGaD (Compound → Gene → Disease)")
print("  Testing: Can we predict C→G→D counts from edge-level features?")

print("\nLoading edge matrices...")
t0 = time.time()
edge1_mat = load_edge_matrix('CbG')
edge2_mat = load_edge_matrix('GaD')
t_load = time.time() - t0

print(f"  CbG: {edge1_mat.shape}, {edge1_mat.nnz:,} edges")
print(f"  GaD: {edge2_mat.shape}, {edge2_mat.nnz:,} edges")
print(f"  Load time: {t_load:.3f}s")

print("\nComputing length-2 pathway counts (ground truth)...")
t0 = time.time()
pathway_matrix = edge1_mat @ edge2_mat
t_pathways = time.time() - t0

print(f"  Pathway matrix: {pathway_matrix.shape}")
print(f"  Non-zero pathways: {pathway_matrix.nnz:,}")
print(f"  Computation time: {t_pathways:.3f}s")

print("\nSampling node pairs...")
sources_train, targets_train, y_train = sample_node_pairs(
    pathway_matrix, n_samples=5000, target_ratio=0.5, random_state=42
)
sources_test, targets_test, y_test = sample_node_pairs(
    pathway_matrix, n_samples=5000, target_ratio=0.5, random_state=123
)

print(f"  Train: {len(y_train)} pairs, {np.sum(y_train > 0)} with paths")
print(f"  Test: {len(y_test)} pairs, {np.sum(y_test > 0)} with paths")
print(f"  Train range: [{y_train.min()}, {y_train.max()}]")
print(f"  Test range: [{y_test.min()}, {y_test.max()}]")

print("\nBuilding features...")

# Get degrees
deg_source_train = np.array(edge1_mat.sum(axis=1)).flatten()[sources_train]
deg_target_train = np.array(edge2_mat.sum(axis=1)).flatten()[targets_train]

deg_source_test = np.array(edge1_mat.sum(axis=1)).flatten()[sources_test]
deg_target_test = np.array(edge2_mat.sum(axis=1)).flatten()[targets_test]

# Get total edge counts per node (aggregated connectivity)
# For each source: total number of edges from that source in edge1
count_edge1_train = deg_source_train
count_edge2_train = deg_target_train

count_edge1_test = deg_source_test
count_edge2_test = deg_target_test

# Build feature matrix
X_train = np.column_stack([
    count_edge1_train,           # Total CbG edges from compound
    count_edge2_train,           # Total GaD edges from disease
    deg_source_train,             # Compound degree
    deg_target_train,             # Disease degree
    count_edge1_train * count_edge2_train,  # Interaction
    deg_source_train * deg_target_train,    # Degree product
    deg_source_train ** 2,        # Squared terms
    deg_target_train ** 2
])

X_test = np.column_stack([
    count_edge1_test,
    count_edge2_test,
    deg_source_test,
    deg_target_test,
    count_edge1_test * count_edge2_test,
    deg_source_test * deg_target_test,
    deg_source_test ** 2,
    deg_target_test ** 2
])

feature_names = [
    'count_CbG', 'count_GaD', 'deg_C', 'deg_D',
    'count_CbG×count_GaD', 'deg_C×deg_D',
    'deg_C²', 'deg_D²'
]

print(f"  Feature matrix: {X_train.shape}")
print(f"  Features: {feature_names}")

print("\nTraining linear regression model...")
t0 = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
t_train = time.time() - t0

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
t_pred = time.time() - t0 - t_train

r_train = np.corrcoef(y_train, y_train_pred)[0, 1]
r_test = np.corrcoef(y_test, y_test_pred)[0, 1]
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"  Training time: {t_train:.4f}s")
print(f"  Prediction time: {t_pred:.4f}s")

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nTrain Performance:")
print(f"  r = {r_train:.4f}")
print(f"  R² = {r2_train:.4f}")
print(f"  MAE = {mae_train:.2f}")

print(f"\nTest Performance:")
print(f"  r = {r_test:.4f}")
print(f"  R² = {r2_test:.4f}")
print(f"  MAE = {mae_test:.2f}")

print(f"\nFeature Coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name:20s}: {coef:12.6f}")
print(f"  {'Intercept':20s}: {model.intercept_:12.6f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if r_test > 0.95:
    result_status = "SUCCESS"
    interpretation = (
        "Hierarchical prediction is highly effective for length-2 paths. "
        "The model achieves r > 0.95, indicating that pathway counts can be "
        "accurately predicted from node-level edge count features."
    )
elif r_test > 0.8:
    result_status = "PARTIAL SUCCESS"
    interpretation = (
        "Hierarchical prediction shows moderate effectiveness with 0.8 < r < 0.95. "
        "The approach may be useful for approximation but may need refinement "
        "for production use."
    )
else:
    result_status = "FAILURE"
    interpretation = (
        "Hierarchical prediction does not work well for this test case with r < 0.8. "
        "This is unexpected for a length-2 sanity check and may indicate issues "
        "with feature engineering or data quality."
    )

print(f"\nResult: {result_status} (r = {r_test:.4f})")
print(f"\n{interpretation}")

print("\nComputational Cost Comparison:")
print(f"  Direct counting (matrix mult): {t_pathways:.3f}s")
print(f"  Feature computation: ~{t_load:.3f}s (edge loading)")
print(f"  Model training: {t_train:.4f}s")
print(f"  Prediction for {len(y_test)} pairs: {t_pred:.4f}s")

if t_pathways > t_train + t_pred:
    speedup = t_pathways / (t_train + t_pred)
    print(f"  Speedup: {speedup:.2f}x for prediction phase")
    print("  Note: For this simple case, direct counting is competitive")
    print("  Speedup matters more for length-3+ paths")

# Save results
print("\nSaving results...")

results = {
    'experiment': 'Experiment 1',
    'metapath': 'CbGaD',
    'description': 'Length-2 from Length-1 (sanity check)',
    'n_train': len(y_train),
    'n_test': len(y_test),
    'r_train': r_train,
    'r_test': r_test,
    'r2_train': r2_train,
    'r2_test': r2_test,
    'mae_train': mae_train,
    'mae_test': mae_test,
    'time_pathways': t_pathways,
    'time_train': t_train,
    'time_pred': t_pred,
    'status': result_status,
    'success': r_test > 0.95
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment1_results.csv', index=False)

# Save feature importance
df_features = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
})
df_features = df_features.sort_values('abs_coefficient', ascending=False)
df_features.to_csv(results_dir / 'experiment1_features.csv', index=False)

print("\nCreating visualizations...")

fig = plt.figure(figsize=(18, 12))

# Plot 1: Train predictions
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_train, y_train_pred, alpha=0.3, s=20)
ax1.plot([0, y_train.max()], [0, y_train.max()], 'r--', linewidth=2,
         label='Perfect prediction')
ax1.set_xlabel('True Path Count')
ax1.set_ylabel('Predicted Path Count')
ax1.set_title(f'Train Set (r={r_train:.3f})')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Test predictions
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test, y_test_pred, alpha=0.3, s=20)
ax2.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2,
         label='Perfect prediction')
ax2.set_xlabel('True Path Count')
ax2.set_ylabel('Predicted Path Count')
ax2.set_title(f'Test Set (r={r_test:.3f})')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Feature importance
ax3 = plt.subplot(2, 3, 3)
sorted_idx = np.argsort(np.abs(model.coef_))[::-1]
y_pos = np.arange(len(feature_names))
ax3.barh(y_pos, np.abs(model.coef_)[sorted_idx], edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels([feature_names[i] for i in sorted_idx])
ax3.set_xlabel('|Coefficient|')
ax3.set_title('Feature Importance')
ax3.grid(alpha=0.3, axis='x')

# Plot 4: Residuals vs predicted (train)
ax4 = plt.subplot(2, 3, 4)
residuals_train = y_train - y_train_pred
ax4.scatter(y_train_pred, residuals_train, alpha=0.3, s=20)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Path Count')
ax4.set_ylabel('Residual (True - Predicted)')
ax4.set_title('Train Residuals')
ax4.grid(alpha=0.3)

# Plot 5: Residuals vs predicted (test)
ax5 = plt.subplot(2, 3, 5)
residuals_test = y_test - y_test_pred
ax5.scatter(y_test_pred, residuals_test, alpha=0.3, s=20)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Path Count')
ax5.set_ylabel('Residual (True - Predicted)')
ax5.set_title('Test Residuals')
ax5.grid(alpha=0.3)

# Plot 6: Error distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax6.set_xlabel('Residual (True - Predicted)')
ax6.set_ylabel('Frequency')
ax6.set_title(f'Test Error Distribution (MAE={mae_test:.2f})')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'experiment1_plots.png', dpi=150, bbox_inches='tight')
print("  Saved plots")

print("\n" + "="*80)
print("EXPERIMENT 1 COMPLETE")
print("="*80)
print(f"\nResults: {result_status}")
print(f"Test correlation: r = {r_test:.4f}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment1_results.csv'}")
print(f"  {results_dir / 'experiment1_features.csv'}")
print(f"  {results_dir / 'experiment1_plots.png'}")
