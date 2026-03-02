"""
Hierarchical Path Prediction - Experiment 2A

Test whether we can predict length-3 pathway counts from aggregated length-2
subpath counts using learned composition functions.

This is fundamentally different from Experiment 1:
- Experiment 1: Aggregate degrees -> pair counts (FAILED r=0.32)
- Experiment 2A: Subpath counts -> full path counts (TESTING)

Metapath: CbGiGpPW (Compound -> Gene -> Gene -> Pathway)
Subpaths: CbGiG and GiGpPW

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
print("HIERARCHICAL PATH PREDICTION - EXPERIMENT 2A")
print("Predict Length-3 from Length-2 Subpath Counts")
print("="*80)


def load_edge_matrix(edge_abbrev):
    """Load edge matrix."""
    edge_file = data_dir / 'edges' / f'{edge_abbrev}.sparse.npz'

    if edge_file.exists():
        return sp.load_npz(str(edge_file))

    raise FileNotFoundError(f"Could not find edge file for {edge_abbrev}")


def sample_node_pairs(path_count_matrix, n_samples=5000, target_ratio=0.5,
                      random_state=42):
    """Sample node pairs for training/testing."""
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

    path_count_lil = path_count_matrix.tolil()
    path_counts = np.array([
        path_count_lil[s, t] for s, t in zip(all_sources, all_targets)
    ], dtype=float).flatten()

    return all_sources, all_targets, path_counts


print("\nMetapath: CbGiGpPW (Compound -> Gene -> Gene -> Pathway)")
print("  Length-3 path predicted from length-2 subpath counts")

print("\nLoading edge matrices...")
t0 = time.time()
CbG = load_edge_matrix('CbG')
GiG = load_edge_matrix('GiG')
GpPW = load_edge_matrix('GpPW')
t_load = time.time() - t0

print(f"  CbG: {CbG.shape}, {CbG.nnz:,} edges")
print(f"  GiG: {GiG.shape}, {GiG.nnz:,} edges")
print(f"  GpPW: {GpPW.shape}, {GpPW.nnz:,} edges")
print(f"  Load time: {t_load:.3f}s")

print("\nComputing subpath counts (length-2)...")
t0 = time.time()
CbGiG = CbG @ GiG
t_CbGiG = time.time() - t0

print(f"  CbGiG: {CbGiG.shape}, {CbGiG.nnz:,} non-zero")
print(f"  Computation time: {t_CbGiG:.3f}s")

t0 = time.time()
GiGpPW = GiG @ GpPW
t_GiGpPW = time.time() - t0

print(f"  GiGpPW: {GiGpPW.shape}, {GiGpPW.nnz:,} non-zero")
print(f"  Computation time: {t_GiGpPW:.3f}s")

print("\nComputing full path counts (length-3, ground truth)...")
t0 = time.time()
CbGiGpPW = CbGiG @ GpPW
t_CbGiGpPW = time.time() - t0

print(f"  CbGiGpPW: {CbGiGpPW.shape}, {CbGiGpPW.nnz:,} non-zero")
print(f"  Computation time: {t_CbGiGpPW:.3f}s")

print("\nSampling node pairs...")
sources_train, targets_train, y_train = sample_node_pairs(
    CbGiGpPW, n_samples=5000, target_ratio=0.5, random_state=42
)
sources_test, targets_test, y_test = sample_node_pairs(
    CbGiGpPW, n_samples=5000, target_ratio=0.5, random_state=123
)

print(f"  Train: {len(y_train)} pairs, {np.sum(y_train > 0)} with paths")
print(f"  Test: {len(y_test)} pairs, {np.sum(y_test > 0)} with paths")
print(f"  Train range: [{y_train.min():.1f}, {y_train.max():.1f}]")
print(f"  Test range: [{y_test.min():.1f}, {y_test.max():.1f}]")

print("\nBuilding features from subpath counts...")

# Convert to CSR for efficient row operations
CbGiG_csr = CbGiG.tocsr()
GiGpPW_csr = GiGpPW.tocsr()

# Build features for training set
print("  Processing training features...")
X_train_list = []
for src, tgt in zip(sources_train, targets_train):
    # Subpath counts
    CbGiG_row = CbGiG_csr.getrow(src).toarray().flatten()
    GiGpPW_col = GiGpPW_csr.getcol(tgt).toarray().flatten()

    # Aggregated statistics
    total_CbGiG = CbGiG_row.sum()
    total_GiGpPW = GiGpPW_col.sum()
    max_CbGiG = CbGiG_row.max()
    max_GiGpPW = GiGpPW_col.max()
    n_nonzero_CbGiG = np.count_nonzero(CbGiG_row)
    n_nonzero_GiGpPW = np.count_nonzero(GiGpPW_col)

    # Degrees
    deg_C = CbG.getrow(src).nnz
    deg_PW = GpPW.getcol(tgt).nnz

    # Interaction terms
    interaction = total_CbGiG * total_GiGpPW
    deg_product = deg_C * deg_PW

    # Naive compositional baseline
    naive_composition = np.sum(CbGiG_row * GiGpPW_col)

    features = [
        total_CbGiG,
        total_GiGpPW,
        max_CbGiG,
        max_GiGpPW,
        n_nonzero_CbGiG,
        n_nonzero_GiGpPW,
        deg_C,
        deg_PW,
        interaction,
        deg_product,
        naive_composition,
        deg_C ** 2,
        deg_PW ** 2
    ]

    X_train_list.append(features)

X_train = np.array(X_train_list)

print("  Processing test features...")
X_test_list = []
for src, tgt in zip(sources_test, targets_test):
    CbGiG_row = CbGiG_csr.getrow(src).toarray().flatten()
    GiGpPW_col = GiGpPW_csr.getcol(tgt).toarray().flatten()

    total_CbGiG = CbGiG_row.sum()
    total_GiGpPW = GiGpPW_col.sum()
    max_CbGiG = CbGiG_row.max()
    max_GiGpPW = GiGpPW_col.max()
    n_nonzero_CbGiG = np.count_nonzero(CbGiG_row)
    n_nonzero_GiGpPW = np.count_nonzero(GiGpPW_col)

    deg_C = CbG.getrow(src).nnz
    deg_PW = GpPW.getcol(tgt).nnz

    interaction = total_CbGiG * total_GiGpPW
    deg_product = deg_C * deg_PW
    naive_composition = np.sum(CbGiG_row * GiGpPW_col)

    features = [
        total_CbGiG, total_GiGpPW, max_CbGiG, max_GiGpPW,
        n_nonzero_CbGiG, n_nonzero_GiGpPW, deg_C, deg_PW,
        interaction, deg_product, naive_composition,
        deg_C ** 2, deg_PW ** 2
    ]

    X_test_list.append(features)

X_test = np.array(X_test_list)

feature_names = [
    'total_CbGiG', 'total_GiGpPW', 'max_CbGiG', 'max_GiGpPW',
    'n_nonzero_CbGiG', 'n_nonzero_GiGpPW', 'deg_C', 'deg_PW',
    'total_CbGiG×total_GiGpPW', 'deg_C×deg_PW', 'naive_composition',
    'deg_C²', 'deg_PW²'
]

print(f"  Feature matrix: {X_train.shape}")
print(f"  Features: {len(feature_names)}")

print("\nTraining linear regression model...")
t0 = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
t_train = time.time() - t0

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

r_train = np.corrcoef(y_train, y_train_pred)[0, 1]
r_test = np.corrcoef(y_test, y_test_pred)[0, 1]
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"  Training time: {t_train:.4f}s")

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

print(f"\nTop 5 Feature Coefficients (by absolute value):")
abs_coefs = np.abs(model.coef_)
top_idx = np.argsort(abs_coefs)[::-1][:5]
for idx in top_idx:
    print(f"  {feature_names[idx]:25s}: {model.coef_[idx]:12.6f}")
print(f"  {'Intercept':25s}: {model.intercept_:12.6f}")

print("\n" + "="*80)
print("BASELINE COMPARISONS")
print("="*80)

# Baseline 1: Degrees only (like Experiment 1)
X_baseline_deg = X_train[:, [6, 7, 9, 11, 12]]  # deg_C, deg_PW, deg_product, squares
model_deg = LinearRegression()
model_deg.fit(X_baseline_deg, y_train)
y_test_deg = model_deg.predict(X_test[:, [6, 7, 9, 11, 12]])
r_deg = np.corrcoef(y_test, y_test_deg)[0, 1]

# Baseline 2: Naive composition only
r_naive = np.corrcoef(y_test, X_test[:, 10])[0, 1]  # naive_composition feature

print(f"\nBaseline 1 (Degrees only): r = {r_deg:.4f}")
print(f"Baseline 2 (Naive composition): r = {r_naive:.4f}")
print(f"Full Model (Subpath features): r = {r_test:.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if r_test > 0.90:
    result_status = "SUCCESS"
    interpretation = (
        f"Hierarchical prediction WORKS with r = {r_test:.4f} > 0.90. "
        "Aggregated subpath counts contain sufficient information to accurately "
        "predict full path counts. This approach is viable for scaling to longer paths."
    )
elif r_test > 0.70:
    result_status = "PARTIAL SUCCESS"
    interpretation = (
        f"Hierarchical prediction shows promise with r = {r_test:.4f} (0.70-0.90 range). "
        "The approach captures significant information but may need refinement "
        "(non-linear models, additional features) for production use."
    )
else:
    result_status = "FAILURE"
    interpretation = (
        f"Hierarchical prediction does not work well with r = {r_test:.4f} < 0.70. "
        "Subpath aggregates do not contain sufficient information to predict "
        "full path counts. May need alternative approaches."
    )

print(f"\nResult: {result_status}")
print(f"\n{interpretation}")

print(f"\nComparison to Experiment 1:")
print(f"  Experiment 1 (aggregate degrees): r = 0.32 (FAILURE)")
print(f"  Experiment 2A (subpath counts): r = {r_test:.4f}")
if r_test > 0.32:
    improvement = ((r_test - 0.32) / 0.32) * 100
    print(f"  Improvement: {improvement:.1f}%")
else:
    print(f"  No improvement over Experiment 1")

# Save results
print("\nSaving results...")

results = {
    'experiment': 'Experiment 2A',
    'metapath': 'CbGiGpPW',
    'description': 'Length-3 from length-2 subpath counts',
    'n_train': len(y_train),
    'n_test': len(y_test),
    'r_train': r_train,
    'r_test': r_test,
    'r2_train': r2_train,
    'r2_test': r2_test,
    'mae_train': mae_train,
    'mae_test': mae_test,
    'r_baseline_degrees': r_deg,
    'r_baseline_naive': r_naive,
    'time_subpaths': t_CbGiG + t_GiGpPW,
    'time_fullpath': t_CbGiGpPW,
    'time_train': t_train,
    'status': result_status,
    'success': r_test > 0.90
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2a_results.csv', index=False)

df_features = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
})
df_features = df_features.sort_values('abs_coefficient', ascending=False)
df_features.to_csv(results_dir / 'experiment2a_features.csv', index=False)

print("\nCreating visualizations...")

fig = plt.figure(figsize=(18, 12))

# Plot 1: Train predictions
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_train, y_train_pred, alpha=0.3, s=20)
ax1.plot([0, y_train.max()], [0, y_train.max()], 'r--', linewidth=2)
ax1.set_xlabel('True Path Count')
ax1.set_ylabel('Predicted Path Count')
ax1.set_title(f'Train Set (r={r_train:.3f})')
ax1.grid(alpha=0.3)

# Plot 2: Test predictions
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test, y_test_pred, alpha=0.3, s=20)
ax2.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
ax2.set_xlabel('True Path Count')
ax2.set_ylabel('Predicted Path Count')
ax2.set_title(f'Test Set (r={r_test:.3f})')
ax2.grid(alpha=0.3)

# Plot 3: Feature importance
ax3 = plt.subplot(2, 3, 3)
sorted_idx = np.argsort(np.abs(model.coef_))[::-1]
y_pos = np.arange(len(feature_names))
ax3.barh(y_pos, np.abs(model.coef_)[sorted_idx], edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
ax3.set_xlabel('|Coefficient|')
ax3.set_title('Feature Importance')
ax3.grid(alpha=0.3, axis='x')

# Plot 4: Baseline comparison
ax4 = plt.subplot(2, 3, 4)
models = ['Degrees\nOnly', 'Naive\nComposition', 'Full Model\n(Subpath)']
correlations = [r_deg, r_naive, r_test]
colors = ['red' if r < 0.7 else 'orange' if r < 0.9 else 'green' for r in correlations]
ax4.bar(models, correlations, color=colors, edgecolor='black', alpha=0.7)
ax4.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='Success (r>0.9)')
ax4.axhline(y=0.70, color='orange', linestyle='--', linewidth=2, label='Partial (r>0.7)')
ax4.set_ylabel('Test Correlation (r)')
ax4.set_title('Model Comparison')
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

# Plot 5: Residuals
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
ax6.set_title(f'Error Distribution (MAE={mae_test:.2f})')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'experiment2a_plots.png', dpi=150, bbox_inches='tight')
print("  Saved plots")

print("\n" + "="*80)
print("EXPERIMENT 2A COMPLETE")
print("="*80)
print(f"\nResult: {result_status}")
print(f"Test correlation: r = {r_test:.4f}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2a_results.csv'}")
print(f"  {results_dir / 'experiment2a_features.csv'}")
print(f"  {results_dir / 'experiment2a_plots.png'}")
