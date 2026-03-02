"""
Minimal Permutations Test: 5 Permutations (0-4) to Predict Mean(5-20)

Test if 5 permutations are sufficient for training a null model.
75% reduction from 20 permutations.

Date: 2025-11-03
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import time
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'pair_level_minimal_perms'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MINIMAL PERMUTATIONS TEST: 5 PERMS (0-4) → MEAN(5-20)")
print("="*80)


def load_edge_matrix(edge_type, perm_id='original'):
    """Load edge matrix."""
    if perm_id == 'original':
        edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    return sp.load_npz(str(edge_file)).astype(np.int32)


def compute_pathway_count(source_idx, target_idx, edge1, edge2):
    """Compute pathway count."""
    neighbors_source = set(edge1.getrow(source_idx).nonzero()[1])
    neighbors_target = set(edge2.getcol(target_idx).nonzero()[0])
    return len(neighbors_source & neighbors_target)


def sample_pairs_stratified(edge1, edge2, n_samples=10000, random_state=42):
    """Sample pairs stratified by pathway count."""
    np.random.seed(random_state)

    pathway_matrix = edge1 @ edge2
    sources_nonzero, targets_nonzero = pathway_matrix.nonzero()

    n_nonzero = min(int(n_samples * 0.5), len(sources_nonzero))
    if len(sources_nonzero) > 0:
        idx_nonzero = np.random.choice(len(sources_nonzero), n_nonzero, replace=False)
        sampled_sources = list(sources_nonzero[idx_nonzero])
        sampled_targets = list(targets_nonzero[idx_nonzero])
    else:
        sampled_sources = []
        sampled_targets = []

    n_random = n_samples - len(sampled_sources)
    random_sources = np.random.choice(edge1.shape[0], n_random)
    random_targets = np.random.choice(edge2.shape[1], n_random)

    sampled_sources.extend(random_sources)
    sampled_targets.extend(random_targets)

    pairs = list(zip(sampled_sources, sampled_targets))
    return pairs


def extract_degree_features(source_idx, target_idx, edge1, edge2):
    """Extract 5 degree features."""
    d_u = edge1.getrow(source_idx).nnz
    d_v = edge2.getcol(target_idx).nnz

    return np.array([
        d_u, d_v, d_u * d_v, d_u ** 2, d_v ** 2
    ], dtype=np.float64)


print("\nCbGpPW metapath analysis...")
print("-" * 80)

# Load original edges
print("  Loading original Hetionet edges...")
edge1_orig = load_edge_matrix('CbG', perm_id='original')
edge2_orig = load_edge_matrix('GpPW', perm_id='original')

# Sample pairs
print(f"\n  Sampling {10000} pairs...")
pairs = sample_pairs_stratified(edge1_orig, edge2_orig, n_samples=10000)
print(f"    Sampled {len(pairs)} pairs")

# Extract features from original
print(f"\n  Extracting degree features from original...")
start = time.time()
X = []
for source_idx, target_idx in pairs:
    features = extract_degree_features(source_idx, target_idx, edge1_orig, edge2_orig)
    X.append(features)

X = np.array(X)
print(f"    Extracted {X.shape} features in {time.time()-start:.1f}s")

# Compute training target: mean(perms 0-4)
print(f"\n  Computing training target: mean(perms 0-4)...")
start = time.time()
train_perm_counts = []

for perm_id in range(0, 5):
    edge1_perm = load_edge_matrix('CbG', perm_id)
    edge2_perm = load_edge_matrix('GpPW', perm_id)

    counts = []
    for source_idx, target_idx in pairs:
        count = compute_pathway_count(source_idx, target_idx, edge1_perm, edge2_perm)
        counts.append(count)

    train_perm_counts.append(counts)
    print(f"      Processed perm {perm_id}")

y_train_target = np.mean(train_perm_counts, axis=0)
print(f"    Computed in {time.time()-start:.1f}s")
print(f"    Mean pathway count (perms 0-4): {y_train_target.mean():.4f}")

# Compute validation target: mean(perms 5-20)
print(f"\n  Computing validation target: mean(perms 5-20)...")
start = time.time()
val_perm_counts = []

for perm_id in range(5, 21):
    if perm_id % 5 == 0:
        print(f"      Processing perm {perm_id}...")

    edge1_perm = load_edge_matrix('CbG', perm_id)
    edge2_perm = load_edge_matrix('GpPW', perm_id)

    counts = []
    for source_idx, target_idx in pairs:
        count = compute_pathway_count(source_idx, target_idx, edge1_perm, edge2_perm)
        counts.append(count)

    val_perm_counts.append(counts)

y_val_target = np.mean(val_perm_counts, axis=0)
print(f"    Computed in {time.time()-start:.1f}s")
print(f"    Mean pathway count (perms 5-20): {y_val_target.mean():.4f}")

# Check correlation between train and val targets
r_targets = pearsonr(y_train_target, y_val_target)[0]
print(f"\n  Correlation between targets:")
print(f"    r(mean_perms_0-4, mean_perms_5-20) = {r_targets:.4f}")

# Train/test split
print(f"\n  Splitting into train (80%) and test (20%)...")
X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
    X, y_train_target, y_val_target, test_size=0.2, random_state=42
)
print(f"    Train: {len(X_train)} pairs")
print(f"    Test: {len(X_test)} pairs")

# Train model
print(f"\n  Training Linear Regression...")
start = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
print(f"    Trained in {time.time()-start:.3f}s")

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate on training target (mean perms 0-4)
r_train = pearsonr(y_pred_train, y_train)[0]
rmse_train = np.sqrt(np.mean((y_pred_train - y_train)**2))

r_test = pearsonr(y_pred_test, y_test)[0]
rmse_test = np.sqrt(np.mean((y_pred_test - y_test)**2))

print(f"\n  Performance on Training Target (mean perms 0-4):")
print(f"    Train: r = {r_train:.4f}, RMSE = {rmse_train:.4f}")
print(f"    Test:  r = {r_test:.4f}, RMSE = {rmse_test:.4f}")

# Evaluate on validation target (mean perms 5-20)
r_val_train = pearsonr(y_pred_train, y_val_train)[0]
rmse_val_train = np.sqrt(np.mean((y_pred_train - y_val_train)**2))
bias_val_train = np.mean(y_pred_train - y_val_train)

r_val_test = pearsonr(y_pred_test, y_val_test)[0]
rmse_val_test = np.sqrt(np.mean((y_pred_test - y_val_test)**2))
bias_val_test = np.mean(y_pred_test - y_val_test)

print(f"\n  Performance on Validation Target (mean perms 5-20):")
print(f"    Train: r = {r_val_train:.4f}, RMSE = {rmse_val_train:.4f}, Bias = {bias_val_train:+.4f}")
print(f"    Test:  r = {r_val_test:.4f}, RMSE = {rmse_val_test:.4f}, Bias = {bias_val_test:+.4f}")

# Feature importance
print(f"\n  Feature Coefficients:")
feature_names = ['d_u', 'd_v', 'd_u*d_v', 'd_u²', 'd_v²']
for name, coef in zip(feature_names, model.coef_):
    print(f"    {name:12s}: {coef:+.6f}")
print(f"    Intercept: {model.intercept_:+.6f}")

# Save results
results = {
    'r_targets_correlation': r_targets,
    'train_r_perms04': r_train,
    'test_r_perms04': r_test,
    'train_r_perms520': r_val_train,
    'test_r_perms520': r_val_test,
    'rmse_val_test': rmse_val_test,
    'bias_val_test': bias_val_test,
    'mean_train_target': y_train_target.mean(),
    'mean_val_target': y_val_target.mean()
}

pd.DataFrame([results]).to_csv(results_dir / 'minimal_perms_results.csv', index=False)

# Visualization
print(f"\n  Creating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Target correlation and predictions
ax = axes[0, 0]
ax.scatter(y_val_target, y_train_target, alpha=0.3, s=10)
ax.plot([0, y_val_target.max()], [0, y_val_target.max()], 'r--', label='Perfect agreement')
ax.set_xlabel('Mean Perms 5-20')
ax.set_ylabel('Mean Perms 0-4')
ax.set_title(f'Training vs Validation Targets (r = {r_targets:.4f})')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.scatter(y_test, y_pred_test, alpha=0.3, s=10)
ax.plot([0, y_test.max()], [0, y_test.max()], 'r--', label='Perfect prediction')
ax.set_xlabel('Observed (Mean Perms 0-4)')
ax.set_ylabel('Predicted')
ax.set_title(f'Test vs Training Target (r = {r_test:.4f})')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.scatter(y_val_test, y_pred_test, alpha=0.3, s=10, color='green')
ax.plot([0, y_val_test.max()], [0, y_val_test.max()], 'r--', label='Perfect prediction')
ax.set_xlabel('Observed (Mean Perms 5-20)')
ax.set_ylabel('Predicted')
ax.set_title(f'Test vs Validation Target (r = {r_val_test:.4f})')
ax.legend()
ax.grid(alpha=0.3)

# Row 2: Residuals
ax = axes[1, 0]
residuals_target = y_train_target - y_val_target
ax.scatter(y_val_target, residuals_target, alpha=0.3, s=10)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Mean Perms 5-20')
ax.set_ylabel('Residual (Perms 0-4 - Perms 5-20)')
ax.set_title('Target Difference')
ax.grid(alpha=0.3)

ax = axes[1, 1]
residuals_test = y_test - y_pred_test
ax.scatter(y_pred_test, residuals_test, alpha=0.3, s=10)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Residual (Observed - Predicted)')
ax.set_title('Model Residuals (vs Perms 0-4)')
ax.grid(alpha=0.3)

ax = axes[1, 2]
residuals_val = y_val_test - y_pred_test
ax.scatter(y_pred_test, residuals_val, alpha=0.3, s=10, color='green')
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Residual (Perms 5-20 - Predicted)')
ax.set_title(f'Validation Residuals (Bias = {bias_val_test:+.4f})')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'minimal_perms_analysis.png', dpi=150)
print(f"    Saved visualization")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Target correlation: r(perms 0-4, perms 5-20) = {r_targets:.4f}")
print(f"  Model performance:")
print(f"    vs Training target (perms 0-4):  r = {r_test:.4f}")
print(f"    vs Validation target (perms 5-20): r = {r_val_test:.4f}")

if r_val_test > 0.85:
    print(f"\n  SUCCESS: Validation r = {r_val_test:.4f} > 0.85!")
    print(f"  5 permutations (0-4) are sufficient for training!")
else:
    print(f"\n  Validation r = {r_val_test:.4f} < 0.85")
    print(f"  5 permutations may not be sufficient")
