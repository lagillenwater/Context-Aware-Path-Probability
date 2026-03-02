"""
Pair-Level Pathway Prediction with Jaccard Similarity

Train Linear Regression on original Hetionet using Jaccard + degree features.
Validate on permutation data.

Goal: Predict null pathway distributions without independence assumption.

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
results_dir = repo_dir / 'results' / 'pair_level_jaccard'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PAIR-LEVEL PATHWAY PREDICTION WITH JACCARD SIMILARITY")
print("="*80)


def load_edge_matrix(edge_type, perm_id='original'):
    """
    Load edge matrix from original Hetionet or permutation.

    Parameters
    ----------
    edge_type : str
        Edge type (e.g., 'CbG', 'GpPW')
    perm_id : str or int
        'original' for Hetionet, or 1-200 for permutations
    """
    if perm_id == 'original':
        edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    # Convert to int32 to avoid boolean bug
    return sp.load_npz(str(edge_file)).astype(np.int32)


def extract_jaccard_features(source_idx, target_idx, edge1, edge2):
    """
    Extract Jaccard similarity and degree features for a pair.

    Features:
    1-5: d_u, d_v, d_u*d_v, d_u^2, d_v^2
    6: jaccard = |neighbors_u ∩ neighbors_v| / |neighbors_u ∪ neighbors_v|
    7-8: jaccard*d_u, jaccard*d_v

    Parameters
    ----------
    source_idx : int
        Source node index
    target_idx : int
        Target node index
    edge1 : sparse matrix
        First edge type (source → intermediate)
    edge2 : sparse matrix
        Second edge type (intermediate → target)

    Returns
    -------
    features : array
        8-dimensional feature vector
    """
    # Get neighbor sets
    neighbors_source = set(edge1.getrow(source_idx).nonzero()[1])
    neighbors_target = set(edge2.getcol(target_idx).nonzero()[0])

    # Degrees
    d_u = len(neighbors_source)
    d_v = len(neighbors_target)

    # Jaccard similarity
    if d_u == 0 or d_v == 0:
        jaccard = 0.0
    else:
        intersection = len(neighbors_source & neighbors_target)
        union = len(neighbors_source | neighbors_target)
        jaccard = intersection / union if union > 0 else 0.0

    # Feature vector
    features = np.array([
        d_u,
        d_v,
        d_u * d_v,
        d_u ** 2,
        d_v ** 2,
        jaccard,
        jaccard * d_u,
        jaccard * d_v
    ], dtype=np.float64)

    return features


def compute_pathway_count(source_idx, target_idx, edge1, edge2):
    """
    Compute actual pathway count for a pair.

    Returns number of shared intermediate nodes.
    """
    neighbors_source = set(edge1.getrow(source_idx).nonzero()[1])
    neighbors_target = set(edge2.getcol(target_idx).nonzero()[0])
    return len(neighbors_source & neighbors_target)


def sample_pairs_stratified(edge1, edge2, n_samples=50000, random_state=42):
    """
    Sample pairs stratified by pathway count.

    Returns list of (source_idx, target_idx) tuples.
    """
    np.random.seed(random_state)

    # Compute pathway matrix
    pathway_matrix = edge1 @ edge2

    # Get pairs with nonzero pathways
    sources_nonzero, targets_nonzero = pathway_matrix.nonzero()

    # Sample 50% from nonzero pairs
    n_nonzero = min(int(n_samples * 0.5), len(sources_nonzero))
    if len(sources_nonzero) > 0:
        idx_nonzero = np.random.choice(len(sources_nonzero), n_nonzero, replace=False)
        sampled_sources = list(sources_nonzero[idx_nonzero])
        sampled_targets = list(targets_nonzero[idx_nonzero])
    else:
        sampled_sources = []
        sampled_targets = []

    # Sample random pairs (mostly zeros)
    n_random = n_samples - len(sampled_sources)
    random_sources = np.random.choice(edge1.shape[0], n_random)
    random_targets = np.random.choice(edge2.shape[1], n_random)

    sampled_sources.extend(random_sources)
    sampled_targets.extend(random_targets)

    pairs = list(zip(sampled_sources, sampled_targets))

    return pairs


print("\nTesting on CbGpPW metapath...")
print("-" * 80)

# Load original Hetionet edges
print("  Loading original Hetionet edges...")
edge1_orig = load_edge_matrix('CbG', perm_id='original')
edge2_orig = load_edge_matrix('GpPW', perm_id='original')
print(f"    CbG: {edge1_orig.shape}, {edge1_orig.nnz} edges")
print(f"    GpPW: {edge2_orig.shape}, {edge2_orig.nnz} edges")

# Sample pairs
print(f"\n  Sampling {50000} pairs from original graph...")
start = time.time()
pairs = sample_pairs_stratified(edge1_orig, edge2_orig, n_samples=50000)
print(f"    Sampled {len(pairs)} pairs in {time.time()-start:.1f}s")

# Extract features and targets from ORIGINAL graph
print(f"\n  Extracting features and targets from original Hetionet...")
start = time.time()
X = []
y = []

for source_idx, target_idx in pairs:
    features = extract_jaccard_features(source_idx, target_idx, edge1_orig, edge2_orig)
    target = compute_pathway_count(source_idx, target_idx, edge1_orig, edge2_orig)

    X.append(features)
    y.append(target)

X = np.array(X)
y = np.array(y)
print(f"    Extracted {X.shape} features in {time.time()-start:.1f}s")
print(f"    Pathway count range: {y.min():.0f} - {y.max():.0f}")
print(f"    Mean pathway count: {y.mean():.2f}")
print(f"    Pairs with pathways: {(y > 0).sum()} / {len(y)}")

# Train/test split
print(f"\n  Splitting into train (80%) and test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Train: {len(X_train)} pairs")
print(f"    Test: {len(X_test)} pairs")

# Train Linear Regression
print(f"\n  Training Linear Regression...")
start = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
print(f"    Trained in {time.time()-start:.3f}s")

# Evaluate on train set
y_pred_train = model.predict(X_train)
r_train = pearsonr(y_pred_train, y_train)[0]
rmse_train = np.sqrt(np.mean((y_pred_train - y_train)**2))
bias_train = np.mean(y_pred_train - y_train)
print(f"\n  Train Performance:")
print(f"    r = {r_train:.4f}")
print(f"    RMSE = {rmse_train:.4f}")
print(f"    Bias = {bias_train:+.4f}")

# Evaluate on test set
y_pred_test = model.predict(X_test)
r_test = pearsonr(y_pred_test, y_test)[0]
rmse_test = np.sqrt(np.mean((y_pred_test - y_test)**2))
bias_test = np.mean(y_pred_test - y_test)
print(f"\n  Test Performance:")
print(f"    r = {r_test:.4f}")
print(f"    RMSE = {rmse_test:.4f}")
print(f"    Bias = {bias_test:+.4f}")

# Compute permutation validation
print(f"\n  Validating on permutations 1-20...")
print(f"    Computing mean pathway counts from permutations...")
start = time.time()

# Use test pairs for validation
test_pairs = [(pairs[i][0], pairs[i][1]) for i in range(len(pairs)) if i >= int(0.8*len(pairs))]

perm_counts = []
for perm_id in range(1, 21):
    if perm_id % 5 == 0:
        print(f"      Processing permutation {perm_id}...")

    edge1_perm = load_edge_matrix('CbG', perm_id)
    edge2_perm = load_edge_matrix('GpPW', perm_id)

    counts = []
    for source_idx, target_idx in test_pairs:
        count = compute_pathway_count(source_idx, target_idx, edge1_perm, edge2_perm)
        counts.append(count)

    perm_counts.append(counts)

y_perm_mean = np.mean(perm_counts, axis=0)
print(f"    Computed in {time.time()-start:.1f}s")
print(f"    Mean permutation pathway count: {y_perm_mean.mean():.2f}")

# Validation: Compare model predictions (from original features) vs permutation means
r_val = pearsonr(y_pred_test, y_perm_mean)[0]
rmse_val = np.sqrt(np.mean((y_pred_test - y_perm_mean)**2))
bias_val = np.mean(y_pred_test - y_perm_mean)
print(f"\n  Validation Performance (Model vs Permutation Mean):")
print(f"    r = {r_val:.4f}")
print(f"    RMSE = {rmse_val:.4f}")
print(f"    Bias = {bias_val:+.4f}")

# Feature importance
print(f"\n  Feature Importance (Coefficients):")
feature_names = ['d_u', 'd_v', 'd_u*d_v', 'd_u²', 'd_v²', 'jaccard', 'jaccard*d_u', 'jaccard*d_v']
for name, coef in zip(feature_names, model.coef_):
    print(f"    {name:12s}: {coef:+.6f}")
print(f"    Intercept: {model.intercept_:+.6f}")

# Save results
results = {
    'train_r': r_train,
    'train_rmse': rmse_train,
    'train_bias': bias_train,
    'test_r': r_test,
    'test_rmse': rmse_test,
    'test_bias': bias_test,
    'validation_r': r_val,
    'validation_rmse': rmse_val,
    'validation_bias': bias_val,
    'n_train': len(X_train),
    'n_test': len(X_test)
}

results_df = pd.DataFrame([results])
results_df.to_csv(results_dir / 'jaccard_results.csv', index=False)
print(f"\n  Saved results to {results_dir / 'jaccard_results.csv'}")

# Visualizations
print(f"\n  Creating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Predicted vs Observed
# Train
ax = axes[0, 0]
ax.scatter(y_train, y_pred_train, alpha=0.3, s=5)
ax.plot([0, y_train.max()], [0, y_train.max()], 'r--', label='Perfect prediction')
ax.set_xlabel('Observed Pathway Count')
ax.set_ylabel('Predicted Pathway Count')
ax.set_title(f'Train Set (r = {r_train:.4f})')
ax.legend()
ax.grid(alpha=0.3)

# Test
ax = axes[0, 1]
ax.scatter(y_test, y_pred_test, alpha=0.3, s=5)
ax.plot([0, y_test.max()], [0, y_test.max()], 'r--', label='Perfect prediction')
ax.set_xlabel('Observed Pathway Count')
ax.set_ylabel('Predicted Pathway Count')
ax.set_title(f'Test Set (r = {r_test:.4f})')
ax.legend()
ax.grid(alpha=0.3)

# Validation
ax = axes[0, 2]
ax.scatter(y_perm_mean, y_pred_test, alpha=0.3, s=5, color='green')
ax.plot([0, y_perm_mean.max()], [0, y_perm_mean.max()], 'r--', label='Perfect prediction')
ax.set_xlabel('Permutation Mean Pathway Count')
ax.set_ylabel('Predicted Pathway Count')
ax.set_title(f'Validation (r = {r_val:.4f})')
ax.legend()
ax.grid(alpha=0.3)

# Row 2: Residuals
# Train
ax = axes[1, 0]
residuals_train = y_train - y_pred_train
ax.scatter(y_pred_train, residuals_train, alpha=0.3, s=5)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted Pathway Count')
ax.set_ylabel('Residual')
ax.set_title(f'Train Residuals (Bias = {bias_train:+.4f})')
ax.grid(alpha=0.3)

# Test
ax = axes[1, 1]
residuals_test = y_test - y_pred_test
ax.scatter(y_pred_test, residuals_test, alpha=0.3, s=5)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted Pathway Count')
ax.set_ylabel('Residual')
ax.set_title(f'Test Residuals (Bias = {bias_test:+.4f})')
ax.grid(alpha=0.3)

# Validation
ax = axes[1, 2]
residuals_val = y_perm_mean - y_pred_test
ax.scatter(y_pred_test, residuals_val, alpha=0.3, s=5, color='green')
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted Pathway Count')
ax.set_ylabel('Residual (Perm Mean - Predicted)')
ax.set_title(f'Validation Residuals (Bias = {bias_val:+.4f})')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'jaccard_analysis.png', dpi=150, bbox_inches='tight')
print(f"    Saved visualization to {results_dir / 'jaccard_analysis.png'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Train:      r = {r_train:.4f}")
print(f"  Test:       r = {r_test:.4f}")
print(f"  Validation: r = {r_val:.4f}")

if r_test > 0.85 and r_val > 0.85:
    print(f"\n  SUCCESS: Both test and validation exceed r > 0.85 target!")
elif r_test > 0.85:
    print(f"\n  PARTIAL: Test passes r > 0.85, but validation r = {r_val:.4f}")
else:
    print(f"\n  Need improvement: Test r = {r_test:.4f}, target is r > 0.85")
