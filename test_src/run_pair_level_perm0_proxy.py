"""
Test Permutation 0 as Proxy for Null Distribution

Phase 1: Check if perm 0 correlates with mean(perms 1-20)
Phase 2 (conditional): If Phase 1 passes, learn transformation original → perm 0

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
results_dir = repo_dir / 'results' / 'pair_level_perm0_proxy'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PERMUTATION 0 AS PROXY FOR NULL DISTRIBUTION")
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
    """Compute pathway count for a pair."""
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


def extract_jaccard_features(source_idx, target_idx, edge1, edge2):
    """Extract 8 features (degrees + Jaccard)."""
    neighbors_source = set(edge1.getrow(source_idx).nonzero()[1])
    neighbors_target = set(edge2.getcol(target_idx).nonzero()[0])

    d_u = len(neighbors_source)
    d_v = len(neighbors_target)

    if d_u == 0 or d_v == 0:
        jaccard = 0.0
    else:
        intersection = len(neighbors_source & neighbors_target)
        union = len(neighbors_source | neighbors_target)
        jaccard = intersection / union if union > 0 else 0.0

    return np.array([
        d_u, d_v, d_u * d_v, d_u ** 2, d_v ** 2,
        jaccard, jaccard * d_u, jaccard * d_v
    ], dtype=np.float64)


print("\nPHASE 1: Testing if Permutation 0 is Proxy for Null Distribution")
print("-" * 80)

# Load edges
print("  Loading original Hetionet edges...")
edge1_orig = load_edge_matrix('CbG', perm_id='original')
edge2_orig = load_edge_matrix('GpPW', perm_id='original')

# Sample pairs
print(f"\n  Sampling {10000} pairs...")
pairs = sample_pairs_stratified(edge1_orig, edge2_orig, n_samples=10000)
print(f"    Sampled {len(pairs)} pairs")

# Compute permutation 0 counts
print(f"\n  Computing pathway counts in permutation 0...")
start = time.time()
edge1_perm0 = load_edge_matrix('CbG', perm_id=0)
edge2_perm0 = load_edge_matrix('GpPW', perm_id=0)

perm0_counts = []
for source_idx, target_idx in pairs:
    count = compute_pathway_count(source_idx, target_idx, edge1_perm0, edge2_perm0)
    perm0_counts.append(count)

perm0_counts = np.array(perm0_counts)
print(f"    Computed in {time.time()-start:.1f}s")
print(f"    Mean perm 0 count: {perm0_counts.mean():.2f}")

# Compute permutations 1-20 average
print(f"\n  Computing average pathway counts from permutations 1-20...")
start = time.time()
perm_counts = []

for perm_id in range(1, 21):
    if perm_id % 5 == 0:
        print(f"      Processing permutation {perm_id}...")

    edge1_perm = load_edge_matrix('CbG', perm_id)
    edge2_perm = load_edge_matrix('GpPW', perm_id)

    counts = []
    for source_idx, target_idx in pairs:
        count = compute_pathway_count(source_idx, target_idx, edge1_perm, edge2_perm)
        counts.append(count)

    perm_counts.append(counts)

perm_mean = np.mean(perm_counts, axis=0)
print(f"    Computed in {time.time()-start:.1f}s")
print(f"    Mean perms 1-20 count: {perm_mean.mean():.2f}")

# Phase 1 validation: Correlation between perm 0 and mean(perms 1-20)
r_phase1 = pearsonr(perm0_counts, perm_mean)[0]
rmse_phase1 = np.sqrt(np.mean((perm0_counts - perm_mean)**2))
bias_phase1 = np.mean(perm0_counts - perm_mean)

print(f"\n  PHASE 1 RESULTS:")
print(f"    Correlation: r(perm0, mean_perms) = {r_phase1:.4f}")
print(f"    RMSE: {rmse_phase1:.4f}")
print(f"    Bias: {bias_phase1:+.4f}")

# Decision point
if r_phase1 > 0.9:
    print(f"\n    PASS: r = {r_phase1:.4f} > 0.9")
    print(f"    Permutation 0 is good proxy for null distribution")
    print(f"    Proceeding to Phase 2...")
    proceed_to_phase2 = True
else:
    print(f"\n    FAIL: r = {r_phase1:.4f} < 0.9")
    print(f"    Permutation 0 is NOT a good proxy for null distribution")
    print(f"    Skipping Phase 2")
    proceed_to_phase2 = False

# Visualization for Phase 1
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(perm_mean, perm0_counts, alpha=0.3, s=10)
ax.plot([0, perm_mean.max()], [0, perm_mean.max()], 'r--', label='Perfect agreement')
ax.set_xlabel('Mean Pathway Count (Perms 1-20)')
ax.set_ylabel('Pathway Count (Perm 0)')
ax.set_title(f'Phase 1: Perm 0 vs Null Mean (r = {r_phase1:.4f})')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
residuals = perm0_counts - perm_mean
ax.scatter(perm_mean, residuals, alpha=0.3, s=10)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Mean Pathway Count (Perms 1-20)')
ax.set_ylabel('Residual (Perm 0 - Mean)')
ax.set_title(f'Phase 1 Residuals (Bias = {bias_phase1:+.4f})')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'phase1_perm0_proxy_test.png', dpi=150)
print(f"\n  Saved Phase 1 visualization")

# Save Phase 1 results
phase1_results = {
    'r_perm0_vs_mean': r_phase1,
    'rmse': rmse_phase1,
    'bias': bias_phase1,
    'perm0_mean': perm0_counts.mean(),
    'perms_mean': perm_mean.mean(),
    'proceed_to_phase2': proceed_to_phase2
}

pd.DataFrame([phase1_results]).to_csv(results_dir / 'phase1_results.csv', index=False)

# Phase 2: Learn transformation (only if Phase 1 passed)
if proceed_to_phase2:
    print("\n" + "="*80)
    print("PHASE 2: Learning Transformation Original → Perm 0")
    print("="*80)

    # Extract features from original
    print("\n  Extracting features from original Hetionet...")
    start = time.time()
    X_degrees = []
    X_jaccard = []

    for source_idx, target_idx in pairs:
        feat_deg = extract_degree_features(source_idx, target_idx, edge1_orig, edge2_orig)
        feat_jac = extract_jaccard_features(source_idx, target_idx, edge1_orig, edge2_orig)
        X_degrees.append(feat_deg)
        X_jaccard.append(feat_jac)

    X_degrees = np.array(X_degrees)
    X_jaccard = np.array(X_jaccard)
    y_perm0 = perm0_counts  # Target: perm 0 counts
    y_validation = perm_mean  # Validation: mean of perms 1-20

    print(f"    Extracted in {time.time()-start:.1f}s")

    # Train/test split
    X_deg_train, X_deg_test, X_jac_train, X_jac_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
        X_degrees, X_jaccard, y_perm0, y_validation, test_size=0.2, random_state=42
    )

    # Model 1: Degrees only
    print("\n  Model 1: Degrees Only (5 features)")
    model_deg = LinearRegression()
    model_deg.fit(X_deg_train, y_train)
    y_pred_deg = model_deg.predict(X_deg_test)

    r_deg_test = pearsonr(y_pred_deg, y_test)[0]  # vs perm 0
    r_deg_val = pearsonr(y_pred_deg, y_val_test)[0]  # vs mean(perms 1-20)

    print(f"    r(predicted, perm0) = {r_deg_test:.4f}")
    print(f"    r(predicted, mean_perms) = {r_deg_val:.4f}")

    # Model 2: Degrees + Jaccard
    print("\n  Model 2: Degrees + Jaccard (8 features)")
    model_jac = LinearRegression()
    model_jac.fit(X_jac_train, y_train)
    y_pred_jac = model_jac.predict(X_jac_test)

    r_jac_test = pearsonr(y_pred_jac, y_test)[0]  # vs perm 0
    r_jac_val = pearsonr(y_pred_jac, y_val_test)[0]  # vs mean(perms 1-20)

    print(f"    r(predicted, perm0) = {r_jac_test:.4f}")
    print(f"    r(predicted, mean_perms) = {r_jac_val:.4f}")

    # Phase 2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Degrees only - vs perm 0
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred_deg, alpha=0.3, s=10)
    ax.plot([0, y_test.max()], [0, y_test.max()], 'r--')
    ax.set_xlabel('Perm 0 Count')
    ax.set_ylabel('Predicted Count')
    ax.set_title(f'Degrees Only vs Perm 0 (r = {r_deg_test:.4f})')
    ax.grid(alpha=0.3)

    # Degrees only - vs mean(perms 1-20)
    ax = axes[0, 1]
    ax.scatter(y_val_test, y_pred_deg, alpha=0.3, s=10, color='green')
    ax.plot([0, y_val_test.max()], [0, y_val_test.max()], 'r--')
    ax.set_xlabel('Mean Perms 1-20 Count')
    ax.set_ylabel('Predicted Count')
    ax.set_title(f'Degrees Only vs Mean Perms (r = {r_deg_val:.4f})')
    ax.grid(alpha=0.3)

    # Jaccard - vs perm 0
    ax = axes[1, 0]
    ax.scatter(y_test, y_pred_jac, alpha=0.3, s=10)
    ax.plot([0, y_test.max()], [0, y_test.max()], 'r--')
    ax.set_xlabel('Perm 0 Count')
    ax.set_ylabel('Predicted Count')
    ax.set_title(f'Degrees+Jaccard vs Perm 0 (r = {r_jac_test:.4f})')
    ax.grid(alpha=0.3)

    # Jaccard - vs mean(perms 1-20)
    ax = axes[1, 1]
    ax.scatter(y_val_test, y_pred_jac, alpha=0.3, s=10, color='green')
    ax.plot([0, y_val_test.max()], [0, y_val_test.max()], 'r--')
    ax.set_xlabel('Mean Perms 1-20 Count')
    ax.set_ylabel('Predicted Count')
    ax.set_title(f'Degrees+Jaccard vs Mean Perms (r = {r_jac_val:.4f})')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'phase2_transformation_learning.png', dpi=150)
    print(f"\n  Saved Phase 2 visualization")

    # Save Phase 2 results
    phase2_results = {
        'degrees_r_perm0': r_deg_test,
        'degrees_r_mean_perms': r_deg_val,
        'jaccard_r_perm0': r_jac_test,
        'jaccard_r_mean_perms': r_jac_val
    }

    pd.DataFrame([phase2_results]).to_csv(results_dir / 'phase2_results.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nPhase 1 Summary:")
print(f"  r(perm0, mean_perms) = {r_phase1:.4f}")
print(f"  Threshold: 0.9")
print(f"  Result: {'PASS' if r_phase1 > 0.9 else 'FAIL'}")

if proceed_to_phase2:
    print(f"\nPhase 2 Summary:")
    print(f"  Degrees only:")
    print(f"    r(predicted, mean_perms) = {r_deg_val:.4f}")
    print(f"  Degrees + Jaccard:")
    print(f"    r(predicted, mean_perms) = {r_jac_val:.4f}")

    if r_deg_val > 0.85 or r_jac_val > 0.85:
        print(f"\n  SUCCESS: Transformation approach works!")
    else:
        print(f"\n  Transformation learned but validation r < 0.85")
else:
    print(f"\n  Phase 2 skipped (perm 0 not good proxy)")
