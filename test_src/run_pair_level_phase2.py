"""
Pair-Level Phase 2: Degree-Aware Correction

Complete implementation that regenerates Phase 1 data and applies correction.

Goal: r > 0.90 for null distribution pathway prediction
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import time
import warnings
warnings.filterwarnings('ignore')

# Setup
repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'pair_level_phase2'
results_dir.mkdir(parents=True, exist_ok=True)

# Add src modules
sys.path.insert(0, str(repo_dir / 'src'))

print("="*80)
print("PAIR-LEVEL PHASE 2: DEGREE-AWARE CORRECTION")
print("="*80)


def load_edge_matrix(edge_type, perm_id='original'):
    """
    Load edge matrix.

    Parameters
    ----------
    edge_type : str
        Edge type code (e.g., 'CbG')
    perm_id : str or int
        'original' for original Hetionet, or 0-199 for permutations
    """
    if perm_id == 'original':
        edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
    # Convert from bool to int to enable pathway counting
    return sp.load_npz(edge_file).astype(np.int32)


def extract_pair_features_simple(edge1, edge2, source_idx, target_idx):
    """
    Extract 5 degree features for a pair.

    Features:
    1. source_degree
    2. target_degree
    3. source × target
    4. source²
    5. target²
    """
    source_deg = edge1.getrow(source_idx).nnz
    target_deg = edge2.getcol(target_idx).nnz

    return np.array([
        source_deg,
        target_deg,
        source_deg * target_deg,
        source_deg ** 2,
        target_deg ** 2
    ])


def sample_pairs_by_pathway_count(edge1, edge2, n_samples=50000, random_state=42):
    """
    Sample pairs stratified by pathway count.

    Returns sampled (source_idx, target_idx) pairs.
    """
    np.random.seed(random_state)

    # Compute pathway matrix
    pathway_matrix = edge1 @ edge2

    # Get all pairs with nonzero pathways
    sources, targets = pathway_matrix.nonzero()
    pathway_counts = np.array(pathway_matrix[sources, targets]).flatten()

    # Stratify by pathway count
    zero_pairs = min(int(n_samples * 0.5), len(sources))  # 50% from nonzero

    # Sample from nonzero pairs
    if len(sources) > zero_pairs:
        sample_idx = np.random.choice(len(sources), zero_pairs, replace=False)
        sampled_sources = sources[sample_idx]
        sampled_targets = targets[sample_idx]
    else:
        sampled_sources = sources
        sampled_targets = targets

    # Add random zero pairs
    n_zero = n_samples - len(sampled_sources)
    if n_zero > 0:
        all_sources = np.arange(edge1.shape[0])
        all_targets = np.arange(edge2.shape[1])

        # Random pairs
        zero_sources = np.random.choice(all_sources, n_zero)
        zero_targets = np.random.choice(all_targets, n_zero)

        sampled_sources = np.concatenate([sampled_sources, zero_sources])
        sampled_targets = np.concatenate([sampled_targets, zero_targets])

    pair_indices = list(zip(sampled_sources, sampled_targets))

    return pair_indices


def compute_pathway_counts_for_pairs(edge1, edge2, pair_indices):
    """Compute pathway counts for specific pairs."""
    pathway_matrix = edge1 @ edge2

    counts = np.zeros(len(pair_indices))
    for i, (source_idx, target_idx) in enumerate(pair_indices):
        counts[i] = pathway_matrix[source_idx, target_idx]

    return counts


print("\nTesting on CbGpPW metapath...")
print("-" * 80)

# Load edges from ORIGINAL Hetionet
print("  Loading edge matrices from original Hetionet...")
edge1 = load_edge_matrix('CbG', perm_id='original')
edge2 = load_edge_matrix('GpPW', perm_id='original')
print(f"    CbG: {edge1.shape}, {edge1.nnz} edges")
print(f"    GpPW: {edge2.shape}, {edge2.nnz} edges")

# Sample pairs
print(f"\n  Sampling {10000} pairs (quick test)...")
start = time.time()
pair_indices = sample_pairs_by_pathway_count(edge1, edge2, n_samples=10000)
print(f"    Sampled {len(pair_indices)} pairs in {time.time()-start:.1f}s")

# Extract features
print(f"\n  Extracting features...")
start = time.time()
X = []
for source_idx, target_idx in pair_indices:
    features = extract_pair_features_simple(edge1, edge2, source_idx, target_idx)
    X.append(features)
X = np.array(X)
print(f"    Features extracted: {X.shape} in {time.time()-start:.1f}s")

# Compute targets from permutations
print(f"\n  Computing targets from permutations 1-20...")
start = time.time()
perm_counts = []
for perm_id in range(1, 21):
    edge1_perm = load_edge_matrix('CbG', perm_id)
    edge2_perm = load_edge_matrix('GpPW', perm_id)
    counts = compute_pathway_counts_for_pairs(edge1_perm, edge2_perm, pair_indices)
    perm_counts.append(counts)

y_validation = np.mean(perm_counts, axis=0)
print(f"    Computed in {time.time()-start:.1f}s")
print(f"    Mean pathway count: {y_validation.mean():.4f}")
print(f"    Pairs with paths: {(y_validation > 0).sum()} / {len(y_validation)}")

# Compute permutation 000 counts (for correction)
print(f"\n  Computing permutation 000 counts (for correction)...")
start = time.time()
edge1_perm0 = load_edge_matrix('CbG', perm_id=0)  # Loads from permutations/000.hetmat/
edge2_perm0 = load_edge_matrix('GpPW', perm_id=0)
y_perm0 = compute_pathway_counts_for_pairs(edge1_perm0, edge2_perm0, pair_indices)
print(f"    Computed in {time.time()-start:.1f}s")
print(f"    Mean pathway count: {y_perm0.mean():.4f}")

# Train baseline model
print(f"\n  Stage 1: Training baseline model...")
base_model = LinearRegression()
base_model.fit(X, y_validation)
y_pred_base = base_model.predict(X)

r_base = pearsonr(y_pred_base, y_validation)[0]
rmse_base = np.sqrt(np.mean((y_pred_base - y_validation)**2))
bias_base = np.mean(y_pred_base - y_validation)

print(f"    Baseline: r = {r_base:.4f}, RMSE = {rmse_base:.4f}, bias = {bias_base:+.4f}")

# Extract correction features
print(f"\n  Stage 2: Training correction model...")
def extract_correction_features(X, y_pred):
    """Extract 15 correction features."""
    source_deg = X[:, 0]
    target_deg = X[:, 1]

    features = [
        source_deg, target_deg, source_deg * target_deg,
        source_deg ** 2, target_deg ** 2,
        np.sqrt(source_deg + 1), np.sqrt(target_deg + 1),
        y_pred, y_pred ** 2, np.log1p(y_pred),
        y_pred * source_deg, y_pred * target_deg,
        y_pred * source_deg * target_deg,
        np.sqrt(y_pred + 1) * source_deg,
        np.sqrt(y_pred + 1) * target_deg
    ]

    return np.column_stack(features)

correction_features = extract_correction_features(X, y_pred_base)
correction_target = y_perm0 - y_pred_base

correction_model = LinearRegression()
correction_model.fit(correction_features, correction_target)
correction = correction_model.predict(correction_features)

y_pred_corrected = y_pred_base + correction

r_corrected = pearsonr(y_pred_corrected, y_validation)[0]
rmse_corrected = np.sqrt(np.mean((y_pred_corrected - y_validation)**2))
bias_corrected = np.mean(y_pred_corrected - y_validation)

print(f"    Corrected: r = {r_corrected:.4f}, RMSE = {rmse_corrected:.4f}, bias = {bias_corrected:+.4f}")
print(f"\n    Improvement:")
print(f"      Δr    = {r_corrected - r_base:+.4f} ({100*(r_corrected/r_base - 1):+.1f}%)")
print(f"      ΔRMSE = {rmse_corrected - rmse_base:+.4f} ({100*(rmse_corrected/rmse_base - 1):+.1f}%)")
print(f"      Δbias = {bias_corrected - bias_base:+.4f}")

# Visualization
print(f"\n  Creating visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Baseline
ax = axes[0]
ax.scatter(y_validation, y_pred_base, alpha=0.3, s=5)
ax.plot([0, y_validation.max()], [0, y_validation.max()], 'r--', label='Perfect prediction')
ax.set_xlabel('True Pathway Count')
ax.set_ylabel('Predicted Pathway Count')
ax.set_title(f'Baseline (r = {r_base:.4f})')
ax.legend()
ax.grid(alpha=0.3)

# Corrected
ax = axes[1]
ax.scatter(y_validation, y_pred_corrected, alpha=0.3, s=5)
ax.plot([0, y_validation.max()], [0, y_validation.max()], 'r--', label='Perfect prediction')
ax.set_xlabel('True Pathway Count')
ax.set_ylabel('Predicted Pathway Count')
ax.set_title(f'Corrected (r = {r_corrected:.4f})')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'CbGpPW_phase2_quick_test.png', dpi=150)
print(f"    Saved: {results_dir / 'CbGpPW_phase2_quick_test.png'}")

print("\n" + "="*80)
print("QUICK TEST COMPLETE")
print("="*80)
print(f"\nResults:")
print(f"  Baseline:  r = {r_base:.4f}, RMSE = {rmse_base:.4f}")
print(f"  Corrected: r = {r_corrected:.4f}, RMSE = {rmse_corrected:.4f}")
print(f"  Improvement: {100*(r_corrected/r_base - 1):+.1f}%")

if r_corrected > 0.90:
    print(f"\n SUCCESS: Achieved r > 0.90 target!")
else:
    print(f"\n  Need more improvement to reach r > 0.90 target")
    print(f"  Gap: {0.90 - r_corrected:.4f}")
