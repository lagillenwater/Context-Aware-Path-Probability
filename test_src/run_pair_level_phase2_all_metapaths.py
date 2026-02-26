"""
Pair-Level Phase 2: Complete Analysis on All Metapaths

Tests degree-aware correction on CbGpPW, CtDaG, and CrCbG.
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
results_dir = repo_dir / 'results' / 'pair_level_phase2_all'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PAIR-LEVEL PHASE 2: ALL METAPATHS")
print("="*80)

# Configuration
METAPATHS = [
    ('CbG', 'GpPW', 'CbGpPW'),
    ('CtD', 'DaG', 'CtDaG'),
    ('CrC', 'CbG', 'CrCbG')
]
N_SAMPLES = 10000  # Quick test
PERM_VALIDATION = list(range(1, 21))

# Reuse functions from previous script
def load_edge_matrix(edge_type, perm_id='original'):
    if perm_id == 'original':
        edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
    # Convert from bool to int to enable pathway counting
    return sp.load_npz(edge_file).astype(np.int32)

def extract_pair_features_simple(edge1, edge2, source_idx, target_idx):
    source_deg = edge1.getrow(source_idx).nnz
    target_deg = edge2.getcol(target_idx).nnz
    return np.array([source_deg, target_deg, source_deg * target_deg,
                     source_deg ** 2, target_deg ** 2])

def sample_pairs_by_pathway_count(edge1, edge2, n_samples, random_state=42):
    np.random.seed(random_state)
    pathway_matrix = edge1 @ edge2
    sources, targets = pathway_matrix.nonzero()

    zero_pairs = min(int(n_samples * 0.5), len(sources))
    if len(sources) > zero_pairs:
        sample_idx = np.random.choice(len(sources), zero_pairs, replace=False)
        sampled_sources = sources[sample_idx]
        sampled_targets = targets[sample_idx]
    else:
        sampled_sources = sources
        sampled_targets = targets

    n_zero = n_samples - len(sampled_sources)
    if n_zero > 0:
        all_sources = np.arange(edge1.shape[0])
        all_targets = np.arange(edge2.shape[1])
        zero_sources = np.random.choice(all_sources, n_zero)
        zero_targets = np.random.choice(all_targets, n_zero)
        sampled_sources = np.concatenate([sampled_sources, zero_sources])
        sampled_targets = np.concatenate([sampled_targets, zero_targets])

    return list(zip(sampled_sources, sampled_targets))

def compute_pathway_counts_for_pairs(edge1, edge2, pair_indices):
    pathway_matrix = edge1 @ edge2
    counts = np.zeros(len(pair_indices))
    for i, (source_idx, target_idx) in enumerate(pair_indices):
        counts[i] = pathway_matrix[source_idx, target_idx]
    return counts

def extract_correction_features(X, y_pred):
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

# Process each metapath
results = []

for edge1_type, edge2_type, metapath_name in METAPATHS:
    print(f"\n{'='*80}")
    print(f"METAPATH: {metapath_name}")
    print(f"{'='*80}")

    # Load edges
    print("  Loading edges...")
    edge1 = load_edge_matrix(edge1_type, perm_id='original')
    edge2 = load_edge_matrix(edge2_type, perm_id='original')
    print(f"    {edge1_type}: {edge1.shape}, {edge1.nnz} edges")
    print(f"    {edge2_type}: {edge2.shape}, {edge2.nnz} edges")

    # Sample pairs
    print(f"  Sampling {N_SAMPLES} pairs...")
    pair_indices = sample_pairs_by_pathway_count(edge1, edge2, N_SAMPLES)

    # Extract features
    print("  Extracting features...")
    X = np.array([extract_pair_features_simple(edge1, edge2, s, t)
                  for s, t in pair_indices])

    # Compute targets
    print("  Computing targets from permutations 1-20...")
    perm_counts = []
    for perm_id in range(1, 21):
        edge1_perm = load_edge_matrix(edge1_type, perm_id)
        edge2_perm = load_edge_matrix(edge2_type, perm_id)
        counts = compute_pathway_counts_for_pairs(edge1_perm, edge2_perm, pair_indices)
        perm_counts.append(counts)
    y_validation = np.mean(perm_counts, axis=0)

    # Compute permutation 000 counts
    print("  Computing permutation 000 counts...")
    edge1_perm0 = load_edge_matrix(edge1_type, perm_id=0)
    edge2_perm0 = load_edge_matrix(edge2_type, perm_id=0)
    y_perm0 = compute_pathway_counts_for_pairs(edge1_perm0, edge2_perm0, pair_indices)

    # Train baseline
    print("  Training baseline model...")
    base_model = LinearRegression()
    base_model.fit(X, y_validation)
    y_pred_base = base_model.predict(X)
    r_base = pearsonr(y_pred_base, y_validation)[0]
    rmse_base = np.sqrt(np.mean((y_pred_base - y_validation)**2))
    bias_base = np.mean(y_pred_base - y_validation)

    # Train correction
    print("  Training correction model...")
    correction_features = extract_correction_features(X, y_pred_base)
    correction_target = y_perm0 - y_pred_base
    correction_model = LinearRegression()
    correction_model.fit(correction_features, correction_target)
    correction = correction_model.predict(correction_features)
    y_pred_corrected = y_pred_base + correction

    r_corrected = pearsonr(y_pred_corrected, y_validation)[0]
    rmse_corrected = np.sqrt(np.mean((y_pred_corrected - y_validation)**2))
    bias_corrected = np.mean(y_pred_corrected - y_validation)

    # Store results
    results.append({
        'metapath': metapath_name,
        'r_base': r_base,
        'r_corrected': r_corrected,
        'rmse_base': rmse_base,
        'rmse_corrected': rmse_corrected,
        'bias_base': bias_base,
        'bias_corrected': bias_corrected,
        'improvement_r': r_corrected - r_base,
        'improvement_rmse': rmse_corrected - rmse_base,
        'improvement_pct': 100 * (r_corrected / r_base - 1)
    })

    print(f"\n  RESULTS:")
    print(f"    Baseline:  r = {r_base:.4f}, RMSE = {rmse_base:.4f}, bias = {bias_base:+.4f}")
    print(f"    Corrected: r = {r_corrected:.4f}, RMSE = {rmse_corrected:.4f}, bias = {bias_corrected:+.4f}")
    print(f"    Improvement: Δr = {r_corrected - r_base:+.4f} ({100*(r_corrected/r_base - 1):+.1f}%)")
    print(f"    Target (r > 0.90): {'✓ PASS' if r_corrected > 0.90 else 'FAIL'}")

# Summary
print("\n" + "="*80)
print("SUMMARY: ALL METAPATHS")
print("="*80)

df = pd.DataFrame(results)
print(f"\n{df.to_string(index=False)}")

# Save results
df.to_csv(results_dir / 'phase2_all_metapaths_results.csv', index=False)
print(f"\nSaved: {results_dir / 'phase2_all_metapaths_results.csv'}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Correlation comparison
ax = axes[0]
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, df['r_base'], width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, df['r_corrected'], width, label='Corrected', alpha=0.8)
ax.axhline(0.90, color='r', linestyle='--', label='Target (r > 0.90)')
ax.set_ylabel('Correlation (r)')
ax.set_title('Correlation: Baseline vs Corrected')
ax.set_xticks(x)
ax.set_xticklabels(df['metapath'])
ax.legend()
ax.grid(alpha=0.3)

# RMSE comparison
ax = axes[1]
ax.bar(x - width/2, df['rmse_base'], width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, df['rmse_corrected'], width, label='Corrected', alpha=0.8)
ax.set_ylabel('RMSE')
ax.set_title('RMSE: Baseline vs Corrected')
ax.set_xticks(x)
ax.set_xticklabels(df['metapath'])
ax.legend()
ax.grid(alpha=0.3)

# Improvement
ax = axes[2]
ax.bar(x, df['improvement_pct'], alpha=0.8)
ax.set_ylabel('Improvement (%)')
ax.set_title('Relative Improvement')
ax.set_xticks(x)
ax.set_xticklabels(df['metapath'])
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'phase2_all_metapaths_comparison.png', dpi=150)
print(f"Saved: {results_dir / 'phase2_all_metapaths_comparison.png'}")

# Success summary
success_count = (df['r_corrected'] > 0.90).sum()
print(f"\n{'='*80}")
print(f"SUCCESS RATE: {success_count}/{len(df)} metapaths achieve r > 0.90")
print(f"{'='*80}")
