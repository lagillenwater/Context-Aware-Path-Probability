"""
Minimum Permutations Analysis: Degrees vs Jaccard across Multiple Metapaths

Test minimum permutations (1-5) needed to achieve r > 0.95 using:
1. Degree features only (5 features)
2. Jaccard features (8 features)

Across 5 metapaths:
- CbGpPW (Compound-binds-Gene-participates-Pathway)
- CtDaG (Compound-treats-Disease-associates-Gene)
- CrCbG (Compound-resembles-Compound-binds-Gene)
- CbGaD (Compound-binds-Gene-associates-Disease)
- CpDaG (Compound-palliates-Disease-associates-Gene)

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
results_dir = repo_dir / 'results' / 'minimum_perms_comparison'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MINIMUM PERMUTATIONS ANALYSIS: DEGREES vs JACCARD")
print("="*80)

# Metapaths to test
METAPATHS = [
    ('CbGpPW', 'CbG', 'GpPW'),
    ('CtDaG', 'CtD', 'DaG'),
    ('CrCbG', 'CrC', 'CbG'),
    ('CbGaD', 'CbG', 'GaD'),
    ('CpDaG', 'CpD', 'DaG')
]

N_SAMPLES = 10000
N_PERMS_LIST = [1, 2, 3, 4, 5]
VALIDATION_PERMS = range(6, 21)  # Perms 6-20 for validation


def load_edge_matrix(edge_type, perm_id='original'):
    """
    Load edge matrix, handling bidirectional edges via transpose.

    For bidirectional edges (e.g., GaD), if the file doesn't exist,
    load the reverse edge (DaG) and transpose it.
    """
    if perm_id == 'original':
        edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)

    # Try reverse edge for bidirectional relationships
    # Map of edge types to their reverses (based on metagraph.json "both" edges)
    reverse_map = {
        'GaD': 'DaG',
        'GdD': 'DdG',
        'GuD': 'DuG',
        'GbC': 'CbG',
        'GeA': 'AeG',
        'DaG': 'GaD',
        'DdG': 'GdD',
        'DuG': 'GuD',
        'CbG': 'GbC',
        'AeG': 'GeA',
    }

    if edge_type in reverse_map:
        reverse_type = reverse_map[edge_type]
        if perm_id == 'original':
            reverse_file = data_dir / 'edges' / f'{reverse_type}.sparse.npz'
        else:
            reverse_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{reverse_type}.sparse.npz'

        if reverse_file.exists():
            matrix = sp.load_npz(str(reverse_file)).astype(np.int32)
            return matrix.T.tocsr()

    return None


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


def extract_jaccard_features(source_idx, target_idx, edge1, edge2):
    """Extract Jaccard similarity and degree features (8 features)."""
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


def test_metapath(metapath_name, edge1_type, edge2_type):
    """Test one metapath with varying numbers of permutations."""
    print(f"\n{'='*80}")
    print(f"METAPATH: {metapath_name} ({edge1_type} → {edge2_type})")
    print(f"{'='*80}")

    # Load original edges for sampling
    edge1_orig = load_edge_matrix(edge1_type, perm_id='original')
    edge2_orig = load_edge_matrix(edge2_type, perm_id='original')

    if edge1_orig is None or edge2_orig is None:
        print(f"  SKIP: Edge files not found")
        return None

    # Sample pairs
    print(f"  Sampling {N_SAMPLES} pairs...")
    pairs = sample_pairs_stratified(edge1_orig, edge2_orig, n_samples=N_SAMPLES)
    print(f"    Sampled {len(pairs)} pairs")

    # Compute validation target (perms 6-20)
    print(f"  Computing validation target (perms 6-20)...")
    val_perm_counts = []
    for perm_id in VALIDATION_PERMS:
        edge1_perm = load_edge_matrix(edge1_type, perm_id)
        edge2_perm = load_edge_matrix(edge2_type, perm_id)

        if edge1_perm is None or edge2_perm is None:
            print(f"    SKIP: Perm {perm_id} not found")
            return None

        counts = []
        for source_idx, target_idx in pairs:
            count = compute_pathway_count(source_idx, target_idx, edge1_perm, edge2_perm)
            counts.append(count)

        val_perm_counts.append(counts)

    y_val_target = np.mean(val_perm_counts, axis=0)
    print(f"    Validation mean: {y_val_target.mean():.4f} pathways/pair")

    results = []

    # Test different numbers of training permutations
    for n_train_perms in N_PERMS_LIST:
        print(f"\n  Testing {n_train_perms} training permutation(s)...")

        # Compute training target (perms 0 to n_train_perms-1)
        train_perm_counts = []
        for perm_id in range(0, n_train_perms):
            edge1_perm = load_edge_matrix(edge1_type, perm_id)
            edge2_perm = load_edge_matrix(edge2_type, perm_id)

            counts = []
            for source_idx, target_idx in pairs:
                count = compute_pathway_count(source_idx, target_idx, edge1_perm, edge2_perm)
                counts.append(count)

            train_perm_counts.append(counts)

        y_train_target = np.mean(train_perm_counts, axis=0)
        r_targets = pearsonr(y_train_target, y_val_target)[0]
        print(f"    Target correlation: r = {r_targets:.4f}")

        # Test with degree features
        print(f"    Extracting degree features from perm 0...")
        edge1_perm0 = load_edge_matrix(edge1_type, perm_id=0)
        edge2_perm0 = load_edge_matrix(edge2_type, perm_id=0)

        X_degree = []
        for source_idx, target_idx in pairs:
            features = extract_degree_features(source_idx, target_idx, edge1_perm0, edge2_perm0)
            X_degree.append(features)
        X_degree = np.array(X_degree)

        # Train/test split for degree model
        X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
            X_degree, y_train_target, y_val_target, test_size=0.2, random_state=42
        )

        model_degree = LinearRegression()
        model_degree.fit(X_train, y_train)
        y_pred_test = model_degree.predict(X_test)

        r_val_degree = pearsonr(y_pred_test, y_val_test)[0]
        print(f"    Degree features: r = {r_val_degree:.4f}")

        # Test with Jaccard features
        print(f"    Extracting Jaccard features from perm 0...")
        X_jaccard = []
        for source_idx, target_idx in pairs:
            features = extract_jaccard_features(source_idx, target_idx, edge1_perm0, edge2_perm0)
            X_jaccard.append(features)
        X_jaccard = np.array(X_jaccard)

        # Train/test split for Jaccard model
        X_train_j, X_test_j, y_train_j, y_test_j, y_val_train_j, y_val_test_j = train_test_split(
            X_jaccard, y_train_target, y_val_target, test_size=0.2, random_state=42
        )

        model_jaccard = LinearRegression()
        model_jaccard.fit(X_train_j, y_train_j)
        y_pred_test_j = model_jaccard.predict(X_test_j)

        r_val_jaccard = pearsonr(y_pred_test_j, y_val_test_j)[0]
        print(f"    Jaccard features: r = {r_val_jaccard:.4f}")

        results.append({
            'metapath': metapath_name,
            'n_train_perms': n_train_perms,
            'r_targets': r_targets,
            'r_val_degree': r_val_degree,
            'r_val_jaccard': r_val_jaccard,
            'train_mean': y_train_target.mean(),
            'val_mean': y_val_target.mean()
        })

    return results


# Run analysis for all metapaths
all_results = []

for metapath_name, edge1_type, edge2_type in METAPATHS:
    results = test_metapath(metapath_name, edge1_type, edge2_type)
    if results:
        all_results.extend(results)

# Save results
df_results = pd.DataFrame(all_results)
df_results.to_csv(results_dir / 'minimum_perms_comparison.csv', index=False)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

# Summary by metapath
print("\n\nSUMMARY BY METAPATH:")
print("="*80)

for metapath_name in df_results['metapath'].unique():
    df_mp = df_results[df_results['metapath'] == metapath_name]
    print(f"\n{metapath_name}:")
    print("  N_perms | r_targets | r_degree | r_jaccard | degree>0.95? | jaccard>0.95?")
    print("  " + "-"*75)

    for _, row in df_mp.iterrows():
        degree_pass = "YES" if row['r_val_degree'] > 0.95 else "NO"
        jaccard_pass = "YES" if row['r_val_jaccard'] > 0.95 else "NO"
        print(f"     {row['n_train_perms']:2d}   |   {row['r_targets']:.4f}  | {row['r_val_degree']:.4f}  |  {row['r_val_jaccard']:.4f}  |     {degree_pass}      |      {jaccard_pass}")

# Find minimum permutations needed for each approach
print("\n\nMINIMUM PERMUTATIONS TO ACHIEVE r > 0.95:")
print("="*80)

for metapath_name in df_results['metapath'].unique():
    df_mp = df_results[df_results['metapath'] == metapath_name]

    # Find minimum for degree
    degree_passes = df_mp[df_mp['r_val_degree'] > 0.95]
    if len(degree_passes) > 0:
        min_degree = degree_passes['n_train_perms'].min()
        r_degree = degree_passes[degree_passes['n_train_perms'] == min_degree]['r_val_degree'].values[0]
    else:
        min_degree = None
        r_degree = df_mp['r_val_degree'].max()

    # Find minimum for Jaccard
    jaccard_passes = df_mp[df_mp['r_val_jaccard'] > 0.95]
    if len(jaccard_passes) > 0:
        min_jaccard = jaccard_passes['n_train_perms'].min()
        r_jaccard = jaccard_passes[jaccard_passes['n_train_perms'] == min_jaccard]['r_val_jaccard'].values[0]
    else:
        min_jaccard = None
        r_jaccard = df_mp['r_val_jaccard'].max()

    print(f"\n{metapath_name}:")
    if min_degree:
        print(f"  Degree features:  {min_degree} perms (r = {r_degree:.4f})")
    else:
        print(f"  Degree features:  >5 perms needed (best r = {r_degree:.4f})")

    if min_jaccard:
        print(f"  Jaccard features: {min_jaccard} perms (r = {r_jaccard:.4f})")
    else:
        print(f"  Jaccard features: >5 perms needed (best r = {r_jaccard:.4f})")

# Create visualization
print("\n\nCreating visualization...")

successful_metapaths = df_results['metapath'].unique()
n_successful = len(successful_metapaths)
fig, axes = plt.subplots(n_successful, 1, figsize=(12, 4*n_successful))
if n_successful == 1:
    axes = [axes]

for idx, metapath_name in enumerate(successful_metapaths):
    ax = axes[idx]
    df_mp = df_results[df_results['metapath'] == metapath_name]

    ax.plot(df_mp['n_train_perms'], df_mp['r_targets'], 'o-', label='Target correlation', linewidth=2)
    ax.plot(df_mp['n_train_perms'], df_mp['r_val_degree'], 's-', label='Degree features', linewidth=2)
    ax.plot(df_mp['n_train_perms'], df_mp['r_val_jaccard'], '^-', label='Jaccard features', linewidth=2)
    ax.axhline(y=0.95, color='r', linestyle='--', label='r=0.95 threshold')

    ax.set_xlabel('Number of Training Permutations')
    ax.set_ylabel('Validation Correlation (r)')
    ax.set_title(f'{metapath_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(N_PERMS_LIST)

plt.tight_layout()
plt.savefig(results_dir / 'minimum_perms_comparison.png', dpi=150)
print("  Saved visualization")

print("\n" + "="*80)
print("DONE")
print("="*80)
