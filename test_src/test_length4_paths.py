"""
Test Minimum Permutations on Length-4 Paths

Test 5 metapaths with 3 edges each to validate:
1. Degree features still work for longer paths
2. Heuristic predictions based on bottleneck edge

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
results_dir = repo_dir / 'results' / 'length4_paths'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("LENGTH-4 PATH ANALYSIS: 5 METAPATHS")
print("="*80)

# Select 5 diverse length-4 metapaths
LENGTH4_METAPATHS = [
    ('CbGiGpPW', ['CbG', 'GiG', 'GpPW']),     # Compound-Gene-Gene-Pathway
    ('CbGaD', ['CbG', 'GaD']),                 # Keep one length-2 for comparison
    ('CtDaGiG', ['CtD', 'DaG', 'GiG']),       # Compound-Disease-Gene-Gene
    ('CrCbGaD', ['CrC', 'CbG', 'GaD']),       # Compound-Compound-Gene-Disease
    ('CbGdGaD', ['CbG', 'GdG', 'GaD']),       # Compound-Gene-Gene-Disease (downregulates)
]

N_SAMPLES = 10000
N_PERMS_LIST = [1, 2, 3, 4, 5]
VALIDATION_PERMS = range(6, 21)


def load_edge_matrix(edge_type, perm_id='original'):
    """Load edge matrix, handling bidirectional edges via transpose."""
    if perm_id == 'original':
        edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)

    # Try reverse edge for bidirectional relationships
    reverse_map = {
        'GaD': 'DaG', 'GdD': 'DdG', 'GuD': 'DuG',
        'GbC': 'CbG', 'GeA': 'AeG',
        'DaG': 'GaD', 'DdG': 'GdD', 'DuG': 'GuD',
        'CbG': 'GbC', 'AeG': 'GeA',
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


def compute_pathway_counts_batch(source_indices, target_indices, edges_list):
    """
    Compute pathway counts for multiple pairs using sparse matrix operations.

    Much faster than iterating pairs individually.
    """
    if len(edges_list) == 2:
        # Length-3 path: source -e1-> intermediate -e2-> target
        pathway_matrix = edges_list[0] @ edges_list[1]
    else:
        # Length-4 path: source -e1-> m1 -e2-> m2 -e3-> target
        pathway_matrix = edges_list[0] @ edges_list[1] @ edges_list[2]

    # Extract counts for specified pairs
    counts = []
    for src, tgt in zip(source_indices, target_indices):
        counts.append(pathway_matrix[src, tgt])

    return np.array(counts)


def compute_pathway_count_length2(source_idx, target_idx, edge1, edge2):
    """Compute pathway count for length-3 path (2 edges)."""
    neighbors_source = set(edge1.getrow(source_idx).nonzero()[1])
    neighbors_target = set(edge2.getcol(target_idx).nonzero()[0])
    return len(neighbors_source & neighbors_target)


def sample_pairs_stratified(edges_list, n_samples=10000, random_state=42):
    """Sample pairs stratified by pathway count."""
    np.random.seed(random_state)

    # For length-4, compute pathway matrix
    if len(edges_list) == 3:
        # source -e1-> mid1 -e2-> mid2 -e3-> target
        pathway_matrix = edges_list[0] @ edges_list[1] @ edges_list[2]
    else:  # length-2
        pathway_matrix = edges_list[0] @ edges_list[1]

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
    random_sources = np.random.choice(edges_list[0].shape[0], n_random)
    random_targets = np.random.choice(edges_list[-1].shape[1], n_random)

    sampled_sources.extend(random_sources)
    sampled_targets.extend(random_targets)

    pairs = list(zip(sampled_sources, sampled_targets))
    return pairs


def extract_degree_features(source_idx, target_idx, edges_list):
    """Extract degree features for endpoints."""
    d_u = edges_list[0].getrow(source_idx).nnz
    d_v = edges_list[-1].getcol(target_idx).nnz

    return np.array([
        d_u, d_v, d_u * d_v, d_u ** 2, d_v ** 2
    ], dtype=np.float64)


def test_metapath(metapath_name, edge_types):
    """Test one metapath."""
    print(f"\n{'='*80}")
    print(f"METAPATH: {metapath_name} ({' → '.join(edge_types)})")
    print(f"  Length: {len(edge_types) + 1} (edges: {len(edge_types)})")
    print(f"{'='*80}")

    # Load original edges
    edges_orig = []
    for edge_type in edge_types:
        edge = load_edge_matrix(edge_type, perm_id='original')
        if edge is None:
            print(f"  SKIP: {edge_type} not found")
            return None
        edges_orig.append(edge)

    # Get edge counts for heuristic prediction
    edge_counts = [e.nnz for e in edges_orig]
    bottleneck_count = min(edge_counts)
    bottleneck_idx = edge_counts.index(bottleneck_count)

    # Predict permutations needed
    if bottleneck_count >= 1000:
        pred_perms = 1
    elif bottleneck_count >= 500:
        pred_perms = 2
    elif bottleneck_count >= 250:
        pred_perms = 3
    else:
        pred_perms = min(int(np.ceil(1000 / bottleneck_count)), 10)

    print(f"  Edge counts: {[f'{et}({c:,})' for et, c in zip(edge_types, edge_counts)]}")
    print(f"  Bottleneck: {edge_types[bottleneck_idx]} ({bottleneck_count:,} edges)")
    print(f"  Predicted min perms: {pred_perms}")

    # Sample pairs
    print(f"\n  Sampling {N_SAMPLES} pairs...")
    pairs = sample_pairs_stratified(edges_orig, n_samples=N_SAMPLES)
    print(f"    Sampled {len(pairs)} pairs")

    # Compute validation target (perms 6-20)
    print(f"  Computing validation target (perms 6-20)...")
    val_perm_counts = []
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]

    for perm_id in VALIDATION_PERMS:
        edges_perm = [load_edge_matrix(et, perm_id) for et in edge_types]
        if None in edges_perm:
            print(f"    SKIP: Perm {perm_id} edges not found")
            return None

        counts = compute_pathway_counts_batch(sources, targets, edges_perm)
        val_perm_counts.append(counts)

    y_val_target = np.mean(val_perm_counts, axis=0)
    print(f"    Validation mean: {y_val_target.mean():.4f} pathways/pair")

    results = []

    # Test different numbers of training permutations
    for n_train_perms in N_PERMS_LIST:
        print(f"\n  Testing {n_train_perms} training permutation(s)...")

        # Compute training target
        train_perm_counts = []
        for perm_id in range(0, n_train_perms):
            edges_perm = [load_edge_matrix(et, perm_id) for et in edge_types]
            counts = compute_pathway_counts_batch(sources, targets, edges_perm)
            train_perm_counts.append(counts)

        y_train_target = np.mean(train_perm_counts, axis=0)
        r_targets = pearsonr(y_train_target, y_val_target)[0]
        print(f"    Target correlation: r = {r_targets:.4f}")

        # Extract degree features from perm 0
        edges_perm0 = [load_edge_matrix(et, perm_id=0) for et in edge_types]

        X_degree = []
        for source_idx, target_idx in pairs:
            features = extract_degree_features(source_idx, target_idx, edges_perm0)
            X_degree.append(features)
        X_degree = np.array(X_degree)

        # Train/test split
        X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
            X_degree, y_train_target, y_val_target, test_size=0.2, random_state=42
        )

        model_degree = LinearRegression()
        model_degree.fit(X_train, y_train)
        y_pred_test = model_degree.predict(X_test)

        r_val_degree = pearsonr(y_pred_test, y_val_test)[0]
        print(f"    Degree features: r = {r_val_degree:.4f}")

        results.append({
            'metapath': metapath_name,
            'length': len(edge_types) + 1,
            'n_train_perms': n_train_perms,
            'r_targets': r_targets,
            'r_val_degree': r_val_degree,
            'train_mean': y_train_target.mean(),
            'val_mean': y_val_target.mean(),
            'bottleneck_count': bottleneck_count,
            'pred_perms': pred_perms
        })

    return results


# Run analysis
all_results = []

for metapath_name, edge_types in LENGTH4_METAPATHS:
    results = test_metapath(metapath_name, edge_types)
    if results:
        all_results.extend(results)

# Save results
df_results = pd.DataFrame(all_results)
df_results.to_csv(results_dir / 'length4_paths_results.csv', index=False)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

# Summary
print("\n\nSUMMARY - MINIMUM PERMUTATIONS FOR r > 0.95:")
print("="*80)

for metapath_name in df_results['metapath'].unique():
    df_mp = df_results[df_results['metapath'] == metapath_name]

    # Find minimum for degree
    degree_passes = df_mp[df_mp['r_val_degree'] > 0.95]
    if len(degree_passes) > 0:
        min_degree = degree_passes['n_train_perms'].min()
        r_degree = degree_passes[degree_passes['n_train_perms'] == min_degree]['r_val_degree'].values[0]
        pred_perms = df_mp['pred_perms'].values[0]
        bottleneck = df_mp['bottleneck_count'].values[0]
        length = df_mp['length'].values[0]

        print(f"\n{metapath_name} (length {length}):")
        print(f"  Bottleneck: {bottleneck:,} edges")
        print(f"  Predicted: {pred_perms} perms")
        print(f"  Observed: {min_degree} perms (r = {r_degree:.4f})")
        print(f"  Prediction error: {abs(pred_perms - min_degree)}")
    else:
        print(f"\n{metapath_name}: >5 perms needed (best r = {df_mp['r_val_degree'].max():.4f})")

print("\n" + "="*80)
print("DONE")
print("="*80)
