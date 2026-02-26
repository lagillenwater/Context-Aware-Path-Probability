"""
Heuristic for Predicting Minimum Permutations from Edge Characteristics

Based on empirical findings from minimum permutations analysis:
- Edge count matters more than density
- Bottleneck edge dominates requirements
- Need >1,000 edges in bottleneck for 1-perm sufficiency

Date: 2025-11-03
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'permutation_heuristic'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PERMUTATION PREDICTION HEURISTIC")
print("="*80)


def load_edge_matrix(edge_type):
    """Load edge matrix."""
    edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    if not edge_file.exists():
        return None
    return sp.load_npz(str(edge_file))


def load_metagraph():
    """Load metagraph to get all edge types."""
    with open(data_dir / 'metagraph.json', 'r') as f:
        return json.load(f)


def get_edge_stats(edge_abbrev):
    """Get edge statistics."""
    edge_matrix = load_edge_matrix(edge_abbrev)
    if edge_matrix is None:
        return None

    n_edges = edge_matrix.nnz
    n_possible = edge_matrix.shape[0] * edge_matrix.shape[1]
    density = n_edges / n_possible if n_possible > 0 else 0

    return {
        'edge_abbrev': edge_abbrev,
        'n_edges': n_edges,
        'density': density
    }


def predict_min_permutations(edge_counts, method='edge_count'):
    """
    Predict minimum permutations needed for a metapath.

    Args:
        edge_counts: List of edge counts for each edge in the path
        method: 'edge_count', 'density', or 'hybrid'

    Returns:
        Predicted minimum permutations (integer)
    """
    if len(edge_counts) == 0:
        return None

    bottleneck_edges = min(edge_counts)

    if method == 'edge_count':
        # Simple heuristic based on empirical findings
        if bottleneck_edges >= 1000:
            return 1
        elif bottleneck_edges >= 500:
            return 2
        elif bottleneck_edges >= 250:
            return 3
        else:
            # Scale inversely with edge count
            return min(int(np.ceil(1000 / bottleneck_edges)), 10)

    elif method == 'sqrt':
        # Square root scaling (less aggressive)
        if bottleneck_edges >= 1000:
            return 1
        else:
            return min(int(np.ceil(np.sqrt(1000 / bottleneck_edges))), 10)

    elif method == 'log':
        # Logarithmic scaling (very conservative)
        if bottleneck_edges >= 1000:
            return 1
        else:
            ratio = 1000 / bottleneck_edges
            return min(int(np.ceil(1 + np.log2(ratio))), 10)

    return None


# Load validated metapath results
validated_results = {
    'CbGpPW': {'edges': ['CbG', 'GpPW'], 'observed_min': 1},
    'CtDaG': {'edges': ['CtD', 'DaG'], 'observed_min': 1},
    'CrCbG': {'edges': ['CrC', 'CbG'], 'observed_min': 1},
    'CbGaD': {'edges': ['CbG', 'DaG'], 'observed_min': 1},
    'CpDaG': {'edges': ['CpD', 'DaG'], 'observed_min': 6}  # >5 means at least 6
}

print("\nTesting heuristic on validated length-2 paths...")
print("-"*80)

results = []
for metapath, data in validated_results.items():
    edge_types = data['edges']
    observed_min = data['observed_min']

    # Get edge counts
    edge_counts = []
    edge_stats = []
    for edge_type in edge_types:
        stats = get_edge_stats(edge_type)
        if stats:
            edge_counts.append(stats['n_edges'])
            edge_stats.append(stats)

    if len(edge_counts) != len(edge_types):
        print(f"\n{metapath}: SKIP (edge files not found)")
        continue

    # Predict with different methods
    pred_edge_count = predict_min_permutations(edge_counts, method='edge_count')
    pred_sqrt = predict_min_permutations(edge_counts, method='sqrt')
    pred_log = predict_min_permutations(edge_counts, method='log')

    bottleneck = edge_stats[0] if edge_counts[0] < edge_counts[1] else edge_stats[1]

    print(f"\n{metapath}:")
    print(f"  Edges: {edge_types[0]} ({edge_counts[0]:,}) → {edge_types[1]} ({edge_counts[1]:,})")
    print(f"  Bottleneck: {bottleneck['edge_abbrev']} ({bottleneck['n_edges']:,} edges)")
    print(f"  Observed min perms: {observed_min}")
    print(f"  Predicted (edge_count): {pred_edge_count}")
    print(f"  Predicted (sqrt): {pred_sqrt}")
    print(f"  Predicted (log): {pred_log}")

    results.append({
        'metapath': metapath,
        'length': 2,
        'bottleneck_edge': bottleneck['edge_abbrev'],
        'bottleneck_count': bottleneck['n_edges'],
        'observed_min_perms': observed_min,
        'pred_edge_count': pred_edge_count,
        'pred_sqrt': pred_sqrt,
        'pred_log': pred_log,
        'error_edge_count': abs(pred_edge_count - observed_min),
        'error_sqrt': abs(pred_sqrt - observed_min),
        'error_log': abs(pred_log - observed_min)
    })

# Evaluate methods
df_results = pd.DataFrame(results)
print("\n" + "="*80)
print("HEURISTIC EVALUATION ON LENGTH-2 PATHS")
print("="*80)

for method in ['edge_count', 'sqrt', 'log']:
    pred_col = f'pred_{method}'
    error_col = f'error_{method}'

    mae = df_results[error_col].mean()
    correct = (df_results[error_col] == 0).sum()
    within_1 = (df_results[error_col] <= 1).sum()

    print(f"\n{method.upper()} method:")
    print(f"  Mean Absolute Error: {mae:.2f}")
    print(f"  Exact matches: {correct}/{len(df_results)}")
    print(f"  Within ±1: {within_1}/{len(df_results)}")

# Save results
df_results.to_csv(results_dir / 'heuristic_validation_length2.csv', index=False)

# Recommend best method
best_method = None
best_mae = float('inf')
for method in ['edge_count', 'sqrt', 'log']:
    mae = df_results[f'error_{method}'].mean()
    if mae < best_mae:
        best_mae = mae
        best_method = method

print(f"\n{'='*80}")
print(f"BEST METHOD: {best_method.upper()} (MAE = {best_mae:.2f})")
print(f"{'='*80}")

# Define recommended heuristic
if best_method == 'edge_count':
    print("\nRecommended Heuristic (edge_count method):")
    print("  if bottleneck_edges >= 1000: return 1")
    print("  elif bottleneck_edges >= 500: return 2")
    print("  elif bottleneck_edges >= 250: return 3")
    print("  else: return min(ceil(1000 / bottleneck_edges), 10)")
elif best_method == 'sqrt':
    print("\nRecommended Heuristic (sqrt method):")
    print("  if bottleneck_edges >= 1000: return 1")
    print("  else: return min(ceil(sqrt(1000 / bottleneck_edges)), 10)")
elif best_method == 'log':
    print("\nRecommended Heuristic (log method):")
    print("  if bottleneck_edges >= 1000: return 1")
    print("  else: return min(ceil(1 + log2(1000 / bottleneck_edges)), 10)")

print("\n" + "="*80)
print("DONE - Ready to test on length-4 paths")
print("="*80)
