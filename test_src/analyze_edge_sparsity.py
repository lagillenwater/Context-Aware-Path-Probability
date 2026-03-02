"""
Edge-Level Sparsity Analysis

Analyze sparsity characteristics for all edge types in Hetionet to predict
minimum permutation requirements for metapaths.

Key metrics:
- Edge count
- Density (edges / possible_edges)
- Degree distributions
- Predicted minimum permutations needed

Date: 2025-11-03
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import json
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'edge_sparsity_analysis'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EDGE-LEVEL SPARSITY ANALYSIS")
print("="*80)


def load_metagraph():
    """Load metagraph to get all edge types."""
    with open(data_dir / 'metagraph.json', 'r') as f:
        metagraph_data = json.load(f)
    return metagraph_data


def load_edge_matrix(edge_type):
    """Load edge matrix."""
    edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'

    if not edge_file.exists():
        return None

    return sp.load_npz(str(edge_file))


def get_node_counts():
    """Get node counts for each node type."""
    node_counts = {}
    nodes_dir = data_dir / 'nodes'

    for node_file in nodes_dir.glob('*.tsv'):
        node_type = node_file.stem
        with open(node_file, 'r') as f:
            # Count lines minus header
            n_nodes = sum(1 for _ in f) - 1
        node_counts[node_type] = n_nodes

    return node_counts


def analyze_edge_type(edge_abbrev, source_type, target_type, node_counts):
    """
    Analyze sparsity characteristics for one edge type.

    Returns dict with edge statistics.
    """
    print(f"\n  Analyzing {edge_abbrev} ({source_type} → {target_type})...")

    edge_matrix = load_edge_matrix(edge_abbrev)

    if edge_matrix is None:
        print(f"    SKIP: File not found")
        return None

    n_edges = edge_matrix.nnz
    n_source = node_counts.get(source_type, edge_matrix.shape[0])
    n_target = node_counts.get(target_type, edge_matrix.shape[1])
    n_possible = n_source * n_target
    density = n_edges / n_possible if n_possible > 0 else 0

    # Degree statistics
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    avg_source_degree = np.mean(source_degrees)
    avg_target_degree = np.mean(target_degrees)
    std_source_degree = np.std(source_degrees)
    std_target_degree = np.std(target_degrees)

    # Effective sample size
    eff_sample_size = n_edges * density

    # Predicted minimum permutations (heuristic)
    # Based on empirical findings: need ~500 effective samples
    if eff_sample_size < 500:
        pred_min_perms = int(np.ceil(500 / eff_sample_size))
    else:
        pred_min_perms = 1

    # Alternative: edge count based (need > 1000 edges for stable estimates)
    if n_edges < 1000:
        pred_min_perms_alt = int(np.ceil(1000 / n_edges))
    else:
        pred_min_perms_alt = 1

    pred_min_perms_final = max(pred_min_perms, pred_min_perms_alt, 1)

    print(f"    Edges: {n_edges:,}")
    print(f"    Density: {density:.6f}")
    print(f"    Avg degrees: {avg_source_degree:.2f} (source), {avg_target_degree:.2f} (target)")
    print(f"    Predicted min perms: {pred_min_perms_final}")

    return {
        'edge_abbrev': edge_abbrev,
        'source_type': source_type,
        'target_type': target_type,
        'n_source': n_source,
        'n_target': n_target,
        'n_edges': n_edges,
        'n_possible': n_possible,
        'density': density,
        'avg_source_degree': avg_source_degree,
        'avg_target_degree': avg_target_degree,
        'std_source_degree': std_source_degree,
        'std_target_degree': std_target_degree,
        'eff_sample_size': eff_sample_size,
        'pred_min_perms': pred_min_perms_final
    }


print("\nLoading metagraph...")
metagraph = load_metagraph()

print("\nCounting nodes...")
node_counts = get_node_counts()
print(f"  Found {len(node_counts)} node types")

print("\nAnalyzing edge types...")
all_results = []

for edge_tuple in metagraph['metaedge_tuples']:
    source_type = edge_tuple[0]
    target_type = edge_tuple[1]
    rel_type = edge_tuple[2]
    direction = edge_tuple[3]

    # Create edge abbreviation
    edge_abbrev = source_type[0] + rel_type[0] + target_type[0]

    result = analyze_edge_type(edge_abbrev, source_type, target_type, node_counts)

    if result:
        all_results.append(result)

# Save results
df_results = pd.DataFrame(all_results)
df_results = df_results.sort_values('density', ascending=False)
df_results.to_csv(results_dir / 'edge_sparsity_analysis.csv', index=False)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

# Print summary table
print("\n\nEDGE SPARSITY SUMMARY (sorted by density):")
print("="*80)
print(f"{'Edge':<8} {'Edges':>8} {'Density':>12} {'Avg Deg (S→T)':>18} {'Pred Perms':>12}")
print("-"*80)

for _, row in df_results.iterrows():
    print(f"{row['edge_abbrev']:<8} {row['n_edges']:>8,} {row['density']:>12.6f} "
          f"{row['avg_source_degree']:>7.2f}→{row['avg_target_degree']:<7.2f} {row['pred_min_perms']:>12}")

# Identify bottleneck edges
print("\n\nBOTTLENECK EDGES (predicted to need >1 permutation):")
print("="*80)

bottlenecks = df_results[df_results['pred_min_perms'] > 1]
if len(bottlenecks) > 0:
    for _, row in bottlenecks.iterrows():
        print(f"\n{row['edge_abbrev']} ({row['source_type']} → {row['target_type']}):")
        print(f"  Edges: {row['n_edges']:,}")
        print(f"  Density: {row['density']:.6f}")
        print(f"  Effective sample size: {row['eff_sample_size']:.2f}")
        print(f"  Predicted min perms: {row['pred_min_perms']}")
else:
    print("  None found (all edges predicted to need only 1 permutation)")

# Analyze tested metapaths
print("\n\nTESTED METAPATHS - BOTTLENECK ANALYSIS:")
print("="*80)

tested_metapaths = {
    'CbGpPW': ['CbG', 'GpPW'],
    'CtDaG': ['CtD', 'DaG'],
    'CrCbG': ['CrC', 'CbG'],
    'CbGaD': ['CbG', 'GaD'],
    'CpDaG': ['CpD', 'DaG']
}

observed_min_perms = {
    'CbGpPW': 1,
    'CtDaG': 1,
    'CrCbG': 1,
    'CbGaD': 1,
    'CpDaG': '>5'
}

for metapath, edges in tested_metapaths.items():
    print(f"\n{metapath}: {edges[0]} → {edges[1]}")

    edge1_data = df_results[df_results['edge_abbrev'] == edges[0]]
    edge2_data = df_results[df_results['edge_abbrev'] == edges[1]]

    if len(edge1_data) == 0:
        # Try reverse (for bidirectional edges like GaD)
        reverse_map = {'GaD': 'DaG', 'GbC': 'CbG', 'GeA': 'AeG'}
        if edges[0] in reverse_map:
            edge1_data = df_results[df_results['edge_abbrev'] == reverse_map[edges[0]]]

    if len(edge2_data) == 0:
        reverse_map = {'GaD': 'DaG', 'GbC': 'CbG', 'GeA': 'AeG'}
        if edges[1] in reverse_map:
            edge2_data = df_results[df_results['edge_abbrev'] == reverse_map[edges[1]]]

    if len(edge1_data) > 0 and len(edge2_data) > 0:
        edge1 = edge1_data.iloc[0]
        edge2 = edge2_data.iloc[0]

        bottleneck = edge1 if edge1['n_edges'] < edge2['n_edges'] else edge2

        pred_perms = max(edge1['pred_min_perms'], edge2['pred_min_perms'])

        print(f"  Edge 1 ({edge1['edge_abbrev']}): {edge1['n_edges']:,} edges, density={edge1['density']:.6f}")
        print(f"  Edge 2 ({edge2['edge_abbrev']}): {edge2['n_edges']:,} edges, density={edge2['density']:.6f}")
        print(f"  Bottleneck: {bottleneck['edge_abbrev']} ({bottleneck['n_edges']:,} edges)")
        print(f"  Predicted min perms: {pred_perms}")
        print(f"  Observed min perms: {observed_min_perms[metapath]}")

# Create visualizations
print("\n\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Edge count distribution
ax = axes[0, 0]
edge_counts = df_results['n_edges'].values
ax.hist(np.log10(edge_counts + 1), bins=20, edgecolor='black')
ax.set_xlabel('log10(Edge Count)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Edge Counts')
ax.grid(alpha=0.3)

# Annotate CpD
cpd_count = df_results[df_results['edge_abbrev'] == 'CpD']['n_edges'].values
if len(cpd_count) > 0:
    ax.axvline(np.log10(cpd_count[0] + 1), color='r', linestyle='--', label=f'CpD ({cpd_count[0]} edges)')
    ax.legend()

# Plot 2: Density distribution
ax = axes[0, 1]
densities = df_results['density'].values
ax.hist(np.log10(densities + 1e-9), bins=20, edgecolor='black')
ax.set_xlabel('log10(Density)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Edge Densities')
ax.grid(alpha=0.3)

# Plot 3: Edge count vs density
ax = axes[1, 0]
ax.scatter(df_results['n_edges'], df_results['density'], alpha=0.6, s=50)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Edge Count')
ax.set_ylabel('Density')
ax.set_title('Edge Count vs Density')
ax.grid(alpha=0.3, which='both')

# Annotate tested edges
tested_edges = set(e for edges in tested_metapaths.values() for e in edges)
for edge_abbrev in tested_edges:
    edge_data = df_results[df_results['edge_abbrev'] == edge_abbrev]
    if len(edge_data) > 0:
        edge = edge_data.iloc[0]
        color = 'red' if edge_abbrev == 'CpD' else 'blue'
        ax.annotate(edge_abbrev, (edge['n_edges'], edge['density']),
                   xytext=(5, 5), textcoords='offset points',
                   color=color, fontweight='bold' if edge_abbrev == 'CpD' else 'normal')

# Plot 4: Predicted permutations
ax = axes[1, 1]
pred_perms = df_results['pred_min_perms'].value_counts().sort_index()
ax.bar(pred_perms.index, pred_perms.values, edgecolor='black')
ax.set_xlabel('Predicted Minimum Permutations')
ax.set_ylabel('Number of Edge Types')
ax.set_title('Predicted Permutation Requirements')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_dir / 'edge_sparsity_analysis.png', dpi=150)
print("  Saved visualization")

print("\n" + "="*80)
print("DONE")
print("="*80)
