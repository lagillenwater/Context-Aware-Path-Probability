"""
Complete Edge-Level Sparsity Analysis

Analyze ALL 24 edge types in Hetionet by discovering edge files directly
rather than inferring names from metagraph.

Key metrics:
- Edge count
- Density (edges / possible_edges)
- Degree distributions
- Predicted minimum permutations needed

Date: 2025-11-04
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'edge_sparsity_analysis'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPLETE EDGE-LEVEL SPARSITY ANALYSIS - ALL 24 EDGES")
print("="*80)


def get_node_counts():
    """Get node counts for each node type."""
    node_counts = {}
    nodes_dir = data_dir / 'nodes'

    for node_file in nodes_dir.glob('*.tsv'):
        node_type = node_file.stem
        with open(node_file, 'r') as f:
            n_nodes = sum(1 for _ in f) - 1
        node_counts[node_type] = n_nodes

    return node_counts


def infer_node_types(edge_abbrev):
    """
    Infer source and target node types from edge abbreviation.

    Returns (source_type, target_type) or (None, None) if cannot infer.
    """
    # Map first letters to node types
    node_type_map = {
        'A': 'Anatomy',
        'C': 'Compound',
        'D': 'Disease',
        'G': 'Gene',
        'P': 'PharmacologicClass',  # For PCiC
        'S': 'Symptom'
    }

    # Special cases with 4 letters (need to handle BP, CC, MF, PW, SE)
    if len(edge_abbrev) >= 4:
        source_letter = edge_abbrev[0]
        target_letters = edge_abbrev[2:]

        if target_letters == 'BP':
            target_type = 'BiologicalProcess'
        elif target_letters == 'CC':
            target_type = 'CellularComponent'
        elif target_letters == 'MF':
            target_type = 'MolecularFunction'
        elif target_letters == 'PW':
            target_type = 'Pathway'
        elif target_letters == 'SE':
            target_type = 'SideEffect'
        else:
            return (None, None)

        source_type = node_type_map.get(source_letter)
        return (source_type, target_type)

    # Standard 3-letter abbreviations
    if len(edge_abbrev) >= 3:
        source_letter = edge_abbrev[0]
        target_letter = edge_abbrev[2]

        source_type = node_type_map.get(source_letter)
        target_type = node_type_map.get(target_letter)

        return (source_type, target_type)

    return (None, None)


def analyze_edge_file(edge_file, node_counts):
    """
    Analyze one edge file.

    Returns dict with edge statistics.
    """
    edge_abbrev = edge_file.stem.replace('.sparse', '')

    print(f"\n  Analyzing {edge_abbrev}...")

    edge_matrix = sp.load_npz(str(edge_file))

    n_edges = edge_matrix.nnz
    n_source = edge_matrix.shape[0]
    n_target = edge_matrix.shape[1]

    # Infer node types
    source_type, target_type = infer_node_types(edge_abbrev)

    # Verify dimensions match known node counts
    if source_type and source_type in node_counts:
        expected_source = node_counts[source_type]
        if n_source != expected_source:
            print(f"    WARNING: Shape mismatch for {source_type}: "
                  f"{n_source} != {expected_source}")

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

    # Predicted minimum permutations using edge count heuristic
    if n_edges >= 1000:
        pred_min_perms = 1
    elif n_edges >= 500:
        pred_min_perms = 2
    elif n_edges >= 250:
        pred_min_perms = 3
    else:
        pred_min_perms = min(int(np.ceil(1000 / n_edges)), 10)

    print(f"    Edges: {n_edges:,}")
    print(f"    Shape: {n_source} x {n_target}")
    print(f"    Density: {density:.6f}")
    print(f"    Avg degrees: {avg_source_degree:.2f} (source), "
          f"{avg_target_degree:.2f} (target)")
    print(f"    Predicted min perms: {pred_min_perms}")

    return {
        'edge_abbrev': edge_abbrev,
        'source_type': source_type or 'Unknown',
        'target_type': target_type or 'Unknown',
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
        'pred_min_perms': pred_min_perms
    }


print("\nCounting nodes...")
node_counts = get_node_counts()
print(f"  Found {len(node_counts)} node types:")
for node_type, count in sorted(node_counts.items()):
    print(f"    {node_type}: {count:,}")

print("\nDiscovering edge files...")
edge_files = sorted((data_dir / 'edges').glob('*.sparse.npz'))
print(f"  Found {len(edge_files)} edge files")

print("\nAnalyzing all edge types...")
all_results = []

for edge_file in edge_files:
    result = analyze_edge_file(edge_file, node_counts)
    all_results.append(result)

# Save results
df_results = pd.DataFrame(all_results)
df_results = df_results.sort_values('density', ascending=False)
df_results.to_csv(results_dir / 'all_edges_sparsity_analysis.csv', index=False)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

# Print summary table
print("\n\nCOMPLETE EDGE SPARSITY SUMMARY (sorted by density):")
print("="*80)
print(f"{'Edge':<8} {'Source→Target':<30} {'Edges':>8} {'Density':>12} "
      f"{'Pred Perms':>12}")
print("-"*80)

for _, row in df_results.iterrows():
    edge_name = f"{row['source_type']}→{row['target_type']}"
    print(f"{row['edge_abbrev']:<8} {edge_name:<30} {row['n_edges']:>8,} "
          f"{row['density']:>12.6f} {row['pred_min_perms']:>12}")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")
print(f"Total edges analyzed: {len(df_results)}")
print(f"Total edge count: {df_results['n_edges'].sum():,}")
print(f"Median edge count: {df_results['n_edges'].median():,.0f}")
print(f"Min edge count: {df_results['n_edges'].min():,} "
      f"({df_results[df_results['n_edges'] == df_results['n_edges'].min()]['edge_abbrev'].values[0]})")
print(f"Max edge count: {df_results['n_edges'].max():,} "
      f"({df_results[df_results['n_edges'] == df_results['n_edges'].max()]['edge_abbrev'].values[0]})")
print(f"\nEdges needing only 1 permutation: "
      f"{len(df_results[df_results['pred_min_perms'] == 1])} / {len(df_results)} "
      f"({100*len(df_results[df_results['pred_min_perms'] == 1])/len(df_results):.1f}%)")
print(f"Edges needing >1 permutation: "
      f"{len(df_results[df_results['pred_min_perms'] > 1])} / {len(df_results)}")

# Bottleneck edges
print(f"\n{'='*80}")
print("BOTTLENECK EDGES (predicted to need >1 permutation):")
print(f"{'='*80}")

bottlenecks = df_results[df_results['pred_min_perms'] > 1]
if len(bottlenecks) > 0:
    for _, row in bottlenecks.iterrows():
        print(f"\n{row['edge_abbrev']} ({row['source_type']} → {row['target_type']}):")
        print(f"  Edges: {row['n_edges']:,}")
        print(f"  Density: {row['density']:.6f}")
        print(f"  Effective sample size: {row['eff_sample_size']:.2f}")
        print(f"  Predicted min perms: {row['pred_min_perms']}")
else:
    print("  None found")

# Create comprehensive visualizations
print("\n\nCreating visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Edge count distribution (log scale)
ax1 = fig.add_subplot(gs[0, 0])
edge_counts = df_results['n_edges'].values
ax1.hist(np.log10(edge_counts + 1), bins=20, edgecolor='black')
ax1.set_xlabel('log10(Edge Count)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Edge Counts')
ax1.grid(alpha=0.3)

# Plot 2: Density distribution
ax2 = fig.add_subplot(gs[0, 1])
densities = df_results['density'].values
ax2.hist(np.log10(densities + 1e-9), bins=20, edgecolor='black')
ax2.set_xlabel('log10(Density)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Edge Densities')
ax2.grid(alpha=0.3)

# Plot 3: Predicted permutations distribution
ax3 = fig.add_subplot(gs[0, 2])
pred_perms = df_results['pred_min_perms'].value_counts().sort_index()
ax3.bar(pred_perms.index, pred_perms.values, edgecolor='black')
ax3.set_xlabel('Predicted Minimum Permutations')
ax3.set_ylabel('Number of Edge Types')
ax3.set_title('Predicted Permutation Requirements')
ax3.grid(alpha=0.3, axis='y')

# Plot 4: Edge count vs density
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(df_results['n_edges'], df_results['density'],
            alpha=0.6, s=80, edgecolor='black')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('Edge Count')
ax4.set_ylabel('Density')
ax4.set_title('Edge Count vs Density')
ax4.grid(alpha=0.3, which='both')

# Annotate extremes
for _, row in df_results.iterrows():
    if row['n_edges'] < 1000 or row['edge_abbrev'] in ['AeG', 'GiG', 'GpBP']:
        color = 'red' if row['pred_min_perms'] > 1 else 'blue'
        ax4.annotate(row['edge_abbrev'],
                    (row['n_edges'], row['density']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color=color)

# Plot 5: Edges sorted by count
ax5 = fig.add_subplot(gs[1, 1:])
df_sorted = df_results.sort_values('n_edges', ascending=True)
colors = ['red' if p > 1 else 'green' for p in df_sorted['pred_min_perms']]
ax5.barh(range(len(df_sorted)), df_sorted['n_edges'],
         color=colors, edgecolor='black', alpha=0.7)
ax5.set_yticks(range(len(df_sorted)))
ax5.set_yticklabels(df_sorted['edge_abbrev'], fontsize=9)
ax5.set_xlabel('Edge Count')
ax5.set_title('All Edges Sorted by Edge Count (Red: >1 perm needed, Green: 1 perm sufficient)')
ax5.grid(alpha=0.3, axis='x')
ax5.axvline(1000, color='black', linestyle='--', linewidth=2, alpha=0.5,
            label='1000 edge threshold')
ax5.legend()

# Plot 6: Degree distributions
ax6 = fig.add_subplot(gs[2, :])
x_pos = np.arange(len(df_results))
width = 0.35
ax6.bar(x_pos - width/2, df_results['avg_source_degree'], width,
        label='Avg Source Degree', alpha=0.7, edgecolor='black')
ax6.bar(x_pos + width/2, df_results['avg_target_degree'], width,
        label='Avg Target Degree', alpha=0.7, edgecolor='black')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(df_results['edge_abbrev'], rotation=45, ha='right', fontsize=9)
ax6.set_ylabel('Average Degree')
ax6.set_title('Average Node Degrees by Edge Type')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')
ax6.set_yscale('log')

plt.savefig(results_dir / 'all_edges_sparsity_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved comprehensive visualization")

print("\n" + "="*80)
print("ALL 24 EDGES ANALYZED SUCCESSFULLY")
print("="*80)
print(f"\nResults saved to:")
print(f"  {results_dir / 'all_edges_sparsity_analysis.csv'}")
print(f"  {results_dir / 'all_edges_sparsity_analysis.png'}")
