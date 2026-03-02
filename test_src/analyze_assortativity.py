#!/usr/bin/env python3
"""
Assortativity Analysis for Hetionet

Compute edge-specific and pathway-specific assortativity coefficients.
Compare original graph to permutation 000.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import networkx as nx

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))


def compute_edge_assortativity(edge_matrix):
    """
    Compute degree assortativity coefficient for a bipartite edge type.

    For bipartite graphs, compute correlation between source node degrees
    and target node degrees across all edges.

    Parameters
    ----------
    edge_matrix : sp.csr_matrix
        Adjacency matrix for edge type (source_nodes × target_nodes)

    Returns
    -------
    float
        Assortativity coefficient (Pearson correlation of node degrees)
    """
    # Compute degrees
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    # Get edge list
    edge_coo = edge_matrix.tocoo()

    # For each edge, record (source_degree, target_degree)
    source_deg_list = []
    target_deg_list = []

    for source_idx, target_idx in zip(edge_coo.row, edge_coo.col):
        source_deg_list.append(source_degrees[source_idx])
        target_deg_list.append(target_degrees[target_idx])

    # Compute Pearson correlation
    r = np.corrcoef(source_deg_list, target_deg_list)[0, 1]

    return r


def compute_pathway_assortativity(edge1, edge2):
    """
    Compute pathway-specific assortativity for a 2-step metapath.

    For paths source → intermediate → target, compute:
    1. Correlation between deg(source) and deg(intermediate)
    2. Correlation between deg(intermediate) and deg(target)
    3. Weighted correlation between deg(source) and deg(target)

    Parameters
    ----------
    edge1 : sp.csr_matrix
        First edge matrix (source → intermediate)
    edge2 : sp.csr_matrix
        Second edge matrix (intermediate → target)

    Returns
    -------
    dict
        Assortativity metrics
    """
    # Compute pathway matrix
    pathway_matrix = edge1 @ edge2

    # Compute degrees
    source_out_degrees = np.array(edge1.sum(axis=1)).flatten()
    intermediate_in_degrees = np.array(edge1.sum(axis=0)).flatten()
    intermediate_out_degrees = np.array(edge2.sum(axis=1)).flatten()
    target_in_degrees = np.array(edge2.sum(axis=0)).flatten()

    # Enumerate all paths: source → intermediate → target
    # For each edge in edge1: source → intermediate
    # For each edge in edge2: intermediate → target
    # If both exist, there's a path

    source_degrees_list = []
    intermediate_degrees_list_1 = []  # For source-intermediate correlation
    intermediate_degrees_list_2 = []  # For intermediate-target correlation
    target_degrees_list = []
    pathway_counts_list = []

    # Get all paths
    edge1_coo = edge1.tocoo()
    edge2_coo = edge2.tocoo()

    # Build intermediate node → (sources, targets) mappings
    intermediate_to_sources = {}
    for i, j in zip(edge1_coo.row, edge1_coo.col):
        if j not in intermediate_to_sources:
            intermediate_to_sources[j] = []
        intermediate_to_sources[j].append(i)

    intermediate_to_targets = {}
    for i, j in zip(edge2_coo.row, edge2_coo.col):
        if i not in intermediate_to_targets:
            intermediate_to_targets[i] = []
        intermediate_to_targets[i].append(j)

    # For each intermediate node, enumerate paths through it
    for intermediate_idx in range(edge1.shape[1]):
        if intermediate_idx not in intermediate_to_sources:
            continue
        if intermediate_idx not in intermediate_to_targets:
            continue

        sources = intermediate_to_sources[intermediate_idx]
        targets = intermediate_to_targets[intermediate_idx]

        for source in sources:
            for target in targets:
                # Path: source → intermediate → target
                source_degrees_list.append(source_out_degrees[source])
                intermediate_degrees_list_1.append(intermediate_out_degrees[intermediate_idx])
                intermediate_degrees_list_2.append(intermediate_in_degrees[intermediate_idx])
                target_degrees_list.append(target_in_degrees[target])
                pathway_counts_list.append(pathway_matrix[source, target])

    # Convert to arrays
    source_degrees = np.array(source_degrees_list)
    intermediate_degrees_1 = np.array(intermediate_degrees_list_1)
    intermediate_degrees_2 = np.array(intermediate_degrees_list_2)
    target_degrees = np.array(target_degrees_list)
    pathway_counts = np.array(pathway_counts_list)

    # Compute correlations
    r_source_intermediate = np.corrcoef(source_degrees, intermediate_degrees_1)[0, 1]
    r_intermediate_target = np.corrcoef(intermediate_degrees_2, target_degrees)[0, 1]

    # Weighted correlation: for each (source, target) pair, weight by pathway count
    # Aggregate by (source, target) pair
    pair_source_degrees = {}
    pair_target_degrees = {}
    pair_pathway_counts = {}

    for i in range(len(source_degrees_list)):
        source = source_degrees_list[i]
        target = target_degrees_list[i]
        pathway_count = pathway_counts_list[i]

        # Use pathway count as identifier (approximation)
        # Better: use actual (source_idx, target_idx) pair
        # Let's iterate differently

    # Alternative: just use pathway_matrix directly
    pathway_coo = pathway_matrix.tocoo()
    source_degrees_weighted = []
    target_degrees_weighted = []
    weights = []

    for source_idx, target_idx, count in zip(pathway_coo.row, pathway_coo.col, pathway_coo.data):
        if count > 0:
            source_degrees_weighted.append(source_out_degrees[source_idx])
            target_degrees_weighted.append(target_in_degrees[target_idx])
            weights.append(count)

    source_degrees_weighted = np.array(source_degrees_weighted)
    target_degrees_weighted = np.array(target_degrees_weighted)
    weights = np.array(weights)

    # Weighted correlation
    if len(weights) > 0:
        # Weighted Pearson correlation
        mean_source = np.average(source_degrees_weighted, weights=weights)
        mean_target = np.average(target_degrees_weighted, weights=weights)

        cov = np.average(
            (source_degrees_weighted - mean_source) * (target_degrees_weighted - mean_target),
            weights=weights
        )
        var_source = np.average((source_degrees_weighted - mean_source) ** 2, weights=weights)
        var_target = np.average((target_degrees_weighted - mean_target) ** 2, weights=weights)

        r_source_target_weighted = cov / np.sqrt(var_source * var_target)
    else:
        r_source_target_weighted = np.nan

    return {
        'r_source_intermediate': r_source_intermediate,
        'r_intermediate_target': r_intermediate_target,
        'r_source_target_weighted': r_source_target_weighted,
        'n_paths': len(source_degrees_list),
        'n_pairs_with_paths': len(weights)
    }


def main():
    data_dir = repo_dir / 'data'
    edges_dir = data_dir / 'edges'
    perm_dir = data_dir / 'permutations' / '000.hetmat' / 'edges'

    print("=" * 80)
    print("ASSORTATIVITY ANALYSIS")
    print("=" * 80)
    print()

    # Load edge matrices
    print("Loading edge matrices...")
    CbG_original = sp.load_npz(edges_dir / 'CbG.sparse.npz')
    GpPW_original = sp.load_npz(edges_dir / 'GpPW.sparse.npz')
    CbG_perm = sp.load_npz(perm_dir / 'CbG.sparse.npz')
    GpPW_perm = sp.load_npz(perm_dir / 'GpPW.sparse.npz')
    print(f"  CbG: {CbG_original.shape}, {CbG_original.nnz} edges")
    print(f"  GpPW: {GpPW_original.shape}, {GpPW_original.nnz} edges")
    print()

    # 1. Edge-specific assortativity
    print("=" * 80)
    print("EDGE-SPECIFIC ASSORTATIVITY")
    print("=" * 80)
    print()

    print("CbG (Compound-binds-Gene)")
    print("-" * 80)
    r_CbG_original = compute_edge_assortativity(CbG_original)
    print(f"  Original Hetionet:  r = {r_CbG_original:.6f}")

    r_CbG_perm = compute_edge_assortativity(CbG_perm)
    print(f"  Permutation 000:    r = {r_CbG_perm:.6f}")

    diff_CbG = r_CbG_perm - r_CbG_original
    print(f"  Difference:         Δr = {diff_CbG:+.6f}")

    if abs(diff_CbG) < 0.01:
        print("  → Assortativity PRESERVED by permutation")
    else:
        print("  → Assortativity CHANGED by permutation")
    print()

    print("GpPW (Gene-participates-Pathway)")
    print("-" * 80)
    r_GpPW_original = compute_edge_assortativity(GpPW_original)
    print(f"  Original Hetionet:  r = {r_GpPW_original:.6f}")

    r_GpPW_perm = compute_edge_assortativity(GpPW_perm)
    print(f"  Permutation 000:    r = {r_GpPW_perm:.6f}")

    diff_GpPW = r_GpPW_perm - r_GpPW_original
    print(f"  Difference:         Δr = {diff_GpPW:+.6f}")

    if abs(diff_GpPW) < 0.01:
        print("  → Assortativity PRESERVED by permutation")
    else:
        print("  → Assortativity CHANGED by permutation")
    print()

    # 2. Pathway-specific assortativity
    print("=" * 80)
    print("PATHWAY-SPECIFIC ASSORTATIVITY")
    print("=" * 80)
    print()

    print("CbGpPW (Compound → Gene → Pathway)")
    print("-" * 80)
    print()

    print("Original Hetionet:")
    pathway_assort_original = compute_pathway_assortativity(CbG_original, GpPW_original)
    print(f"  Source-Intermediate correlation:      r = {pathway_assort_original['r_source_intermediate']:.6f}")
    print(f"  Intermediate-Target correlation:      r = {pathway_assort_original['r_intermediate_target']:.6f}")
    print(f"  Source-Target correlation (weighted): r = {pathway_assort_original['r_source_target_weighted']:.6f}")
    print(f"  Number of paths analyzed:             {pathway_assort_original['n_paths']:,}")
    print(f"  Number of (source, target) pairs:     {pathway_assort_original['n_pairs_with_paths']:,}")
    print()

    print("Permutation 000:")
    pathway_assort_perm = compute_pathway_assortativity(CbG_perm, GpPW_perm)
    print(f"  Source-Intermediate correlation:      r = {pathway_assort_perm['r_source_intermediate']:.6f}")
    print(f"  Intermediate-Target correlation:      r = {pathway_assort_perm['r_intermediate_target']:.6f}")
    print(f"  Source-Target correlation (weighted): r = {pathway_assort_perm['r_source_target_weighted']:.6f}")
    print(f"  Number of paths analyzed:             {pathway_assort_perm['n_paths']:,}")
    print(f"  Number of (source, target) pairs:     {pathway_assort_perm['n_pairs_with_paths']:,}")
    print()

    print("Differences:")
    diff_SI = pathway_assort_perm['r_source_intermediate'] - pathway_assort_original['r_source_intermediate']
    diff_IT = pathway_assort_perm['r_intermediate_target'] - pathway_assort_original['r_intermediate_target']
    diff_ST = pathway_assort_perm['r_source_target_weighted'] - pathway_assort_original['r_source_target_weighted']

    print(f"  Source-Intermediate:      Δr = {diff_SI:+.6f}")
    print(f"  Intermediate-Target:      Δr = {diff_IT:+.6f}")
    print(f"  Source-Target (weighted): Δr = {diff_ST:+.6f}")
    print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print("Edge-Specific Assortativity:")
    print(f"  {'Metric':<30} {'Original':>12} {'Perm 000':>12} {'Difference':>12} {'Preserved?':>12}")
    print("-" * 80)
    print(f"  {'r_CbG':<30} {r_CbG_original:>12.6f} {r_CbG_perm:>12.6f} {diff_CbG:>+12.6f} {'YES' if abs(diff_CbG) < 0.01 else 'NO':>12}")
    print(f"  {'r_GpPW':<30} {r_GpPW_original:>12.6f} {r_GpPW_perm:>12.6f} {diff_GpPW:>+12.6f} {'YES' if abs(diff_GpPW) < 0.01 else 'NO':>12}")
    print()

    print("Pathway-Specific Assortativity:")
    print(f"  {'Metric':<30} {'Original':>12} {'Perm 000':>12} {'Difference':>12} {'Preserved?':>12}")
    print("-" * 80)
    print(f"  {'r_source_intermediate':<30} {pathway_assort_original['r_source_intermediate']:>12.6f} {pathway_assort_perm['r_source_intermediate']:>12.6f} {diff_SI:>+12.6f} {'YES' if abs(diff_SI) < 0.01 else 'NO':>12}")
    print(f"  {'r_intermediate_target':<30} {pathway_assort_original['r_intermediate_target']:>12.6f} {pathway_assort_perm['r_intermediate_target']:>12.6f} {diff_IT:>+12.6f} {'YES' if abs(diff_IT) < 0.01 else 'NO':>12}")
    print(f"  {'r_source_target_weighted':<30} {pathway_assort_original['r_source_target_weighted']:>12.6f} {pathway_assort_perm['r_source_target_weighted']:>12.6f} {diff_ST:>+12.6f} {'YES' if abs(diff_ST) < 0.01 else 'NO':>12}")
    print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    print("Edge Assortativity:")
    if r_CbG_original < 0:
        print(f"  CbG is DISASSORTATIVE (r = {r_CbG_original:.3f})")
        print("    → High-degree compounds bind to low-degree genes")
    else:
        print(f"  CbG is ASSORTATIVE (r = {r_CbG_original:.3f})")
        print("    → High-degree compounds bind to high-degree genes")

    if r_GpPW_original < 0:
        print(f"  GpPW is DISASSORTATIVE (r = {r_GpPW_original:.3f})")
        print("    → High-degree genes participate in low-degree pathways")
    else:
        print(f"  GpPW is ASSORTATIVE (r = {r_GpPW_original:.3f})")
        print("    → High-degree genes participate in high-degree pathways")
    print()

    print("Pathway Assortativity:")
    if pathway_assort_original['r_source_intermediate'] < 0:
        print(f"  Source-Intermediate is DISASSORTATIVE (r = {pathway_assort_original['r_source_intermediate']:.3f})")
        print("    → High-degree compounds connect through low-degree genes")
    else:
        print(f"  Source-Intermediate is ASSORTATIVE (r = {pathway_assort_original['r_source_intermediate']:.3f})")
        print("    → High-degree compounds connect through high-degree genes")

    if pathway_assort_original['r_source_target_weighted'] > 0:
        print(f"  Source-Target is POSITIVE (r = {pathway_assort_original['r_source_target_weighted']:.3f})")
        print("    → High-degree source/target pairs tend to have more pathways")
        print("    → This is expected and captured by deg_source × deg_target feature")
    print()

    print("Permutation Effects:")
    if abs(diff_CbG) > 0.01 or abs(diff_GpPW) > 0.01:
        print("  ⚠ Permutation ALTERS edge assortativity")
        print("  → XSwap preserves degree sequence but not assortativity")
    else:
        print("  ✓ Permutation PRESERVES edge assortativity")

    if abs(diff_SI) > 0.01 or abs(diff_IT) > 0.01:
        print("  ⚠ Permutation ALTERS pathway assortativity")
        print("  → This may explain imperfect pair-level model performance (r < 1.0)")
    else:
        print("  ✓ Permutation PRESERVES pathway assortativity")

    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
