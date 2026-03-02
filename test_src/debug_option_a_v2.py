#!/usr/bin/env python3
"""
Debug why Option A probabilistic combination isn't working - version 2.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# Setup paths
repo_dir = Path.cwd()
data_dir = repo_dir / 'data'

def load_edge_matrix(edge_type: str, perm_id: int = 0) -> sp.csr_matrix:
    """Load edge matrix for given edge type and permutation."""
    edge_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
    return sp.load_npz(edge_file)

def analytical_prior(u, v, m):
    """Analytical formula for edge probability."""
    uv = u * v
    denominator = np.sqrt(uv**2 + (m - u - v + 1)**2)
    return uv / denominator if denominator > 0 else 0.0

def debug_option_a_v2():
    print("DEBUG V2: More comprehensive investigation...")

    # Load edge matrices
    edge1_matrix = load_edge_matrix('CbG', 0)
    edge2_matrix = load_edge_matrix('GpPW', 0)

    print(f"Edge matrices: {edge1_matrix.shape} @ {edge2_matrix.shape}")

    # Get degrees and filter
    compound_degrees = np.array(edge1_matrix.sum(axis=1)).flatten()
    pathway_degrees = np.array(edge2_matrix.sum(axis=0)).flatten()

    compound_nonzero = np.where(compound_degrees > 0)[0]
    pathway_nonzero = np.where(pathway_degrees > 0)[0]

    edge1_aligned = edge1_matrix[compound_nonzero, :]
    edge2_aligned = edge2_matrix[:, pathway_nonzero]

    print(f"Aligned matrices: {edge1_aligned.shape} @ {edge2_aligned.shape}")

    # Compute metapath matrix to find actual pairs
    metapath_matrix = edge1_aligned @ edge2_aligned
    print(f"Metapath matrix: {metapath_matrix.shape}, {metapath_matrix.nnz} nonzero")

    # Compute edge priors (simplified)
    print("Computing edge priors...")

    # Edge1 priors
    m1 = edge1_aligned.nnz
    source_degrees_1 = np.array(edge1_aligned.sum(axis=1)).flatten()
    target_degrees_1 = np.array(edge1_aligned.sum(axis=0)).flatten()

    edge1_priors = {}
    rows, cols = edge1_aligned.nonzero()
    for i, j in zip(rows, cols):
        u, v = source_degrees_1[i], target_degrees_1[j]
        if u > 0 and v > 0:
            edge1_priors[(i, j)] = analytical_prior(u, v, m1)

    # Edge2 priors
    m2 = edge2_aligned.nnz
    source_degrees_2 = np.array(edge2_aligned.sum(axis=1)).flatten()
    target_degrees_2 = np.array(edge2_aligned.sum(axis=0)).flatten()

    edge2_priors = {}
    rows, cols = edge2_aligned.nonzero()
    for i, j in zip(rows, cols):
        u, v = source_degrees_2[i], target_degrees_2[j]
        if u > 0 and v > 0:
            edge2_priors[(i, j)] = analytical_prior(u, v, m2)

    print(f"Edge1 priors: {len(edge1_priors)}")
    print(f"Edge2 priors: {len(edge2_priors)}")

    # Now test actual metapath pairs
    test_cases = []
    counter = 0

    for i, j in zip(*metapath_matrix.nonzero()):
        if counter >= 20:  # Test first 20 cases
            break

        compound_genes = edge1_aligned.getrow(i).nonzero()[1]
        pathway_genes = edge2_aligned.getcol(j).nonzero()[0]
        shared_genes = set(compound_genes) & set(pathway_genes)

        if len(shared_genes) > 0:
            # Compute old method (sum)
            old_sum = 0.0
            individual_probs = []

            for gene in shared_genes:
                p_edge1 = edge1_priors.get((i, gene), 0.0)
                p_edge2 = edge2_priors.get((gene, j), 0.0)
                individual_prob = p_edge1 * p_edge2
                individual_probs.append(individual_prob)
                old_sum += individual_prob

            # Compute Option A
            failure_prob = 1.0
            for prob in individual_probs:
                failure_prob *= (1 - prob)
            option_a_result = 1 - failure_prob

            test_cases.append({
                'i': i, 'j': j,
                'n_genes': len(shared_genes),
                'individual_probs': individual_probs,
                'old_sum': old_sum,
                'option_a': option_a_result,
                'old_exceeds_1': old_sum > 1.0,
                'option_a_exceeds_1': option_a_result > 1.0
            })

            counter += 1

    print(f"\nAnalyzed {len(test_cases)} metapath pairs:")

    over_1_cases = [case for case in test_cases if case['old_exceeds_1']]
    option_a_over_1 = [case for case in test_cases if case['option_a_exceeds_1']]

    print(f"Old method > 1.0: {len(over_1_cases)} cases")
    print(f"Option A > 1.0: {len(option_a_over_1)} cases")

    if over_1_cases:
        print(f"\nFirst few cases where old method > 1.0:")
        for case in over_1_cases[:3]:
            print(f"  i={case['i']}, j={case['j']}: {case['n_genes']} genes")
            print(f"    Individual: {[f'{p:.4f}' for p in case['individual_probs']]}")
            print(f"    Old sum: {case['old_sum']:.6f}")
            print(f"    Option A: {case['option_a']:.6f}")
            if case['option_a_exceeds_1']:
                print(f"    ERROR: Option A still > 1!")

    if option_a_over_1:
        print(f"\nCases where Option A > 1.0:")
        for case in option_a_over_1:
            print(f"  MAJOR BUG: Option A = {case['option_a']:.6f} > 1.0")

if __name__ == "__main__":
    debug_option_a_v2()