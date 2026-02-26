#!/usr/bin/env python3
"""
Debug why Option A probabilistic combination isn't working.
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

def debug_option_a():
    print("DEBUG: Investigating Option A implementation...")

    # Load small sample
    edge1_matrix = load_edge_matrix('CbG', 0)
    edge2_matrix = load_edge_matrix('GpPW', 0)

    # Get degrees
    compound_degrees = np.array(edge1_matrix.sum(axis=1)).flatten()
    pathway_degrees = np.array(edge2_matrix.sum(axis=0)).flatten()

    compound_nonzero = np.where(compound_degrees > 0)[0]
    pathway_nonzero = np.where(pathway_degrees > 0)[0]

    edge1_aligned = edge1_matrix[compound_nonzero, :]
    edge2_aligned = edge2_matrix[:, pathway_nonzero]

    # Compute edge priors
    print("Computing edge priors...")

    # Edge1 priors
    m1 = edge1_aligned.nnz
    source_degrees = np.array(edge1_aligned.sum(axis=1)).flatten()
    target_degrees = np.array(edge1_aligned.sum(axis=0)).flatten()

    edge1_priors = {}
    rows, cols = edge1_aligned.nonzero()
    max_prior_1 = 0
    for i, j in zip(rows, cols):
        u, v = source_degrees[i], target_degrees[j]
        if u > 0 and v > 0:
            prior = analytical_prior(u, v, m1)
            edge1_priors[(i, j)] = prior
            max_prior_1 = max(max_prior_1, prior)

    print(f"Edge1 priors: {len(edge1_priors)}, max = {max_prior_1:.6f}")

    # Edge2 priors
    m2 = edge2_aligned.nnz
    source_degrees = np.array(edge2_aligned.sum(axis=1)).flatten()
    target_degrees = np.array(edge2_aligned.sum(axis=0)).flatten()

    edge2_priors = {}
    rows, cols = edge2_aligned.nonzero()
    max_prior_2 = 0
    for i, j in zip(rows, cols):
        u, v = source_degrees[i], target_degrees[j]
        if u > 0 and v > 0:
            prior = analytical_prior(u, v, m2)
            edge2_priors[(i, j)] = prior
            max_prior_2 = max(max_prior_2, prior)

    print(f"Edge2 priors: {len(edge2_priors)}, max = {max_prior_2:.6f}")

    # Test compositional calculation for a few cases
    print("\nTesting compositional calculation...")

    n_compounds = edge1_aligned.shape[0]
    n_pathways = edge2_aligned.shape[1]

    test_cases = []
    for i in range(min(10, n_compounds)):
        compound_genes = edge1_aligned.getrow(i).nonzero()[1]

        for j in range(min(10, n_pathways)):
            pathway_genes = edge2_aligned.getcol(j).nonzero()[0]
            shared_genes = set(compound_genes) & set(pathway_genes)

            if shared_genes and len(shared_genes) >= 2:  # Find cases with multiple paths

                # Current (buggy) method - sum
                old_sum = 0.0
                individual_probs = []

                for gene in shared_genes:
                    p_edge1 = edge1_priors.get((i, gene), 0.0)
                    p_edge2 = edge2_priors.get((gene, j), 0.0)
                    individual_prob = p_edge1 * p_edge2
                    individual_probs.append(individual_prob)
                    old_sum += individual_prob

                # Option A method - probabilistic combination
                failure_prob = 1.0
                for prob in individual_probs:
                    failure_prob *= (1 - prob)
                option_a_result = 1 - failure_prob

                test_cases.append({
                    'compound_i': i,
                    'pathway_j': j,
                    'n_genes': len(shared_genes),
                    'individual_probs': individual_probs,
                    'old_sum': old_sum,
                    'option_a': option_a_result,
                    'sum_exceeds_1': old_sum > 1.0,
                    'option_a_exceeds_1': option_a_result > 1.0
                })

                if len(test_cases) >= 5:
                    break
        if len(test_cases) >= 5:
            break

    print(f"\nFound {len(test_cases)} test cases:")
    for case in test_cases:
        print(f"\nCase: compound {case['compound_i']}, pathway {case['pathway_j']}")
        print(f"  Genes: {case['n_genes']}")
        print(f"  Individual probs: {[f'{p:.6f}' for p in case['individual_probs']]}")
        print(f"  Old sum: {case['old_sum']:.6f} (>1: {case['sum_exceeds_1']})")
        print(f"  Option A: {case['option_a']:.6f} (>1: {case['option_a_exceeds_1']})")

        if case['option_a_exceeds_1']:
            print(f"  ERROR: Option A still > 1.0!")

if __name__ == "__main__":
    debug_option_a()