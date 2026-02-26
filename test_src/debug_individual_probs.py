#!/usr/bin/env python3
"""
Debug individual probabilities to find why Option A isn't working.
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

def debug_individual_probs():
    print("DEBUG: Finding individual probabilities that exceed 1.0...")

    # Load edge matrices
    edge1_matrix = load_edge_matrix('CbG', 0)
    edge2_matrix = load_edge_matrix('GpPW', 0)

    # Get degrees and filter
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
    source_degrees_1 = np.array(edge1_aligned.sum(axis=1)).flatten()
    target_degrees_1 = np.array(edge1_aligned.sum(axis=0)).flatten()

    edge1_priors = {}
    rows, cols = edge1_aligned.nonzero()
    print(f"Computing {len(rows)} edge1 priors...")

    edge1_over_1 = []
    for i, j in zip(rows, cols):
        u, v = source_degrees_1[i], target_degrees_1[j]
        if u > 0 and v > 0:
            prior = analytical_prior(u, v, m1)
            edge1_priors[(i, j)] = prior
            if prior > 1.0:
                edge1_over_1.append((i, j, u, v, prior))

    print(f"Edge1 priors > 1.0: {len(edge1_over_1)}")
    if edge1_over_1:
        for i, j, u, v, prior in edge1_over_1[:5]:
            print(f"  ({i},{j}): u={u}, v={v}, prior={prior:.6f}")

    # Edge2 priors
    m2 = edge2_aligned.nnz
    source_degrees_2 = np.array(edge2_aligned.sum(axis=1)).flatten()
    target_degrees_2 = np.array(edge2_aligned.sum(axis=0)).flatten()

    edge2_priors = {}
    rows, cols = edge2_aligned.nonzero()
    print(f"Computing {len(rows)} edge2 priors...")

    edge2_over_1 = []
    for i, j in zip(rows, cols):
        u, v = source_degrees_2[i], target_degrees_2[j]
        if u > 0 and v > 0:
            prior = analytical_prior(u, v, m2)
            edge2_priors[(i, j)] = prior
            if prior > 1.0:
                edge2_over_1.append((i, j, u, v, prior))

    print(f"Edge2 priors > 1.0: {len(edge2_over_1)}")
    if edge2_over_1:
        for i, j, u, v, prior in edge2_over_1[:5]:
            print(f"  ({i},{j}): u={u}, v={v}, prior={prior:.6f}")

    # Now check for individual pathway probabilities > 1.0
    print("Checking individual pathway probabilities...")

    metapath_matrix = edge1_aligned @ edge2_aligned
    over_1_individual = []

    for count, (i, j) in enumerate(zip(*metapath_matrix.nonzero())):
        if count >= 10000:  # Check first 10k pairs
            break

        compound_genes = edge1_aligned.getrow(i).nonzero()[1]
        pathway_genes = edge2_aligned.getcol(j).nonzero()[0]
        shared_genes = set(compound_genes) & set(pathway_genes)

        for gene in shared_genes:
            p_edge1 = edge1_priors.get((i, gene), 0.0)
            p_edge2 = edge2_priors.get((gene, j), 0.0)
            individual_prob = p_edge1 * p_edge2

            if individual_prob > 1.0:
                over_1_individual.append((i, j, gene, p_edge1, p_edge2, individual_prob))

    print(f"Individual pathway probabilities > 1.0: {len(over_1_individual)}")
    if over_1_individual:
        print("Found individual probabilities > 1.0! This is the bug:")
        for i, j, gene, p1, p2, prod in over_1_individual[:5]:
            print(f"  Compound {i} → Gene {gene} → Pathway {j}")
            print(f"    P1={p1:.6f}, P2={p2:.6f}, Product={prod:.6f}")

    # Test the mathematical bounds of analytical_prior
    print("\nTesting analytical_prior mathematical bounds...")
    test_cases = [
        (1, 1, 100),
        (10, 10, 100),
        (50, 50, 100),
        (100, 100, 100),
        (1000, 1000, 100),  # Extreme case
    ]

    for u, v, m in test_cases:
        result = analytical_prior(u, v, m)
        print(f"  analytical_prior({u}, {v}, {m}) = {result:.6f} (>1: {result > 1.0})")

if __name__ == "__main__":
    debug_individual_probs()