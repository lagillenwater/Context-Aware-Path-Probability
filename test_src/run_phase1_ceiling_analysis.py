"""
Run Phase 1: Ceiling Analysis

Executes oracle upper bound, binning resolution, and feature sufficiency tests.

Usage:
    python run_phase1_ceiling_analysis.py

Outputs:
    results/ceiling_analysis/
        - oracle_exact_r.txt
        - oracle_binned_r.txt
        - binning_resolution.csv
        - binning_resolution.png
        - feature_sufficiency.csv
        - feature_sufficiency.png
        - ceiling_analysis_results.pkl
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import sys

from src.ceiling_analysis import run_ceiling_analysis


def load_hetionet_data(data_dir='data', edge_type='CbG'):
    """
    Load original Hetionet graph data.

    Parameters:
    - data_dir: Path to data directory
    - edge_type: Edge type abbreviation (e.g., 'CbG' for Compound-binds-Gene)

    Returns:
    - edge_matrix: Sparse adjacency matrix for edge type
    """
    edge_path = os.path.join(data_dir, 'edges', f'{edge_type}.sparse.npz')

    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge file not found: {edge_path}")

    edge_matrix = sp.load_npz(edge_path)

    print(f"Loaded {edge_type} edge matrix: {edge_matrix.shape}")
    print(f"  Edges: {edge_matrix.nnz}")
    print(f"  Density: {edge_matrix.nnz / (edge_matrix.shape[0] * edge_matrix.shape[1]):.6f}")

    return edge_matrix


def load_permutation_pathway_counts(data_dir='data',
                                      metapath='CbGaD',
                                      permutation_ids=range(1, 21)):
    """
    Load pathway counts from permuted graphs.

    For ceiling analysis, we use empirical pathway counts from permutations.

    Parameters:
    - data_dir: Path to data directory
    - metapath: Metapath string (e.g., 'CbGaD')
    - permutation_ids: Which permutations to load

    Returns:
    - pathway_data: List of DataFrames with columns
                     [source_idx, target_idx, source_deg, target_deg, count]
    """
    pathway_data = []

    for perm_id in permutation_ids:
        perm_dir = os.path.join(data_dir, 'permutations', f'perm_{perm_id}')

        if not os.path.exists(perm_dir):
            print(f"Warning: Permutation {perm_id} not found, skipping")
            continue

        edge1_type = metapath[:3]
        edge2_type = metapath[3:6]

        edge1_path = os.path.join(perm_dir, f'{edge1_type}.sparse.npz')
        edge2_path = os.path.join(perm_dir, f'{edge2_type}.sparse.npz')

        if not os.path.exists(edge1_path) or not os.path.exists(edge2_path):
            print(f"Warning: Edge files for perm {perm_id} not found, skipping")
            continue

        edge1 = sp.load_npz(edge1_path)
        edge2 = sp.load_npz(edge2_path)

        pathway_matrix = edge1.dot(edge2)

        if sp.issparse(pathway_matrix):
            pathway_matrix = pathway_matrix.toarray()

        source_degrees = np.array(edge1.sum(axis=1)).flatten()
        target_degrees = np.array(edge2.sum(axis=0)).flatten()

        n_sources = pathway_matrix.shape[0]
        n_targets = pathway_matrix.shape[1]

        source_idx_all = []
        target_idx_all = []
        counts_all = []
        source_deg_all = []
        target_deg_all = []

        for i in range(n_sources):
            for j in range(n_targets):
                if pathway_matrix[i, j] > 0:
                    source_idx_all.append(i)
                    target_idx_all.append(j)
                    counts_all.append(pathway_matrix[i, j])
                    source_deg_all.append(source_degrees[i])
                    target_deg_all.append(target_degrees[j])

        perm_df = pd.DataFrame({
            'source_idx': source_idx_all,
            'target_idx': target_idx_all,
            'source_deg': source_deg_all,
            'target_deg': target_deg_all,
            'count': counts_all,
            'permutation': perm_id
        })

        pathway_data.append(perm_df)

        print(f"Loaded permutation {perm_id}: {len(perm_df)} pathways")

    return pathway_data


def main():
    """Run Phase 1 ceiling analysis."""
    print("=" * 70)
    print("RUNNING PHASE 1: CEILING ANALYSIS")
    print("=" * 70)

    data_dir = 'data'
    metapath = 'CbGaD'

    edge1_type = 'CbG'
    edge2_type = 'GaD'

    print("\nLoading Hetionet edge matrices...")
    edge1_matrix = load_hetionet_data(data_dir, edge1_type)
    edge2_matrix = load_hetionet_data(data_dir, edge2_type)

    print("\nLoading permutation pathway counts...")
    print("Training data: permutations 1-20")
    train_permutations = load_permutation_pathway_counts(
        data_dir, metapath, permutation_ids=range(1, 21)
    )

    print("\nTest data: permutations 21-25")
    test_permutations = load_permutation_pathway_counts(
        data_dir, metapath, permutation_ids=range(21, 26)
    )

    if len(train_permutations) == 0 or len(test_permutations) == 0:
        print("\nERROR: No permutation data found!")
        print("Please run data preparation first to generate permuted graphs.")
        sys.exit(1)

    print(f"\nTotal training pathways: {sum(len(df) for df in train_permutations)}")
    print(f"Total test pathways: {sum(len(df) for df in test_permutations)}")

    print("\nRunning ceiling analysis...")
    results = run_ceiling_analysis(
        edge1_matrix=edge1_matrix,
        edge2_matrix=edge2_matrix,
        train_permutations=train_permutations,
        test_permutations=test_permutations,
        output_dir='results/ceiling_analysis',
        n_bins=10
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Oracle (exact): r = {results['oracle_exact']['r']:.4f}")
    print(f"Oracle (binned): r = {results['oracle_binned']['r']:.4f}")
    print(f"Gap: {results['gap_exact_binned']:.4f}")

    print("\nBest feature set:")
    best_idx = results['feature_results']['r_mean'].idxmax()
    best_row = results['feature_results'].iloc[best_idx]
    print(f"  Set {best_row['feature_set']}: {best_row['n_features']} features")
    print(f"  r = {best_row['r_mean']:.4f} ± {best_row['r_std']:.4f}")

    gap_to_oracle = results['oracle_binned']['r'] - best_row['r_mean']
    print(f"\nGap to oracle (binned): {gap_to_oracle:.4f}")

    if gap_to_oracle < 0.01:
        print("CONCLUSION: Near-optimal performance (within 1% of ceiling)")
    elif gap_to_oracle < 0.03:
        print("CONCLUSION: Close to ceiling (1-3% gap)")
    else:
        print("CONCLUSION: Significant headroom for improvement (>3% gap)")

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("Results saved to: results/ceiling_analysis/")


if __name__ == "__main__":
    main()
