"""
Debug script to examine oracle training and test data.

Check if reference binning is working correctly and why oracle r is so high.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import sys

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(__file__))

from run_phase1_ceiling_single_metapath import (
    load_edge_matrix_from_hetmat,
    load_pathway_counts_from_permutations
)


def debug_oracle_data(metapath='AdGpMF', perm_range_1=(0, 3), perm_range_2=(3, 5)):
    """
    Load and compare pathway data from two permutation ranges.

    Parameters:
    - metapath: Metapath to test
    - perm_range_1: First range (e.g., 0-2 for "training")
    - perm_range_2: Second range (e.g., 3-4 for "testing")
    """
    if len(metapath) > 2 and metapath[2] == '>':
        edge1_abbrev = metapath[:4]
        edge2_node_type = metapath[3]
        edge2_abbrev = edge2_node_type + metapath[4:]
    else:
        edge1_abbrev = metapath[:3]
        edge2_node_type = metapath[2]
        edge2_abbrev = edge2_node_type + metapath[3:]

    print(f"Metapath: {metapath}")
    print(f"  Edge 1: {edge1_abbrev}")
    print(f"  Edge 2: {edge2_abbrev}")
    print()

    print("Loading data from two permutation ranges...")
    print(f"  Range 1: perms {perm_range_1[0]:03d}-{perm_range_1[1]-1:03d}")
    print(f"  Range 2: perms {perm_range_2[0]:03d}-{perm_range_2[1]-1:03d}")
    print()

    data1 = load_pathway_counts_from_permutations(
        'data/permutations', edge1_abbrev, edge2_abbrev,
        range(perm_range_1[0], perm_range_1[1]),
        data_dir='data', n_bins=10
    )

    data2 = load_pathway_counts_from_permutations(
        'data/permutations', edge1_abbrev, edge2_abbrev,
        range(perm_range_2[0], perm_range_2[1]),
        data_dir='data', n_bins=10
    )

    df1 = pd.concat(data1, ignore_index=True)
    df2 = pd.concat(data2, ignore_index=True)

    print("=" * 70)
    print("RANGE 1 DATA")
    print("=" * 70)
    print(df1.head(20))
    print(f"\nShape: {df1.shape}")
    print(f"Mean count range: {df1['mean_count'].min():.4f} - {df1['mean_count'].max():.4f}")
    print(f"Std of mean counts: {df1['mean_count'].std():.4f}")
    print()

    print("=" * 70)
    print("RANGE 2 DATA")
    print("=" * 70)
    print(df2.head(20))
    print(f"\nShape: {df2.shape}")
    print(f"Mean count range: {df2['mean_count'].min():.4f} - {df2['mean_count'].max():.4f}")
    print(f"Std of mean counts: {df2['mean_count'].std():.4f}")
    print()

    # Compute bin-level averages for each range
    group_cols = ['source_bin', 'target_bin']

    avg1 = df1.groupby(group_cols)['mean_count'].agg(['mean', 'std', 'count']).reset_index()
    avg1.columns = group_cols + ['range1_mean', 'range1_std', 'range1_n']

    avg2 = df2.groupby(group_cols)['mean_count'].agg(['mean', 'std', 'count']).reset_index()
    avg2.columns = group_cols + ['range2_mean', 'range2_std', 'range2_n']

    comparison = avg1.merge(avg2, on=group_cols, how='outer')

    print("=" * 70)
    print("BIN-LEVEL COMPARISON")
    print("=" * 70)
    print(comparison)
    print()

    # Compute correlation
    valid_mask = comparison['range1_mean'].notna() & comparison['range2_mean'].notna()
    if valid_mask.sum() > 0:
        from scipy.stats import pearsonr
        r = pearsonr(comparison.loc[valid_mask, 'range1_mean'],
                     comparison.loc[valid_mask, 'range2_mean'])[0]
        print(f"Correlation between range averages: r = {r:.4f}")
        print()

        # Check if values are nearly identical
        diff = (comparison['range1_mean'] - comparison['range2_mean']).abs()
        print(f"Mean absolute difference: {diff.mean():.6f}")
        print(f"Max absolute difference: {diff.max():.6f}")
    else:
        print("No overlapping bins to compare")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if valid_mask.sum() > 0 and r > 0.99:
        print("PROBLEM: Bin means are nearly identical across permutation ranges!")
        print("This suggests reference binning is NOT working as intended.")
        print()
        print("Possible causes:")
        print("1. Bins still contain the same nodes across permutations")
        print("2. Pathway formation is so degree-deterministic that")
        print("   even with shuffled edges, the same patterns emerge")
        print("3. Bug in binning logic")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metapath', type=str, default='AdGpMF')
    args = parser.parse_args()

    debug_oracle_data(args.metapath, perm_range_1=(0, 3), perm_range_2=(3, 5))
