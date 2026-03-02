#!/usr/bin/env python3
"""
Debug validation to understand what's being compared.

This script traces through the validation process to show exactly what
features and targets are being used.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import (
    extract_features_from_permutation,
    compute_degree_bins
)


def main():
    """
    Debug validation process.
    """
    data_dir = repo_dir / 'data'
    edge1_type = 'CbG'
    edge2_type = 'GpPW'
    n_bins = 10

    print("=" * 80)
    print("DEBUG: Understanding Validation Process")
    print("=" * 80)
    print()

    # Step 1: Extract features and targets from permutation 000
    print("Step 1: Extract features from PERMUTATION 000")
    X_perm000, y_perm000, meta_perm000 = extract_features_from_permutation(
        edge1_type, edge2_type, 0, data_dir, n_bins, feature_set='A'
    )
    print(f"  Features shape: {X_perm000.shape}")
    print(f"  Targets (perm 000 pathway counts): shape {y_perm000.shape}")
    print(f"  Targets mean: {y_perm000.mean():.4f}")
    print()

    # Step 2: Get bin metadata (which bins were created)
    bin_metadata = meta_perm000['bin_metadata']
    print(f"Step 2: Bin structure from permutation 000")
    print(f"  Number of bins: {len(bin_metadata)}")
    print(f"  Example bins:")
    for i in range(min(5, len(bin_metadata))):
        bm = bin_metadata[i]
        print(f"    Bin {i}: source_bin={bm['source_bin']}, "
              f"target_bin={bm['target_bin']}, "
              f"n_pairs={bm['n_pairs']}")
    print()

    # Step 3: Compute pathway counts for validation permutations
    # using the BIN STRUCTURE from permutation 000
    print("Step 3: Compute pathway counts from permutations 001-020")
    print("  Using BIN STRUCTURE from permutation 000")
    print()

    perm_ids = list(range(1, 21))
    permutation_averages = []

    for perm_id in perm_ids:
        perm_dir = (data_dir / 'permutations' /
                    f'{perm_id:03d}.hetmat' / 'edges')
        edge1_file = perm_dir / f'{edge1_type}.sparse.npz'
        edge2_file = perm_dir / f'{edge2_type}.sparse.npz'

        edge1 = sp.load_npz(edge1_file)
        edge2 = sp.load_npz(edge2_file)
        pathway_matrix = edge1 @ edge2

        # Compute degrees for THIS permutation
        source_degrees = np.array(edge1.sum(axis=1)).flatten()
        target_degrees = np.array(edge2.sum(axis=0)).flatten()

        # Compute bins for THIS permutation
        source_bins, _ = compute_degree_bins(source_degrees, n_bins)
        target_bins, _ = compute_degree_bins(target_degrees, n_bins)

        # Extract pathway counts using bin structure from PERMUTATION 000
        perm_counts = []
        for bin_meta in bin_metadata:
            src_bin = bin_meta['source_bin']
            tgt_bin = bin_meta['target_bin']

            # Find nodes in THIS permutation that fall into these bins
            src_indices = np.where(source_bins == src_bin)[0]
            tgt_indices = np.where(target_bins == tgt_bin)[0]

            if len(src_indices) == 0 or len(tgt_indices) == 0:
                perm_counts.append(0.0)
                continue

            submatrix = pathway_matrix[np.ix_(src_indices, tgt_indices)]
            if isinstance(submatrix, sp.spmatrix):
                counts = submatrix.toarray().flatten()
            else:
                counts = submatrix.flatten()

            perm_counts.append(counts.mean())

        permutation_averages.append(perm_counts)

    permutation_averages = np.array(permutation_averages)
    y_validation = permutation_averages.mean(axis=0)

    print(f"  Validation targets shape: {y_validation.shape}")
    print(f"  Validation targets mean: {y_validation.mean():.4f}")
    print()

    # Step 4: Compare targets
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()

    print("What we're doing:")
    print("  1. Train model: X (perm 000 features) -> y (perm 000 pathway counts)")
    print(f"     Training targets mean: {y_perm000.mean():.4f}")
    print()
    print("  2. Validation: X (perm 000 features) -> y (avg perm 001-020 counts)")
    print(f"     Validation targets mean: {y_validation.mean():.4f}")
    print()

    # Check correlation between training and validation targets
    r_targets, p_targets = pearsonr(y_perm000, y_validation)
    print(f"Correlation between training and validation TARGETS: r = {r_targets:.4f}")
    print()

    print("INSIGHT:")
    print("  The targets are highly correlated (from Phase 0b, we know r=0.997)")
    print("  So the issue is NOT with the targets.")
    print()
    print("  The issue is that the model learns to map features to pathway counts,")
    print("  but permutation 000's features are specific to its random structure.")
    print()
    print("  The features capture things like:")
    print("    - Intermediate node degree distributions")
    print("    - Which specific nodes participate in paths")
    print()
    print("  These are random in each permutation and don't generalize.")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
