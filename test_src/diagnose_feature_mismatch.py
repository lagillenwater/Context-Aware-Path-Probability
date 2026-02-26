#!/usr/bin/env python3
"""
Diagnostic script to confirm feature extraction mismatch.

This script compares features extracted from the original graph vs
permutation 000 to show they are dramatically different.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import (
    extract_features_from_original,
    extract_features_from_permutation
)


def main():
    """
    Compare features from original graph vs permutation 000.
    """
    data_dir = repo_dir / 'data'
    edge1_type = 'CbG'
    edge2_type = 'GpPW'
    n_bins = 10

    print("=" * 80)
    print("DIAGNOSTIC: Feature Extraction Mismatch")
    print("=" * 80)
    print()

    # Extract features from original graph
    print("Extracting features from ORIGINAL graph...")
    X_original, y_original, meta_original = extract_features_from_original(
        edge1_type, edge2_type, data_dir, n_bins, feature_set='A'
    )
    print(f"  Shape: {X_original.shape}")
    print(f"  Target mean: {y_original.mean():.4f}")
    print()

    # Extract features from permutation 000
    print("Extracting features from PERMUTATION 000...")
    X_perm000, y_perm000, meta_perm000 = extract_features_from_permutation(
        edge1_type, edge2_type, 0, data_dir, n_bins, feature_set='A'
    )
    print(f"  Shape: {X_perm000.shape}")
    print(f"  Target mean: {y_perm000.mean():.4f}")
    print()

    # Compare features
    print("=" * 80)
    print("FEATURE COMPARISON")
    print("=" * 80)
    print()

    # First 2 features are degree bins (should be identical)
    print("Degree bin features (first 2 features):")
    print(f"  Original: {X_original[:, :2]}")
    print(f"  Perm 000: {X_perm000[:, :2]}")
    degree_bins_match = np.allclose(X_original[:, :2], X_perm000[:, :2])
    print(f"  Match: {degree_bins_match}")
    print()

    # Next 100 features are intermediate signatures (should be different)
    print("Intermediate signature features (next 100 features):")
    sig_original = X_original[:, 2:]
    sig_perm000 = X_perm000[:, 2:]

    # Compute correlation for each bin
    correlations = []
    for i in range(len(X_original)):
        if sig_original[i].std() > 0 and sig_perm000[i].std() > 0:
            r, _ = pearsonr(sig_original[i], sig_perm000[i])
            correlations.append(r)

    print(f"  Mean correlation across bins: {np.mean(correlations):.4f}")
    print(f"  Median correlation: {np.median(correlations):.4f}")
    print(f"  Min correlation: {np.min(correlations):.4f}")
    print(f"  Max correlation: {np.max(correlations):.4f}")
    print()

    # L2 distance between feature vectors
    print("Feature vector distances:")
    l2_distances = np.linalg.norm(X_original - X_perm000, axis=1)
    print(f"  Mean L2 distance: {l2_distances.mean():.4f}")
    print(f"  Max L2 distance: {l2_distances.max():.4f}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("The degree bin features match (as expected - XSwap preserves degrees).")
    print(f"But the intermediate signatures are very different:")
    print(f"  Mean correlation: {np.mean(correlations):.4f}")
    print()
    print("This explains why Phase 1b fails:")
    print("  1. Model trained on PERMUTATION 000 features")
    print("  2. Validation uses ORIGINAL GRAPH features (hardcoded)")
    print("  3. Features are completely different")
    print("  4. Model cannot generalize")
    print()
    print("The validation function in src/pathway_evaluation_v2.py always")
    print("uses extract_features_from_original(), regardless of what was")
    print("used for training. This is the BUG.")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
