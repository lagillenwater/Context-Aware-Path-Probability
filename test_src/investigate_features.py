#!/usr/bin/env python3
"""
Investigate what features are actually predicting pathway counts.

Questions:
1. Is the model using intermediate signatures or just bin indices?
2. How do intermediate signatures differ between original and permutations?
3. What's the correlation between original features and permutation targets?
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import (
    extract_features_from_original,
    extract_features_from_permutation,
    compute_degree_bins
)

def compare_intermediate_signatures(edge1_type, edge2_type, data_dir, n_bins=10):
    """
    Compare intermediate signatures between original and permutations.
    """
    data_dir = Path(data_dir)

    # Extract features from original
    X_orig, y_orig, meta_orig = extract_features_from_original(
        edge1_type, edge2_type, data_dir, n_bins, feature_set='A'
    )

    # Extract intermediate signatures (skip first 2 features which are bin indices)
    sig_orig = X_orig[:, 2:]  # 100 features per bin

    # Extract features from multiple permutations
    print("Extracting features from permutations...")
    sig_perms = []
    y_perms = []

    for perm_id in range(5):  # First 5 permutations
        X_perm, y_perm, _ = extract_features_from_permutation(
            edge1_type, edge2_type, perm_id, data_dir, n_bins, feature_set='A'
        )
        sig_perms.append(X_perm[:, 2:])
        y_perms.append(y_perm)

    sig_perms = np.array(sig_perms)  # (5, 100, 100)
    y_perms = np.array(y_perms)  # (5, 100)

    # Compute average intermediate signature across permutations
    sig_perm_avg = sig_perms.mean(axis=0)  # (100, 100)
    y_perm_avg = y_perms.mean(axis=0)  # (100,)

    print("\n" + "="*80)
    print("INTERMEDIATE SIGNATURE ANALYSIS")
    print("="*80)

    # Compare signatures: original vs permutation average
    correlations = []
    for i in range(len(sig_orig)):
        r, _ = pearsonr(sig_orig[i], sig_perm_avg[i])
        correlations.append(r)

    print(f"\nCorrelation between original and perm-avg intermediate signatures:")
    print(f"  Mean: {np.mean(correlations):.4f}")
    print(f"  Std: {np.std(correlations):.4f}")
    print(f"  Min: {np.min(correlations):.4f}")
    print(f"  Max: {np.max(correlations):.4f}")

    # Compare targets: original vs permutation average
    r_targets, p_targets = pearsonr(y_orig, y_perm_avg)
    print(f"\nCorrelation between original and perm-avg pathway counts:")
    print(f"  r = {r_targets:.4f}, p = {p_targets:.4e}")

    # Check if bin indices alone predict permutation averages
    # Bin indices are just 0-9 for each, so let's see if they correlate with targets
    bin_indices = X_orig[:, :2]  # (100, 2)

    # Use just source bin to predict
    source_bins = bin_indices[:, 0]
    r_source, _ = pearsonr(source_bins, y_perm_avg)
    print(f"\nCorrelation between source_bin and perm-avg pathway count:")
    print(f"  r = {r_source:.4f}")

    # Use just target bin to predict
    target_bins = bin_indices[:, 1]
    r_target, _ = pearsonr(target_bins, y_perm_avg)
    print(f"\nCorrelation between target_bin and perm-avg pathway count:")
    print(f"  r = {r_target:.4f}")

    # Try predicting using just bin indices (linear combination)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(bin_indices, y_perm_avg)
    y_pred_bins_only = lr.predict(bin_indices)
    r_bins_only, _ = pearsonr(y_perm_avg, y_pred_bins_only)

    print(f"\nCorrelation using ONLY bin indices (source + target):")
    print(f"  r = {r_bins_only:.4f}")
    print(f"  Coefficients: source={lr.coef_[0]:.6f}, target={lr.coef_[1]:.6f}")

    # Now use original intermediate signature to predict permutation average
    lr_full = LinearRegression()
    lr_full.fit(X_orig, y_perm_avg)
    y_pred_full = lr_full.predict(X_orig)
    r_full, _ = pearsonr(y_perm_avg, y_pred_full)

    print(f"\nCorrelation using ALL features (bins + intermediate signature):")
    print(f"  r = {r_full:.4f}")
    print(f"  Improvement over bins alone: {r_full - r_bins_only:.4f}")

    # Compare within-permutation variance to between-permutation variance
    print("\n" + "="*80)
    print("VARIANCE ANALYSIS")
    print("="*80)

    # For each bin, compute variance across permutations
    bin_variances = y_perms.var(axis=0)  # Variance across 5 perms for each bin
    print(f"\nBin-level variance across permutations:")
    print(f"  Mean: {bin_variances.mean():.6f}")
    print(f"  Std: {bin_variances.std():.6f}")

    # Compare to overall variance
    print(f"\nOverall pathway count variance:")
    print(f"  Original: {y_orig.var():.6f}")
    print(f"  Perm average: {y_perm_avg.var():.6f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    print("\n1. Do intermediate signatures differ between original and permutations?")
    if np.mean(correlations) < 0.5:
        print("   YES - very different (mean correlation < 0.5)")
    elif np.mean(correlations) < 0.8:
        print("   SOMEWHAT - moderately different (mean correlation 0.5-0.8)")
    else:
        print("   NO - quite similar (mean correlation > 0.8)")

    print("\n2. Do bin indices alone explain permutation averages?")
    if r_bins_only > 0.8:
        print(f"   YES - bins alone achieve r={r_bins_only:.3f}")
        print("   Intermediate signature may not be needed!")
    else:
        print(f"   NO - bins alone only achieve r={r_bins_only:.3f}")

    print("\n3. Does original intermediate signature improve prediction?")
    improvement = r_full - r_bins_only
    if improvement > 0.1:
        print(f"   YES - substantial improvement of {improvement:.3f}")
    elif improvement > 0.01:
        print(f"   MODEST - small improvement of {improvement:.3f}")
    else:
        print(f"   NO - negligible improvement of {improvement:.3f}")


if __name__ == '__main__':
    data_dir = repo_dir / 'data'
    edge1_type = 'CbG'
    edge2_type = 'GpPW'

    print("Investigating feature relationships for CbGpPW metapath")
    print("="*80)

    compare_intermediate_signatures(edge1_type, edge2_type, data_dir, n_bins=10)
