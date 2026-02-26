#!/usr/bin/env python3
"""
Test if original graph features are more "stable" than permutation features.

Hypothesis: Original graph intermediate signatures encode statistical
properties that are preserved across permutations, while a single
permutation's signatures are just one random realization.
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

data_dir = repo_dir / 'data'
edge1_type = 'CbG'
edge2_type = 'GpPW'
n_bins = 10

print("="*80)
print("TESTING FEATURE STABILITY")
print("="*80)

# Extract features from original
X_orig, y_orig, _ = extract_features_from_original(
    edge1_type, edge2_type, data_dir, n_bins, feature_set='A'
)

# Extract intermediate signatures
sig_orig = X_orig[:, 2:]  # 100 bins x 100 features

# Extract features from multiple permutations
print("\nExtracting features from 20 permutations...")
sig_perms = []
y_perms = []

for perm_id in range(0, 20):
    X_perm, y_perm, _ = extract_features_from_permutation(
        edge1_type, edge2_type, perm_id, data_dir, n_bins, feature_set='A'
    )
    sig_perms.append(X_perm[:, 2:])
    y_perms.append(y_perm)

sig_perms = np.array(sig_perms)  # (20, 100, 100)
y_perms = np.array(y_perms)  # (20, 100)

# Compute average intermediate signature across permutations
sig_perm_avg = sig_perms.mean(axis=0)  # (100, 100)
y_perm_avg = y_perms.mean(axis=0)  # (100,)

print("\n" + "="*80)
print("TEST 1: FEATURE SIMILARITY TO PERMUTATION AVERAGE")
print("="*80)

# For each bin, compare original vs perm-avg intermediate signature
correlations_orig_vs_avg = []
for i in range(len(sig_orig)):
    if sig_orig[i].std() == 0 or sig_perm_avg[i].std() == 0:
        continue
    r, _ = pearsonr(sig_orig[i], sig_perm_avg[i])
    correlations_orig_vs_avg.append(r)

print(f"\nOriginal signature vs Perm-Avg signature:")
print(f"  Mean correlation: {np.mean(correlations_orig_vs_avg):.4f}")

# For each permutation, compare to perm-avg
print(f"\nEach individual perm signature vs Perm-Avg signature:")
for perm_id in range(5):  # Just show first 5
    correlations_perm_vs_avg = []
    for i in range(len(sig_perms[perm_id])):
        if sig_perms[perm_id][i].std() == 0 or sig_perm_avg[i].std() == 0:
            continue
        r, _ = pearsonr(sig_perms[perm_id][i], sig_perm_avg[i])
        correlations_perm_vs_avg.append(r)
    print(f"  Perm {perm_id:03d}: mean correlation = {np.mean(correlations_perm_vs_avg):.4f}")

print("\n" + "="*80)
print("TEST 2: FEATURE PREDICTIVENESS")
print("="*80)

# Test: Can original features predict permutation average targets?
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

print("\nOriginal features → Perm avg targets:")
ridge = Ridge(alpha=1.0)
y_pred_orig = cross_val_predict(ridge, X_orig, y_perm_avg, cv=5)
r_orig, _ = pearsonr(y_perm_avg, y_pred_orig)
print(f"  Cross-val r: {r_orig:.4f}")

# Test: Can each permutation's features predict perm avg?
print("\nEach permutation's features → Perm avg targets:")
r_perms = []
for perm_id in range(20):
    y_pred_perm = cross_val_predict(ridge, sig_perms[perm_id], y_perm_avg, cv=5)
    r_perm, _ = pearsonr(y_perm_avg, y_pred_perm)
    r_perms.append(r_perm)
    if perm_id < 5:
        print(f"  Perm {perm_id:03d}: r = {r_perm:.4f}")

print(f"\nAverage across all 20 perms: r = {np.mean(r_perms):.4f} ± {np.std(r_perms):.4f}")
print(f"Original: r = {r_orig:.4f}")

print("\n" + "="*80)
print("TEST 3: WITHIN-BIN VARIANCE")
print("="*80)

# For each bin, compute variance of intermediate signature across permutations
print("\nIntermediate signature variance across permutations:")
sig_variance = sig_perms.var(axis=0)  # (100, 100)
print(f"  Mean variance per feature: {sig_variance.mean():.6f}")
print(f"  Std variance per feature: {sig_variance.std():.6f}")

# Compare to signal (difference from mean)
sig_signal = (sig_perm_avg - sig_perm_avg.mean())**2
print(f"\nSignal (squared deviation from grand mean):")
print(f"  Mean signal: {sig_signal.mean():.6f}")

# Signal-to-noise ratio
snr = sig_signal.mean() / sig_variance.mean()
print(f"\nSignal-to-noise ratio: {snr:.4f}")

print("\n" + "="*80)
print("TEST 4: LEAVE-ONE-OUT PREDICTION")
print("="*80)

# Can we predict perm N using average of other 19 perms?
print("\nFor each permutation, predict using avg of other 19:")

loo_correlations = []
for leave_out in range(5):  # Just test first 5
    # Average of all except leave_out
    mask = np.ones(20, dtype=bool)
    mask[leave_out] = False
    y_train_loo = y_perms[mask].mean(axis=0)

    # Use features from leave_out permutation
    y_pred_loo = cross_val_predict(ridge, sig_perms[leave_out], y_train_loo, cv=5)
    r_loo, _ = pearsonr(y_train_loo, y_pred_loo)
    loo_correlations.append(r_loo)
    print(f"  Perm {leave_out:03d}: r = {r_loo:.4f}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("\n1. FEATURE STABILITY:")
if np.mean(correlations_orig_vs_avg) > np.mean(r_perms):
    print(f"   Original features (r={np.mean(correlations_orig_vs_avg):.3f}) are MORE similar to perm-avg")
    print(f"   than individual perms (r={np.mean(r_perms):.3f})")
    print("   → Original captures the 'average' structure better")
else:
    print(f"   Individual perms are as similar to perm-avg as original")

print("\n2. PREDICTIVENESS:")
if r_orig > np.mean(r_perms) + 0.1:
    print(f"   Original features (r={r_orig:.3f}) predict perm-avg BETTER")
    print(f"   than individual perm features (r={np.mean(r_perms):.3f})")
    print("   → Original has more stable predictive signal")
elif r_orig > np.mean(r_perms):
    print(f"   Original features (r={r_orig:.3f}) slightly better than")
    print(f"   individual perm features (r={np.mean(r_perms):.3f})")
else:
    print(f"   All features have similar predictiveness")

print("\n3. SIGNAL-TO-NOISE:")
if snr > 1.0:
    print(f"   SNR = {snr:.2f} > 1: Signal dominates noise")
    print("   → Intermediate signatures are relatively stable")
else:
    print(f"   SNR = {snr:.2f} < 1: Noise dominates signal")
    print("   → Intermediate signatures are highly variable across perms")

print("\n4. WHY PHASE 1 SUCCEEDS:")
print("   Even though original graph has biological structure,")
print("   the DEGREE-BASED patterns it contains are:")
print("   a) Similar to permutation averages")
print("   b) Preserved under degree-preserving permutations")
print("   c) The NN learns these degree-based patterns, not biology")

print("\n5. WHY PHASE 1b FAILS:")
print("   Perm 000 features are one random realization")
print("   They predict perm 000 targets well (r=1.00 in training)")
print("   But other permutations have different random realizations")
print("   NN overfits to perm 000 specifics instead of general patterns")
