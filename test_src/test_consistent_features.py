#!/usr/bin/env python3
"""
Test predictiveness using consistent features (full features for both).
"""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

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
print("CONSISTENT FEATURE COMPARISON")
print("="*80)

# Extract features from original (FULL FEATURES)
X_orig, y_orig, _ = extract_features_from_original(
    edge1_type, edge2_type, data_dir, n_bins, feature_set='A'
)

# Extract features from multiple permutations (FULL FEATURES)
X_perms = []
y_perms = []

for perm_id in range(0, 20):
    X_perm, y_perm, _ = extract_features_from_permutation(
        edge1_type, edge2_type, perm_id, data_dir, n_bins, feature_set='A'
    )
    X_perms.append(X_perm)  # Keep FULL features
    y_perms.append(y_perm)

X_perms = np.array(X_perms)  # (20, 100, 102)
y_perms = np.array(y_perms)  # (20, 100)
y_perm_avg = y_perms.mean(axis=0)

print(f"\nFeature shape: {X_orig.shape[1]} dimensions")
print(f"  - 2 bin indices")
print(f"  - 100 intermediate signature features")

ridge = Ridge(alpha=1.0)

print("\n" + "="*80)
print("FULL FEATURES: Ridge CV Prediction")
print("="*80)

# Original
y_pred_orig = cross_val_predict(ridge, X_orig, y_perm_avg, cv=5)
r_orig, _ = pearsonr(y_perm_avg, y_pred_orig)
print(f"\nOriginal features → Perm avg: r = {r_orig:.4f}")

# Each permutation
print(f"\nEach permutation features → Perm avg:")
r_perms = []
for perm_id in range(20):
    y_pred_perm = cross_val_predict(ridge, X_perms[perm_id], y_perm_avg, cv=5)
    r_perm, _ = pearsonr(y_perm_avg, y_pred_perm)
    r_perms.append(r_perm)
    if perm_id < 5:
        print(f"  Perm {perm_id:03d}: r = {r_perm:.4f}")

print(f"\nAverage across 20 perms: r = {np.mean(r_perms):.4f} ± {np.std(r_perms):.4f}")

print("\n" + "="*80)
print("BIN INDICES ONLY: Ridge CV Prediction")
print("="*80)

# Original
y_pred_orig_bins = cross_val_predict(ridge, X_orig[:, :2], y_perm_avg, cv=5)
r_orig_bins, _ = pearsonr(y_perm_avg, y_pred_orig_bins)
print(f"\nOriginal bin indices → Perm avg: r = {r_orig_bins:.4f}")

# Perm 000
y_pred_perm000_bins = cross_val_predict(ridge, X_perms[0, :, :2], y_perm_avg, cv=5)
r_perm000_bins, _ = pearsonr(y_perm_avg, y_pred_perm000_bins)
print(f"Perm 000 bin indices → Perm avg: r = {r_perm000_bins:.4f}")

print("\n" + "="*80)
print("INTERMEDIATE SIGNATURES ONLY: Ridge CV Prediction")
print("="*80)

# Original
y_pred_orig_sig = cross_val_predict(ridge, X_orig[:, 2:], y_perm_avg, cv=5)
r_orig_sig, _ = pearsonr(y_perm_avg, y_pred_orig_sig)
print(f"\nOriginal signatures → Perm avg: r = {r_orig_sig:.4f}")

# Each permutation
print(f"\nEach permutation signatures → Perm avg:")
r_perms_sig = []
for perm_id in range(20):
    y_pred_perm = cross_val_predict(ridge, X_perms[perm_id, :, 2:], y_perm_avg, cv=5)
    r_perm, _ = pearsonr(y_perm_avg, y_pred_perm)
    r_perms_sig.append(r_perm)
    if perm_id < 5:
        print(f"  Perm {perm_id:03d}: r = {r_perm:.4f}")

print(f"\nAverage across 20 perms: r = {np.mean(r_perms_sig):.4f} ± {np.std(r_perms_sig):.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nWith FULL features:")
print(f"  Original: r = {r_orig:.4f}")
print(f"  Perm 000: r = {r_perms[0]:.4f}")
print(f"  Perm avg: r = {np.mean(r_perms):.4f}")

print("\nWith ONLY bin indices:")
print(f"  Original: r = {r_orig_bins:.4f}")
print(f"  Perm 000: r = {r_perm000_bins:.4f}")

print("\nWith ONLY intermediate signatures:")
print(f"  Original: r = {r_orig_sig:.4f}")
print(f"  Perm 000: r = {r_perms_sig[0]:.4f}")
print(f"  Perm avg: r = {np.mean(r_perms_sig):.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if abs(r_perms[0] - r_orig) < 0.05:
    print("\nFull features: Original and Perm 000 perform SIMILARLY")
else:
    print("\nFull features: Original performs DIFFERENTLY than Perm 000")

if abs(r_perm000_bins - r_orig_bins) < 0.01:
    print("Bin indices: Identical for both (makes sense - same binning)")
else:
    print("Bin indices: Different (unexpected)")

if abs(r_perms_sig[0] - r_orig_sig) < 0.05:
    print("Signatures only: Original and Perm 000 perform SIMILARLY")
else:
    print("Signatures only: Original performs DIFFERENTLY than Perm 000")
    if r_orig_sig > 0 and r_perms_sig[0] < 0:
        print("  → Original signatures have positive predictive power")
        print("  → Perm 000 signatures have NEGATIVE correlation")
