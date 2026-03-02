#!/usr/bin/env python3
"""
Verify what targets are used for training in Phase 1 vs Phase 1b.
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
print("CHECKING TRAINING TARGETS")
print("="*80)

# Phase 1: Extract features from original
X_orig, y_orig, _ = extract_features_from_original(
    edge1_type, edge2_type, data_dir, n_bins, feature_set='A'
)

print("\nPhase 1 Training:")
print(f"  Features extracted from: ORIGINAL graph")
print(f"  Targets are: ???")
print(f"  y_train mean: {y_orig.mean():.6f}")
print(f"  y_train std: {y_orig.std():.6f}")

# Get permutation averages
y_perms = []
for perm_id in range(0, 20):
    _, y_perm, _ = extract_features_from_permutation(
        edge1_type, edge2_type, perm_id, data_dir, n_bins, feature_set='A'
    )
    y_perms.append(y_perm)

y_perm_avg = np.array(y_perms).mean(axis=0)

print(f"\nPermutation Average (perms 000-019):")
print(f"  y_perm_avg mean: {y_perm_avg.mean():.6f}")
print(f"  y_perm_avg std: {y_perm_avg.std():.6f}")

# Compare
r, p = pearsonr(y_orig, y_perm_avg)
print(f"\nCorrelation between y_orig and y_perm_avg: r = {r:.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if r > 0.95:
    print("\nTraining targets (y_orig) are ESSENTIALLY THE SAME as validation targets")
    print("The model is trained on original graph pathway counts,")
    print("which are highly correlated with permutation averages.")
    print("\nThis means Phase 1 is NOT learning biological specifics!")
    print("It's learning the degree-based statistical patterns that")
    print("are preserved across original and permuted graphs.")

elif r > 0.7:
    print("\nTraining targets are SIMILAR but not identical to validation targets")
    print("The model learns patterns from original that generalize to permutations.")

else:
    print("\nTraining targets are DIFFERENT from validation targets")
    print("The model must learn a transformation that generalizes.")

print(f"\nMean difference: {(y_orig - y_perm_avg).mean():.6f}")
print(f"  Original graph has {'higher' if y_orig.mean() > y_perm_avg.mean() else 'lower'} counts on average")

# Phase 1b comparison
X_perm000, y_perm000, _ = extract_features_from_permutation(
    edge1_type, edge2_type, 0, data_dir, n_bins, feature_set='A'
)

# Validation target for Phase 1b is average of perms 1-20
y_perms_1to20 = []
for perm_id in range(1, 21):
    _, y_perm, _ = extract_features_from_permutation(
        edge1_type, edge2_type, perm_id, data_dir, n_bins, feature_set='A'
    )
    y_perms_1to20.append(y_perm)

y_perm_avg_1to20 = np.array(y_perms_1to20).mean(axis=0)

r_1b, _ = pearsonr(y_perm000, y_perm_avg_1to20)

print("\n" + "="*80)
print("PHASE 1b COMPARISON")
print("="*80)
print(f"\nTraining targets (perm 000): mean = {y_perm000.mean():.6f}")
print(f"Validation targets (avg 001-020): mean = {y_perm_avg_1to20.mean():.6f}")
print(f"Correlation: r = {r_1b:.4f}")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print(f"\nPhase 1: Training vs validation target correlation: r = {r:.4f}")
print(f"Phase 1b: Training vs validation target correlation: r = {r_1b:.4f}")

if r > r_1b + 0.01:
    print("\nPhase 1 training targets are CLOSER to validation targets!")
    print("This partially explains why Phase 1 succeeds.")
else:
    print("\nBoth phases have similar train-validation target correlation.")
    print("The difference must be in the features, not the targets.")
