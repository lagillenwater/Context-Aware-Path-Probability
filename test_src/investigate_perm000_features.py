#!/usr/bin/env python3
"""
Compare feature-target relationships for original vs perm 000.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

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
print("COMPARING ORIGINAL vs PERM 000 FEATURE-TARGET RELATIONSHIPS")
print("="*80)

# Extract features from original
X_orig, y_orig, _ = extract_features_from_original(
    edge1_type, edge2_type, data_dir, n_bins, feature_set='A'
)

# Extract features from perm 000
X_perm000, y_perm000, _ = extract_features_from_permutation(
    edge1_type, edge2_type, 0, data_dir, n_bins, feature_set='A'
)

# Get permutation average targets (perms 1-20)
y_perms = []
for perm_id in range(1, 21):
    _, y_perm, _ = extract_features_from_permutation(
        edge1_type, edge2_type, perm_id, data_dir, n_bins, feature_set='A'
    )
    y_perms.append(y_perm)

y_perm_avg = np.array(y_perms).mean(axis=0)

print("\n1. ORIGINAL FEATURES → PERM AVG TARGETS")
print("-" * 80)

# Linear regression: original features → perm avg targets
lr_orig = LinearRegression()
lr_orig.fit(X_orig, y_perm_avg)
y_pred_orig = lr_orig.predict(X_orig)
r_orig, _ = pearsonr(y_perm_avg, y_pred_orig)

print(f"Linear regression r: {r_orig:.4f}")
print(f"This is what Phase 1 model learns (approximately)")

# Check bins alone
lr_orig_bins = LinearRegression()
lr_orig_bins.fit(X_orig[:, :2], y_perm_avg)
y_pred_orig_bins = lr_orig_bins.predict(X_orig[:, :2])
r_orig_bins, _ = pearsonr(y_perm_avg, y_pred_orig_bins)
print(f"Using only bin indices: r = {r_orig_bins:.4f}")
print(f"Intermediate signature adds: {r_orig - r_orig_bins:.4f}")

print("\n2. PERM 000 FEATURES → PERM AVG TARGETS (1-20)")
print("-" * 80)

# Linear regression: perm 000 features → perm avg targets
lr_perm000 = LinearRegression()
lr_perm000.fit(X_perm000, y_perm_avg)
y_pred_perm000 = lr_perm000.predict(X_perm000)
r_perm000, _ = pearsonr(y_perm_avg, y_pred_perm000)

print(f"Linear regression r: {r_perm000:.4f}")
print(f"This is what Phase 1b model tries to learn")

# Check bins alone
lr_perm000_bins = LinearRegression()
lr_perm000_bins.fit(X_perm000[:, :2], y_perm_avg)
y_pred_perm000_bins = lr_perm000_bins.predict(X_perm000[:, :2])
r_perm000_bins, _ = pearsonr(y_perm_avg, y_pred_perm000_bins)
print(f"Using only bin indices: r = {r_perm000_bins:.4f}")
print(f"Intermediate signature adds: {r_perm000 - r_perm000_bins:.4f}")

print("\n3. PERM 000 FEATURES → PERM 000 TARGETS")
print("-" * 80)

# What if we predict perm 000's own targets?
lr_perm000_self = LinearRegression()
lr_perm000_self.fit(X_perm000, y_perm000)
y_pred_perm000_self = lr_perm000_self.predict(X_perm000)
r_perm000_self, _ = pearsonr(y_perm000, y_pred_perm000_self)

print(f"Linear regression r: {r_perm000_self:.4f}")
print(f"This is what the model was trained on in Phase 1b")

print("\n4. COMPARING INTERMEDIATE SIGNATURES")
print("-" * 80)

# Compare intermediate signatures
sig_orig = X_orig[:, 2:]
sig_perm000 = X_perm000[:, 2:]

# Correlation for each bin
correlations = []
for i in range(len(sig_orig)):
    # Check if either signature is constant
    if sig_orig[i].std() == 0 or sig_perm000[i].std() == 0:
        continue
    r, _ = pearsonr(sig_orig[i], sig_perm000[i])
    correlations.append(r)

print(f"Correlation between original and perm 000 intermediate signatures:")
print(f"  Mean: {np.mean(correlations):.4f} (across {len(correlations)} non-constant bins)")
print(f"  Std: {np.std(correlations):.4f}")
print(f"  Range: [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print(f"\n1. Original features → Perm avg: r = {r_orig:.4f} (LINEAR FIT)")
print("   Phase 1 NN achieves r = 0.93 (close to linear baseline)")

print(f"\n2. Perm 000 features → Perm avg: r = {r_perm000:.4f} (LINEAR FIT)")
print("   Phase 1b NN achieves r = -0.09 (COMPLETE FAILURE)")

print(f"\n3. Perm 000 features → Perm 000 targets: r = {r_perm000_self:.4f}")
print("   (Training target for Phase 1b)")

print("\n4. DIAGNOSIS:")
if r_perm000 > 0.85:
    print("   Perm 000 features CAN predict perm avg with linear regression")
    print("   BUT the neural network FAILS to learn this relationship")
    print("   Possible reasons:")
    print("   - Too few training samples (80 bins) for complex NN")
    print("   - NN is overfitting to noise in perm 000")
    print("   - Need simpler model or more regularization")
else:
    print("   Perm 000 features CANNOT reliably predict perm avg")
    print("   The relationship is too noisy/unstable")
    print("   Each permutation's intermediate signature is independent")

print(f"\n5. Intermediate signature correlation (original vs perm 000): {np.mean(correlations):.4f}")
print("   Original and perm 000 have different intermediate signatures")
print("   But ORIGINAL signatures have a strong linear relationship to perm avg")
print("   While PERM 000 signatures do not")
