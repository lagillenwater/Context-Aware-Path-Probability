#!/usr/bin/env python3
"""
Test if simple linear models can predict permutation averages.

Key questions:
1. Do linear models overfit with 102 features on 100 samples?
2. Is biological structure actually needed, or is it just statistics?
3. Should we use simpler models instead of neural networks?
"""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score

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
print("TESTING LINEAR MODELS: ORIGINAL vs PERM000")
print("="*80)

# Extract features
X_orig, y_orig, _ = extract_features_from_original(
    edge1_type, edge2_type, data_dir, n_bins, feature_set='A'
)

X_perm000, y_perm000, _ = extract_features_from_permutation(
    edge1_type, edge2_type, 0, data_dir, n_bins, feature_set='A'
)

# Get permutation average targets
y_perms = []
for perm_id in range(0, 20):
    _, y_perm, _ = extract_features_from_permutation(
        edge1_type, edge2_type, perm_id, data_dir, n_bins, feature_set='A'
    )
    y_perms.append(y_perm)

y_perm_avg = np.array(y_perms).mean(axis=0)

print(f"\nDataset: {len(X_orig)} samples, {X_orig.shape[1]} features")
print(f"WARNING: More features (102) than samples (100) - expect overfitting!")

# Test with train/test split
print("\n" + "="*80)
print("TEST 1: TRAIN/TEST SPLIT (80/20)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_orig, y_perm_avg, test_size=0.2, random_state=789
)

print("\n1. Original Features → Perm Avg (with held-out test set)")
print("-" * 80)

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

r_train, _ = pearsonr(y_train, y_pred_train)
r_test, _ = pearsonr(y_test, y_pred_test)

print(f"Linear Regression:")
print(f"  Training r: {r_train:.4f}")
print(f"  Test r: {r_test:.4f}")
print(f"  Overfit indicator: {r_train - r_test:.4f}")

# Ridge regression (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_train_ridge = ridge.predict(X_train)
y_pred_test_ridge = ridge.predict(X_test)

r_train_ridge, _ = pearsonr(y_train, y_pred_train_ridge)
r_test_ridge, _ = pearsonr(y_test, y_pred_test_ridge)

print(f"\nRidge Regression (alpha=1.0):")
print(f"  Training r: {r_train_ridge:.4f}")
print(f"  Test r: {r_test_ridge:.4f}")
print(f"  Overfit indicator: {r_train_ridge - r_test_ridge:.4f}")

# Test with just bin indices
lr_bins = LinearRegression()
lr_bins.fit(X_train[:, :2], y_train)
y_pred_test_bins = lr_bins.predict(X_test[:, :2])
r_test_bins, _ = pearsonr(y_test, y_pred_test_bins)

print(f"\nUsing ONLY bin indices (2 features):")
print(f"  Test r: {r_test_bins:.4f}")

# Now test perm000
print("\n2. Perm000 Features → Perm Avg (with held-out test set)")
print("-" * 80)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_perm000, y_perm_avg, test_size=0.2, random_state=789
)

lr_p = LinearRegression()
lr_p.fit(X_train_p, y_train_p)
y_pred_train_p = lr_p.predict(X_train_p)
y_pred_test_p = lr_p.predict(X_test_p)

r_train_p, _ = pearsonr(y_train_p, y_pred_train_p)
r_test_p, _ = pearsonr(y_test_p, y_pred_test_p)

print(f"Linear Regression:")
print(f"  Training r: {r_train_p:.4f}")
print(f"  Test r: {r_test_p:.4f}")
print(f"  Overfit indicator: {r_train_p - r_test_p:.4f}")

# Ridge
ridge_p = Ridge(alpha=1.0)
ridge_p.fit(X_train_p, y_train_p)
y_pred_test_ridge_p = ridge_p.predict(X_test_p)
r_test_ridge_p, _ = pearsonr(y_test_p, y_pred_test_ridge_p)

print(f"\nRidge Regression (alpha=1.0):")
print(f"  Test r: {r_test_ridge_p:.4f}")

# Compare to NN results
print("\n" + "="*80)
print("COMPARISON TO NEURAL NETWORK RESULTS")
print("="*80)

print("\nPhase 1 (Original → Perm Avg):")
print(f"  Neural Network (reported): r = 0.9266")
print(f"  Linear Regression (test): r = {r_test:.4f}")
print(f"  Ridge Regression (test): r = {r_test_ridge:.4f}")
print(f"  Bins Only (test): r = {r_test_bins:.4f}")

print("\nPhase 1b (Perm000 → Perm Avg):")
print(f"  Neural Network (reported): r = -0.0912")
print(f"  Linear Regression (test): r = {r_test_p:.4f}")
print(f"  Ridge Regression (test): r = {r_test_ridge_p:.4f}")

# Cross-validation for more robust estimate
print("\n" + "="*80)
print("TEST 2: 5-FOLD CROSS-VALIDATION")
print("="*80)

from sklearn.model_selection import cross_val_predict

print("\nOriginal Features → Perm Avg:")
lr_cv = LinearRegression()
y_pred_cv_orig = cross_val_predict(lr_cv, X_orig, y_perm_avg, cv=5)
r_cv_orig, _ = pearsonr(y_perm_avg, y_pred_cv_orig)
print(f"  Linear Regression CV r: {r_cv_orig:.4f}")

ridge_cv = Ridge(alpha=1.0)
y_pred_cv_orig_ridge = cross_val_predict(ridge_cv, X_orig, y_perm_avg, cv=5)
r_cv_orig_ridge, _ = pearsonr(y_perm_avg, y_pred_cv_orig_ridge)
print(f"  Ridge Regression CV r: {r_cv_orig_ridge:.4f}")

print("\nPerm000 Features → Perm Avg:")
y_pred_cv_perm = cross_val_predict(lr_cv, X_perm000, y_perm_avg, cv=5)
r_cv_perm, _ = pearsonr(y_perm_avg, y_pred_cv_perm)
print(f"  Linear Regression CV r: {r_cv_perm:.4f}")

ridge_cv_p = Ridge(alpha=1.0)
y_pred_cv_perm_ridge = cross_val_predict(ridge_cv_p, X_perm000, y_perm_avg, cv=5)
r_cv_perm_ridge, _ = pearsonr(y_perm_avg, y_pred_cv_perm_ridge)
print(f"  Ridge Regression CV r: {r_cv_perm_ridge:.4f}")

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("\n1. OVERFITTING CHECK:")
if r_train - r_test > 0.1:
    print("   YES - Linear regression overfits (train >> test)")
else:
    print("   NO - Linear regression generalizes well")

print("\n2. DO LINEAR MODELS WORK?")
print(f"   Original features: r = {r_cv_orig_ridge:.4f} (Ridge CV)")
print(f"   Perm000 features: r = {r_cv_perm_ridge:.4f} (Ridge CV)")

if r_cv_orig_ridge > 0.85 and r_cv_perm_ridge > 0.85:
    print("   BOTH work well with linear models!")
    print("   → Biological structure is NOT required")
    print("   → It's purely a statistical relationship")
elif r_cv_orig_ridge > 0.85 and r_cv_perm_ridge < 0.85:
    print("   Original works, Perm000 doesn't")
    print("   → Biological structure DOES matter")
else:
    print("   Neither works well")
    print("   → Need more complex models")

print("\n3. NEURAL NETWORK vs LINEAR MODEL:")
print(f"   Original: NN (r=0.93) vs Ridge (r={r_cv_orig_ridge:.4f})")
if abs(0.93 - r_cv_orig_ridge) < 0.05:
    print("   → Similar performance, NN is overkill")
else:
    print("   → NN provides benefit")

print(f"\n   Perm000: NN (r=-0.09) vs Ridge (r={r_cv_perm_ridge:.4f})")
if r_cv_perm_ridge > 0.5:
    print("   → NN FAILED to learn what Ridge easily finds")
    print("   → NN is too complex for this task")
else:
    print("   → Both models struggle equally")

print("\n4. RECOMMENDATION:")
if r_cv_perm_ridge > 0.85:
    print("   Use Ridge regression instead of neural networks")
    print("   Simpler, faster, more interpretable, better performance")
else:
    print("   Linear models also fail on permutations")
    print("   Problem may be fundamental lack of signal")
