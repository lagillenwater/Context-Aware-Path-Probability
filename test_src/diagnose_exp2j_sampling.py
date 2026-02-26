"""
Diagnostic script to check if Exp 2J sampling is causing constant correlations.

Check:
1. Are test indices the same across all K?
2. Are predictions actually different?
3. Is y_val_test actually the same?
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))

print("="*80)
print("DIAGNOSING EXP 2J SAMPLING ISSUE")
print("="*80)

# Simulate the experiment structure
np.random.seed(42)

n_samples = 10000
n_features = 2

# Fixed features (same across all K)
X_features = np.random.randn(n_samples, n_features)

# Fixed validation target (same across all K)
y_val_target = np.random.randn(n_samples)

# Different training targets for each K
K_values = [1, 2, 3, 4, 6, 8, 10]

print("\n" + "="*80)
print("TEST 1: Are test indices identical across K?")
print("="*80)

test_indices_by_K = {}

for K in K_values:
    # Simulate different training target
    y_train_target_K = np.random.randn(n_samples) + K*0.1  # Different for each K

    # This is what the code does
    X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
        X_features, y_train_target_K, y_val_target, test_size=0.2, random_state=42
    )

    # Get indices (reconstruct from values)
    test_idx = []
    for i in range(len(X_features)):
        if np.allclose(X_features[i], X_test[0]):
            test_idx.append(i)
            break

    test_indices_by_K[K] = (X_test, y_val_test, y_test)

    print(f"K={K}: X_test shape={X_test.shape}, y_val_test shape={y_val_test.shape}")
    print(f"       X_test[0]={X_test[0, :]}")
    print(f"       y_val_test mean={y_val_test.mean():.4f}, std={y_val_test.std():.4f}")

print("\n" + "="*80)
print("TEST 2: Are X_test identical across K?")
print("="*80)

X_test_K1 = test_indices_by_K[1][0]
for K in K_values[1:]:
    X_test_K = test_indices_by_K[K][0]
    are_identical = np.allclose(X_test_K1, X_test_K)
    print(f"K=1 vs K={K}: X_test identical? {are_identical}")

print("\n" + "="*80)
print("TEST 3: Are y_val_test identical across K?")
print("="*80)

y_val_test_K1 = test_indices_by_K[1][1]
for K in K_values[1:]:
    y_val_test_K = test_indices_by_K[K][1]
    are_identical = np.allclose(y_val_test_K1, y_val_test_K)
    print(f"K=1 vs K={K}: y_val_test identical? {are_identical}")

print("\n" + "="*80)
print("TEST 4: Are y_test (training targets) different across K?")
print("="*80)

y_test_K1 = test_indices_by_K[1][2]
for K in K_values[1:]:
    y_test_K = test_indices_by_K[K][2]
    are_identical = np.allclose(y_test_K1, y_test_K)
    correlation = pearsonr(y_test_K1, y_test_K)[0]
    print(f"K=1 vs K={K}: y_test identical? {are_identical}, correlation={correlation:.4f}")

print("\n" + "="*80)
print("TEST 5: Does using same random_state cause constant predictions?")
print("="*80)

from sklearn.linear_model import LinearRegression

predictions_by_K = {}

for K in K_values:
    y_train_target_K = np.random.randn(n_samples) + K*0.1

    X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
        X_features, y_train_target_K, y_val_target, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate vs validation target
    r_val = pearsonr(y_pred, y_val_test)[0]
    r_train = pearsonr(y_pred, y_test)[0]

    predictions_by_K[K] = y_pred

    print(f"K={K}: r vs y_val_test={r_val:.4f}, r vs y_test={r_train:.4f}")
    print(f"       coef={model.coef_}, intercept={model.intercept_:.4f}")

print("\n" + "="*80)
print("TEST 6: Are predictions different across K?")
print("="*80)

y_pred_K1 = predictions_by_K[1]
for K in K_values[1:]:
    y_pred_K = predictions_by_K[K]
    are_identical = np.allclose(y_pred_K1, y_pred_K)
    correlation = pearsonr(y_pred_K1, y_pred_K)[0]
    print(f"K=1 vs K={K}: predictions identical? {are_identical}, correlation={correlation:.4f}")

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

print("\nExpected behavior:")
print("- X_test should be SAME across K (same feature subset)")
print("- y_val_test should be SAME across K (same validation target subset)")
print("- y_test should be DIFFERENT across K (different training target subset)")
print("- Predictions should be DIFFERENT across K (model trained on different targets)")
print("- r vs y_val_test should VARY with K (if training target quality matters)")

print("\nActual behavior (if bug exists):")
print("- All of the above are true EXCEPT:")
print("- If predictions are nearly identical, then r won't change")
print("- This could happen if X features dominate over y_train variation")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("\nThe constant r=0.778 across all K could be caused by:")
print("1. Using random_state=42 means same train/test split (CORRECT)")
print("2. y_val_test is identical across K (CORRECT - same subset)")
print("3. But predictions SHOULD differ if training targets differ")
print("4. If predictions are very similar, r will be constant")
print("\nThis could mean:")
print("a) Bug: Features don't actually change predictions much")
print("b) Reality: Model is overfitting to features, ignoring training target")
print("c) Bug: Training target variation is too small to affect learned model")
