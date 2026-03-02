"""
Direct check: reproduce exactly what Exp 2J does and verify the constant r.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'

print("="*80)
print("DIRECT REPRODUCTION OF EXP 2J LOGIC")
print("="*80)

def load_edge_matrix(edge_abbrev, perm_num):
    edge_file = data_dir / 'permutations' / f'{perm_num:03d}.hetmat' / 'edges' / f'{edge_abbrev}.sparse.npz'
    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)
    return None

def analytical_edge_probability(deg_u, deg_v, n_u, n_v, m):
    if m == 0:
        return 0.0
    expected_multiedges = deg_u * deg_v / m
    if expected_multiedges > 10:
        return 1.0
    return 1.0 - np.exp(-expected_multiedges)

# Use small sample for speed
n_pairs = 1000

print(f"\nUsing {n_pairs} pairs for quick verification")

print("\nStep 1: Load perm 0 and extract features...")
CbG_0 = load_edge_matrix('CbG', 0)
GiG_0 = load_edge_matrix('GiG', 0)
GpPW_0 = load_edge_matrix('GpPW', 0)

CbGiG_0 = CbG_0 @ GiG_0
CbGiGpPW_0 = CbGiG_0 @ GpPW_0

gene_degrees = np.array(GiG_0.sum(axis=1)).flatten()
pathway_degrees = np.array(GpPW_0.sum(axis=0)).flatten()
n_genes = GpPW_0.shape[0]
n_pathways = GpPW_0.shape[1]
m_GpPW = GpPW_0.nnz

# Sample pairs
np.random.seed(42)
pairs = [(np.random.randint(0, CbG_0.shape[0]), np.random.randint(0, GpPW_0.shape[1])) for _ in range(n_pairs)]

# Extract features
CbGiG_0_lil = CbGiG_0.tolil()
GpPW_0_csr = GpPW_0.tocsr()

X_features = []
for C, PW in pairs:
    genes_to_PW = GpPW_0_csr[:, PW].nonzero()[0]
    composition_sum = 0.0

    for G2 in genes_to_PW:
        CbGiG_count = CbGiG_0_lil[C, G2]
        if CbGiG_count == 0:
            continue

        deg_G2 = gene_degrees[G2]
        deg_PW = pathway_degrees[PW]
        P_edge = analytical_edge_probability(deg_G2, deg_PW, n_genes, n_pathways, m_GpPW)
        composition_sum += CbGiG_count * P_edge

    X_features.append(composition_sum)

X_features = np.array(X_features).reshape(-1, 1)
print(f"X_features shape: {X_features.shape}")

print("\nStep 2: Compute validation target (mean of perms 11-20)...")

def compute_pathway_counts(perm_num, pairs):
    CbG = load_edge_matrix('CbG', perm_num)
    GiG = load_edge_matrix('GiG', perm_num)
    GpPW = load_edge_matrix('GpPW', perm_num)
    CbGiG = CbG @ GiG
    CbGiGpPW = CbGiG @ GpPW
    CbGiGpPW_lil = CbGiGpPW.tolil()
    return np.array([float(CbGiGpPW_lil[C, PW]) for C, PW in pairs])

val_counts = [compute_pathway_counts(p, pairs) for p in range(11, 21)]
y_val_target = np.mean(val_counts, axis=0)

print(f"y_val_target shape: {y_val_target.shape}, mean: {y_val_target.mean():.3f}")

print("\nStep 3: Test K=1 and K=10...")

results = {}

for K in [1, 10]:
    print(f"\n--- K = {K} ---")

    # Compute training target
    train_counts = [compute_pathway_counts(p, pairs) for p in range(1, K+1)]
    y_train_target_K = np.mean(train_counts, axis=0)

    print(f"y_train_target mean: {y_train_target_K.mean():.3f}")

    # Split with same random_state (THIS IS THE KEY ISSUE?)
    X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
        X_features, y_train_target_K, y_val_target, test_size=0.2, random_state=42
    )

    print(f"After split:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  X_test[0]: {X_test[0, 0]:.6f}")
    print(f"  y_val_test mean: {y_val_test.mean():.3f}")
    print(f"  y_train mean: {y_train.mean():.3f}")

    # Train baseline model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Model coefficients:")
    print(f"  coef: {model.coef_[0]:.6f}")
    print(f"  intercept: {model.intercept_:.6f}")

    # Predict
    y_pred = model.predict(X_test)

    print(f"Predictions:")
    print(f"  mean: {y_pred.mean():.3f}")
    print(f"  first 5: {y_pred[:5]}")

    # Evaluate
    r_val = pearsonr(y_pred, y_val_test)[0]
    r_train = pearsonr(y_pred, y_test)[0]

    print(f"Correlations:")
    print(f"  r vs y_val_test: {r_val:.10f}")
    print(f"  r vs y_test: {r_train:.10f}")

    results[K] = {
        'X_test': X_test,
        'y_val_test': y_val_test,
        'y_pred': y_pred,
        'r_val': r_val,
        'coef': model.coef_[0],
        'intercept': model.intercept_
    }

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print("\nAre X_test the same?")
print(f"  Identical: {np.allclose(results[1]['X_test'], results[10]['X_test'])}")

print("\nAre y_val_test the same?")
print(f"  Identical: {np.allclose(results[1]['y_val_test'], results[10]['y_val_test'])}")

print("\nAre predictions the same?")
pred_K1 = results[1]['y_pred']
pred_K10 = results[10]['y_pred']
print(f"  Identical: {np.allclose(pred_K1, pred_K10)}")
print(f"  Correlation: {pearsonr(pred_K1, pred_K10)[0]:.10f}")
print(f"  Mean diff: {np.abs(pred_K1 - pred_K10).mean():.6f}")
print(f"  First 5 K=1:  {pred_K1[:5]}")
print(f"  First 5 K=10: {pred_K10[:5]}")

print("\nCorrelations with y_val_test:")
print(f"  K=1:  r = {results[1]['r_val']:.10f}")
print(f"  K=10: r = {results[10]['r_val']:.10f}")
print(f"  Difference: {abs(results[1]['r_val'] - results[10]['r_val']):.10f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if abs(results[1]['r_val'] - results[10]['r_val']) < 1e-8:
    print("\nCONFIRMED: r values are essentially identical!")

    if pearsonr(pred_K1, pred_K10)[0] > 0.9999:
        print("\nROOT CAUSE: Predictions are perfectly correlated")
        print("This happens because:")
        print("1. X_test is the same (same random_state)")
        print("2. Single feature regression: pred = coef * X + intercept")
        print("3. Even though coef and intercept differ, the predictions")
        print("   are just linear rescalings of the same X values")
        print("4. Linear rescalings preserve correlation perfectly!")

        print("\nThis is NOT a bug in the code - it's a mathematical property")
        print("of single-feature linear regression.")

        print("\nHowever, the INTERPRETATION may be wrong:")
        print("- We claimed composition features have a ceiling at r=0.78")
        print("- But actually we only tested ONE way to use composition_sum")
        print("- Different splits, non-linear models, or additional features")
        print("  might break through this ceiling")
    else:
        print("\nPredictions differ but correlations are the same")
        print("This is unexpected - investigating further...")
else:
    print(f"\nCorrelations DO differ: {abs(results[1]['r_val'] - results[10]['r_val']):.6f}")
    print("This suggests K does matter, but the effect is small")
