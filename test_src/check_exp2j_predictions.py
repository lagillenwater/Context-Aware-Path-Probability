"""
Check if baseline predictions are actually identical across K in Exp 2J.

This would explain the constant r=0.778.
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
print("CHECKING IF BASELINE PREDICTIONS ARE IDENTICAL ACROSS K")
print("="*80)

def load_edge_matrix(edge_abbrev, perm_num='original'):
    if perm_num == 'original':
        edge_file = data_dir / 'edges' / f'{edge_abbrev}.sparse.npz'
    else:
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

print("\nLoading perm 0 topology...")
CbG_0 = load_edge_matrix('CbG', perm_num=0)
GiG_0 = load_edge_matrix('GiG', perm_num=0)
GpPW_0 = load_edge_matrix('GpPW', perm_num=0)

CbGiG_0 = CbG_0 @ GiG_0

gene_degrees = np.array(GiG_0.sum(axis=1)).flatten()
pathway_degrees = np.array(GpPW_0.sum(axis=0)).flatten()
n_genes = GpPW_0.shape[0]
n_pathways = GpPW_0.shape[1]
m_GpPW = GpPW_0.nnz

# Sample 100 pairs for quick check
np.random.seed(42)
test_pairs = [(np.random.randint(0, CbG_0.shape[0]), np.random.randint(0, GpPW_0.shape[1])) for _ in range(100)]

print(f"\nExtracting features for {len(test_pairs)} test pairs...")

CbGiG_0_lil = CbGiG_0.tolil()
GpPW_0_csr = GpPW_0.tocsr()

features_list = []

for C, PW in test_pairs:
    genes_to_PW = GpPW_0_csr[:, PW].nonzero()[0]

    composition_sum = 0.0
    n_intermediates = 0

    for G2 in genes_to_PW:
        CbGiG_count = CbGiG_0_lil[C, G2]
        if CbGiG_count == 0:
            continue

        n_intermediates += 1
        deg_G2 = gene_degrees[G2]
        deg_PW = pathway_degrees[PW]
        P_edge = analytical_edge_probability(deg_G2, deg_PW, n_genes, n_pathways, m_GpPW)
        composition_sum += CbGiG_count * P_edge

    features_list.append([composition_sum, n_intermediates])

X_features = np.array(features_list)

print(f"X_features shape: {X_features.shape}")
print(f"  composition_sum: min={X_features[:, 0].min():.2f}, max={X_features[:, 0].max():.2f}")
print(f"  n_intermediates: min={X_features[:, 1].min():.0f}, max={X_features[:, 1].max():.0f}")

# Now load different training targets for K=1 and K=10
print("\n" + "="*80)
print("LOADING TRAINING TARGETS")
print("="*80)

def load_pathway_counts(perm_num, pairs):
    CbG = load_edge_matrix('CbG', perm_num=perm_num)
    GiG = load_edge_matrix('GiG', perm_num=perm_num)
    GpPW = load_edge_matrix('GpPW', perm_num=perm_num)

    CbGiG = CbG @ GiG
    CbGiGpPW = CbGiG @ GpPW
    CbGiGpPW_lil = CbGiGpPW.tolil()

    return np.array([float(CbGiGpPW_lil[C, PW]) for C, PW in pairs])

print("\nLoading perm 1 (K=1 training target)...")
y_train_K1 = load_pathway_counts(1, test_pairs)

print("Loading perms 1-10 (K=10 training target)...")
counts_list = []
for perm in range(1, 11):
    print(f"  Perm {perm}...")
    counts_list.append(load_pathway_counts(perm, test_pairs))
y_train_K10 = np.mean(counts_list, axis=0)

print("\nTraining target comparison:")
print(f"  K=1 mean: {y_train_K1.mean():.3f}")
print(f"  K=10 mean: {y_train_K10.mean():.3f}")
print(f"  Correlation: {pearsonr(y_train_K1, y_train_K10)[0]:.4f}")

# Now train baseline models
print("\n" + "="*80)
print("TRAINING BASELINE MODELS (composition only)")
print("="*80)

# No train/test split for this diagnostic - use all data
X_comp = X_features[:, 0:1]

print("\nModel for K=1:")
model_K1 = LinearRegression()
model_K1.fit(X_comp, y_train_K1)
print(f"  Coefficient: {model_K1.coef_[0]:.6f}")
print(f"  Intercept: {model_K1.intercept_:.6f}")

pred_K1 = model_K1.predict(X_comp)

print("\nModel for K=10:")
model_K10 = LinearRegression()
model_K10.fit(X_comp, y_train_K10)
print(f"  Coefficient: {model_K10.coef_[0]:.6f}")
print(f"  Intercept: {model_K10.intercept_:.6f}")

pred_K10 = model_K10.predict(X_comp)

print("\n" + "="*80)
print("COMPARING PREDICTIONS")
print("="*80)

print(f"\nPredictions for K=1:")
print(f"  Mean: {pred_K1.mean():.3f}")
print(f"  Std: {pred_K1.std():.3f}")
print(f"  Range: [{pred_K1.min():.2f}, {pred_K1.max():.2f}]")

print(f"\nPredictions for K=10:")
print(f"  Mean: {pred_K10.mean():.3f}")
print(f"  Std: {pred_K10.std():.3f}")
print(f"  Range: [{pred_K10.min():.2f}, {pred_K10.max():.2f}]")

print(f"\nAre predictions identical? {np.allclose(pred_K1, pred_K10)}")
print(f"Correlation between predictions: {pearsonr(pred_K1, pred_K10)[0]:.10f}")
print(f"Mean absolute difference: {np.abs(pred_K1 - pred_K10).mean():.6f}")
print(f"Max absolute difference: {np.abs(pred_K1 - pred_K10).max():.6f}")

# Now check with validation target
print("\n" + "="*80)
print("CHECKING WITH VALIDATION TARGET")
print("="*80)

print("\nLoading validation target (mean of perms 11-20)...")
val_counts_list = []
for perm in range(11, 21):
    print(f"  Perm {perm}...")
    val_counts_list.append(load_pathway_counts(perm, test_pairs))
y_val = np.mean(val_counts_list, axis=0)

print(f"\nValidation target:")
print(f"  Mean: {y_val.mean():.3f}")
print(f"  Std: {y_val.std():.3f}")

r_K1 = pearsonr(pred_K1, y_val)[0]
r_K10 = pearsonr(pred_K10, y_val)[0]

print(f"\nCorrelation with validation target:")
print(f"  K=1 predictions: r = {r_K1:.10f}")
print(f"  K=10 predictions: r = {r_K10:.10f}")
print(f"  Difference: {abs(r_K1 - r_K10):.10f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if np.allclose(pred_K1, pred_K10, rtol=1e-6):
    print("\nBUG FOUND: Predictions are essentially identical!")
    print("This explains the constant r=0.778 across all K.")
    print("\nRoot cause: Baseline model (composition_sum only) produces")
    print("nearly identical predictions regardless of training target.")
else:
    print("\nNo bug: Predictions are different across K.")
    print(f"But correlation is very high ({pearsonr(pred_K1, pred_K10)[0]:.4f}),")
    print("which means they rank pairs similarly even if values differ.")

print("\nPossible reasons:")
print("1. Features dominate: X (composition_sum) is so predictive that")
print("   the training target (y_train) barely affects learned coefficients")
print("2. Training targets are very similar: y_train_K1 and y_train_K10")
print(f"   correlate at r={pearsonr(y_train_K1, y_train_K10)[0]:.4f}")
print("3. Linear regression finds similar weights regardless of exact target")
