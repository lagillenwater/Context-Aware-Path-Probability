"""
Experiment 2G: Focused Composition Model

Improve Exp 2E (r=0.855) by learning to weight the composition term
based on endpoint degrees and intermediate sparsity.

Minimal feature set:
- Endpoint degrees: deg_C, deg_PW, deg_C × deg_PW, deg_C², deg_PW²
- Sparsity: n_intermediates (genes connecting to both C and PW)
- Composition: sum(CbGiG × P_edge) from Exp 2E
- Interaction: n_intermediates × composition_sum

Date: 2025-11-04
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPERIMENT 2G: FOCUSED COMPOSITION MODEL")
print("Learn to weight Exp 2E formula with endpoint degrees + sparsity")
print("="*80)


def load_edge_matrix(edge_abbrev, perm_num='original'):
    """Load edge matrix for specific permutation."""
    if perm_num == 'original':
        edge_file = data_dir / 'edges' / f'{edge_abbrev}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_num:03d}.hetmat' / 'edges' / f'{edge_abbrev}.sparse.npz'

    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)
    else:
        print(f"  Warning: {edge_file} not found")
        return None


def sample_pairs_stratified(pathway_matrix, n_samples=10000, random_state=42):
    """Sample node pairs stratified by pathway count."""
    np.random.seed(random_state)

    sources_nonzero, targets_nonzero = pathway_matrix.nonzero()
    n_nonzero = len(sources_nonzero)

    n_nonzero_sample = min(int(n_samples * 0.5), n_nonzero)
    if n_nonzero > 0:
        idx_nonzero = np.random.choice(n_nonzero, n_nonzero_sample, replace=False)
        sampled_sources = list(sources_nonzero[idx_nonzero])
        sampled_targets = list(targets_nonzero[idx_nonzero])
    else:
        sampled_sources = []
        sampled_targets = []

    n_random = n_samples - len(sampled_sources)
    random_sources = np.random.randint(0, pathway_matrix.shape[0], n_random)
    random_targets = np.random.randint(0, pathway_matrix.shape[1], n_random)

    sampled_sources.extend(random_sources)
    sampled_targets.extend(random_targets)

    return list(zip(sampled_sources, sampled_targets))


print("\n" + "="*80)
print("PHASE 1: Load Data and Compute Empirical Frequencies")
print("="*80)

print("\nLoading perm 0 edges...")
CbG_0 = load_edge_matrix('CbG', perm_num=0)
GiG_0 = load_edge_matrix('GiG', perm_num=0)
GpPW_0 = load_edge_matrix('GpPW', perm_num=0)

print(f"  CbG: {CbG_0.shape}, {CbG_0.nnz:,} edges")
print(f"  GiG: {GiG_0.shape}, {GiG_0.nnz:,} edges")
print(f"  GpPW: {GpPW_0.shape}, {GpPW_0.nnz:,} edges")

print("\nComputing pathway matrices...")
CbGiG_0 = CbG_0 @ GiG_0
CbGiGpPW_0 = CbGiG_0 @ GpPW_0
print(f"  CbGiG: {CbGiG_0.nnz:,} non-zero")
print(f"  CbGiGpPW: {CbGiGpPW_0.nnz:,} non-zero")

# Get degrees
compound_degrees = np.array(CbG_0.sum(axis=1)).flatten()
gene_degrees = np.array(GiG_0.sum(axis=1)).flatten()
pathway_degrees = np.array(GpPW_0.sum(axis=0)).flatten()

print("\nSampling pairs...")
pairs = sample_pairs_stratified(CbGiGpPW_0, n_samples=10000, random_state=42)
print(f"  Sampled {len(pairs)} pairs")

print("\nComputing empirical GpPW edge frequencies...")
print("  (Using perms 5, 10, 15, 20)")

edge_counts_by_degree = {}
total_counts_by_degree = {}

for perm_num in [5, 10, 15, 20]:
    print(f"  Processing perm {perm_num}...")
    GpPW_perm = load_edge_matrix('GpPW', perm_num=perm_num)

    if GpPW_perm is None:
        continue

    GpPW_perm_lil = GpPW_perm.tolil()

    for i in range(GpPW_perm.shape[0]):
        deg_i = gene_degrees[i]
        for j in range(GpPW_perm.shape[1]):
            deg_j = pathway_degrees[j]
            deg_pair = (int(deg_i), int(deg_j))

            if deg_pair not in total_counts_by_degree:
                total_counts_by_degree[deg_pair] = 0
                edge_counts_by_degree[deg_pair] = 0

            total_counts_by_degree[deg_pair] += 1
            if GpPW_perm_lil[i, j] > 0:
                edge_counts_by_degree[deg_pair] += 1

empirical_freq = {}
for deg_pair, total in total_counts_by_degree.items():
    if total > 0:
        empirical_freq[deg_pair] = edge_counts_by_degree[deg_pair] / total
    else:
        empirical_freq[deg_pair] = 0.0

print(f"  Computed frequencies for {len(empirical_freq)} degree pairs")


print("\n" + "="*80)
print("PHASE 2: Compute Features for Each Pair")
print("="*80)

print("\nExtracting features...")

CbGiG_0_lil = CbGiG_0.tolil()
CbGiGpPW_0_lil = CbGiGpPW_0.tolil()
GpPW_0_csr = GpPW_0.tocsr()

features_list = []
targets = []

for idx, (C, PW) in enumerate(pairs):
    if idx % 2000 == 0:
        print(f"  Processed {idx}/{len(pairs)} pairs...")

    deg_C = compound_degrees[C]
    deg_PW = pathway_degrees[PW]

    # Get genes that connect to PW
    genes_to_PW = GpPW_0_csr[:, PW].nonzero()[0]

    # Compute Exp 2E composition term
    composition_sum = 0.0
    n_intermediates = 0

    for G2 in genes_to_PW:
        CbGiG_count = CbGiG_0_lil[C, G2]

        if CbGiG_count == 0:
            continue

        n_intermediates += 1
        deg_G2 = gene_degrees[G2]
        deg_pair = (int(deg_G2), int(deg_PW))
        P_edge = empirical_freq.get(deg_pair, 0.0)

        composition_sum += CbGiG_count * P_edge

    # Build feature vector
    features = np.array([
        deg_C,                              # 0: Source degree
        deg_PW,                             # 1: Target degree
        deg_C * deg_PW,                     # 2: Degree product
        deg_C ** 2,                         # 3: Source degree squared
        deg_PW ** 2,                        # 4: Target degree squared
        n_intermediates,                    # 5: Sparsity (key feature)
        composition_sum,                    # 6: Exp 2E formula
        n_intermediates * composition_sum,  # 7: Interaction
    ], dtype=np.float64)

    target = float(CbGiGpPW_0_lil[C, PW])

    features_list.append(features)
    targets.append(target)

X_features = np.array(features_list)
y_targets = np.array(targets)

print(f"\nFeature matrix: {X_features.shape}")
print(f"Target shape: {y_targets.shape}")

feature_names = [
    'deg_C', 'deg_PW', 'deg_C*deg_PW', 'deg_C^2', 'deg_PW^2',
    'n_intermediates', 'composition_sum', 'n_inter*comp'
]


print("\n" + "="*80)
print("PHASE 3: Train Model")
print("="*80)

print("\nSplitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_targets, test_size=0.2, random_state=42
)

print(f"  Train: {len(X_train)} pairs")
print(f"  Test: {len(X_test)} pairs")

print("\nTraining linear regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

r_train = pearsonr(y_train_pred, y_train)[0]
r_test = pearsonr(y_test_pred, y_test)[0]

print(f"\nTrain r: {r_train:.4f}")
print(f"Test r: {r_test:.4f}")

print("\nModel coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name:20s}: {coef:10.6f}")
print(f"  {'intercept':20s}: {model.intercept_:10.6f}")


print("\n" + "="*80)
print("PHASE 4: Compare to Baselines")
print("="*80)

# Exp 2E baseline: just use composition_sum
composition_sum_test = X_test[:, 6]
r_exp2e = pearsonr(composition_sum_test, y_test)[0]

print(f"\nBaseline (Exp 2E - composition only): r = {r_exp2e:.4f}")
print(f"Exp 2G (with degrees + sparsity):     r = {r_test:.4f}")
print(f"Improvement: {r_test - r_exp2e:.4f}")

if r_test > 0.95:
    status = "SUCCESS"
    print(f"\nSTATUS: SUCCESS (r > 0.95)")
elif r_test > 0.90:
    status = "PROMISING"
    print(f"\nSTATUS: PROMISING (r > 0.90)")
else:
    status = "PARTIAL"
    print(f"\nSTATUS: PARTIAL (r < 0.90)")


print("\n" + "="*80)
print("PHASE 5: Compute Performance Metrics")
print("="*80)

from sklearn.metrics import mean_absolute_error

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"\nTrain MAE: {mae_train:.4f}")
print(f"Test MAE: {mae_test:.4f}")

print(f"\nMean true (test): {y_test.mean():.4f}")
print(f"Mean pred (test): {y_test_pred.mean():.4f}")
print(f"Ratio (pred/true): {y_test_pred.mean() / y_test.mean():.4f}")


print("\n" + "="*80)
print("PHASE 6: Save Results")
print("="*80)

results = {
    'experiment': 'Experiment 2G',
    'description': 'Focused composition model with degrees + sparsity',
    'n_pairs': len(pairs),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'r_train': r_train,
    'r_test': r_test,
    'r_baseline_exp2e': r_exp2e,
    'improvement': r_test - r_exp2e,
    'mae_train': mae_train,
    'mae_test': mae_test,
    'mean_true': y_test.mean(),
    'mean_pred': y_test_pred.mean(),
    'ratio': y_test_pred.mean() / y_test.mean(),
    'status': status
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2g_results.csv', index=False)

# Save feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_
})
feature_importance['abs_coefficient'] = np.abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
feature_importance.to_csv(results_dir / 'experiment2g_feature_importance.csv', index=False)

print("\nFeature importance (by absolute coefficient):")
print(feature_importance[['feature', 'coefficient']].to_string(index=False))


print("\n" + "="*80)
print("PHASE 7: Create Visualization")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Exp 2E baseline (composition only)
ax = axes[0, 0]
ax.scatter(y_test, composition_sum_test, alpha=0.3, s=20)
max_val = max(y_test.max(), composition_sum_test.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax.set_xlabel('True CbGiGpPW Count')
ax.set_ylabel('Exp 2E Prediction (composition only)')
ax.set_title(f'Exp 2E Baseline: r={r_exp2e:.3f}')
ax.grid(alpha=0.3)

# Plot 2: Exp 2G (with degrees + sparsity)
ax = axes[0, 1]
ax.scatter(y_test, y_test_pred, alpha=0.3, s=20)
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax.set_xlabel('True CbGiGpPW Count')
ax.set_ylabel('Exp 2G Prediction (with features)')
ax.set_title(f'Exp 2G: r={r_test:.3f}', fontweight='bold')
ax.grid(alpha=0.3)

# Plot 3: Feature importance
ax = axes[1, 0]
y_pos = np.arange(len(feature_importance))
ax.barh(y_pos, feature_importance['coefficient'])
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_importance['feature'])
ax.set_xlabel('Coefficient')
ax.set_title('Feature Importance')
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3, axis='x')

# Plot 4: Summary
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
EXPERIMENT 2G: FOCUSED COMPOSITION

Features (8 total):
  - Endpoint degrees (5)
  - Sparsity (n_intermediates)
  - Composition sum (Exp 2E)
  - Interaction term

Results:
  Baseline (Exp 2E): r = {r_exp2e:.4f}
  Exp 2G (focused):  r = {r_test:.4f}
  Improvement:       {r_test - r_exp2e:.4f}

Status: {status}

Mean true:  {y_test.mean():.2f}
Mean pred:  {y_test_pred.mean():.2f}
MAE:        {mae_test:.2f}
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig(results_dir / 'experiment2g_plots.png', dpi=150, bbox_inches='tight')

print(f"\nSaved: {results_dir / 'experiment2g_plots.png'}")


print("\n" + "="*80)
print("EXPERIMENT 2G COMPLETE")
print("="*80)
print(f"\nFinal r: {r_test:.4f}")
print(f"Improvement over Exp 2E: {r_test - r_exp2e:.4f}")
print(f"Status: {status}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2g_results.csv'}")
print(f"  {results_dir / 'experiment2g_feature_importance.csv'}")
print(f"  {results_dir / 'experiment2g_plots.png'}")
