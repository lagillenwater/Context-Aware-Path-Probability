"""
Experiment 2H: Analytical Prior Composition with Sparsity

Test whether sparsity effect is real when using ANALYTICAL edge priors
instead of empirical frequencies from multiple permutations.

Features:
- composition_sum (using analytical prior)
- n_intermediates

Compare to baseline: composition_sum only

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
print("EXPERIMENT 2H: ANALYTICAL PRIOR COMPOSITION WITH SPARSITY")
print("Use analytical edge prior instead of empirical frequencies")
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


def analytical_edge_probability(deg_u, deg_v, n_u, n_v, m):
    """
    Analytical edge probability from configuration model.

    P(edge exists) = 1 - exp(-deg_u * deg_v / m)

    where m = total edges in network
    """
    if m == 0:
        return 0.0

    # Avoid overflow for large degree products
    expected_multiedges = deg_u * deg_v / m
    if expected_multiedges > 10:
        return 1.0  # Saturates at 1

    return 1.0 - np.exp(-expected_multiedges)


print("\n" + "="*80)
print("PHASE 1: Load Data")
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

# Network sizes for analytical prior
n_genes = GpPW_0.shape[0]
n_pathways = GpPW_0.shape[1]
m_GpPW = GpPW_0.nnz

print(f"\nNetwork statistics for analytical prior:")
print(f"  n_genes: {n_genes}")
print(f"  n_pathways: {n_pathways}")
print(f"  m_GpPW (total edges): {m_GpPW}")

print("\nSampling pairs...")
pairs = sample_pairs_stratified(CbGiGpPW_0, n_samples=10000, random_state=42)
print(f"  Sampled {len(pairs)} pairs")


print("\n" + "="*80)
print("PHASE 2: Compute Features with Analytical Prior")
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

    # Get genes that connect to PW
    genes_to_PW = GpPW_0_csr[:, PW].nonzero()[0]

    # Compute composition term with ANALYTICAL prior
    composition_sum = 0.0
    n_intermediates = 0

    for G2 in genes_to_PW:
        CbGiG_count = CbGiG_0_lil[C, G2]

        if CbGiG_count == 0:
            continue

        n_intermediates += 1
        deg_G2 = gene_degrees[G2]
        deg_PW = pathway_degrees[PW]

        # Use ANALYTICAL edge probability
        P_edge = analytical_edge_probability(deg_G2, deg_PW, n_genes, n_pathways, m_GpPW)

        composition_sum += CbGiG_count * P_edge

    # Build feature vector (ONLY 2 features)
    features = np.array([
        composition_sum,    # 0: Composition with analytical prior
        n_intermediates,    # 1: Sparsity
    ], dtype=np.float64)

    target = float(CbGiGpPW_0_lil[C, PW])

    features_list.append(features)
    targets.append(target)

X_features = np.array(features_list)
y_targets = np.array(targets)

print(f"\nFeature matrix: {X_features.shape}")
print(f"Target shape: {y_targets.shape}")

feature_names = ['composition_sum', 'n_intermediates']


print("\n" + "="*80)
print("PHASE 3: Train Models")
print("="*80)

print("\nSplitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_targets, test_size=0.2, random_state=42
)

print(f"  Train: {len(X_train)} pairs")
print(f"  Test: {len(X_test)} pairs")

# Model 1: Baseline (composition only)
print("\nTraining baseline (composition only)...")
X_train_comp = X_train[:, 0:1]
X_test_comp = X_test[:, 0:1]

model_baseline = LinearRegression()
model_baseline.fit(X_train_comp, y_train)

y_train_baseline = model_baseline.predict(X_train_comp)
y_test_baseline = model_baseline.predict(X_test_comp)

r_train_baseline = pearsonr(y_train_baseline, y_train)[0]
r_test_baseline = pearsonr(y_test_baseline, y_test)[0]

print(f"  Train r: {r_train_baseline:.4f}")
print(f"  Test r: {r_test_baseline:.4f}")
print(f"  Coefficient: {model_baseline.coef_[0]:.6f}")
print(f"  Intercept: {model_baseline.intercept_:.6f}")

# Model 2: With sparsity (composition + n_intermediates)
print("\nTraining with sparsity (composition + n_intermediates)...")
model_full = LinearRegression()
model_full.fit(X_train, y_train)

y_train_full = model_full.predict(X_train)
y_test_full = model_full.predict(X_test)

r_train_full = pearsonr(y_train_full, y_train)[0]
r_test_full = pearsonr(y_test_full, y_test)[0]

print(f"  Train r: {r_train_full:.4f}")
print(f"  Test r: {r_test_full:.4f}")

print("\n  Model coefficients:")
for name, coef in zip(feature_names, model_full.coef_):
    print(f"    {name:20s}: {coef:10.6f}")
print(f"    {'intercept':20s}: {model_full.intercept_:10.6f}")


print("\n" + "="*80)
print("PHASE 4: Compare Results")
print("="*80)

print(f"\nBaseline (composition only): r = {r_test_baseline:.4f}")
print(f"With sparsity (+ n_inter):   r = {r_test_full:.4f}")
print(f"Improvement:                  {r_test_full - r_test_baseline:.4f}")

if r_test_full > r_test_baseline + 0.05:
    status = "CONFIRMED"
    print(f"\nSTATUS: CONFIRMED - Sparsity effect is REAL (improvement > 0.05)")
elif r_test_full > r_test_baseline:
    status = "WEAK"
    print(f"\nSTATUS: WEAK - Small improvement from sparsity")
else:
    status = "NONE"
    print(f"\nSTATUS: NONE - No improvement from sparsity")


print("\n" + "="*80)
print("PHASE 5: Compute Performance Metrics")
print("="*80)

from sklearn.metrics import mean_absolute_error

mae_baseline = mean_absolute_error(y_test, y_test_baseline)
mae_full = mean_absolute_error(y_test, y_test_full)

print(f"\nBaseline MAE: {mae_baseline:.4f}")
print(f"Full model MAE: {mae_full:.4f}")
print(f"MAE reduction: {mae_baseline - mae_full:.4f}")

print(f"\nMean true (test): {y_test.mean():.4f}")
print(f"Mean pred baseline: {y_test_baseline.mean():.4f}")
print(f"Mean pred full: {y_test_full.mean():.4f}")


print("\n" + "="*80)
print("PHASE 6: Save Results")
print("="*80)

results = {
    'experiment': 'Experiment 2H',
    'description': 'Analytical prior composition with sparsity',
    'n_pairs': len(pairs),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'r_train_baseline': r_train_baseline,
    'r_test_baseline': r_test_baseline,
    'r_train_full': r_train_full,
    'r_test_full': r_test_full,
    'improvement': r_test_full - r_test_baseline,
    'mae_baseline': mae_baseline,
    'mae_full': mae_full,
    'mean_true': y_test.mean(),
    'mean_pred_baseline': y_test_baseline.mean(),
    'mean_pred_full': y_test_full.mean(),
    'status': status
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2h_results.csv', index=False)

# Save feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model_full.coef_
})
feature_importance['abs_coefficient'] = np.abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
feature_importance.to_csv(results_dir / 'experiment2h_feature_importance.csv', index=False)

print("\nFeature importance:")
print(feature_importance[['feature', 'coefficient']].to_string(index=False))


print("\n" + "="*80)
print("PHASE 7: Create Visualization")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Baseline (composition only)
ax = axes[0, 0]
ax.scatter(y_test, y_test_baseline, alpha=0.3, s=20)
max_val = max(y_test.max(), y_test_baseline.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax.set_xlabel('True CbGiGpPW Count')
ax.set_ylabel('Predicted (composition only)')
ax.set_title(f'Baseline (Analytical Prior): r={r_test_baseline:.3f}')
ax.grid(alpha=0.3)

# Plot 2: With sparsity
ax = axes[0, 1]
ax.scatter(y_test, y_test_full, alpha=0.3, s=20)
max_val = max(y_test.max(), y_test_full.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax.set_xlabel('True CbGiGpPW Count')
ax.set_ylabel('Predicted (comp + sparsity)')
ax.set_title(f'With Sparsity: r={r_test_full:.3f}', fontweight='bold')
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
EXPERIMENT 2H: ANALYTICAL PRIOR

Features (2 total):
  - composition_sum (analytical prior)
  - n_intermediates

Results:
  Baseline (comp only): r = {r_test_baseline:.4f}
  With sparsity:        r = {r_test_full:.4f}
  Improvement:          {r_test_full - r_test_baseline:.4f}

Status: {status}

Coefficients:
  composition_sum: {model_full.coef_[0]:.3f}
  n_intermediates: {model_full.coef_[1]:.3f}

Mean true:  {y_test.mean():.2f}
Mean pred:  {y_test_full.mean():.2f}
MAE:        {mae_full:.2f}
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig(results_dir / 'experiment2h_plots.png', dpi=150, bbox_inches='tight')

print(f"\nSaved: {results_dir / 'experiment2h_plots.png'}")


print("\n" + "="*80)
print("EXPERIMENT 2H COMPLETE")
print("="*80)
print(f"\nImprovement from sparsity: {r_test_full - r_test_baseline:.4f}")
print(f"Status: {status}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2h_results.csv'}")
print(f"  {results_dir / 'experiment2h_feature_importance.csv'}")
print(f"  {results_dir / 'experiment2h_plots.png'}")
