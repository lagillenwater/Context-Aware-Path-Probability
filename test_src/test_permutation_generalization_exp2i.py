"""
Experiment 2I: Permutation Generalization Test

Test whether models trained on perm 0 generalize to predict
mean pathway counts across permutations 5-20.

Training: Perm 0 data with analytical prior
Validation: Mean of perms 5-20

Compare:
- Baseline: composition_sum only
- With sparsity: composition_sum + n_intermediates

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
print("EXPERIMENT 2I: PERMUTATION GENERALIZATION TEST")
print("Train on perm 0, validate on mean of perms 5-20")
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
    """Analytical edge probability from configuration model."""
    if m == 0:
        return 0.0

    expected_multiedges = deg_u * deg_v / m
    if expected_multiedges > 10:
        return 1.0

    return 1.0 - np.exp(-expected_multiedges)


print("\n" + "="*80)
print("PHASE 1: Load Perm 0 Data and Extract Features")
print("="*80)

print("\nLoading perm 0 edges...")
CbG_0 = load_edge_matrix('CbG', perm_num=0)
GiG_0 = load_edge_matrix('GiG', perm_num=0)
GpPW_0 = load_edge_matrix('GpPW', perm_num=0)

print(f"  CbG: {CbG_0.shape}, {CbG_0.nnz:,} edges")
print(f"  GiG: {GiG_0.shape}, {GiG_0.nnz:,} edges")
print(f"  GpPW: {GpPW_0.shape}, {GpPW_0.nnz:,} edges")

CbGiG_0 = CbG_0 @ GiG_0
CbGiGpPW_0 = CbGiG_0 @ GpPW_0

compound_degrees = np.array(CbG_0.sum(axis=1)).flatten()
gene_degrees = np.array(GiG_0.sum(axis=1)).flatten()
pathway_degrees = np.array(GpPW_0.sum(axis=0)).flatten()

n_genes = GpPW_0.shape[0]
n_pathways = GpPW_0.shape[1]
m_GpPW = GpPW_0.nnz

print("\nSampling pairs...")
pairs = sample_pairs_stratified(CbGiGpPW_0, n_samples=10000, random_state=42)
print(f"  Sampled {len(pairs)} pairs")

print("\nExtracting features from perm 0...")
CbGiG_0_lil = CbGiG_0.tolil()
CbGiGpPW_0_lil = CbGiGpPW_0.tolil()
GpPW_0_csr = GpPW_0.tocsr()

features_list = []
y_train_target = []  # Perm 0 counts

for idx, (C, PW) in enumerate(pairs):
    if idx % 2000 == 0:
        print(f"  Processed {idx}/{len(pairs)} pairs...")

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

    features = np.array([composition_sum, n_intermediates], dtype=np.float64)
    target_perm0 = float(CbGiGpPW_0_lil[C, PW])

    features_list.append(features)
    y_train_target.append(target_perm0)

X_features = np.array(features_list)
y_train_target = np.array(y_train_target)

print(f"\nFeature matrix: {X_features.shape}")


print("\n" + "="*80)
print("PHASE 2: Compute Validation Target (Mean of Perms 5-20)")
print("="*80)

print("\nComputing pathway counts for perms 5-20...")
val_perm_counts = []

for perm_num in [5, 10, 15, 20]:
    print(f"  Processing permutation {perm_num}...")

    CbG_perm = load_edge_matrix('CbG', perm_num=perm_num)
    GiG_perm = load_edge_matrix('GiG', perm_num=perm_num)
    GpPW_perm = load_edge_matrix('GpPW', perm_num=perm_num)

    if CbG_perm is None or GiG_perm is None or GpPW_perm is None:
        continue

    CbGiG_perm = CbG_perm @ GiG_perm
    CbGiGpPW_perm = CbGiG_perm @ GpPW_perm
    CbGiGpPW_perm_lil = CbGiGpPW_perm.tolil()

    counts = np.array([
        float(CbGiGpPW_perm_lil[C, PW]) for C, PW in pairs
    ], dtype=float)

    val_perm_counts.append(counts)

y_val_target = np.mean(val_perm_counts, axis=0)

print(f"\nValidation target (mean of {len(val_perm_counts)} perms):")
print(f"  Shape: {y_val_target.shape}")
print(f"  Non-zero: {np.sum(y_val_target > 0)}")
print(f"  Range: [{y_val_target.min():.1f}, {y_val_target.max():.1f}]")
print(f"  Mean: {y_val_target.mean():.3f}")

r_targets = pearsonr(y_train_target, y_val_target)[0]
print(f"\nTarget correlation (perm 0 vs mean 5-20): r = {r_targets:.4f}")


print("\n" + "="*80)
print("PHASE 3: Train Models on Perm 0")
print("="*80)

print("\nSplitting into train/test (80/20)...")
X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
    X_features, y_train_target, y_val_target, test_size=0.2, random_state=42
)

print(f"  Train: {len(X_train)} pairs")
print(f"  Test: {len(X_test)} pairs")

# Model 1: Baseline (composition only)
print("\nTraining baseline (composition only) on PERM 0...")
X_train_comp = X_train[:, 0:1]
X_test_comp = X_test[:, 0:1]

model_baseline = LinearRegression()
model_baseline.fit(X_train_comp, y_train)

y_train_pred_baseline = model_baseline.predict(X_train_comp)
y_test_pred_baseline = model_baseline.predict(X_test_comp)

r_train_train_baseline = pearsonr(y_train_pred_baseline, y_train)[0]
r_train_val_baseline = pearsonr(y_train_pred_baseline, y_val_train)[0]
r_test_train_baseline = pearsonr(y_test_pred_baseline, y_test)[0]
r_test_val_baseline = pearsonr(y_test_pred_baseline, y_val_test)[0]

print(f"  Train r vs perm 0: {r_train_train_baseline:.4f}")
print(f"  Train r vs mean 5-20: {r_train_val_baseline:.4f}")
print(f"  Test r vs perm 0: {r_test_train_baseline:.4f}")
print(f"  Test r vs mean 5-20: {r_test_val_baseline:.4f}")

# Model 2: With sparsity
print("\nTraining with sparsity (composition + n_intermediates) on PERM 0...")
model_full = LinearRegression()
model_full.fit(X_train, y_train)

y_train_pred_full = model_full.predict(X_train)
y_test_pred_full = model_full.predict(X_test)

r_train_train_full = pearsonr(y_train_pred_full, y_train)[0]
r_train_val_full = pearsonr(y_train_pred_full, y_val_train)[0]
r_test_train_full = pearsonr(y_test_pred_full, y_test)[0]
r_test_val_full = pearsonr(y_test_pred_full, y_val_test)[0]

print(f"  Train r vs perm 0: {r_train_train_full:.4f}")
print(f"  Train r vs mean 5-20: {r_train_val_full:.4f}")
print(f"  Test r vs perm 0: {r_test_train_full:.4f}")
print(f"  Test r vs mean 5-20: {r_test_val_full:.4f}")


print("\n" + "="*80)
print("PHASE 4: Performance Comparison")
print("="*80)

print("\nValidation Performance (Test set, Mean of Perms 5-20):")
print(f"  Baseline (composition only):  r = {r_test_val_baseline:.4f}")
print(f"  With sparsity (+ n_inter):    r = {r_test_val_full:.4f}")
print(f"  Improvement:                   {r_test_val_full - r_test_val_baseline:.4f}")

if r_test_val_baseline > 0.95 and r_test_val_full > 0.95:
    status = "BOTH SUCCESS"
    print(f"\nSTATUS: BOTH SUCCESS (both > 0.95)")
elif r_test_val_full > 0.95:
    status = "SPARSITY SUCCESS"
    print(f"\nSTATUS: SPARSITY NEEDED (sparsity reaches > 0.95)")
elif r_test_val_baseline > 0.95:
    status = "BASELINE SUCCESS"
    print(f"\nSTATUS: BASELINE SUCCESS (baseline alone > 0.95)")
else:
    status = "BOTH FAIL"
    print(f"\nSTATUS: BOTH FAIL (neither reaches > 0.95)")


print("\n" + "="*80)
print("PHASE 5: Residual Analysis")
print("="*80)

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Compute residuals
residual_baseline = y_val_test - y_test_pred_baseline
residual_full = y_val_test - y_test_pred_full

mae_baseline = mean_absolute_error(y_val_test, y_test_pred_baseline)
mae_full = mean_absolute_error(y_val_test, y_test_pred_full)

rmse_baseline = np.sqrt(mean_squared_error(y_val_test, y_test_pred_baseline))
rmse_full = np.sqrt(mean_squared_error(y_val_test, y_test_pred_full))

print(f"\nBaseline (composition only):")
print(f"  MAE: {mae_baseline:.4f}")
print(f"  RMSE: {rmse_baseline:.4f}")
print(f"  Mean pred: {y_test_pred_baseline.mean():.4f}")

print(f"\nWith sparsity:")
print(f"  MAE: {mae_full:.4f}")
print(f"  RMSE: {rmse_full:.4f}")
print(f"  Mean pred: {y_test_pred_full.mean():.4f}")

print(f"\nValidation target:")
print(f"  Mean: {y_val_test.mean():.4f}")

# Residuals by sparsity bin
df_test = pd.DataFrame({
    'n_intermediates': X_test[:, 1],
    'composition_sum': X_test[:, 0],
    'y_val': y_val_test,
    'y_perm0': y_test,
    'pred_baseline': y_test_pred_baseline,
    'pred_full': y_test_pred_full,
    'residual_baseline': residual_baseline,
    'residual_full': residual_full
})

df_test['sparsity_bin'] = pd.cut(df_test['n_intermediates'],
                                   bins=[0, 1, 3, 5, 10, 100],
                                   labels=['0-1', '2-3', '4-5', '6-10', '10+'])

print("\nResiduals by sparsity (n_intermediates):")
print()
print(f"{'Sparsity':10s} {'N':>6s} {'Baseline MAE':>14s} {'Full MAE':>10s} {'Reduction':>12s}")
print("-" * 56)

for sparsity_bin in ['0-1', '2-3', '4-5', '6-10', '10+']:
    mask = df_test['sparsity_bin'] == sparsity_bin
    if mask.sum() < 10:
        continue

    subset = df_test[mask]
    n = len(subset)

    mae_b = np.abs(subset['residual_baseline']).mean()
    mae_f = np.abs(subset['residual_full']).mean()

    print(f"{sparsity_bin:10s} {n:6d} {mae_b:14.2f} {mae_f:10.2f} {mae_b - mae_f:12.2f}")


print("\n" + "="*80)
print("PHASE 6: Save Results")
print("="*80)

results = {
    'experiment': 'Experiment 2I',
    'description': 'Permutation generalization test',
    'n_pairs': len(pairs),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'r_targets': r_targets,
    'baseline_r_test_perm0': r_test_train_baseline,
    'baseline_r_test_val': r_test_val_baseline,
    'full_r_test_perm0': r_test_train_full,
    'full_r_test_val': r_test_val_full,
    'improvement': r_test_val_full - r_test_val_baseline,
    'baseline_mae': mae_baseline,
    'full_mae': mae_full,
    'baseline_rmse': rmse_baseline,
    'full_rmse': rmse_full,
    'status': status
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2i_results.csv', index=False)

df_test.to_csv(results_dir / 'experiment2i_test_predictions.csv', index=False)

print(f"\nSaved: {results_dir / 'experiment2i_results.csv'}")
print(f"Saved: {results_dir / 'experiment2i_test_predictions.csv'}")


print("\n" + "="*80)
print("PHASE 7: Create Visualizations")
print("="*80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Baseline predictions vs validation target
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_val_test, y_test_pred_baseline, alpha=0.3, s=20)
max_val = max(y_val_test.max(), y_test_pred_baseline.max())
ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax1.set_xlabel('True (Mean Perms 5-20)')
ax1.set_ylabel('Predicted (Baseline)')
ax1.set_title(f'Baseline: r={r_test_val_baseline:.3f}')
ax1.grid(alpha=0.3)

# Plot 2: Full model predictions vs validation target
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_val_test, y_test_pred_full, alpha=0.3, s=20)
max_val = max(y_val_test.max(), y_test_pred_full.max())
ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax2.set_xlabel('True (Mean Perms 5-20)')
ax2.set_ylabel('Predicted (With Sparsity)')
ax2.set_title(f'With Sparsity: r={r_test_val_full:.3f}', fontweight='bold')
ax2.grid(alpha=0.3)

# Plot 3: Target correlation (perm 0 vs mean 5-20)
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_val_test, y_test, alpha=0.3, s=20, c='purple')
max_val = max(y_val_test.max(), y_test.max())
ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax3.set_xlabel('Mean Perms 5-20')
ax3.set_ylabel('Perm 0')
ax3.set_title(f'Target Correlation: r={r_targets:.3f}')
ax3.grid(alpha=0.3)

# Plot 4: Baseline residuals vs sparsity
ax4 = fig.add_subplot(gs[1, 0])
scatter = ax4.scatter(df_test['n_intermediates'], df_test['residual_baseline'],
                      alpha=0.3, s=20, c=df_test['y_val'], cmap='viridis')
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('n_intermediates')
ax4.set_ylabel('Residual (true - pred)')
ax4.set_title('Baseline Residuals vs Sparsity')
ax4.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='True count')

# Plot 5: Full model residuals vs sparsity
ax5 = fig.add_subplot(gs[1, 1])
scatter = ax5.scatter(df_test['n_intermediates'], df_test['residual_full'],
                      alpha=0.3, s=20, c=df_test['y_val'], cmap='viridis')
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('n_intermediates')
ax5.set_ylabel('Residual (true - pred)')
ax5.set_title('With Sparsity: Residuals vs Sparsity')
ax5.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax5, label='True count')

# Plot 6: MAE by sparsity bin
ax6 = fig.add_subplot(gs[1, 2])
sparsity_bins = ['0-1', '2-3', '4-5', '6-10', '10+']
mae_baseline_by_bin = []
mae_full_by_bin = []

for sparsity_bin in sparsity_bins:
    mask = df_test['sparsity_bin'] == sparsity_bin
    if mask.sum() < 10:
        mae_baseline_by_bin.append(np.nan)
        mae_full_by_bin.append(np.nan)
    else:
        subset = df_test[mask]
        mae_baseline_by_bin.append(np.abs(subset['residual_baseline']).mean())
        mae_full_by_bin.append(np.abs(subset['residual_full']).mean())

x = np.arange(len(sparsity_bins))
width = 0.35

ax6.bar(x - width/2, mae_baseline_by_bin, width, label='Baseline', alpha=0.8)
ax6.bar(x + width/2, mae_full_by_bin, width, label='With Sparsity', alpha=0.8)
ax6.set_xlabel('Sparsity bin')
ax6.set_ylabel('Mean Absolute Error')
ax6.set_title('MAE by Sparsity (Validation Target)')
ax6.set_xticks(x)
ax6.set_xticklabels(sparsity_bins)
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

# Plot 7: Residual distributions
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(df_test['residual_baseline'], bins=50, alpha=0.5, label='Baseline', density=True)
ax7.hist(df_test['residual_full'], bins=50, alpha=0.5, label='With Sparsity', density=True)
ax7.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax7.set_xlabel('Residual (true - pred)')
ax7.set_ylabel('Density')
ax7.set_title('Residual Distributions')
ax7.legend()
ax7.grid(alpha=0.3)

# Plot 8: Q-Q plot baseline
ax8 = fig.add_subplot(gs[2, 1])
from scipy import stats
stats.probplot(df_test['residual_baseline'], dist="norm", plot=ax8)
ax8.set_title('Q-Q Plot: Baseline Residuals')
ax8.grid(alpha=0.3)

# Plot 9: Q-Q plot full
ax9 = fig.add_subplot(gs[2, 2])
stats.probplot(df_test['residual_full'], dist="norm", plot=ax9)
ax9.set_title('Q-Q Plot: With Sparsity Residuals')
ax9.grid(alpha=0.3)

plt.savefig(results_dir / 'experiment2i_analysis.png', dpi=150, bbox_inches='tight')

print(f"\nSaved: {results_dir / 'experiment2i_analysis.png'}")


print("\n" + "="*80)
print("EXPERIMENT 2I COMPLETE")
print("="*80)
print(f"\nValidation Performance (Mean of Perms 5-20):")
print(f"  Baseline: r = {r_test_val_baseline:.4f}")
print(f"  With sparsity: r = {r_test_val_full:.4f}")
print(f"  Improvement: {r_test_val_full - r_test_val_baseline:.4f}")
print(f"\nStatus: {status}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2i_results.csv'}")
print(f"  {results_dir / 'experiment2i_test_predictions.csv'}")
print(f"  {results_dir / 'experiment2i_analysis.png'}")
