"""
Experiment 2J: Minimum Permutations for Cross-Permutation Generalization

Test how many training permutations are needed to achieve r>0.95
on held-out validation permutations.

Training: Mean of perms 1-K (varying K)
Validation: Mean of perms 11-20 (fixed, held-out)

Test K in {1, 2, 3, 4, 6, 8, 10} with early stopping if r>0.95

Compare:
- Baseline: composition_sum only
- With sparsity: composition_sum + n_intermediates

Note: This is not really "model training" in the ML sense - it's more like
parameter estimation from averaged observations. We're computing empirical
expectations across permutations and fitting simple linear coefficients.

Date: 2025-11-05
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
print("EXPERIMENT 2J: MINIMUM PERMUTATIONS FOR GENERALIZATION")
print("Test K permutations for training, validate on perms 11-20")
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
print("PHASE 1: Load Perm 0 Topology and Extract Features")
print("="*80)

print("\nLoading perm 0 edges for topology...")
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

print("\nSampling pairs (using perm 0 for stratification)...")
pairs = sample_pairs_stratified(CbGiGpPW_0, n_samples=10000, random_state=42)
print(f"  Sampled {len(pairs)} pairs")

print("\nExtracting features from perm 0 topology...")
print("(Features: composition_sum with analytical prior, n_intermediates)")

CbGiG_0_lil = CbGiG_0.tolil()
GpPW_0_csr = GpPW_0.tocsr()

features_list = []

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
    features_list.append(features)

X_features = np.array(features_list)

print(f"\nFeature matrix: {X_features.shape}")
print(f"  composition_sum range: [{X_features[:, 0].min():.2f}, {X_features[:, 0].max():.2f}]")
print(f"  n_intermediates range: [{X_features[:, 1].min():.0f}, {X_features[:, 1].max():.0f}]")


print("\n" + "="*80)
print("PHASE 2: Compute Validation Target (Mean of Perms 11-20)")
print("="*80)

print("\nComputing pathway counts for perms 11-20...")
val_perm_counts = []

for perm_num in range(11, 21):
    print(f"  Processing permutation {perm_num}...")

    CbG_perm = load_edge_matrix('CbG', perm_num=perm_num)
    GiG_perm = load_edge_matrix('GiG', perm_num=perm_num)
    GpPW_perm = load_edge_matrix('GpPW', perm_num=perm_num)

    if CbG_perm is None or GiG_perm is None or GpPW_perm is None:
        print(f"    WARNING: Could not load perm {perm_num}, skipping")
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


print("\n" + "="*80)
print("PHASE 3: Compute Training Targets for Each K")
print("="*80)

K_values = [1, 2, 3, 4, 6, 8, 10]
training_targets = {}

for K in K_values:
    print(f"\n--- K = {K} ---")
    print(f"Computing mean pathway counts for perms 1-{K}...")

    train_perm_counts = []

    for perm_num in range(1, K+1):
        print(f"  Loading permutation {perm_num}...")

        CbG_perm = load_edge_matrix('CbG', perm_num=perm_num)
        GiG_perm = load_edge_matrix('GiG', perm_num=perm_num)
        GpPW_perm = load_edge_matrix('GpPW', perm_num=perm_num)

        if CbG_perm is None or GiG_perm is None or GpPW_perm is None:
            print(f"    WARNING: Could not load perm {perm_num}, skipping")
            continue

        CbGiG_perm = CbG_perm @ GiG_perm
        CbGiGpPW_perm = CbGiG_perm @ GpPW_perm
        CbGiGpPW_perm_lil = CbGiGpPW_perm.tolil()

        counts = np.array([
            float(CbGiGpPW_perm_lil[C, PW]) for C, PW in pairs
        ], dtype=float)

        train_perm_counts.append(counts)

    y_train_target_K = np.mean(train_perm_counts, axis=0)

    print(f"  Training target statistics:")
    print(f"    Mean: {y_train_target_K.mean():.3f}")
    print(f"    Non-zero: {np.sum(y_train_target_K > 0)}")
    print(f"    Range: [{y_train_target_K.min():.1f}, {y_train_target_K.max():.1f}]")

    r_target_K = pearsonr(y_train_target_K, y_val_target)[0]
    print(f"  Target correlation (train vs val): r = {r_target_K:.4f}")

    training_targets[K] = {
        'y_train': y_train_target_K,
        'r_target': r_target_K,
        'n_perms': len(train_perm_counts)
    }


print("\n" + "="*80)
print("PHASE 4: Train Models for Each K")
print("="*80)

results_list = []

for K in K_values:
    print(f"\n{'='*80}")
    print(f"K = {K}: Training on mean of perms 1-{K}")
    print(f"{'='*80}")

    y_train_target_K = training_targets[K]['y_train']
    r_target_K = training_targets[K]['r_target']

    print(f"\nTarget correlation: r = {r_target_K:.4f}")

    # Split into train/test
    X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
        X_features, y_train_target_K, y_val_target, test_size=0.2, random_state=42
    )

    print(f"  Train: {len(X_train)} pairs")
    print(f"  Test: {len(X_test)} pairs")

    # Model 1: Baseline (composition only)
    print("\n  Training baseline (composition only)...")
    X_train_comp = X_train[:, 0:1]
    X_test_comp = X_test[:, 0:1]

    model_baseline = LinearRegression()
    model_baseline.fit(X_train_comp, y_train)

    y_test_pred_baseline = model_baseline.predict(X_test_comp)

    r_test_train_baseline = pearsonr(y_test_pred_baseline, y_test)[0]
    r_test_val_baseline = pearsonr(y_test_pred_baseline, y_val_test)[0]

    from sklearn.metrics import mean_absolute_error
    mae_baseline = mean_absolute_error(y_val_test, y_test_pred_baseline)

    print(f"    Test r vs train target: {r_test_train_baseline:.4f}")
    print(f"    Test r vs val target: {r_test_val_baseline:.4f}")
    print(f"    MAE: {mae_baseline:.3f}")

    # Model 2: With sparsity
    print("\n  Training with sparsity (composition + n_intermediates)...")
    model_full = LinearRegression()
    model_full.fit(X_train, y_train)

    y_test_pred_full = model_full.predict(X_test)

    r_test_train_full = pearsonr(y_test_pred_full, y_test)[0]
    r_test_val_full = pearsonr(y_test_pred_full, y_val_test)[0]

    mae_full = mean_absolute_error(y_val_test, y_test_pred_full)

    print(f"    Test r vs train target: {r_test_train_full:.4f}")
    print(f"    Test r vs val target: {r_test_val_full:.4f}")
    print(f"    MAE: {mae_full:.3f}")

    print(f"\n  Coefficients (with sparsity):")
    print(f"    composition_sum: {model_full.coef_[0]:.4f}")
    print(f"    n_intermediates: {model_full.coef_[1]:.4f}")
    print(f"    intercept: {model_full.intercept_:.4f}")

    # Save results
    mean_pred_baseline = y_test_pred_baseline.mean()
    mean_pred_full = y_test_pred_full.mean()
    mean_true = y_val_test.mean()

    results_list.append({
        'K': K,
        'n_train_perms': training_targets[K]['n_perms'],
        'r_target': r_target_K,
        'r_baseline': r_test_val_baseline,
        'r_full': r_test_val_full,
        'improvement': r_test_val_full - r_test_val_baseline,
        'mae_baseline': mae_baseline,
        'mae_full': mae_full,
        'coef_composition': model_full.coef_[0],
        'coef_n_intermediates': model_full.coef_[1],
        'intercept': model_full.intercept_,
        'mean_pred_baseline': mean_pred_baseline,
        'mean_pred_full': mean_pred_full,
        'mean_true': mean_true,
        'bias_baseline': mean_pred_baseline / mean_true,
        'bias_full': mean_pred_full / mean_true
    })

    print(f"\n  Summary:")
    print(f"    Baseline: r = {r_test_val_baseline:.4f}, MAE = {mae_baseline:.3f}")
    print(f"    With sparsity: r = {r_test_val_full:.4f}, MAE = {mae_full:.3f}")
    print(f"    Improvement: {r_test_val_full - r_test_val_baseline:.4f}")

    # Early stopping check
    if r_test_val_full > 0.95:
        print(f"\n  *** SUCCESS: r > 0.95 achieved with K = {K} ***")
        print(f"  Early stopping - no need to test larger K")
        break
    elif r_test_val_baseline > 0.95:
        print(f"\n  *** SUCCESS: Baseline r > 0.95 achieved with K = {K} ***")
        print(f"  Early stopping - no need to test larger K")
        break
    else:
        print(f"\n  Not yet at r > 0.95 threshold, continuing...")


print("\n" + "="*80)
print("PHASE 5: Summary and Analysis")
print("="*80)

df_results = pd.DataFrame(results_list)
print("\nResults Summary:")
print(df_results.to_string(index=False))

# Save results
df_results.to_csv(results_dir / 'experiment2j_results.csv', index=False)
print(f"\nSaved: {results_dir / 'experiment2j_results.csv'}")


print("\n" + "="*80)
print("PHASE 6: Create Visualizations")
print("="*80)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Convergence curves (r vs K)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_results['K'], df_results['r_baseline'], 'o-', linewidth=2,
         markersize=8, label='Baseline (composition only)')
ax1.plot(df_results['K'], df_results['r_full'], 's-', linewidth=2,
         markersize=8, label='With sparsity')
ax1.plot(df_results['K'], df_results['r_target'], '^--', linewidth=2,
         markersize=8, label='Target correlation', alpha=0.7)
ax1.axhline(y=0.95, color='red', linestyle='--', linewidth=2,
            label='Threshold (r=0.95)')
ax1.set_xlabel('Number of training permutations (K)', fontsize=11)
ax1.set_ylabel('Correlation (r)', fontsize=11)
ax1.set_title('Convergence: r vs K', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_ylim([0.7, 1.0])

# Plot 2: MAE vs K
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df_results['K'], df_results['mae_baseline'], 'o-', linewidth=2,
         markersize=8, label='Baseline')
ax2.plot(df_results['K'], df_results['mae_full'], 's-', linewidth=2,
         markersize=8, label='With sparsity')
ax2.set_xlabel('Number of training permutations (K)', fontsize=11)
ax2.set_ylabel('Mean Absolute Error', fontsize=11)
ax2.set_title('Error: MAE vs K', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Improvement from sparsity vs K
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(df_results['K'], df_results['improvement'], 'o-', linewidth=2,
         markersize=8, color='green')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Number of training permutations (K)', fontsize=11)
ax3.set_ylabel('Improvement (r_full - r_baseline)', fontsize=11)
ax3.set_title('Sparsity Benefit vs K', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Coefficient stability
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(df_results['K'], df_results['coef_composition'], 'o-', linewidth=2,
         markersize=8, label='composition_sum')
ax4.plot(df_results['K'], df_results['coef_n_intermediates'], 's-', linewidth=2,
         markersize=8, label='n_intermediates')
ax4.set_xlabel('Number of training permutations (K)', fontsize=11)
ax4.set_ylabel('Coefficient value', fontsize=11)
ax4.set_title('Coefficient Stability vs K', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Plot 5: Systematic bias
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(df_results['K'], df_results['bias_baseline'], 'o-', linewidth=2,
         markersize=8, label='Baseline')
ax5.plot(df_results['K'], df_results['bias_full'], 's-', linewidth=2,
         markersize=8, label='With sparsity')
ax5.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
            label='Perfect calibration')
ax5.set_xlabel('Number of training permutations (K)', fontsize=11)
ax5.set_ylabel('Systematic bias (mean_pred / mean_true)', fontsize=11)
ax5.set_title('Calibration: Bias vs K', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# Plot 6: Target correlation vs K
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(df_results['K'], df_results['r_target'], '^-', linewidth=2,
         markersize=8, color='purple')
ax6.axhline(y=0.95, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('Number of training permutations (K)', fontsize=11)
ax6.set_ylabel('Correlation', fontsize=11)
ax6.set_title('Target Correlation: mean(1-K) vs mean(11-20)',
              fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)
ax6.set_ylim([0.7, 1.0])

# Plot 7-9: Scatter plots for K=1, middle K, and max K tested
tested_K = df_results['K'].values
scatter_K_values = [tested_K[0], tested_K[len(tested_K)//2], tested_K[-1]]
scatter_positions = [(2, 0), (2, 1), (2, 2)]

for k_val, (row, col) in zip(scatter_K_values, scatter_positions):
    ax = fig.add_subplot(gs[row, col])

    # Re-compute predictions for this K for visualization
    # (We'll just use the stored r values and create a simple plot)
    result_row = df_results[df_results['K'] == k_val].iloc[0]

    ax.text(0.5, 0.5, f"K = {k_val}\n\n" +
            f"Baseline: r = {result_row['r_baseline']:.3f}\n" +
            f"With sparsity: r = {result_row['r_full']:.3f}\n" +
            f"Improvement: {result_row['improvement']:.3f}\n\n" +
            f"MAE: {result_row['mae_full']:.3f}",
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Summary: K = {k_val}', fontsize=12, fontweight='bold')

plt.savefig(results_dir / 'experiment2j_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'experiment2j_analysis.png'}")


print("\n" + "="*80)
print("EXPERIMENT 2J COMPLETE")
print("="*80)

print("\nKey Findings:")
print(f"  K values tested: {list(df_results['K'])}")
print(f"\n  Best performance:")
best_idx = df_results['r_full'].idxmax()
best_K = df_results.loc[best_idx, 'K']
best_r = df_results.loc[best_idx, 'r_full']
print(f"    K = {best_K}: r = {best_r:.4f}")

if best_r > 0.95:
    print(f"\n  SUCCESS: Achieved r > 0.95 with K = {best_K}")
    print(f"  Minimum permutations needed: {best_K}")
else:
    print(f"\n  FAILURE: Did not achieve r > 0.95")
    print(f"  Best correlation: r = {best_r:.4f} at K = {best_K}")

print(f"\n  Sparsity effect:")
print(f"    Consistent improvement: {df_results['improvement'].mean():.4f} ± {df_results['improvement'].std():.4f}")

print(f"\n  Target correlation improvement:")
print(f"    K=1: r = {df_results.loc[0, 'r_target']:.4f}")
print(f"    K={best_K}: r = {df_results.loc[best_idx, 'r_target']:.4f}")

print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2j_results.csv'}")
print(f"  {results_dir / 'experiment2j_analysis.png'}")
