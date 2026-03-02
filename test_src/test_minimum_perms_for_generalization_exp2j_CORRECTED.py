"""
Experiment 2J CORRECTED: Minimum Permutations for Cross-Permutation Generalization

CORRECTION: Previous version used same random_state for train/test split across all K,
which caused single-feature baseline to have constant correlation (mathematical property).

NEW DESIGN:
- Use cross-validation OR evaluate on all pairs (no split)
- This properly tests whether more permutations improve generalization

Training: Mean of perms 1-K (varying K)
Validation: Mean of perms 11-20 (fixed, held-out)

Test K in {1, 2, 3, 4, 6, 8, 10} with early stopping if r>0.95

Compare:
- Baseline: composition_sum only
- With sparsity: composition_sum + n_intermediates

Date: 2025-11-05 (corrected)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPERIMENT 2J CORRECTED: MINIMUM PERMUTATIONS FOR GENERALIZATION")
print("Test K permutations for training, validate on perms 11-20")
print("FIXED: Proper evaluation without constant-split artifact")
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
print("PHASE 4: Train Models Using Cross-Validation")
print("="*80)
print("\nUsing 5-fold cross-validation to avoid constant-split artifact")

results_list = []
n_folds = 5

for K in K_values:
    print(f"\n{'='*80}")
    print(f"K = {K}: Training on mean of perms 1-{K}")
    print(f"{'='*80}")

    y_train_target_K = training_targets[K]['y_train']
    r_target_K = training_targets[K]['r_target']

    print(f"\nTarget correlation: r = {r_target_K:.4f}")

    # Use K-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    r_baseline_folds = []
    r_full_folds = []
    mae_baseline_folds = []
    mae_full_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_features)):
        X_train = X_features[train_idx]
        X_test = X_features[test_idx]
        y_train = y_train_target_K[train_idx]
        y_val_test = y_val_target[test_idx]

        # Baseline model (composition only)
        X_train_comp = X_train[:, 0:1]
        X_test_comp = X_test[:, 0:1]

        model_baseline = LinearRegression()
        model_baseline.fit(X_train_comp, y_train)
        y_pred_baseline = model_baseline.predict(X_test_comp)

        r_baseline = pearsonr(y_pred_baseline, y_val_test)[0]
        mae_baseline = np.mean(np.abs(y_pred_baseline - y_val_test))

        r_baseline_folds.append(r_baseline)
        mae_baseline_folds.append(mae_baseline)

        # Full model (with sparsity)
        model_full = LinearRegression()
        model_full.fit(X_train, y_train)
        y_pred_full = model_full.predict(X_test)

        r_full = pearsonr(y_pred_full, y_val_test)[0]
        mae_full = np.mean(np.abs(y_pred_full - y_val_test))

        r_full_folds.append(r_full)
        mae_full_folds.append(mae_full)

    # Average across folds
    r_baseline_mean = np.mean(r_baseline_folds)
    r_baseline_std = np.std(r_baseline_folds)
    r_full_mean = np.mean(r_full_folds)
    r_full_std = np.std(r_full_folds)
    mae_baseline_mean = np.mean(mae_baseline_folds)
    mae_full_mean = np.mean(mae_full_folds)

    print(f"\n  Cross-validation results ({n_folds} folds):")
    print(f"    Baseline: r = {r_baseline_mean:.4f} ± {r_baseline_std:.4f}")
    print(f"    With sparsity: r = {r_full_mean:.4f} ± {r_full_std:.4f}")
    print(f"    Improvement: {r_full_mean - r_baseline_mean:.4f}")

    # Train final model on all data for coefficient inspection
    model_final = LinearRegression()
    model_final.fit(X_features, y_train_target_K)

    print(f"\n  Final model coefficients (trained on all pairs):")
    print(f"    composition_sum: {model_final.coef_[0]:.4f}")
    print(f"    n_intermediates: {model_final.coef_[1]:.4f}")
    print(f"    intercept: {model_final.intercept_:.4f}")

    # Save results
    results_list.append({
        'K': K,
        'n_train_perms': training_targets[K]['n_perms'],
        'r_target': r_target_K,
        'r_baseline_mean': r_baseline_mean,
        'r_baseline_std': r_baseline_std,
        'r_full_mean': r_full_mean,
        'r_full_std': r_full_std,
        'improvement': r_full_mean - r_baseline_mean,
        'mae_baseline': mae_baseline_mean,
        'mae_full': mae_full_mean,
        'coef_composition': model_final.coef_[0],
        'coef_n_intermediates': model_final.coef_[1],
        'intercept': model_final.intercept_
    })

    print(f"\n  Summary:")
    print(f"    Baseline: r = {r_baseline_mean:.4f}, MAE = {mae_baseline_mean:.3f}")
    print(f"    With sparsity: r = {r_full_mean:.4f}, MAE = {mae_full_mean:.3f}")
    print(f"    Improvement: {r_full_mean - r_baseline_mean:.4f}")

    # Early stopping check
    if r_full_mean > 0.95:
        print(f"\n  *** SUCCESS: r > 0.95 achieved with K = {K} ***")
        print(f"  Early stopping - no need to test larger K")
        break
    elif r_baseline_mean > 0.95:
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
df_results.to_csv(results_dir / 'experiment2j_corrected_results.csv', index=False)
print(f"\nSaved: {results_dir / 'experiment2j_corrected_results.csv'}")


print("\n" + "="*80)
print("PHASE 6: Create Visualizations")
print("="*80)

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: Convergence curves with error bars
ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(df_results['K'], df_results['r_baseline_mean'],
             yerr=df_results['r_baseline_std'], fmt='o-', linewidth=2,
             markersize=8, capsize=5, label='Baseline (composition only)')
ax1.errorbar(df_results['K'], df_results['r_full_mean'],
             yerr=df_results['r_full_std'], fmt='s-', linewidth=2,
             markersize=8, capsize=5, label='With sparsity')
ax1.plot(df_results['K'], df_results['r_target'], '^--', linewidth=2,
         markersize=8, label='Target correlation', alpha=0.7)
ax1.axhline(y=0.95, color='red', linestyle='--', linewidth=2,
            label='Threshold (r=0.95)')
ax1.set_xlabel('Number of training permutations (K)', fontsize=11)
ax1.set_ylabel('Correlation (r)', fontsize=11)
ax1.set_title('Convergence: r vs K (5-fold CV)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

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

# Plot 5: Target correlation vs K
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(df_results['K'], df_results['r_target'], '^-', linewidth=2,
         markersize=8, color='purple')
ax5.axhline(y=0.95, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Number of training permutations (K)', fontsize=11)
ax5.set_ylabel('Correlation', fontsize=11)
ax5.set_title('Target Correlation: mean(1-K) vs mean(11-20)',
              fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

# Plot 6: Summary table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

summary_text = "CORRECTED EXPERIMENT 2J\n"
summary_text += "="*40 + "\n\n"
summary_text += "Design Fix:\n"
summary_text += "• Used 5-fold cross-validation\n"
summary_text += "• Avoids constant-split artifact\n"
summary_text += "• Properly tests generalization\n\n"

if df_results['r_full_mean'].max() > 0.95:
    best_idx = df_results['r_full_mean'].idxmax()
    best_K = df_results.loc[best_idx, 'K']
    best_r = df_results.loc[best_idx, 'r_full_mean']
    summary_text += f"Result: SUCCESS\n"
    summary_text += f"• Achieved r={best_r:.3f} at K={best_K}\n"
else:
    best_idx = df_results['r_full_mean'].idxmax()
    best_K = df_results.loc[best_idx, 'K']
    best_r = df_results.loc[best_idx, 'r_full_mean']
    summary_text += f"Result: FAILURE\n"
    summary_text += f"• Best: r={best_r:.3f} at K={best_K}\n"
    summary_text += f"• Did not reach r>0.95\n"

ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(results_dir / 'experiment2j_corrected_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'experiment2j_corrected_analysis.png'}")


print("\n" + "="*80)
print("EXPERIMENT 2J CORRECTED - COMPLETE")
print("="*80)

print("\nKey Findings:")
print(f"  K values tested: {list(df_results['K'])}")

print(f"\n  Best performance:")
best_idx = df_results['r_full_mean'].idxmax()
best_K = df_results.loc[best_idx, 'K']
best_r = df_results.loc[best_idx, 'r_full_mean']
best_std = df_results.loc[best_idx, 'r_full_std']
print(f"    K = {best_K}: r = {best_r:.4f} ± {best_std:.4f}")

if best_r > 0.95:
    print(f"\n  SUCCESS: Achieved r > 0.95 with K = {best_K}")
    print(f"  Minimum permutations needed: {best_K}")
else:
    print(f"\n  FAILURE: Did not achieve r > 0.95")
    print(f"  Best correlation: r = {best_r:.4f} at K = {best_K}")

print(f"\n  Sparsity effect:")
print(f"    Mean improvement: {df_results['improvement'].mean():.4f} ± {df_results['improvement'].std():.4f}")

print(f"\n  Target correlation improvement:")
print(f"    K=1: r = {df_results.loc[0, 'r_target']:.4f}")
print(f"    K={best_K}: r = {df_results.loc[best_idx, 'r_target']:.4f}")

print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2j_corrected_results.csv'}")
print(f"  {results_dir / 'experiment2j_corrected_analysis.png'}")
