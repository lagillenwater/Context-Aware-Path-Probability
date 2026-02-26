"""
Pair-Level Phase 2: Detailed Analysis with Visualizations

Generates comprehensive visualizations for documentation:
1. Predicted vs observed pathway counts (before/after correction)
2. Residual distributions (before/after correction)
3. Residuals vs predicted values (heteroscedasticity check)
4. Correction magnitude analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Setup
repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'pair_level_phase2_detailed'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PAIR-LEVEL PHASE 2: DETAILED ANALYSIS")
print("="*80)

# Configuration
METAPATHS = [
    ('CbG', 'GpPW', 'CbGpPW'),
    ('CtD', 'DaG', 'CtDaG'),
    ('CrC', 'CbG', 'CrCbG')
]
N_SAMPLES = 10000
PERM_VALIDATION = list(range(1, 21))

# Helper functions
def load_edge_matrix(edge_type, perm_id='original'):
    if perm_id == 'original':
        edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
    # Convert from bool to int to enable pathway counting
    return sp.load_npz(edge_file).astype(np.int32)

def extract_pair_features_simple(edge1, edge2, source_idx, target_idx):
    source_deg = edge1.getrow(source_idx).nnz
    target_deg = edge2.getcol(target_idx).nnz
    return np.array([source_deg, target_deg, source_deg * target_deg,
                     source_deg ** 2, target_deg ** 2])

def sample_pairs_by_pathway_count(edge1, edge2, n_samples, random_state=42):
    np.random.seed(random_state)
    pathway_matrix = edge1 @ edge2
    sources, targets = pathway_matrix.nonzero()

    zero_pairs = min(int(n_samples * 0.5), len(sources))
    if len(sources) > zero_pairs:
        sample_idx = np.random.choice(len(sources), zero_pairs, replace=False)
        sampled_sources = sources[sample_idx]
        sampled_targets = targets[sample_idx]
    else:
        sampled_sources = sources
        sampled_targets = targets

    n_zero = n_samples - len(sampled_sources)
    if n_zero > 0:
        all_sources = np.arange(edge1.shape[0])
        all_targets = np.arange(edge2.shape[1])
        zero_sources = np.random.choice(all_sources, n_zero)
        zero_targets = np.random.choice(all_targets, n_zero)
        sampled_sources = np.concatenate([sampled_sources, zero_sources])
        sampled_targets = np.concatenate([sampled_targets, zero_targets])

    return list(zip(sampled_sources, sampled_targets))

def compute_pathway_counts_for_pairs(edge1, edge2, pair_indices):
    pathway_matrix = edge1 @ edge2
    counts = np.zeros(len(pair_indices))
    for i, (source_idx, target_idx) in enumerate(pair_indices):
        counts[i] = pathway_matrix[source_idx, target_idx]
    return counts

def extract_correction_features(X, y_pred):
    source_deg = X[:, 0]
    target_deg = X[:, 1]
    features = [
        source_deg, target_deg, source_deg * target_deg,
        source_deg ** 2, target_deg ** 2,
        np.sqrt(source_deg + 1), np.sqrt(target_deg + 1),
        y_pred, y_pred ** 2, np.log1p(y_pred),
        y_pred * source_deg, y_pred * target_deg,
        y_pred * source_deg * target_deg,
        np.sqrt(y_pred + 1) * source_deg,
        np.sqrt(y_pred + 1) * target_deg
    ]
    return np.column_stack(features)

def create_detailed_visualizations(metapath_name, X, y_validation, y_perm0,
                                   y_pred_base, y_pred_corrected, correction):
    """Create comprehensive visualizations for one metapath."""

    # Calculate residuals
    residuals_base = y_validation - y_pred_base
    residuals_corrected = y_validation - y_pred_corrected

    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle(f'{metapath_name}: Detailed Model Analysis', fontsize=16, fontweight='bold')

    # Row 1: Predicted vs Observed
    # (1,1) Baseline
    ax = axes[0, 0]
    ax.scatter(y_validation, y_pred_base, alpha=0.3, s=10)
    max_val = max(y_validation.max(), y_pred_base.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect prediction')
    r_base = pearsonr(y_pred_base, y_validation)[0]
    rmse_base = np.sqrt(np.mean(residuals_base**2))
    ax.set_xlabel('True Pathway Count', fontsize=11)
    ax.set_ylabel('Predicted Pathway Count', fontsize=11)
    ax.set_title(f'Baseline Model\nr = {r_base:.4f}, RMSE = {rmse_base:.4f}', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,2) Corrected
    ax = axes[0, 1]
    ax.scatter(y_validation, y_pred_corrected, alpha=0.3, s=10, color='green')
    max_val = max(y_validation.max(), y_pred_corrected.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect prediction')
    r_corr = pearsonr(y_pred_corrected, y_validation)[0]
    rmse_corr = np.sqrt(np.mean(residuals_corrected**2))
    ax.set_xlabel('True Pathway Count', fontsize=11)
    ax.set_ylabel('Predicted Pathway Count', fontsize=11)
    ax.set_title(f'Corrected Model\nr = {r_corr:.4f}, RMSE = {rmse_corr:.4f}', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,3) Comparison
    ax = axes[0, 2]
    x = np.arange(2)
    metrics = pd.DataFrame({
        'Correlation': [r_base, r_corr],
        'RMSE': [rmse_base, rmse_corr]
    }, index=['Baseline', 'Corrected'])

    ax2 = ax.twinx()
    width = 0.35
    ax.bar(x - width/2, metrics['Correlation'], width, label='Correlation', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, metrics['RMSE'], width, label='RMSE', alpha=0.8, color='coral')

    ax.set_ylabel('Correlation (r)', fontsize=11, color='steelblue')
    ax2.set_ylabel('RMSE', fontsize=11, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', 'Corrected'])
    ax.set_title('Performance Metrics', fontsize=12)
    ax.set_ylim([0.8, 1.0])
    ax.axhline(0.90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (r=0.90)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # Row 2: Residuals vs Predicted
    # (2,1) Baseline residuals
    ax = axes[1, 0]
    ax.scatter(y_pred_base, residuals_base, alpha=0.3, s=10)
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Pathway Count', fontsize=11)
    ax.set_ylabel('Residuals (True - Predicted)', fontsize=11)
    ax.set_title(f'Baseline Residuals\nBias = {residuals_base.mean():+.4f}', fontsize=12)
    ax.grid(alpha=0.3)

    # Check heteroscedasticity
    abs_residuals = np.abs(residuals_base)
    r_hetero_base = pearsonr(y_pred_base, abs_residuals)[0]
    ax.text(0.05, 0.95, f'Heteroscedasticity: r = {r_hetero_base:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (2,2) Corrected residuals
    ax = axes[1, 1]
    ax.scatter(y_pred_corrected, residuals_corrected, alpha=0.3, s=10, color='green')
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Pathway Count', fontsize=11)
    ax.set_ylabel('Residuals (True - Predicted)', fontsize=11)
    ax.set_title(f'Corrected Residuals\nBias = {residuals_corrected.mean():+.4f}', fontsize=12)
    ax.grid(alpha=0.3)

    # Check heteroscedasticity
    abs_residuals_corr = np.abs(residuals_corrected)
    r_hetero_corr = pearsonr(y_pred_corrected, abs_residuals_corr)[0]
    ax.text(0.05, 0.95, f'Heteroscedasticity: r = {r_hetero_corr:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (2,3) Residual comparison
    ax = axes[1, 2]
    bins = np.linspace(min(residuals_base.min(), residuals_corrected.min()),
                       max(residuals_base.max(), residuals_corrected.max()), 50)
    ax.hist(residuals_base, bins=bins, alpha=0.5, label='Baseline', density=True)
    ax.hist(residuals_corrected, bins=bins, alpha=0.5, label='Corrected', density=True, color='green')
    ax.axvline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residuals', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Residual Distributions', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # Row 3: Correction Analysis
    # (3,1) Correction magnitude
    ax = axes[2, 0]
    ax.scatter(y_pred_base, correction, alpha=0.3, s=10, color='purple')
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Base Prediction', fontsize=11)
    ax.set_ylabel('Correction Applied', fontsize=11)
    ax.set_title(f'Correction Magnitude\nMean = {correction.mean():+.4f}, Std = {correction.std():.4f}', fontsize=12)
    ax.grid(alpha=0.3)

    # (3,2) Correction vs source degree
    ax = axes[2, 1]
    source_deg = X[:, 0]
    ax.scatter(source_deg, correction, alpha=0.3, s=10, color='purple')
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Source Node Degree', fontsize=11)
    ax.set_ylabel('Correction Applied', fontsize=11)
    ax.set_title('Correction vs Source Degree', fontsize=12)
    ax.grid(alpha=0.3)

    # (3,3) Permutation 0 vs validation
    ax = axes[2, 2]
    ax.scatter(y_validation, y_perm0, alpha=0.3, s=10, color='orange')
    max_val = max(y_validation.max(), y_perm0.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect agreement')
    r_perm = pearsonr(y_perm0, y_validation)[0]
    ax.set_xlabel('Validation (Perms 1-20 avg)', fontsize=11)
    ax.set_ylabel('Permutation 000', fontsize=11)
    ax.set_title(f'Perm 000 vs Validation\nr = {r_perm:.4f}', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / f'{metapath_name}_detailed_analysis.png', dpi=200, bbox_inches='tight')
    print(f"    Saved: {metapath_name}_detailed_analysis.png")

    return {
        'r_base': r_base,
        'r_corrected': r_corr,
        'rmse_base': rmse_base,
        'rmse_corrected': rmse_corr,
        'bias_base': residuals_base.mean(),
        'bias_corrected': residuals_corrected.mean(),
        'heteroscedasticity_base': r_hetero_base,
        'heteroscedasticity_corrected': r_hetero_corr,
        'correction_mean': correction.mean(),
        'correction_std': correction.std(),
        'r_perm0_validation': r_perm
    }

# Process each metapath
all_results = []

for edge1_type, edge2_type, metapath_name in METAPATHS:
    print(f"\n{'='*80}")
    print(f"Processing {metapath_name}...")
    print(f"{'='*80}")

    # Load data
    edge1 = load_edge_matrix(edge1_type, perm_id='original')
    edge2 = load_edge_matrix(edge2_type, perm_id='original')

    # Sample pairs
    pair_indices = sample_pairs_by_pathway_count(edge1, edge2, N_SAMPLES)

    # Extract features
    X = np.array([extract_pair_features_simple(edge1, edge2, s, t)
                  for s, t in pair_indices])

    # Compute targets
    print("  Computing targets...")
    perm_counts = []
    for perm_id in range(1, 21):
        edge1_perm = load_edge_matrix(edge1_type, perm_id)
        edge2_perm = load_edge_matrix(edge2_type, perm_id)
        counts = compute_pathway_counts_for_pairs(edge1_perm, edge2_perm, pair_indices)
        perm_counts.append(counts)
    y_validation = np.mean(perm_counts, axis=0)

    # Compute permutation 000 counts
    edge1_perm0 = load_edge_matrix(edge1_type, perm_id=0)
    edge2_perm0 = load_edge_matrix(edge2_type, perm_id=0)
    y_perm0 = compute_pathway_counts_for_pairs(edge1_perm0, edge2_perm0, pair_indices)

    # Train baseline
    base_model = LinearRegression()
    base_model.fit(X, y_validation)
    y_pred_base = base_model.predict(X)

    # Train correction
    correction_features = extract_correction_features(X, y_pred_base)
    correction_target = y_perm0 - y_pred_base
    correction_model = LinearRegression()
    correction_model.fit(correction_features, correction_target)
    correction = correction_model.predict(correction_features)
    y_pred_corrected = y_pred_base + correction

    # Create visualizations
    print("  Creating visualizations...")
    result = create_detailed_visualizations(
        metapath_name, X, y_validation, y_perm0,
        y_pred_base, y_pred_corrected, correction
    )

    result['metapath'] = metapath_name
    all_results.append(result)

# Save summary
df = pd.DataFrame(all_results)
df.to_csv(results_dir / 'detailed_analysis_summary.csv', index=False)

print("\n" + "="*80)
print("DETAILED ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated visualizations for {len(METAPATHS)} metapaths")
print(f"Results saved to: {results_dir}")
