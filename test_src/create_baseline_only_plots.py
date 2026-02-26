"""
Create Baseline-Only Visualizations for Phase 2 Summary

Generates clean plots showing baseline Linear Regression performance
without any correction comparisons (since correction is not needed).
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
results_dir = repo_dir / 'results' / 'pair_level_phase2_baseline_only'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("CREATING BASELINE-ONLY VISUALIZATIONS")
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

# Process each metapath
all_results = []

for edge1_type, edge2_type, metapath_name in METAPATHS:
    print(f"\n{metapath_name}...")

    # Load data
    edge1 = load_edge_matrix(edge1_type, perm_id='original')
    edge2 = load_edge_matrix(edge2_type, perm_id='original')

    # Sample pairs
    pair_indices = sample_pairs_by_pathway_count(edge1, edge2, N_SAMPLES)

    # Extract features
    X = np.array([extract_pair_features_simple(edge1, edge2, s, t)
                  for s, t in pair_indices])

    # Compute targets
    perm_counts = []
    for perm_id in range(1, 21):
        edge1_perm = load_edge_matrix(edge1_type, perm_id)
        edge2_perm = load_edge_matrix(edge2_type, perm_id)
        counts = compute_pathway_counts_for_pairs(edge1_perm, edge2_perm, pair_indices)
        perm_counts.append(counts)
    y_validation = np.mean(perm_counts, axis=0)

    # Train baseline
    base_model = LinearRegression()
    base_model.fit(X, y_validation)
    y_pred = base_model.predict(X)

    # Compute metrics
    r = pearsonr(y_pred, y_validation)[0]
    rmse = np.sqrt(np.mean((y_pred - y_validation)**2))
    bias = np.mean(y_pred - y_validation)
    residuals = y_validation - y_pred

    # Heteroscedasticity
    abs_residuals = np.abs(residuals)
    r_hetero = pearsonr(y_pred, abs_residuals)[0]

    all_results.append({
        'metapath': metapath_name,
        'X': X,
        'y_true': y_validation,
        'y_pred': y_pred,
        'residuals': residuals,
        'r': r,
        'rmse': rmse,
        'bias': bias,
        'r_hetero': r_hetero
    })

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Row 1: Predicted vs Observed for each metapath
for i, result in enumerate(all_results):
    ax = fig.add_subplot(gs[0, i])

    y_true = result['y_true']
    y_pred = result['y_pred']
    r = result['r']
    rmse = result['rmse']

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, edgecolors='none')

    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect prediction')

    ax.set_xlabel('True Pathway Count', fontsize=11)
    ax.set_ylabel('Predicted Pathway Count', fontsize=11)
    ax.set_title(f'{result["metapath"]}\nr = {r:.4f}, RMSE = {rmse:.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

# Row 1, Column 4: Performance comparison
ax = fig.add_subplot(gs[0, 3])
metapath_names = [r['metapath'] for r in all_results]
correlations = [r['r'] for r in all_results]
rmses = [r['rmse'] for r in all_results]

x = np.arange(len(metapath_names))
width = 0.35

ax2 = ax.twinx()
bars1 = ax.bar(x - width/2, correlations, width, label='Correlation (r)',
               alpha=0.8, color='steelblue')
bars2 = ax2.bar(x + width/2, rmses, width, label='RMSE',
                alpha=0.8, color='coral')

ax.axhline(0.90, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Target (r>0.90)')
ax.set_ylabel('Correlation (r)', fontsize=11, color='steelblue')
ax2.set_ylabel('RMSE', fontsize=11, color='coral')
ax.set_xticks(x)
ax.set_xticklabels(metapath_names, fontsize=10)
ax.set_title('Performance Comparison', fontsize=12, fontweight='bold')
ax.set_ylim([0.85, 1.0])
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Row 2: Residuals vs Predicted
for i, result in enumerate(all_results):
    ax = fig.add_subplot(gs[1, i])

    y_pred = result['y_pred']
    residuals = result['residuals']
    bias = result['bias']
    r_hetero = result['r_hetero']

    ax.scatter(y_pred, residuals, alpha=0.3, s=10, edgecolors='none')
    ax.axhline(0, color='r', linestyle='--', lw=2)

    ax.set_xlabel('Predicted Pathway Count', fontsize=11)
    ax.set_ylabel('Residuals (True - Predicted)', fontsize=11)
    ax.set_title(f'{result["metapath"]} Residuals\nBias = {bias:+.4f}, Hetero r = {r_hetero:.3f}',
                 fontsize=11)
    ax.grid(alpha=0.3)

# Row 2, Column 4: Residual distributions
ax = fig.add_subplot(gs[1, 3])
for result in all_results:
    ax.hist(result['residuals'], bins=50, alpha=0.5,
            label=f'{result["metapath"]} (σ={result["residuals"].std():.3f})',
            density=True)

ax.axvline(0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Residuals', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Residual Distributions', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Row 3: Pathway count distributions (from original graph)
for i, result in enumerate(all_results):
    ax = fig.add_subplot(gs[2, i])

    y_true = result['y_true']

    # Histogram
    ax.hist(y_true[y_true > 0], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Pathway Count', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{result["metapath"]} Distribution\nMean = {y_true.mean():.2f}, Max = {y_true.max():.0f}',
                 fontsize=11)
    ax.grid(alpha=0.3, axis='y')

# Row 3, Column 4: Summary statistics table
ax = fig.add_subplot(gs[2, 3])
ax.axis('off')

# Create summary table
summary_data = []
for result in all_results:
    y_true = result['y_true']
    nonzero = y_true[y_true > 0]
    summary_data.append([
        result['metapath'],
        f"{result['r']:.4f}",
        f"{result['rmse']:.4f}",
        f"{result['bias']:+.4f}",
        f"{y_true.max():.0f}",
        f"{nonzero.mean():.2f}"
    ])

table = ax.table(cellText=summary_data,
                colLabels=['Metapath', 'r', 'RMSE', 'Bias', 'Max\nCount', 'Mean\n(nonzero)'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

# Main title
fig.suptitle('Baseline Linear Regression Performance\n(5 Degree Features, No Correction)',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(results_dir / 'baseline_comprehensive_analysis.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: {results_dir / 'baseline_comprehensive_analysis.png'}")

# Create individual metapath plots
for result in all_results:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    metapath_name = result['metapath']
    y_true = result['y_true']
    y_pred = result['y_pred']
    residuals = result['residuals']
    r = result['r']
    rmse = result['rmse']
    bias = result['bias']
    r_hetero = result['r_hetero']

    # (1,1) Predicted vs Observed
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=15)
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('True Pathway Count', fontsize=12)
    ax.set_ylabel('Predicted Pathway Count', fontsize=12)
    ax.set_title(f'Predicted vs Observed\nr = {r:.4f}, RMSE = {rmse:.4f}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,2) Residuals vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=15)
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Pathway Count', fontsize=12)
    ax.set_ylabel('Residuals (True - Predicted)', fontsize=12)
    ax.set_title(f'Residual Plot\nBias = {bias:+.4f}, Heteroscedasticity r = {r_hetero:.3f}',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # (2,1) Residual distribution
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', lw=2)
    ax.axvline(residuals.mean(), color='blue', linestyle='--', lw=2,
               label=f'Mean = {residuals.mean():+.4f}')
    ax.set_xlabel('Residuals', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Residual Distribution\nStd = {residuals.std():.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (2,2) Pathway count distribution
    ax = axes[1, 1]
    nonzero = y_true[y_true > 0]
    ax.hist(y_true, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Pathway Count', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Pathway Count Distribution\nMax = {y_true.max():.0f}, Mean (nonzero) = {nonzero.mean():.2f}',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Add text box with statistics
    stats_text = f'Statistics:\n'
    stats_text += f'  Correlation: r = {r:.4f}\n'
    stats_text += f'  RMSE: {rmse:.4f}\n'
    stats_text += f'  Bias: {bias:+.4f}\n'
    stats_text += f'  Sample size: {len(y_true):,}\n'
    stats_text += f'  Pairs with paths: {(y_true > 0).sum():,}'

    fig.text(0.98, 0.02, stats_text, fontsize=10, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'{metapath_name}: Baseline Linear Regression\n(5 Degree Features)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / f'{metapath_name}_baseline_analysis.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {results_dir / f'{metapath_name}_baseline_analysis.png'}")

# Save summary CSV
summary_df = pd.DataFrame([{
    'metapath': r['metapath'],
    'r': r['r'],
    'rmse': r['rmse'],
    'bias': r['bias'],
    'heteroscedasticity': r['r_hetero'],
    'max_pathway_count': r['y_true'].max(),
    'mean_pathway_count': r['y_true'].mean(),
    'mean_nonzero': r['y_true'][r['y_true'] > 0].mean(),
    'pairs_with_pathways': (r['y_true'] > 0).sum(),
    'total_pairs': len(r['y_true'])
} for r in all_results])

summary_df.to_csv(results_dir / 'baseline_summary.csv', index=False)
print(f"\nSaved: {results_dir / 'baseline_summary.csv'}")

print("\n" + "="*80)
print("BASELINE-ONLY VISUALIZATIONS COMPLETE")
print("="*80)
print(f"\nGenerated:")
print(f"  - 1 comprehensive comparison plot (12 panels)")
print(f"  - 3 individual metapath plots (4 panels each)")
print(f"  - 1 summary CSV")
print(f"\nAll files in: {results_dir}")
