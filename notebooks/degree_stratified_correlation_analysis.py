#!/usr/bin/env python3
"""
Degree-Stratified Correlation Analysis

Analyze how compositional model fit (correlation between observed and compositional probabilities)
varies across degree combinations to test the hypothesis that "degree-aware compositional models may suffice."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def pearsonr(x, y):
    """Simple Pearson correlation implementation."""
    try:
        return pd.Series(x).corr(pd.Series(y)), 0.0  # Return correlation and dummy p-value
    except:
        return np.nan, np.nan

# Setup paths
repo_dir = Path(__file__).parent.parent
results_dir = repo_dir / 'results'

# Load data from notebook 11
print("Loading degree-stratified results from notebook 11...")
hetionet_df = pd.read_csv(results_dir / 'metapath_CbGpPW_hetionet_degree_results.csv')
null_df = pd.read_csv(results_dir / 'metapath_CbGpPW_null_degree_results.csv')

print(f"Loaded Hetionet data: {len(hetionet_df)} metapath pairs")
print(f"Loaded Null data: {len(null_df)} metapath pairs")

# Define degree bins
DEGREE_LABELS = ['Very Low (0-5)', 'Low (5-20)', 'Medium (20-100)', 'High (>100)']

def calculate_stratified_correlations(df, min_samples=50):
    """
    Calculate correlation between observed_freq and compositional_prob
    within each degree combination.
    """
    correlations = {}
    sample_sizes = {}

    for comp_bin in DEGREE_LABELS:
        for path_bin in DEGREE_LABELS:
            subset = df[
                (df['compound_degree_bin'] == comp_bin) &
                (df['pathway_degree_bin'] == path_bin)
            ].copy()

            if len(subset) >= min_samples:
                # Remove any infinite or NaN values
                subset = subset[
                    np.isfinite(subset['observed_freq']) &
                    np.isfinite(subset['compositional_prob'])
                ]

                if len(subset) >= min_samples:
                    try:
                        corr, p_val = pearsonr(subset['observed_freq'], subset['compositional_prob'])
                        correlations[(comp_bin, path_bin)] = corr
                        sample_sizes[(comp_bin, path_bin)] = len(subset)
                    except:
                        correlations[(comp_bin, path_bin)] = np.nan
                        sample_sizes[(comp_bin, path_bin)] = len(subset)
                else:
                    correlations[(comp_bin, path_bin)] = np.nan
                    sample_sizes[(comp_bin, path_bin)] = len(subset)
            else:
                correlations[(comp_bin, path_bin)] = np.nan
                sample_sizes[(comp_bin, path_bin)] = len(subset)

    return correlations, sample_sizes

print("\nCalculating degree-stratified correlations...")

# Calculate correlations for Hetionet
het_correlations, het_sample_sizes = calculate_stratified_correlations(hetionet_df)

# Calculate correlations for Null (average across permutations)
null_correlations = {}
null_sample_sizes = {}

for perm_id in null_df['perm_id'].unique():
    perm_subset = null_df[null_df['perm_id'] == perm_id]
    perm_corrs, perm_sizes = calculate_stratified_correlations(perm_subset)

    for key in perm_corrs:
        if key not in null_correlations:
            null_correlations[key] = []
            null_sample_sizes[key] = []

        if not np.isnan(perm_corrs[key]):
            null_correlations[key].append(perm_corrs[key])
            null_sample_sizes[key].append(perm_sizes[key])

# Average null correlations
null_corr_means = {}
null_corr_stds = {}
for key in null_correlations:
    if len(null_correlations[key]) > 0:
        null_corr_means[key] = np.mean(null_correlations[key])
        null_corr_stds[key] = np.std(null_correlations[key])
        null_sample_sizes[key] = np.mean(null_sample_sizes[key])
    else:
        null_corr_means[key] = np.nan
        null_corr_stds[key] = np.nan

print("Creating correlation matrices...")

# Create correlation matrices
het_corr_matrix = np.full((len(DEGREE_LABELS), len(DEGREE_LABELS)), np.nan)
null_corr_matrix = np.full((len(DEGREE_LABELS), len(DEGREE_LABELS)), np.nan)
sample_size_matrix = np.full((len(DEGREE_LABELS), len(DEGREE_LABELS)), 0)

for i, path_bin in enumerate(DEGREE_LABELS):
    for j, comp_bin in enumerate(DEGREE_LABELS):
        key = (comp_bin, path_bin)

        if key in het_correlations:
            het_corr_matrix[i, j] = het_correlations[key]

        if key in null_corr_means:
            null_corr_matrix[i, j] = null_corr_means[key]

        if key in het_sample_sizes:
            sample_size_matrix[i, j] = het_sample_sizes[key]

# Calculate overall correlations for reference
het_overall_corr = pearsonr(hetionet_df['observed_freq'], hetionet_df['compositional_prob'])[0]
null_overall_corrs = []
for perm_id in null_df['perm_id'].unique():
    perm_subset = null_df[null_df['perm_id'] == perm_id]
    corr = pearsonr(perm_subset['observed_freq'], perm_subset['compositional_prob'])[0]
    null_overall_corrs.append(corr)
null_overall_corr = np.mean(null_overall_corrs)

print(f"\nOverall correlations:")
print(f"  Hetionet: {het_overall_corr:.4f}")
print(f"  Null: {null_overall_corr:.4f} ± {np.std(null_overall_corrs):.4f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Hetionet correlations
ax = axes[0, 0]
mask = np.isnan(het_corr_matrix)
sns.heatmap(het_corr_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
            vmin=-0.5,
            vmax=0.5,
            mask=mask,
            xticklabels=[label.replace(' ', '\n') for label in DEGREE_LABELS],
            yticklabels=[label.replace(' ', '\n') for label in DEGREE_LABELS],
            ax=ax,
            cbar_kws={'label': 'Correlation'})
ax.set_title(f'Hetionet: Degree-Stratified Correlations\n(Overall: {het_overall_corr:.3f})',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Compound Degree', fontsize=12)
ax.set_ylabel('Pathway Degree', fontsize=12)

# Null correlations
ax = axes[0, 1]
mask = np.isnan(null_corr_matrix)
sns.heatmap(null_corr_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
            vmin=-0.5,
            vmax=0.5,
            mask=mask,
            xticklabels=[label.replace(' ', '\n') for label in DEGREE_LABELS],
            yticklabels=[label.replace(' ', '\n') for label in DEGREE_LABELS],
            ax=ax,
            cbar_kws={'label': 'Correlation'})
ax.set_title(f'Null: Degree-Stratified Correlations\n(Overall: {null_overall_corr:.3f})',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Compound Degree', fontsize=12)
ax.set_ylabel('Pathway Degree', fontsize=12)

# Difference (Hetionet - Null)
ax = axes[1, 0]
diff_matrix = het_corr_matrix - null_corr_matrix
mask = np.isnan(diff_matrix)
sns.heatmap(diff_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            vmin=-0.2,
            vmax=0.2,
            mask=mask,
            xticklabels=[label.replace(' ', '\n') for label in DEGREE_LABELS],
            yticklabels=[label.replace(' ', '\n') for label in DEGREE_LABELS],
            ax=ax,
            cbar_kws={'label': 'Correlation Difference'})
ax.set_title('Difference: Hetionet - Null\n(Biological Signal)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Compound Degree', fontsize=12)
ax.set_ylabel('Pathway Degree', fontsize=12)

# Sample sizes
ax = axes[1, 1]
mask = sample_size_matrix == 0
sns.heatmap(sample_size_matrix,
            annot=True,
            fmt='.0f',
            cmap='Greens',
            mask=mask,
            xticklabels=[label.replace(' ', '\n') for label in DEGREE_LABELS],
            yticklabels=[label.replace(' ', '\n') for label in DEGREE_LABELS],
            ax=ax,
            cbar_kws={'label': 'Sample Size'})
ax.set_title('Sample Sizes\n(Reliability Indicator)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Compound Degree', fontsize=12)
ax.set_ylabel('Pathway Degree', fontsize=12)

plt.tight_layout()
plt.savefig(results_dir / 'degree_stratified_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# Analysis summary
print("\n" + "="*80)
print("DEGREE-STRATIFIED CORRELATION ANALYSIS")
print("="*80)

# Count valid correlations
valid_het_corrs = [v for v in het_correlations.values() if not np.isnan(v)]
valid_null_corrs = [v for v in null_corr_means.values() if not np.isnan(v)]

print(f"\nValid correlation measurements:")
print(f"  Hetionet: {len(valid_het_corrs)}/{len(het_correlations)} degree combinations")
print(f"  Null: {len(valid_null_corrs)}/{len(null_corr_means)} degree combinations")

if len(valid_het_corrs) > 0:
    print(f"\nHetionet stratified correlations:")
    print(f"  Mean: {np.mean(valid_het_corrs):.4f}")
    print(f"  Range: {np.min(valid_het_corrs):.4f} to {np.max(valid_het_corrs):.4f}")
    print(f"  Std: {np.std(valid_het_corrs):.4f}")

    # Count improvements
    better_than_overall = sum(1 for v in valid_het_corrs if v > het_overall_corr)
    print(f"  Strata better than overall ({het_overall_corr:.3f}): {better_than_overall}/{len(valid_het_corrs)}")

if len(valid_null_corrs) > 0:
    print(f"\nNull stratified correlations:")
    print(f"  Mean: {np.mean(valid_null_corrs):.4f}")
    print(f"  Range: {np.min(valid_null_corrs):.4f} to {np.max(valid_null_corrs):.4f}")
    print(f"  Std: {np.std(valid_null_corrs):.4f}")

# Interpretation
print(f"\n" + "="*50)
print("INTERPRETATION")
print("="*50)

if len(valid_het_corrs) > 0:
    mean_stratified = np.mean(valid_het_corrs)
    improvement = mean_stratified - het_overall_corr

    print(f"\nModel fit improvement with degree stratification:")
    print(f"  Overall correlation: {het_overall_corr:.4f}")
    print(f"  Mean stratified correlation: {mean_stratified:.4f}")
    print(f"  Improvement: {improvement:+.4f}")

    if improvement > 0.05:
        print(f"\n✓ SUBSTANTIAL IMPROVEMENT with degree stratification")
        print(f"  → Degree-aware compositional models are JUSTIFIED")
        print(f"  → Poor overall fit is due to degree heterogeneity")
    elif improvement > 0.01:
        print(f"\n→ MODEST IMPROVEMENT with degree stratification")
        print(f"  → Some benefit to degree-aware models")
    else:
        print(f"\n✗ MINIMAL IMPROVEMENT with degree stratification")
        print(f"  → Degree-aware models may not suffice")
        print(f"  → Fundamental non-compositionality persists")

# Best and worst performing strata
if len(valid_het_corrs) > 0:
    print(f"\nBest performing degree combinations:")
    sorted_corrs = sorted(het_correlations.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
    for i, ((comp, path), corr) in enumerate(sorted_corrs[:3]):
        if not np.isnan(corr):
            print(f"  {i+1}. {comp} × {path}: r = {corr:.4f} (n = {het_sample_sizes.get((comp, path), 0)})")

print(f"\n" + "="*80)

print(f"\nResults saved to: {results_dir / 'degree_stratified_correlation_heatmaps.png'}")