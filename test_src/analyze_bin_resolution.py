"""
Resolution Analysis: Within-Bin Variance Study

Analyzes how much pathway counts vary within degree bins to understand
the fundamental limitations of bin-based prediction for individual pairs.

Key question: Can a bin-level model provide good individual pair predictions?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def analyze_metapath_resolution(metapath_name):
    """
    Analyze within-bin variance for a single metapath.

    Returns dict with resolution metrics.
    """
    print(f"\nAnalyzing: {metapath_name}")
    print("-" * 70)

    # Load training data (has bin statistics)
    data_file = f'results/phase2_training_data/{metapath_name}_features_A.csv'
    df = pd.read_csv(data_file)

    # Extract statistics
    n_bins = len(df)
    total_pairs = df['n_pairs'].sum()

    bin_means = df['pathway_count_mean'].values
    bin_stds = df['pathway_count_std'].values
    bin_sizes = df['n_pairs'].values

    print(f"  Bins: {n_bins}")
    print(f"  Total pairs: {total_pairs:,}")

    # Coefficient of variation per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_cvs = bin_stds / bin_means
        bin_cvs = np.nan_to_num(bin_cvs, nan=0.0, posinf=0.0)

    # Variance decomposition
    # Total variance = between-bin variance + within-bin variance

    # Between-bin variance (how much bins differ)
    grand_mean = np.average(bin_means, weights=bin_sizes)
    between_var = np.average((bin_means - grand_mean)**2, weights=bin_sizes)

    # Within-bin variance (weighted average of bin variances)
    within_var = np.average(bin_stds**2, weights=bin_sizes)

    total_var = between_var + within_var

    # Explainable variance fraction
    r_squared_max = between_var / total_var if total_var > 0 else 0
    r_max = np.sqrt(r_squared_max)

    print(f"\n  Variance Decomposition:")
    print(f"    Between-bin variance: {between_var:.6f} ({100*between_var/total_var:.1f}%)")
    print(f"    Within-bin variance:  {within_var:.6f} ({100*within_var/total_var:.1f}%)")
    print(f"    Total variance:       {total_var:.6f}")
    print(f"    R² (explainable):     {r_squared_max:.4f}")
    print(f"    Best-case r:          {r_max:.4f}")

    print(f"\n  Within-Bin Variation:")
    print(f"    Mean CV:    {bin_cvs.mean():.4f}")
    print(f"    Median CV:  {np.median(bin_cvs):.4f}")
    print(f"    Max CV:     {bin_cvs.max():.4f}")

    # Identify high-variance bins
    high_var_bins = (bin_cvs > 1.0).sum()
    print(f"    Bins with CV > 1.0: {high_var_bins}/{n_bins} ({100*high_var_bins/n_bins:.1f}%)")

    results = {
        'metapath': metapath_name,
        'n_bins': n_bins,
        'n_pairs': total_pairs,
        'between_var': between_var,
        'within_var': within_var,
        'total_var': total_var,
        'r_squared_max': r_squared_max,
        'r_max': r_max,
        'mean_cv': bin_cvs.mean(),
        'median_cv': np.median(bin_cvs),
        'max_cv': bin_cvs.max(),
        'high_var_bins': high_var_bins,
        'bin_means': bin_means,
        'bin_stds': bin_stds,
        'bin_cvs': bin_cvs,
        'bin_sizes': bin_sizes
    }

    return results


def create_visualizations(results, output_dir='results/resolution_analysis'):
    """Create visualization suite for resolution analysis."""
    os.makedirs(output_dir, exist_ok=True)

    metapath = results['metapath']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Resolution Analysis: {metapath}', fontsize=16, fontweight='bold')

    # Plot 1: Variance by bin (sorted by mean)
    ax = axes[0, 0]
    sorted_idx = np.argsort(results['bin_means'])
    ax.errorbar(range(len(sorted_idx)),
                results['bin_means'][sorted_idx],
                yerr=results['bin_stds'][sorted_idx],
                fmt='o-', capsize=3, alpha=0.7)
    ax.set_xlabel('Bin (sorted by mean pathway count)')
    ax.set_ylabel('Pathway Count')
    ax.set_title('Mean ± Std per Bin')
    ax.grid(True, alpha=0.3)

    # Plot 2: CV distribution
    ax = axes[0, 1]
    ax.hist(results['bin_cvs'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=results['mean_cv'], color='r', linestyle='--',
               linewidth=2, label=f'Mean CV={results["mean_cv"]:.2f}')
    ax.axvline(x=1.0, color='orange', linestyle='--',
               linewidth=2, label='CV=1.0 threshold')
    ax.set_xlabel('Coefficient of Variation (CV)')
    ax.set_ylabel('Number of Bins')
    ax.set_title('CV Distribution Across Bins')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Variance components
    ax = axes[1, 0]
    var_components = [results['between_var'], results['within_var']]
    labels = ['Between-Bin\n(Explainable)', 'Within-Bin\n(Noise)']
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(labels, var_components, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Variance')
    ax.set_title(f'Variance Decomposition\nR²_max = {results["r_squared_max"]:.3f}')
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentages
    total = sum(var_components)
    for bar, val in zip(bars, var_components):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{100*val/total:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # Plot 4: Best-case correlation simulation
    ax = axes[1, 1]

    # Simulate: if model perfectly predicts bin means,
    # what correlation would we get at pair level?

    # Create synthetic pair-level data
    pair_actuals = []
    pair_predictions = []

    for i in range(len(results['bin_means'])):
        n_pairs_in_bin = int(results['bin_sizes'][i])
        if n_pairs_in_bin == 0:
            continue

        # Sample pairs from normal distribution with bin mean and std
        bin_mean = results['bin_means'][i]
        bin_std = results['bin_stds'][i]

        # Actual values (simulated from distribution)
        actual = np.random.normal(bin_mean, bin_std, size=min(n_pairs_in_bin, 1000))
        actual = np.maximum(actual, 0)  # Pathway counts can't be negative

        # Predicted values (all pairs in bin get bin mean)
        predicted = np.full(len(actual), bin_mean)

        pair_actuals.extend(actual)
        pair_predictions.extend(predicted)

    pair_actuals = np.array(pair_actuals)
    pair_predictions = np.array(pair_predictions)

    # Subsample for plotting
    if len(pair_actuals) > 10000:
        idx = np.random.choice(len(pair_actuals), 10000, replace=False)
        pair_actuals = pair_actuals[idx]
        pair_predictions = pair_predictions[idx]

    # Compute correlation
    if len(pair_actuals) > 1:
        best_case_r, _ = pearsonr(pair_actuals, pair_predictions)
    else:
        best_case_r = np.nan

    ax.scatter(pair_actuals, pair_predictions, alpha=0.1, s=5)
    ax.plot([pair_actuals.min(), pair_actuals.max()],
            [pair_actuals.min(), pair_actuals.max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Pathway Count (individual pair)')
    ax.set_ylabel('Predicted (bin mean)')
    ax.set_title(f'Best-Case Pair-Level Prediction\nr = {best_case_r:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metapath}_resolution.png'),
                bbox_inches='tight')
    plt.close()

    print(f"  Visualization saved: {output_dir}/{metapath}_resolution.png")

    return best_case_r


def main():
    """Run resolution analysis for all 5 metapaths."""
    print("=" * 70)
    print("RESOLUTION ANALYSIS: WITHIN-BIN VARIANCE STUDY")
    print("=" * 70)

    metapaths = ['CbGpPW', 'GiGiG', 'CtDaG', 'AdGpBP', 'CrCbG']

    all_results = []

    for metapath in metapaths:
        results = analyze_metapath_resolution(metapath)
        best_case_r = create_visualizations(results)
        results['simulated_r'] = best_case_r
        all_results.append(results)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: CROSS-METAPATH COMPARISON")
    print("=" * 70)
    print()

    df_summary = pd.DataFrame([{
        'Metapath': r['metapath'],
        'Bins': r['n_bins'],
        'Pairs': f"{r['n_pairs']:,}",
        'R²_max': f"{r['r_squared_max']:.3f}",
        'r_ceiling': f"{r['r_max']:.3f}",
        'Simulated_r': f"{r['simulated_r']:.3f}",
        'Mean_CV': f"{r['mean_cv']:.2f}",
        'High_CV_bins': f"{r['high_var_bins']}/{r['n_bins']}"
    } for r in all_results])

    print(df_summary.to_string(index=False))
    print()

    # Save summary
    df_summary.to_csv('results/resolution_analysis/summary.csv', index=False)

    # Decision framework
    print("=" * 70)
    print("DECISION FRAMEWORK")
    print("=" * 70)
    print()

    avg_r_ceiling = np.mean([r['r_max'] for r in all_results])
    avg_simulated_r = np.mean([r['simulated_r'] for r in all_results if not np.isnan(r['simulated_r'])])
    avg_cv = np.mean([r['mean_cv'] for r in all_results])

    print(f"Average r_ceiling across metapaths: {avg_r_ceiling:.3f}")
    print(f"Average simulated r (pair-level):   {avg_simulated_r:.3f}")
    print(f"Average CV:                          {avg_cv:.2f}")
    print()

    if avg_simulated_r >= 0.90:
        print("RECOMMENDATION: ✓ Proceed with feature ablation")
        print("  - Low within-bin variance")
        print("  - Bin predictions are good approximations for pairs")
        print("  - Improving bin model from r=0.94→0.95 is worthwhile")
    elif avg_simulated_r >= 0.70:
        print("RECOMMENDATION: ⚠ Feature ablation with caveats")
        print("  - Moderate within-bin variance")
        print("  - Bin model useful for bin-level predictions only")
        print("  - Limited resolution for individual pair predictions")
    else:
        print("RECOMMENDATION: ✗ Skip feature ablation")
        print("  - High within-bin variance")
        print("  - Bin-based approach fundamentally limited")
        print("  - Need pair-specific features for individual predictions")

    print()
    print(f"Benchmark comparison: Original r=0.94 (bin→bin), Best-case r={avg_simulated_r:.3f} (bin→pairs)")
    print()

    return all_results


if __name__ == "__main__":
    results = main()
