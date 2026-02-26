"""
Continuous Degree Resolution Analysis

Investigates whether using continuous degrees (instead of 10 bins) improves
resolution enough for viable pair-level prediction.

Compares binned vs continuous approaches for CbGpPW and GiGiG metapaths.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from collections import defaultdict
import os

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def load_edge_matrix(data_dir, edge_abbrev):
    """Load edge adjacency matrix."""
    edge_path = os.path.join(data_dir, 'edges', f'{edge_abbrev}.sparse.npz')
    matrix = sp.load_npz(edge_path)
    if matrix.dtype == bool:
        matrix = matrix.astype(np.int32)
    return matrix


def analyze_continuous_degrees(metapath_name, edge1_abbrev, edge2_abbrev):
    """
    Analyze resolution with continuous (exact) degrees vs binned degrees.

    Returns dict with comprehensive statistics.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {metapath_name}")
    print(f"{'='*70}\n")

    # Load edges
    print("Loading edge matrices...")
    edge1 = load_edge_matrix('data', edge1_abbrev)
    edge2 = load_edge_matrix('data', edge2_abbrev)

    print(f"  Edge1 ({edge1_abbrev}): {edge1.shape}, {edge1.nnz:,} edges")
    print(f"  Edge2 ({edge2_abbrev}): {edge2.shape}, {edge2.nnz:,} edges")

    # Compute pathway counts
    print("\nComputing pathway matrix...")
    pathway_matrix = edge1.dot(edge2)
    if sp.issparse(pathway_matrix):
        pathway_matrix = pathway_matrix.toarray()

    # Get degrees
    source_degrees = np.array(edge1.sum(axis=1)).flatten()
    target_degrees = np.array(edge2.sum(axis=0)).flatten()

    print(f"  Source degree range: [{source_degrees.min()}, {source_degrees.max()}]")
    print(f"  Target degree range: [{target_degrees.min()}, {target_degrees.max()}]")
    print(f"  Unique source degrees: {len(np.unique(source_degrees))}")
    print(f"  Unique target degrees: {len(np.unique(target_degrees))}")

    # Group by exact degree combinations
    print("\nGrouping by exact degree combinations...")
    combo_data = defaultdict(list)

    for i in range(pathway_matrix.shape[0]):
        for j in range(pathway_matrix.shape[1]):
            deg_s = source_degrees[i]
            deg_t = target_degrees[j]
            count = pathway_matrix[i, j]
            combo_data[(deg_s, deg_t)].append(count)

    n_combos = len(combo_data)
    print(f"  Unique (source_deg, target_deg) combinations: {n_combos}")

    # Compute statistics per combination
    print("\nComputing combination statistics...")
    combo_stats = []

    for (deg_s, deg_t), counts in combo_data.items():
        counts_array = np.array(counts)
        combo_stats.append({
            'source_deg': deg_s,
            'target_deg': deg_t,
            'n_pairs': len(counts),
            'mean_count': counts_array.mean(),
            'std_count': counts_array.std(),
            'min_count': counts_array.min(),
            'max_count': counts_array.max(),
            'cv': counts_array.std() / counts_array.mean() if counts_array.mean() > 0 else 0
        })

    df_combos = pd.DataFrame(combo_stats)

    # Variance decomposition
    print("\nVariance decomposition (continuous degrees)...")

    # Flatten all pairs with weights
    all_counts = []
    all_combo_means = []
    combo_sizes = []

    for (deg_s, deg_t), counts in combo_data.items():
        all_counts.extend(counts)
        all_combo_means.extend([np.mean(counts)] * len(counts))
        combo_sizes.append(len(counts))

    all_counts = np.array(all_counts)
    all_combo_means = np.array(all_combo_means)
    combo_sizes = np.array(combo_sizes)

    # Total variance
    grand_mean = all_counts.mean()
    total_var = all_counts.var()

    # Between-combo variance (weighted by combo sizes)
    combo_means_array = df_combos['mean_count'].values
    between_var = np.average((combo_means_array - grand_mean)**2,
                              weights=df_combos['n_pairs'].values)

    # Within-combo variance (weighted average)
    combo_vars = df_combos['std_count'].values ** 2
    within_var = np.average(combo_vars, weights=df_combos['n_pairs'].values)

    r_squared = between_var / total_var if total_var > 0 else 0
    r_ceiling = np.sqrt(r_squared)

    print(f"  Between-combo variance: {between_var:.6f} ({100*between_var/total_var:.1f}%)")
    print(f"  Within-combo variance:  {within_var:.6f} ({100*within_var/total_var:.1f}%)")
    print(f"  Total variance:         {total_var:.6f}")
    print(f"  R² explainable:         {r_squared:.4f}")
    print(f"  Best-case r ceiling:    {r_ceiling:.4f}")

    # Distribution analysis
    print("\nSamples per combination:")
    print(f"  Mean:   {df_combos['n_pairs'].mean():.1f}")
    print(f"  Median: {df_combos['n_pairs'].median():.1f}")
    print(f"  Min:    {df_combos['n_pairs'].min()}")
    print(f"  Max:    {df_combos['n_pairs'].max()}")

    sparse_combos = (df_combos['n_pairs'] < 5).sum()
    trainable_combos = (df_combos['n_pairs'] >= 5).sum()
    print(f"  Combos with <5 pairs:  {sparse_combos} ({100*sparse_combos/n_combos:.1f}%)")
    print(f"  Combos with ≥5 pairs:  {trainable_combos} ({100*trainable_combos/n_combos:.1f}%)")

    # CV analysis
    print("\nWithin-combo variation:")
    print(f"  Mean CV:    {df_combos['cv'].mean():.4f}")
    print(f"  Median CV:  {df_combos['cv'].median():.4f}")
    print(f"  Max CV:     {df_combos['cv'].max():.4f}")

    high_cv_combos = (df_combos['cv'] > 1.0).sum()
    print(f"  Combos with CV > 1.0: {high_cv_combos}/{n_combos} ({100*high_cv_combos/n_combos:.1f}%)")

    # Simulated best-case performance
    print("\nSimulating best-case pair-level prediction...")
    pair_actuals = all_counts
    pair_predictions = all_combo_means

    if len(pair_actuals) > 1 and pair_actuals.std() > 0:
        simulated_r, _ = pearsonr(pair_actuals, pair_predictions)
    else:
        simulated_r = np.nan

    print(f"  Simulated r (pairs): {simulated_r:.4f}")

    results = {
        'metapath': metapath_name,
        'n_combos': n_combos,
        'n_unique_source_deg': len(np.unique(source_degrees)),
        'n_unique_target_deg': len(np.unique(target_degrees)),
        'between_var': between_var,
        'within_var': within_var,
        'total_var': total_var,
        'r_squared': r_squared,
        'r_ceiling': r_ceiling,
        'simulated_r': simulated_r,
        'mean_pairs_per_combo': df_combos['n_pairs'].mean(),
        'median_pairs_per_combo': df_combos['n_pairs'].median(),
        'pct_trainable': 100 * trainable_combos / n_combos,
        'mean_cv': df_combos['cv'].mean(),
        'median_cv': df_combos['cv'].median(),
        'pct_high_cv': 100 * high_cv_combos / n_combos,
        'df_combos': df_combos,
        'source_degrees': source_degrees,
        'target_degrees': target_degrees
    }

    return results


def create_visualizations(results_continuous, results_binned, output_dir='results/continuous_degree_analysis'):
    """Create comparative visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    metapath = results_continuous['metapath']
    df_combos = results_continuous['df_combos']

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f'Continuous vs Binned Degree Analysis: {metapath}',
                 fontsize=16, fontweight='bold')

    # 1. Degree combination heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    pivot = df_combos.pivot_table(values='n_pairs',
                                   index='target_deg',
                                   columns='source_deg',
                                   fill_value=0)
    sns.heatmap(pivot, cmap='viridis', ax=ax1, cbar_kws={'label': 'Pairs'},
                robust=True, vmin=0)
    ax1.set_title('Pairs per (Source Deg, Target Deg)')
    ax1.set_xlabel('Source Degree')
    ax1.set_ylabel('Target Degree')

    # 2. Combo size distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_combos['n_pairs'], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=5, color='r', linestyle='--', linewidth=2, label='Trainable threshold')
    ax2.set_xlabel('Pairs per Combination')
    ax2.set_ylabel('Number of Combinations')
    ax2.set_title(f'Distribution of Combo Sizes\n{results_continuous["pct_trainable"]:.1f}% with ≥5 pairs')
    ax2.legend()
    ax2.set_yscale('log')

    # 3. CV distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df_combos['cv'], bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=results_continuous['mean_cv'], color='r', linestyle='--',
                linewidth=2, label=f'Mean={results_continuous["mean_cv"]:.2f}')
    ax3.set_xlabel('Coefficient of Variation')
    ax3.set_ylabel('Number of Combinations')
    ax3.set_title('Within-Combo CV Distribution')
    ax3.legend()

    # 4. Variance comparison (continuous)
    ax4 = fig.add_subplot(gs[1, 0])
    var_cont = [results_continuous['between_var'], results_continuous['within_var']]
    labels_cont = ['Between-Combo\n(Explainable)', 'Within-Combo\n(Noise)']
    colors = ['#2ecc71', '#e74c3c']
    bars1 = ax4.bar(labels_cont, var_cont, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Variance')
    ax4.set_title(f'Continuous Degrees\nR²={results_continuous["r_squared"]:.3f}')
    ax4.grid(True, alpha=0.3, axis='y')

    total_cont = sum(var_cont)
    for bar, val in zip(bars1, var_cont):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{100*val/total_cont:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # 5. Variance comparison (binned)
    ax5 = fig.add_subplot(gs[1, 1])
    var_bin = [results_binned['between_var'], results_binned['within_var']]
    labels_bin = ['Between-Bin\n(Explainable)', 'Within-Bin\n(Noise)']
    bars2 = ax5.bar(labels_bin, var_bin, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Variance')
    ax5.set_title(f'Binned (10×10)\nR²={results_binned["r_squared_max"]:.3f}')
    ax5.grid(True, alpha=0.3, axis='y')

    total_bin = sum(var_bin)
    for bar, val in zip(bars2, var_bin):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{100*val/total_bin:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # 6. R² comparison
    ax6 = fig.add_subplot(gs[1, 2])
    r2_values = [results_binned['r_squared_max'], results_continuous['r_squared']]
    approach_labels = ['Binned\n(10×10)', 'Continuous\n(Exact Degrees)']
    bars3 = ax6.bar(approach_labels, r2_values, color=['#3498db', '#e67e22'],
                    alpha=0.7, edgecolor='black')
    ax6.set_ylabel('R² (Explainable Variance)')
    ax6.set_title('Explainable Variance Comparison')
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars3, r2_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}',
                ha='center', fontweight='bold')

    # 7. Best-case r comparison
    ax7 = fig.add_subplot(gs[2, 0])
    r_values = [results_binned['simulated_r'], results_continuous['simulated_r']]
    bars4 = ax7.bar(approach_labels, r_values, color=['#3498db', '#e67e22'],
                    alpha=0.7, edgecolor='black')
    ax7.axhline(y=0.95, color='g', linestyle='--', linewidth=2, label='Target r=0.95')
    ax7.axhline(y=0.75, color='orange', linestyle='--', linewidth=2, label='Good threshold')
    ax7.set_ylabel('Best-Case Pearson r (Pairs)')
    ax7.set_title('Simulated Pair-Level Performance')
    ax7.set_ylim([0, 1])
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars4, r_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}',
                ha='center', fontweight='bold')

    # 8. Sample size comparison
    ax8 = fig.add_subplot(gs[2, 1])
    n_samples = [results_binned['n_bins'], results_continuous['n_combos']]
    bars5 = ax8.bar(approach_labels, n_samples, color=['#3498db', '#e67e22'],
                    alpha=0.7, edgecolor='black')
    ax8.set_ylabel('Training Samples')
    ax8.set_title('Number of Training Samples')
    ax8.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars5, n_samples):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(n_samples),
                f'{int(val)}',
                ha='center', fontweight='bold')

    # 9. Improvement metrics
    ax9 = fig.add_subplot(gs[2, 2])
    metrics = ['R² Gain', 'r Ceiling\nGain', 'Sample\nIncrease']
    improvements = [
        results_continuous['r_squared'] - results_binned['r_squared_max'],
        results_continuous['simulated_r'] - results_binned['simulated_r'],
        (results_continuous['n_combos'] - results_binned['n_bins']) / results_binned['n_bins']
    ]
    colors_imp = ['g' if x > 0 else 'r' for x in improvements]
    bars6 = ax9.bar(metrics, improvements, color=colors_imp, alpha=0.7, edgecolor='black')
    ax9.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax9.set_ylabel('Change (Continuous - Binned)')
    ax9.set_title('Improvement with Continuous Degrees')
    ax9.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars6, improvements):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.3f}' if abs(val) < 1 else f'{val:+.1f}',
                ha='center', va=va, fontweight='bold')

    plt.savefig(os.path.join(output_dir, f'{metapath}_continuous_vs_binned.png'),
                bbox_inches='tight', dpi=150)
    plt.close()

    print(f"  Visualization saved: {output_dir}/{metapath}_continuous_vs_binned.png")


def main():
    """Run analysis for CbGpPW and GiGiG."""
    print("="*70)
    print("CONTINUOUS DEGREE RESOLUTION ANALYSIS")
    print("="*70)

    metapaths = [
        ('CbGpPW', 'CbG', 'GpPW'),
        ('GiGiG', 'GiG', 'GiG')
    ]

    all_results_continuous = []
    all_results_binned = []

    for metapath_name, edge1_abbrev, edge2_abbrev in metapaths:
        # Continuous degree analysis
        results_cont = analyze_continuous_degrees(metapath_name, edge1_abbrev, edge2_abbrev)
        all_results_continuous.append(results_cont)

        # Load binned results from previous analysis
        binned_file = f'results/phase2_training_data/{metapath_name}_features_A.csv'
        df_binned = pd.read_csv(binned_file)

        # Reconstruct binned variance stats
        bin_means = df_binned['pathway_count_mean'].values
        bin_stds = df_binned['pathway_count_std'].values
        bin_sizes = df_binned['n_pairs'].values

        grand_mean = np.average(bin_means, weights=bin_sizes)
        between_var_bin = np.average((bin_means - grand_mean)**2, weights=bin_sizes)
        within_var_bin = np.average(bin_stds**2, weights=bin_sizes)
        total_var_bin = between_var_bin + within_var_bin
        r_squared_bin = between_var_bin / total_var_bin

        # Simulate binned performance (from previous analysis)
        # Load from resolution analysis results
        if metapath_name == 'CbGpPW':
            simulated_r_bin = 0.524
        elif metapath_name == 'GiGiG':
            simulated_r_bin = 0.606
        else:
            simulated_r_bin = np.nan

        results_bin = {
            'metapath': metapath_name,
            'n_bins': len(df_binned),
            'between_var': between_var_bin,
            'within_var': within_var_bin,
            'total_var': total_var_bin,
            'r_squared_max': r_squared_bin,
            'simulated_r': simulated_r_bin
        }
        all_results_binned.append(results_bin)

        # Create visualizations
        create_visualizations(results_cont, results_bin)

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: CONTINUOUS VS BINNED COMPARISON")
    print("="*70)
    print()

    comparison_data = []
    for cont, binned in zip(all_results_continuous, all_results_binned):
        comparison_data.append({
            'Metapath': cont['metapath'],
            'Approach': 'Binned',
            'Samples': binned['n_bins'],
            'R²': f"{binned['r_squared_max']:.3f}",
            'r_ceiling': f"{binned['simulated_r']:.3f}",
            'Trainable%': '100%'
        })
        comparison_data.append({
            'Metapath': cont['metapath'],
            'Approach': 'Continuous',
            'Samples': cont['n_combos'],
            'R²': f"{cont['r_squared']:.3f}",
            'r_ceiling': f"{cont['simulated_r']:.3f}",
            'Trainable%': f"{cont['pct_trainable']:.1f}%"
        })

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print()

    # Save summary
    df_comparison.to_csv('results/continuous_degree_analysis/comparison_summary.csv', index=False)

    # Decision
    print("="*70)
    print("DECISION FRAMEWORK")
    print("="*70)
    print()

    avg_r_cont = np.mean([r['simulated_r'] for r in all_results_continuous])
    avg_r_bin = np.mean([r['simulated_r'] for r in all_results_binned])
    avg_r2_cont = np.mean([r['r_squared'] for r in all_results_continuous])
    avg_r2_bin = np.mean([r['r_squared_max'] for r in all_results_binned])
    avg_trainable = np.mean([r['pct_trainable'] for r in all_results_continuous])

    print(f"Average R² (continuous): {avg_r2_cont:.3f}")
    print(f"Average R² (binned):     {avg_r2_bin:.3f}")
    print(f"R² improvement:          {avg_r2_cont - avg_r2_bin:+.3f}")
    print()
    print(f"Average r_ceiling (continuous): {avg_r_cont:.3f}")
    print(f"Average r_ceiling (binned):     {avg_r_bin:.3f}")
    print(f"r improvement:                  {avg_r_cont - avg_r_bin:+.3f}")
    print()
    print(f"Average trainable combos: {avg_trainable:.1f}%")
    print()

    # Decision criteria
    r2_improvement = avg_r2_cont - avg_r2_bin
    r_improvement = avg_r_cont - avg_r_bin

    if r2_improvement > 0.15 and avg_r_cont > 0.75 and avg_trainable > 50:
        print("RECOMMENDATION: ✓ Proceed with continuous degree redesign")
        print("  - Substantial R² improvement (>15 percentage points)")
        print("  - Best-case r > 0.75 (viable for pair prediction)")
        print("  - Sufficient trainable combinations")
        print("  - Expected to reach r≥0.95 target with model training")
    elif r2_improvement > 0.05 and avg_r_cont > 0.65:
        print("RECOMMENDATION: ⚠ Proceed with caution")
        print("  - Moderate R² improvement")
        print("  - Best-case r marginal but promising")
        print("  - May reach r=0.85-0.90 with good model")
        print("  - Worth trying but uncertain outcome")
    else:
        print("RECOMMENDATION: ✗ Don't proceed with continuous degrees")
        print("  - Minimal R² improvement")
        print("  - Best-case r still too low")
        print("  - Unlikely to reach r≥0.95 target")
        print("  - Need different approach (pair-specific features)")

    print()
    print("Visualizations saved to results/continuous_degree_analysis/")

    return all_results_continuous, all_results_binned


if __name__ == "__main__":
    main()
