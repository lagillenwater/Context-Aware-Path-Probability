#!/usr/bin/env python3
"""
Test Phase 0b: Permutation stability analysis.

This script tests the correlation between a single permutation (000) and
the average of other permutations (001-020). This evaluates how representative
a single permutation is of the permutation distribution.

Usage:
    python test_src/test_perm000_vs_perms.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from permutation_validation import (
    extract_pathway_bins_from_single_permutation,
    extract_pathway_bins_from_permutations,
    test_original_vs_permutation_correlation,
    analyze_residuals
)


def main():
    """
    Test permutation 000 vs average of 001-020.
    """
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results' / 'perm000_validation'
    results_dir.mkdir(parents=True, exist_ok=True)

    edge1_type = 'CbG'
    edge2_type = 'GpPW'
    metapath = 'CbGpPW'
    n_bins = 10

    print("=" * 80)
    print("Phase 0b: Permutation Stability Analysis")
    print("=" * 80)
    print(f"Metapath: {metapath}")
    print(f"Test: perm 000 vs average(001-020)")
    print(f"Results directory: {results_dir}")
    print()

    # Extract bins from permutation 000
    print("Step 1: Extracting pathway bins from permutation 000...")
    perm000_bins = extract_pathway_bins_from_single_permutation(
        edge1_type, edge2_type, 0, data_dir, n_bins
    )
    print(f"  Extracted {len(perm000_bins)} bins from perm 000")
    print(f"  Mean pathway count: {perm000_bins['pathway_count'].mean():.4f}")
    print(f"  Std pathway count: {perm000_bins['pathway_count'].std():.4f}")
    print()

    # Extract bins from permutations 001-020
    print("Step 2: Extracting average pathway bins from perms 001-020...")
    perm_ids = list(range(1, 21))
    perm_avg_bins = extract_pathway_bins_from_permutations(
        edge1_type, edge2_type, perm_ids, data_dir, n_bins
    )
    print(f"  Averaged over {len(perm_ids)} permutations")
    print(f"  Extracted {len(perm_avg_bins)} bins")
    print(f"  Mean pathway count: "
          f"{perm_avg_bins['mean_pathway_count'].mean():.4f}")
    print(f"  Std across bins: "
          f"{perm_avg_bins['mean_pathway_count'].std():.4f}")
    print()

    # Test correlation
    print("Step 3: Computing correlation...")
    correlation_results = test_original_vs_permutation_correlation(
        perm000_bins, perm_avg_bins
    )

    print(f"  Correlation (r): {correlation_results['correlation']:.4f}")
    print(f"  P-value: {correlation_results['p_value']:.2e}")
    print(f"  RMSE: {correlation_results['rmse']:.4f}")
    print(f"  MAE: {correlation_results['mae']:.4f}")
    print(f"  Hypothesis validated (r > 0.85): "
          f"{correlation_results['hypothesis_valid']}")
    print()

    # Analyze residuals
    print("Step 4: Analyzing residuals...")
    residual_results = analyze_residuals(perm000_bins, perm_avg_bins)
    stats = residual_results['statistics']

    print(f"  Mean residual: {stats['mean_residual']:.4f}")
    print(f"  Std residual: {stats['std_residual']:.4f}")
    print(f"  Mean absolute residual: {stats['mean_abs_residual']:.4f}")
    print(f"  Median absolute residual: {stats['median_abs_residual']:.4f}")
    print(f"  Max absolute residual: {stats['max_abs_residual']:.4f}")
    print(f"  95th percentile absolute residual: "
          f"{stats['q95_abs_residual']:.4f}")
    print()

    # Save results
    print("Step 5: Saving results...")

    # Save summary metrics
    summary = {
        'metapath': metapath,
        'edge1_type': edge1_type,
        'edge2_type': edge2_type,
        'perm000_vs_avg_perms': '001-020',
        'n_bins': n_bins,
        'correlation': float(correlation_results['correlation']),
        'p_value': float(correlation_results['p_value']),
        'rmse': float(correlation_results['rmse']),
        'mae': float(correlation_results['mae']),
        'hypothesis_valid': bool(correlation_results['hypothesis_valid']),
        'residual_stats': {
            k: float(v) for k, v in stats.items()
        }
    }

    summary_file = results_dir / f'{metapath}_perm000_validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {summary_file}")

    # Save merged data
    merged_data = correlation_results['merged_data']
    merged_file = results_dir / f'{metapath}_perm000_vs_perms.csv'
    merged_data.to_csv(merged_file, index=False)
    print(f"  Merged data saved: {merged_file}")

    # Save residual analysis
    residual_data = residual_results['merged_data']
    residual_file = results_dir / f'{metapath}_residual_analysis.csv'
    residual_data.to_csv(residual_file, index=False)
    print(f"  Residual analysis saved: {residual_file}")

    # Create visualization
    print("Step 6: Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1 = axes[0]
    perm000_vals = merged_data['pathway_count'].values
    perm_avg_vals = merged_data['mean_pathway_count'].values

    ax1.scatter(perm000_vals, perm_avg_vals, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)

    min_val = min(perm000_vals.min(), perm_avg_vals.min())
    max_val = max(perm000_vals.max(), perm_avg_vals.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--',
             linewidth=2, label='Perfect agreement')

    ax1.set_xlabel('Perm 000 Pathway Count', fontsize=11)
    ax1.set_ylabel('Average Perm 001-020 Pathway Count', fontsize=11)
    ax1.set_title(
        f'{metapath}: Perm 000 vs Avg(001-020)\n'
        f'r = {correlation_results["correlation"]:.4f}, '
        f'RMSE = {correlation_results["rmse"]:.3f}',
        fontsize=12, fontweight='bold'
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Residual plot
    ax2 = axes[1]
    residuals = residual_data['residual'].values
    ax2.scatter(perm_avg_vals, residuals, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)
    ax2.axhline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Average Perm 001-020 Pathway Count', fontsize=11)
    ax2.set_ylabel('Residual (Perm 000 - Avg)', fontsize=11)
    ax2.set_title(
        f'Residual Analysis\n'
        f'MAE = {correlation_results["mae"]:.3f}, '
        f'Max Error = {stats["max_abs_residual"]:.3f}',
        fontsize=12, fontweight='bold'
    )
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_file = results_dir / f'{metapath}_perm000_vs_perms.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {plot_file}")
    print()

    # Summary
    print("=" * 80)
    print("PHASE 0b RESULTS")
    print("=" * 80)
    print(f"Correlation: r = {correlation_results['correlation']:.4f}")
    print(f"Hypothesis validated (r > 0.85): "
          f"{correlation_results['hypothesis_valid']}")

    if correlation_results['hypothesis_valid']:
        print("\nSUCCESS: Perm 000 is highly correlated with permutation average")
        print(f"  Single permutation is representative of the distribution")
    else:
        print("\nFAILED: Perm 000 does not correlate well with permutation average")
        print(f"  Single permutation may not be representative")

    print("=" * 80)
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
