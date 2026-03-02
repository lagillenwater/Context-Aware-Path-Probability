#!/usr/bin/env python3
"""
Create visualization for Phase 0 correlation.

Plots original graph pathway counts vs permutation average pathway counts.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))


def main():
    """
    Create Phase 0 visualization.
    """
    results_dir = repo_dir / 'results' / 'hypothesis_validation'
    output_dir = repo_dir / 'results' / 'hypothesis_validation'

    # Load Phase 0 results
    results_file = results_dir / 'CbGpPW_hypothesis_validation_summary.json'
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return 1

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Load bin data from CSV
    csv_file = results_dir / 'CbGpPW_original_vs_perms.csv'
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return 1

    import pandas as pd
    df = pd.read_csv(csv_file)
    original_counts = df['original_pathway_count'].values
    perm_avg_counts = df['permutation_avg_pathway_count'].values

    r = results['correlation_r']
    p_value = results['correlation_p_value']

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(original_counts, perm_avg_counts, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)

    min_val = min(original_counts.min(), perm_avg_counts.min())
    max_val = max(original_counts.max(), perm_avg_counts.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--',
             linewidth=2, label='Perfect correlation')

    ax1.set_xlabel('Original Graph Pathway Count', fontsize=11)
    ax1.set_ylabel('Permutation Average Pathway Count', fontsize=11)
    ax1.set_title(
        f'Phase 0: Original vs Permutation Average (CbGpPW)\n'
        f'r = {r:.4f}, p = {p_value:.2e}',
        fontsize=12, fontweight='bold'
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Residual plot
    ax2 = axes[1]
    residuals = perm_avg_counts - original_counts
    ax2.scatter(original_counts, residuals, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)
    ax2.axhline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Original Graph Pathway Count', fontsize=11)
    ax2.set_ylabel('Residual (Perm Avg - Original)', fontsize=11)
    ax2.set_title(
        f'Residual Analysis\n'
        f'Mean Error = {residuals.mean():.4f}, '
        f'Std = {residuals.std():.4f}',
        fontsize=12, fontweight='bold'
    )
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'CbGpPW_phase0_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Phase 0 visualization saved: {output_file}")
    print(f"  Correlation: r = {r:.4f}, p = {p_value:.2e}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
