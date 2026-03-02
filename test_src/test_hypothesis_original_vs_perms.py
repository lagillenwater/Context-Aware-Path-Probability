#!/usr/bin/env python3
"""
Test Phase 0 hypothesis: Original graph predicts permutation averages.

This script validates the core hypothesis that training on the original
Hetionet graph can predict average pathway counts across permutations.

Usage:
    python test_src/test_hypothesis_original_vs_perms.py
    python test_src/test_hypothesis_original_vs_perms.py --metapath CtDaG
    python test_src/test_hypothesis_original_vs_perms.py --n-bins 15
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import json
import argparse

# Add src to path
repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from permutation_validation import (
    extract_pathway_bins_from_graph,
    extract_pathway_bins_from_permutations,
    test_original_vs_permutation_correlation,
    analyze_residuals
)


def parse_metapath(metapath_str):
    """
    Parse metapath string into edge types.

    Parameters
    ----------
    metapath_str : str
        Metapath code (e.g., 'CbGpPW')

    Returns
    -------
    tuple
        (edge1_type, edge2_type)
    """
    # Common 2-hop metapaths
    metapath_map = {
        'CbGpPW': ('CbG', 'GpPW'),
        'GiGiG': ('GiG', 'GiG'),
        'CtDaG': ('CtD', 'DaG'),
        'CbGaD': ('CbG', 'GaD'),
        'CrCbG': ('CrC', 'CbG'),
        'CbGiG': ('CbG', 'GiG'),
        'CpDaG': ('CpD', 'DaG'),
        'CbGpBP': ('CbG', 'GpBP'),
    }

    if metapath_str in metapath_map:
        return metapath_map[metapath_str]
    else:
        raise ValueError(f"Unknown metapath: {metapath_str}")


def load_original_graph_edges(edge1_type, edge2_type, data_dir):
    """
    Load edge matrices from original Hetionet graph.

    Parameters
    ----------
    edge1_type : str
        First edge type
    edge2_type : str
        Second edge type
    data_dir : Path
        Data directory

    Returns
    -------
    tuple
        (edge1_matrix, edge2_matrix)
    """
    edges_dir = data_dir / 'edges'
    edge1_file = edges_dir / f'{edge1_type}.sparse.npz'
    edge2_file = edges_dir / f'{edge2_type}.sparse.npz'

    if not edge1_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge1_file}")
    if not edge2_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge2_file}")

    edge1 = sp.load_npz(edge1_file)
    edge2 = sp.load_npz(edge2_file)

    return edge1, edge2


def plot_original_vs_permutation(results, metapath, output_dir):
    """
    Create scatter plot of original vs permutation averages.

    Parameters
    ----------
    results : dict
        Results from test_original_vs_permutation_correlation
    metapath : str
        Metapath code
    output_dir : Path
        Output directory for plot
    """
    merged = results['merged_data']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(
        merged['mean_pathway_count'],
        merged['pathway_count'],
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )

    # Add diagonal line (perfect correlation)
    min_val = min(
        merged['mean_pathway_count'].min(),
        merged['pathway_count'].min()
    )
    max_val = max(
        merged['mean_pathway_count'].max(),
        merged['pathway_count'].max()
    )
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect correlation')

    ax1.set_xlabel('Permutation Average Pathway Count', fontsize=11)
    ax1.set_ylabel('Original Graph Pathway Count', fontsize=11)
    ax1.set_title(
        f'{metapath}: Original vs Permutation Average\n'
        f'r = {results["correlation"]:.3f} (p = {results["p_value"]:.2e})',
        fontsize=12,
        fontweight='bold'
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Residual plot
    ax2 = axes[1]
    merged['residual'] = (
        merged['pathway_count'] - merged['mean_pathway_count']
    )
    ax2.scatter(
        merged['mean_pathway_count'],
        merged['residual'],
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    ax2.axhline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Permutation Average Pathway Count', fontsize=11)
    ax2.set_ylabel('Residual (Original - Permutation)', fontsize=11)
    ax2.set_title(
        f'Residual Analysis\n'
        f'RMSE = {results["rmse"]:.2f}, MAE = {results["mae"]:.2f}',
        fontsize=12,
        fontweight='bold'
    )
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'{metapath}_original_vs_perms.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_file}")


def save_results(
    results,
    original_bins,
    permutation_bins,
    residual_analysis,
    metapath,
    n_bins,
    output_dir
):
    """
    Save all results to files.

    Parameters
    ----------
    results : dict
        Correlation results
    original_bins : pd.DataFrame
        Original graph bins
    permutation_bins : pd.DataFrame
        Permutation average bins
    residual_analysis : dict
        Residual analysis results
    metapath : str
        Metapath code
    n_bins : int
        Number of bins used
    output_dir : Path
        Output directory
    """
    # Save summary JSON
    summary = {
        'metapath': metapath,
        'n_bins': n_bins,
        'correlation': float(results['correlation']),
        'p_value': float(results['p_value']),
        'rmse': float(results['rmse']),
        'mae': float(results['mae']),
        'n_bin_pairs': int(results['n_bins']),
        'hypothesis_valid': bool(results['hypothesis_valid']),
        'threshold': 0.85,
        'residual_statistics': {
            k: float(v) for k, v in
            residual_analysis['statistics'].items()
        }
    }

    summary_file = output_dir / f'{metapath}_hypothesis_validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")

    # Save merged data CSV
    merged_file = output_dir / f'{metapath}_original_vs_perms.csv'
    results['merged_data'].to_csv(merged_file, index=False)
    print(f"Merged data saved to: {merged_file}")

    # Save residual analysis
    residual_file = output_dir / f'{metapath}_residual_analysis.csv'
    residual_analysis['merged_data'].to_csv(residual_file, index=False)
    print(f"Residual analysis saved to: {residual_file}")


def main():
    """
    Main hypothesis validation function.
    """
    parser = argparse.ArgumentParser(
        description='Test Phase 0 hypothesis: Original graph predicts '
                    'permutation averages'
    )
    parser.add_argument(
        '--metapath',
        default='CbGpPW',
        help='Metapath to test (default: CbGpPW)'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=10,
        help='Number of bins (default: 10)'
    )
    parser.add_argument(
        '--perm-start',
        type=int,
        default=0,
        help='First permutation ID (default: 0)'
    )
    parser.add_argument(
        '--perm-end',
        type=int,
        default=19,
        help='Last permutation ID (default: 19)'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results' / 'hypothesis_validation'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 0: Hypothesis Validation")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Number of bins: {args.n_bins}")
    print(f"Permutations: {args.perm_start:03d}-{args.perm_end:03d}")
    print(f"Results directory: {results_dir}")
    print()

    # Parse metapath
    edge1_type, edge2_type = parse_metapath(args.metapath)
    print(f"Edge types: {edge1_type} -> {edge2_type}")
    print()

    # Step 1: Extract bins from original graph
    print("Step 1: Extracting pathway bins from original Hetionet graph...")
    print("  Loading from: data/edges/")
    edge1, edge2 = load_original_graph_edges(edge1_type, edge2_type, data_dir)
    print(f"  Edge1 shape: {edge1.shape}")
    print(f"  Edge2 shape: {edge2.shape}")
    print(f"  Edge1 density: {edge1.nnz / np.prod(edge1.shape):.6f}")
    print(f"  Edge2 density: {edge2.nnz / np.prod(edge2.shape):.6f}")

    original_bins = extract_pathway_bins_from_graph(edge1, edge2, args.n_bins)
    print(f"  Extracted {len(original_bins)} bin pairs from original graph")
    print(f"  Mean pathway count: {original_bins['pathway_count'].mean():.2f}")
    print(f"  Std pathway count: {original_bins['pathway_count'].std():.2f}")
    print()

    # Step 2: Extract bins from permutations
    print(f"Step 2: Extracting pathway bins from permutations "
          f"{args.perm_start:03d}-{args.perm_end:03d}...")
    perm_ids = list(range(args.perm_start, args.perm_end + 1))
    permutation_bins = extract_pathway_bins_from_permutations(
        edge1_type, edge2_type, perm_ids, data_dir, args.n_bins
    )
    print(f"  Extracted {len(permutation_bins)} bin pairs from permutations")
    print(f"  Mean pathway count: "
          f"{permutation_bins['mean_pathway_count'].mean():.2f}")
    print(f"  Mean std across perms: "
          f"{permutation_bins['std_pathway_count'].mean():.2f}")
    print()

    # Step 3: Test correlation
    print("Step 3: Testing correlation between original and permutation "
          "averages...")
    results = test_original_vs_permutation_correlation(
        original_bins, permutation_bins
    )

    print(f"  Correlation (r): {results['correlation']:.4f}")
    print(f"  P-value: {results['p_value']:.4e}")
    print(f"  RMSE: {results['rmse']:.2f}")
    print(f"  MAE: {results['mae']:.2f}")
    print(f"  Number of bin pairs: {results['n_bins']}")
    print()

    # Step 4: Analyze residuals
    print("Step 4: Analyzing residuals...")
    residual_analysis = analyze_residuals(original_bins, permutation_bins)
    stats = residual_analysis['statistics']
    print(f"  Mean residual: {stats['mean_residual']:.2f}")
    print(f"  Std residual: {stats['std_residual']:.2f}")
    print(f"  Mean absolute residual: {stats['mean_abs_residual']:.2f}")
    print(f"  Median absolute residual: {stats['median_abs_residual']:.2f}")
    print(f"  Max absolute residual: {stats['max_abs_residual']:.2f}")
    print(f"  95th percentile abs residual: {stats['q95_abs_residual']:.2f}")
    print()

    # Step 5: Make decision
    print("=" * 80)
    print("HYPOTHESIS VALIDATION RESULT")
    print("=" * 80)
    print(f"Correlation: r = {results['correlation']:.4f}")
    print(f"Threshold: r > 0.85")
    print()

    if results['hypothesis_valid']:
        print("HYPOTHESIS VALIDATED")
        print("  Original graph bins CAN predict permutation averages")
        print("  Proceed with Phase 1: Train on original Hetionet graph")
    else:
        print("HYPOTHESIS REJECTED")
        print("  Original graph bins CANNOT predict permutation averages")
        print("  Must train on permutations 000-019 instead")
        print("  All subsequent phases need re-architecture")

    print("=" * 80)
    print()

    # Step 6: Save results
    print("Step 5: Saving results...")
    save_results(
        results,
        original_bins,
        permutation_bins,
        residual_analysis,
        args.metapath,
        args.n_bins,
        results_dir
    )

    # Step 7: Create visualization
    print("Step 6: Creating visualization...")
    plot_original_vs_permutation(results, args.metapath, results_dir)

    print()
    print("Phase 0 complete!")

    return 0 if results['hypothesis_valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
