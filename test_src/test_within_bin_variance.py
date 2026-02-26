#!/usr/bin/env python3
"""
Test within-bin variance in pathway counts.

This script investigates whether binning loses important information for
individual node pair predictions by computing within-bin vs between-bin variance.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import compute_degree_bins


def analyze_within_bin_variance(edge1_type, edge2_type, data_dir, n_bins=10):
    """
    Analyze within-bin variance in pathway counts.

    Parameters
    ----------
    edge1_type : str
        First edge type
    edge2_type : str
        Second edge type
    data_dir : Path
        Data directory
    n_bins : int
        Number of bins

    Returns
    -------
    dict
        Variance statistics
    """
    data_dir = Path(data_dir)

    # Load edges
    edges_dir = data_dir / 'edges'
    edge1 = sp.load_npz(edges_dir / f'{edge1_type}.sparse.npz')
    edge2 = sp.load_npz(edges_dir / f'{edge2_type}.sparse.npz')

    # Compute pathways
    pathway_matrix = edge1 @ edge2

    # Compute degrees
    source_degrees = np.array(edge1.sum(axis=1)).flatten()
    target_degrees = np.array(edge2.sum(axis=0)).flatten()

    # Bin degrees
    source_bins, _ = compute_degree_bins(source_degrees, n_bins)
    target_bins, _ = compute_degree_bins(target_degrees, n_bins)

    # Analyze variance for each bin
    bin_stats = []

    for src_bin in range(n_bins):
        src_mask = source_bins == src_bin
        src_indices = np.where(src_mask)[0]

        if len(src_indices) == 0:
            continue

        for tgt_bin in range(n_bins):
            tgt_mask = target_bins == tgt_bin
            tgt_indices = np.where(tgt_mask)[0]

            if len(tgt_indices) == 0:
                continue

            # Extract pathway counts for all pairs in this bin
            pathway_sub = pathway_matrix[np.ix_(src_indices, tgt_indices)]
            pathway_counts = pathway_sub.toarray().flatten()

            # Remove zeros for cleaner statistics
            nonzero_counts = pathway_counts[pathway_counts > 0]

            if len(nonzero_counts) == 0:
                continue

            bin_stats.append({
                'src_bin': src_bin,
                'tgt_bin': tgt_bin,
                'n_pairs': len(pathway_counts),
                'n_nonzero': len(nonzero_counts),
                'mean': np.mean(nonzero_counts),
                'std': np.std(nonzero_counts),
                'min': np.min(nonzero_counts),
                'max': np.max(nonzero_counts),
                'cv': np.std(nonzero_counts) / np.mean(nonzero_counts) if np.mean(nonzero_counts) > 0 else 0,
                'src_degree_range': (source_degrees[src_indices].min(), source_degrees[src_indices].max()),
                'tgt_degree_range': (target_degrees[tgt_indices].min(), target_degrees[tgt_indices].max())
            })

    # Compute overall statistics
    all_means = [s['mean'] for s in bin_stats]
    all_stds = [s['std'] for s in bin_stats]
    all_cvs = [s['cv'] for s in bin_stats]

    # Between-bin variance
    between_bin_var = np.var(all_means)

    # Average within-bin variance
    within_bin_var = np.mean([s['std']**2 for s in bin_stats])

    return {
        'bin_stats': bin_stats,
        'between_bin_var': between_bin_var,
        'within_bin_var': within_bin_var,
        'variance_ratio': within_bin_var / between_bin_var if between_bin_var > 0 else 0,
        'mean_cv': np.mean(all_cvs),
        'n_bins': len(bin_stats)
    }


def main():
    data_dir = repo_dir / 'data'

    print("=" * 80)
    print("Within-Bin Variance Analysis")
    print("=" * 80)
    print()

    metapaths = [
        ('CbGpPW', 'CbG', 'GpPW'),
        ('CtDaG', 'CtD', 'DaG'),
        ('CrCbG', 'CrC', 'CbG')
    ]

    for name, edge1, edge2 in metapaths:
        print(f"Metapath: {name} ({edge1} -> {edge2})")
        print("-" * 80)

        results = analyze_within_bin_variance(edge1, edge2, data_dir)

        print(f"  Populated bins: {results['n_bins']}")
        print(f"  Between-bin variance: {results['between_bin_var']:.6f}")
        print(f"  Within-bin variance (avg): {results['within_bin_var']:.6f}")
        print(f"  Variance ratio (within/between): {results['variance_ratio']:.4f}")
        print(f"  Mean coefficient of variation: {results['mean_cv']:.4f}")
        print()

        # Show examples of high within-bin variance bins
        high_cv_bins = sorted(results['bin_stats'], key=lambda x: x['cv'], reverse=True)[:3]

        print("  Top 3 bins with highest within-bin variation:")
        for i, bin_stat in enumerate(high_cv_bins, 1):
            print(f"    {i}. Bin ({bin_stat['src_bin']}, {bin_stat['tgt_bin']})")
            print(f"       Mean: {bin_stat['mean']:.4f}, Std: {bin_stat['std']:.4f}, CV: {bin_stat['cv']:.4f}")
            print(f"       Pairs: {bin_stat['n_nonzero']}/{bin_stat['n_pairs']} nonzero")
            print(f"       Source degree range: {bin_stat['src_degree_range']}")
            print(f"       Target degree range: {bin_stat['tgt_degree_range']}")
        print()

        print("=" * 80)
        print()

    print("INTERPRETATION:")
    print("-" * 80)
    print("If variance_ratio << 1:")
    print("  Within-bin variance is small relative to between-bin variance")
    print("  Binning captures most of the degree-related variation")
    print("  Bin-level predictions are appropriate")
    print()
    print("If variance_ratio ~ 1 or > 1:")
    print("  Within-bin variance is comparable to between-bin variance")
    print("  Binning loses important information")
    print("  May need pair-level predictions or finer bins")
    print()
    print("Coefficient of Variation (CV = std/mean):")
    print("  CV < 0.5: Low within-bin variation (binning is good)")
    print("  CV > 1.0: High within-bin variation (binning may be problematic)")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
