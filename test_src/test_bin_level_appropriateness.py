#!/usr/bin/env python3
"""
Test whether bin-level predictions are appropriate for anomaly detection.

The key question: For a given bin, do individual node pairs have pathway counts
that are close to the bin mean? Or is there high variation within bins?
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import compute_degree_bins


def analyze_bin_appropriateness(edge1_type, edge2_type, data_dir, n_bins=10):
    """
    Analyze whether bin-level means are good representations of individual pairs.
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

    # For each bin, compute deviations from bin mean
    all_deviations = []
    all_bin_means = []
    bin_info = []

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

            # Compute bin mean (what we predict)
            bin_mean = np.mean(pathway_counts)

            if bin_mean == 0:
                continue

            # Compute deviations from bin mean
            deviations = pathway_counts - bin_mean
            abs_deviations = np.abs(deviations)
            rel_deviations = abs_deviations / bin_mean if bin_mean > 0 else abs_deviations

            all_deviations.extend(abs_deviations.tolist())
            all_bin_means.extend([bin_mean] * len(pathway_counts))

            bin_info.append({
                'src_bin': src_bin,
                'tgt_bin': tgt_bin,
                'mean': bin_mean,
                'n_pairs': len(pathway_counts),
                'mean_abs_dev': np.mean(abs_deviations),
                'max_abs_dev': np.max(abs_deviations),
                'mean_rel_dev': np.mean(rel_deviations),
                'pct_within_2x': np.sum((pathway_counts >= bin_mean/2) & (pathway_counts <= bin_mean*2)) / len(pathway_counts) * 100
            })

    all_deviations = np.array(all_deviations)
    all_bin_means = np.array(all_bin_means)

    return {
        'bin_info': bin_info,
        'all_deviations': all_deviations,
        'all_bin_means': all_bin_means,
        'n_bins': len(bin_info)
    }


def main():
    data_dir = repo_dir / 'data'

    print("=" * 80)
    print("Bin-Level Prediction Appropriateness Analysis")
    print("=" * 80)
    print()
    print("Question: For individual node pairs, how close are their pathway counts")
    print("          to the bin mean that we predict?")
    print()

    metapaths = [
        ('CbGpPW', 'CbG', 'GpPW'),
        ('CtDaG', 'CtD', 'DaG'),
        ('CrCbG', 'CrC', 'CbG')
    ]

    for name, edge1, edge2 in metapaths:
        print(f"Metapath: {name} ({edge1} -> {edge2})")
        print("-" * 80)

        results = analyze_bin_appropriateness(edge1, edge2, data_dir)

        print(f"  Populated bins: {results['n_bins']}")
        print()

        # Overall statistics
        mean_deviations = [b['mean_abs_dev'] for b in results['bin_info']]
        mean_rel_deviations = [b['mean_rel_dev'] for b in results['bin_info']]
        pct_within_2x = [b['pct_within_2x'] for b in results['bin_info']]

        print(f"  Mean absolute deviation from bin mean: {np.mean(mean_deviations):.6f}")
        print(f"  Mean relative deviation: {np.mean(mean_rel_deviations):.4f} ({np.mean(mean_rel_deviations)*100:.1f}%)")
        print(f"  Average % of pairs within 2x of bin mean: {np.mean(pct_within_2x):.1f}%")
        print()

        # Show bins with highest deviation
        high_dev_bins = sorted(results['bin_info'], key=lambda x: x['mean_rel_dev'], reverse=True)[:3]

        print("  Top 3 bins with highest relative deviation from mean:")
        for i, bin_stat in enumerate(high_dev_bins, 1):
            print(f"    {i}. Bin ({bin_stat['src_bin']}, {bin_stat['tgt_bin']})")
            print(f"       Bin mean: {bin_stat['mean']:.6f}")
            print(f"       Mean abs deviation: {bin_stat['mean_abs_dev']:.6f}")
            print(f"       Mean rel deviation: {bin_stat['mean_rel_dev']:.4f} ({bin_stat['mean_rel_dev']*100:.1f}%)")
            print(f"       % within 2x of mean: {bin_stat['pct_within_2x']:.1f}%")
            print(f"       Pairs in bin: {bin_stat['n_pairs']}")
        print()

        print("=" * 80)
        print()

    print("INTERPRETATION:")
    print("-" * 80)
    print("If mean relative deviation < 0.5 (50%):")
    print("  Pairs are generally close to bin mean")
    print("  Bin-level predictions are appropriate for anomaly detection")
    print()
    print("If > 80% of pairs within 2x of bin mean:")
    print("  Most pairs have similar pathway counts")
    print("  Bin represents a meaningful 'degree class'")
    print()
    print("If mean relative deviation > 1.0 (100%) or < 80% within 2x:")
    print("  High within-bin variation")
    print("  May need pair-specific predictions or finer binning")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
