"""
Test if high count outliers are topology-specific.

Key hypothesis: Individual permutations contain high counts driven by specific
topology (triangles, clustering) similar to Hetionet. Training on mean counts
smooths out these topology-specific outliers, so models can't predict them.

This would explain:
- Q-Q plots show under-prediction of high counts
- Training on mean(perms 0-4) vs testing on individual perms
- Why GNN validation was good but test was bad

Analysis:
1. Compare individual perm counts vs mean counts
2. Identify high outliers in individual perms
3. Check if outliers are consistent across perms (degree-driven) or unique (topology-driven)
4. Compare to Hetionet counts

Usage:
    python test_src/test_topology_specific_outliers.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import scipy.stats

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

from test_src.validate_mean_variance_prediction import (
    load_permuted_edge_matrices,
    sample_pairs,
    compute_pathway_counts,
    extract_degree_features
)


def analyze_topology_specific_outliers(edge1_type='CbG', edge2_type='GpPW',
                                      n_samples=10000, random_state=42):
    """
    Analyze if high counts are topology-specific or degree-driven.

    Args:
        edge1_type: First edge type
        edge2_type: Second edge type
        n_samples: Number of pairs to sample
        random_state: Random seed
    """
    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'topology_outliers'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Topology-Specific Outlier Analysis")
    print("="*70)

    print(f"\nLoading data for {edge1_type}+{edge2_type}...")
    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(
        edge1_type, edge2_type, 0, data_dir
    )

    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples=n_samples,
                        random_state=random_state)
    X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)

    print(f"  Sampled {len(pairs)} pairs")

    train_perms = list(range(5))
    test_perms = list(range(15, 21))

    print("\nComputing counts for training and test permutations...")
    counts_train = []
    for perm in train_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_train.append(counts)
        print(f"  Perm {perm}: mean={counts.mean():.2f}, max={counts.max():.0f}")

    counts_train = np.column_stack(counts_train)
    mean_train = counts_train.mean(axis=1)

    counts_test = []
    for perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)
        print(f"  Perm {perm}: mean={counts.mean():.2f}, max={counts.max():.0f}")

    counts_test = np.column_stack(counts_test)

    print("\n" + "="*70)
    print("Analysis 1: Mean vs Individual Permutation Counts")
    print("="*70)

    print(f"\nMean of training perms (0-4):")
    print(f"  Mean: {mean_train.mean():.3f}")
    print(f"  Std: {mean_train.std():.3f}")
    print(f"  Max: {mean_train.max():.0f}")
    print(f"  95th percentile: {np.percentile(mean_train, 95):.1f}")
    print(f"  99th percentile: {np.percentile(mean_train, 99):.1f}")

    print(f"\nIndividual training permutations:")
    for i, perm in enumerate(train_perms):
        counts = counts_train[:, i]
        print(f"  Perm {perm}: mean={counts.mean():.3f}, max={counts.max():.0f}, "
              f"95th={np.percentile(counts, 95):.1f}, 99th={np.percentile(counts, 99):.1f}")

    print("\n" + "="*70)
    print("Analysis 2: High Count Outliers")
    print("="*70)

    high_mean_threshold = np.percentile(mean_train, 99)
    print(f"\n99th percentile of mean counts: {high_mean_threshold:.1f}")

    high_in_mean = mean_train > high_mean_threshold
    print(f"Pairs with high mean count: {high_in_mean.sum()} ({100*high_in_mean.mean():.1f}%)")

    high_in_perms = counts_train > high_mean_threshold
    print(f"\nHigh counts in individual training perms:")
    for i, perm in enumerate(train_perms):
        high_count = high_in_perms[:, i].sum()
        print(f"  Perm {perm}: {high_count} pairs ({100*high_count/len(pairs):.1f}%)")

    print("\n" + "="*70)
    print("Analysis 3: Consistency of High Counts Across Permutations")
    print("="*70)

    print("\nFor pairs with high mean count, how often are they high in each perm?")
    high_mean_pairs = np.where(high_in_mean)[0]

    consistency_counts = []
    for pair_idx in high_mean_pairs:
        n_high = (counts_train[pair_idx, :] > high_mean_threshold).sum()
        consistency_counts.append(n_high)

    consistency_counts = np.array(consistency_counts)
    print(f"  Always high (5/5 perms): {(consistency_counts == 5).sum()} pairs")
    print(f"  Usually high (4/5 perms): {(consistency_counts == 4).sum()} pairs")
    print(f"  Sometimes high (3/5 perms): {(consistency_counts == 3).sum()} pairs")
    print(f"  Rarely high (1-2/5 perms): {(consistency_counts <= 2).sum()} pairs")

    print("\n" + "="*70)
    print("Analysis 4: Topology-Specific Outliers")
    print("="*70)

    print("\nIdentify pairs that are high in some perms but not others:")
    print("(These are topology-specific, not degree-driven)")

    any_high_in_train = (counts_train > high_mean_threshold).any(axis=1)
    print(f"\nPairs high in at least one training perm: {any_high_in_train.sum()}")
    print(f"Pairs high in mean: {high_in_mean.sum()}")
    print(f"Topology-specific outliers: {any_high_in_train.sum() - high_in_mean.sum()}")

    topology_specific = any_high_in_train & ~high_in_mean
    print(f"\nTopology-specific outlier pairs: {topology_specific.sum()} "
          f"({100*topology_specific.mean():.1f}%)")

    print("\n" + "="*70)
    print("Analysis 5: Test Set Outliers")
    print("="*70)

    print("\nHow many high counts in test perms?")
    for i, perm in enumerate(test_perms):
        counts = counts_test[:, i]
        n_high = (counts > high_mean_threshold).sum()
        print(f"  Perm {perm}: {n_high} high counts ({100*n_high/len(pairs):.1f}%)")

    print("\nFor pairs with high mean count (from training), how often are they high in test?")
    test_consistency = []
    for pair_idx in high_mean_pairs:
        n_high = (counts_test[pair_idx, :] > high_mean_threshold).sum()
        test_consistency.append(n_high)

    test_consistency = np.array(test_consistency)
    print(f"  Mean fraction of test perms with high count: {test_consistency.mean()/6:.2f}")
    print(f"  Expected if consistent: ~1.0")
    print(f"  Expected if random: ~0.01")

    print("\n" + "="*70)
    print("Analysis 6: Degree Features of Outliers")
    print("="*70)

    print("\nCompare degree features for:")
    print("1. Consistent high count (high in mean)")
    print("2. Topology-specific high count (high in some perms, not mean)")
    print("3. Never high count")

    never_high = ~any_high_in_train

    print(f"\nDegree product statistics:")
    deg_product = X[:, 2]  # deg_src * deg_tgt
    print(f"  Consistent high: mean={deg_product[high_in_mean].mean():.0f}, "
          f"median={np.median(deg_product[high_in_mean]):.0f}")
    print(f"  Topology-specific: mean={deg_product[topology_specific].mean():.0f}, "
          f"median={np.median(deg_product[topology_specific]):.0f}")
    print(f"  Never high: mean={deg_product[never_high].mean():.0f}, "
          f"median={np.median(deg_product[never_high]):.0f}")

    print("\n" + "="*70)
    print("Analysis 7: Variance Across Permutations")
    print("="*70)

    var_train = counts_train.var(axis=1)

    print(f"\nVariance of counts across training perms:")
    print(f"  Overall mean variance: {var_train.mean():.3f}")
    print(f"  Consistent high pairs: {var_train[high_in_mean].mean():.3f}")
    print(f"  Topology-specific pairs: {var_train[topology_specific].mean():.3f}")
    print(f"  Never high pairs: {var_train[never_high].mean():.3f}")

    print("\nInterpretation:")
    if var_train[topology_specific].mean() > var_train[high_in_mean].mean():
        print("  Topology-specific pairs have HIGHER variance")
        print("  -> Their high counts are driven by permutation-specific topology")
        print("  -> Training on means smooths out these outliers")
        print("  -> Models cannot predict them from degree features alone")
    else:
        print("  Consistent pairs have HIGHER variance")
        print("  -> High counts are more stable across permutations")

    results = {
        'mean_train': mean_train,
        'counts_train': counts_train,
        'counts_test': counts_test,
        'high_in_mean': high_in_mean,
        'topology_specific': topology_specific,
        'X': X,
        'pairs': pairs
    }

    create_plots(results, output_dir)

    return results


def create_plots(results, output_dir):
    """Create visualization plots."""
    mean_train = results['mean_train']
    counts_train = results['counts_train']
    counts_test = results['counts_test']
    high_in_mean = results['high_in_mean']
    topology_specific = results['topology_specific']
    X = results['X']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.hist(mean_train, bins=50, alpha=0.5, label='Mean of perms 0-4', density=True)
    ax.hist(counts_train[:, 0], bins=50, alpha=0.5, label='Perm 0', density=True)
    ax.set_xlabel('Pathway Count')
    ax.set_ylabel('Density')
    ax.set_title('Mean vs Individual Permutation Counts')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[0, 1]
    var_train = counts_train.var(axis=1)
    ax.scatter(mean_train, var_train, alpha=0.1, s=1)
    ax.set_xlabel('Mean Count')
    ax.set_ylabel('Variance Across Perms')
    ax.set_title('Mean-Variance Relationship')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax = axes[1, 0]
    deg_product = X[:, 2]
    ax.scatter(deg_product[~high_in_mean], var_train[~high_in_mean],
              alpha=0.1, s=1, label='Normal', c='blue')
    ax.scatter(deg_product[high_in_mean], var_train[high_in_mean],
              alpha=0.5, s=10, label='Consistent high', c='red')
    ax.scatter(deg_product[topology_specific], var_train[topology_specific],
              alpha=0.5, s=10, label='Topology-specific', c='orange')
    ax.set_xlabel('Degree Product')
    ax.set_ylabel('Variance Across Perms')
    ax.set_title('Variance by Degree and Count Level')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    ax = axes[1, 1]
    consistency = (counts_train > np.percentile(mean_train, 99)).sum(axis=1)
    ax.hist(consistency, bins=np.arange(7)-0.5, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of perms with high count')
    ax.set_ylabel('Number of pairs')
    ax.set_title('Consistency of High Counts Across Perms')
    ax.set_xticks(range(6))

    plt.tight_layout()
    output_file = output_dir / 'topology_specific_outliers.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_file}")
    plt.close()


def main():
    results = analyze_topology_specific_outliers()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    topology_frac = results['topology_specific'].mean()
    high_frac = results['high_in_mean'].mean()

    print(f"\nFraction of pairs with high mean count: {high_frac:.3f}")
    print(f"Fraction of pairs with topology-specific high counts: {topology_frac:.3f}")

    if topology_frac > 0.01:
        print("\nTOPOLOGY-SPECIFIC OUTLIERS ARE SIGNIFICANT")
        print("This explains:")
        print("  1. Q-Q plots show under-prediction of high counts")
        print("  2. Training on mean counts smooths out topology-specific highs")
        print("  3. Testing on individual perms encounters these highs")
        print("  4. Degree features alone cannot predict them")
        print("\nImplication:")
        print("  - r=0.78 ceiling is partly due to unpredictable topology-specific outliers")
        print("  - These outliers are permutation-specific, like Hetionet-specific outliers")
        print("  - To predict them, need topology features or train on individual perms")
    else:
        print("\nHigh counts are mostly consistent across permutations")
        print("Topology-specific outliers are rare")


if __name__ == '__main__':
    main()
