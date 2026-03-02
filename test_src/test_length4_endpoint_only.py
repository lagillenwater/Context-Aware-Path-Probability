"""
Test endpoint-only prediction on length-4 metapath (CbGiGpPW).

This script tests whether the r=0.78 performance on CbGpPW (length-3) extends
to CbGiGpPW (length-4) using only endpoint degrees.

Goal: Quantify performance degradation as path length increases.

Approach:
1. Sample (Compound, Pathway) pairs
2. Compute 3-hop counts: CbG @ GiG @ GpPW for perms 0-4 (train) and 15-19 (test)
3. Train Random Forest on endpoint features: [deg_C, deg_PW, products, squares]
4. Evaluate on test permutations
5. Compare to length-3 baseline (r=0.778, Q-Q=0.815)

Usage:
    python test_src/test_length4_endpoint_only.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

from test_src.validate_mean_variance_prediction import (
    load_permuted_edge_matrices,
    sample_pairs,
    compute_pathway_counts,
    extract_degree_features
)


def compute_3hop_pathway_counts(pairs, edge1, edge2, edge3):
    """
    Compute 3-hop pathway counts: edge1 @ edge2 @ edge3.

    Args:
        pairs: Array of (source_idx, target_idx), shape (n, 2)
        edge1: First edge sparse matrix (e.g., CbG)
        edge2: Second edge sparse matrix (e.g., GiG)
        edge3: Third edge sparse matrix (e.g., GpPW)

    Returns:
        Array of pathway counts, shape (n,)
    """
    pathway = edge1 @ edge2 @ edge3

    counts = np.zeros(len(pairs))
    for i, (src, tgt) in enumerate(pairs):
        counts[i] = pathway[src, tgt]

    return counts


def analyze_length4_endpoint(
    edge1_type='CbG',
    edge2_type='GiG',
    edge3_type='GpPW',
    n_samples=10000,
    random_state=42
):
    """
    Analyze endpoint-only prediction on length-4 metapath.

    Args:
        edge1_type: First edge type
        edge2_type: Second edge type (intermediate)
        edge3_type: Third edge type
        n_samples: Number of pairs to sample
        random_state: Random seed
    """
    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'length4_endpoint'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Length-4 Endpoint-Only Prediction Analysis")
    print("="*70)
    print(f"\nMetapath: {edge1_type}+{edge2_type}+{edge3_type}")
    print(f"Samples: {n_samples}")

    # Load edges from perm 0 for sampling
    print("\nLoading edges for sampling...")
    edge1_perm0, _ = load_permuted_edge_matrices(edge1_type, edge2_type, 0, data_dir)
    _, edge3_perm0 = load_permuted_edge_matrices(edge2_type, edge3_type, 0, data_dir)

    # Sample pairs based on 3-hop paths in perm 0
    print("Sampling (Compound, Pathway) pairs...")
    edge2_perm0, _ = load_permuted_edge_matrices(edge2_type, edge2_type, 0, data_dir)

    # For 3-hop sampling, use edge1 @ edge2 @ edge3
    pathway_perm0 = edge1_perm0 @ edge2_perm0 @ edge3_perm0

    # Sample pairs (50% with pathways, 50% random)
    np.random.seed(random_state)
    pathway_coo = pathway_perm0.tocoo()
    pathway_pairs = np.column_stack([pathway_coo.row, pathway_coo.col])

    n_with_pathways = n_samples // 2
    if len(pathway_pairs) > n_with_pathways:
        idx = np.random.choice(len(pathway_pairs), n_with_pathways, replace=False)
        sampled_with_pathways = pathway_pairs[idx]
    else:
        sampled_with_pathways = pathway_pairs

    n_random = n_samples - len(sampled_with_pathways)
    n_sources = edge1_perm0.shape[0]
    n_targets = edge3_perm0.shape[1]

    random_sources = np.random.randint(0, n_sources, n_random)
    random_targets = np.random.randint(0, n_targets, n_random)
    random_pairs = np.column_stack([random_sources, random_targets])

    pairs = np.vstack([sampled_with_pathways, random_pairs])
    np.random.shuffle(pairs)

    print(f"  Sampled {len(pairs)} pairs")

    # Extract endpoint features (deg_C, deg_PW)
    print("\nExtracting endpoint degree features...")
    source_degrees = np.asarray(edge1_perm0.sum(axis=1)).ravel()
    target_degrees = np.asarray(edge3_perm0.sum(axis=0)).ravel()

    deg_src = source_degrees[pairs[:, 0]]
    deg_tgt = target_degrees[pairs[:, 1]]

    X = np.column_stack([
        deg_src,
        deg_tgt,
        deg_src * deg_tgt,
        deg_src ** 2,
        deg_tgt ** 2
    ])

    print(f"  Feature matrix: {X.shape}")
    print(f"  Source degree range: [{deg_src.min():.0f}, {deg_src.max():.0f}]")
    print(f"  Target degree range: [{deg_tgt.min():.0f}, {deg_tgt.max():.0f}]")

    # Compute training counts (perms 0-4)
    train_perms = list(range(5))
    test_perms = list(range(15, 20))

    print(f"\nComputing 3-hop counts for training perms {train_perms}...")
    counts_train = []
    for perm in train_perms:
        print(f"  Perm {perm}...")
        edge1, _ = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        edge2, _ = load_permuted_edge_matrices(edge2_type, edge3_type, perm, data_dir)
        _, edge3 = load_permuted_edge_matrices(edge2_type, edge3_type, perm, data_dir)

        counts = compute_3hop_pathway_counts(pairs, edge1, edge2, edge3)
        counts_train.append(counts)

    counts_train = np.column_stack(counts_train)
    mean_train = counts_train.mean(axis=1)
    std_train = counts_train.std(axis=1)

    print(f"\nTraining count statistics:")
    print(f"  Mean: {mean_train.mean():.3f} ± {mean_train.std():.3f}")
    print(f"  Range: [{mean_train.min():.3f}, {mean_train.max():.3f}]")
    print(f"  Pairs with pathways: {(mean_train > 0).sum()} ({100*(mean_train > 0).mean():.1f}%)")

    # Train Random Forest (consistent with Nov 11 analysis)
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, mean_train)

    # Compute test counts
    print(f"\nComputing 3-hop counts for test perms {test_perms}...")
    counts_test = []
    for perm in test_perms:
        print(f"  Perm {perm}...")
        edge1, _ = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        edge2, _ = load_permuted_edge_matrices(edge2_type, edge3_type, perm, data_dir)
        _, edge3 = load_permuted_edge_matrices(edge2_type, edge3_type, perm, data_dir)

        counts = compute_3hop_pathway_counts(pairs, edge1, edge2, edge3)
        counts_test.append(counts)

    counts_test = np.column_stack(counts_test)

    # Make predictions
    predictions = model.predict(X)

    # Evaluate on each test permutation
    print("\nEvaluating on test permutations...")
    results = []

    for test_idx, perm in enumerate(test_perms):
        actual = counts_test[:, test_idx]

        # Correlation
        r = np.corrcoef(actual, predictions)[0, 1]

        # MAE
        mae = np.abs(actual - predictions).mean()

        # Q-Q correlation
        qq_corr = stats.probplot(actual - predictions)[1][2]

        # RMSE
        rmse = np.sqrt(((actual - predictions)**2).mean())

        results.append({
            'perm': perm,
            'r': r,
            'mae': mae,
            'rmse': rmse,
            'qq': qq_corr
        })

        print(f"  Perm {perm}: r={r:.3f}, MAE={mae:.3f}, Q-Q={qq_corr:.3f}")

    results_df = pd.DataFrame(results)

    # Summary statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nLength-4 (CbGiGpPW) endpoint-only performance:")
    print(f"  r:     {results_df['r'].mean():.3f} ± {results_df['r'].std():.3f}")
    print(f"  MAE:   {results_df['mae'].mean():.3f} ± {results_df['mae'].std():.3f}")
    print(f"  RMSE:  {results_df['rmse'].mean():.3f} ± {results_df['rmse'].std():.3f}")
    print(f"  Q-Q:   {results_df['qq'].mean():.3f} ± {results_df['qq'].std():.3f}")

    print(f"\nBaseline (length-3 CbGpPW from Nov 11):")
    print(f"  r:     0.778")
    print(f"  Q-Q:   0.815")

    # Compute degradation
    length3_r = 0.778
    length4_r = results_df['r'].mean()
    degradation = length3_r - length4_r
    percent_degradation = 100 * degradation / length3_r

    print(f"\nPerformance degradation:")
    print(f"  Absolute: {degradation:.3f}")
    print(f"  Percent:  {percent_degradation:.1f}%")

    # Save results
    results_file = output_dir / 'CbGiGpPW_endpoint_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(
        pairs, X, mean_train, counts_test, predictions,
        results_df, output_dir
    )

    return results_df


def generate_visualizations(pairs, X, mean_train, counts_test, predictions, results_df, output_dir):
    """Generate comparison visualizations."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Predicted vs Actual (test perm 15)
    ax = axes[0, 0]
    actual_15 = counts_test[:, 0]
    ax.scatter(predictions, actual_15, alpha=0.3, s=10)
    max_val = max(predictions.max(), actual_15.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2)
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Actual Count (Perm 15)')
    ax.set_title(f'Length-4: r={results_df.iloc[0]["r"]:.3f}')
    ax.grid(True, alpha=0.3)

    # Plot 2: Residuals
    ax = axes[0, 1]
    residuals = actual_15 - predictions
    ax.scatter(predictions, residuals, alpha=0.3, s=10)
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Residuals (Perm 15)')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)

    # Plot 3: Q-Q plot
    ax = axes[0, 2]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot: r={results_df.iloc[0]["qq"]:.3f}')
    ax.grid(True, alpha=0.3)

    # Plot 4: Performance across test perms
    ax = axes[1, 0]
    ax.plot(results_df['perm'], results_df['r'], 'o-', label='Length-4 (CbGiGpPW)', lw=2)
    ax.axhline(0.778, color='red', linestyle='--', lw=2, label='Length-3 (CbGpPW)')
    ax.set_xlabel('Test Permutation')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Performance Across Test Permutations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: MAE comparison
    ax = axes[1, 1]
    ax.bar(['Length-3\n(CbGpPW)', 'Length-4\n(CbGiGpPW)'],
           [np.nan, results_df['mae'].mean()],
           color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('MAE Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Degree distribution
    ax = axes[1, 2]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=10, c=mean_train, cmap='viridis')
    ax.set_xlabel('Source Degree (Compound)')
    ax.set_ylabel('Target Degree (Pathway)')
    ax.set_title('Degree Distribution (colored by mean count)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.colorbar(ax.collections[0], ax=ax, label='Mean Count')

    plt.tight_layout()

    output_file = output_dir / 'length3_vs_length4_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    results = analyze_length4_endpoint(
        edge1_type='CbG',
        edge2_type='GiG',
        edge3_type='GpPW',
        n_samples=10000,
        random_state=42
    )
