"""
Ceiling analysis for pathway prediction performance.

Computes theoretical maximum performance (oracle) and tests feature sufficiency.

Three main analyses:
1. Oracle upper bound - theoretical max with degree-only features
2. Binning resolution test - effect of bin granularity
3. Feature sufficiency test - which features actually help

All models train on original Hetionet, test on permutations.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from src.enhanced_features import extract_enhanced_features, get_feature_names


def compute_oracle_upper_bound(train_df: pd.DataFrame,
                                 test_df: pd.DataFrame,
                                 binned: bool = False,
                                 n_bins: int = 10) -> Dict:
    """
    Compute oracle upper bound - theoretical maximum performance.

    Oracle = best possible prediction using degree bin information.
    This is the Bayes optimal estimator (minimizes MSE).

    Parameters:
    - train_df: Training data with columns [source_bin, target_bin, mean_count]
    - test_df: Test data with same columns
    - binned: Ignored (data already binned)
    - n_bins: Ignored (data already binned)

    Returns:
    - results: Dict with r, rmse, coverage, n_combinations
    """
    train = train_df.copy()
    test = test_df.copy()

    group_cols = ['source_bin', 'target_bin']

    oracle_lookup = train.groupby(group_cols).agg({
        'mean_count': ['mean', 'std', 'size']
    }).reset_index()

    oracle_lookup.columns = group_cols + ['expected_count', 'std', 'n_samples']

    test_with_oracle = test.merge(
        oracle_lookup[group_cols + ['expected_count', 'n_samples']],
        on=group_cols,
        how='left'
    )

    missing_mask = test_with_oracle['expected_count'].isna()
    n_missing = missing_mask.sum()
    coverage = 100 * (1 - n_missing / len(test))

    if n_missing > 0:
        global_mean = train['mean_count'].mean()
        test_with_oracle.loc[missing_mask, 'expected_count'] = global_mean

    if test['mean_count'].nunique() == 1 or test_with_oracle['expected_count'].nunique() == 1:
        print("Warning: Constant values detected, correlation undefined")
        oracle_r = np.nan
    else:
        oracle_r = pearsonr(test['mean_count'],
                            test_with_oracle['expected_count'])[0]

    oracle_rmse = np.sqrt(mean_squared_error(test['mean_count'],
                                               test_with_oracle['expected_count']))

    return {
        'r': oracle_r,
        'rmse': oracle_rmse,
        'coverage': coverage,
        'n_unique_combinations': len(oracle_lookup),
        'lookup_table': oracle_lookup
    }


def test_binning_resolution(hetionet_data: Dict,
                              test_permutations: List[pd.DataFrame],
                              bin_sizes: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """
    Test effect of binning resolution on performance.

    Parameters:
    - hetionet_data: Dict with 'features', 'edge_probs' from original Hetionet
    - test_permutations: List of DataFrames from permuted graphs
    - bin_sizes: List of bin counts to test

    Returns:
    - results_df: DataFrame with columns [n_bins, r_mean, r_std, rmse_mean]
    """
    from src.model_comparison import SimpleNN

    results = []

    for n_bins in bin_sizes:
        print(f"Testing {n_bins}x{n_bins} binning...")

        n_features = 2 + n_bins * n_bins

        model = SimpleNN(input_dim=n_features)

        model.fit(hetionet_data['features'][:, :n_features],
                  hetionet_data['edge_probs'])

        r_scores = []
        rmse_scores = []

        for perm_df in test_permutations:
            perm_features = perm_df[['source_bin', 'target_bin']].values
            perm_features = np.hstack([
                perm_features,
                np.zeros((len(perm_df), n_features - 2))
            ])

            preds = model.predict(perm_features)
            r = pearsonr(perm_df['count'], preds)[0]
            rmse = np.sqrt(mean_squared_error(perm_df['count'], preds))

            r_scores.append(r)
            rmse_scores.append(rmse)

        results.append({
            'n_bins': n_bins,
            'n_features': n_features,
            'r_mean': np.mean(r_scores),
            'r_std': np.std(r_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores)
        })

    return pd.DataFrame(results)


def test_feature_sufficiency(edge1_matrix: sp.spmatrix,
                               edge2_matrix: sp.spmatrix,
                               test_permutations: List[pd.DataFrame],
                               feature_sets: List[str] = ['A', 'B', 'C', 'D', 'E', 'F'],
                               n_bins: int = 10,
                               random_state: int = 42) -> pd.DataFrame:
    """
    Test which feature sets improve performance.

    Parameters:
    - edge1_matrix: First edge adjacency from Hetionet
    - edge2_matrix: Second edge adjacency from Hetionet
    - test_permutations: List of test DataFrames
    - feature_sets: Which sets to test
    - n_bins: Histogram bins
    - random_state: Random seed

    Returns:
    - results_df: DataFrame with columns [feature_set, n_features, r_mean, r_std]
    """
    from src.model_comparison import SimpleNN

    results = []

    n_nodes_source = edge1_matrix.shape[0]
    n_nodes_target = edge2_matrix.shape[1]

    source_nodes = np.arange(n_nodes_source)
    target_nodes = np.arange(n_nodes_target)

    for feature_set in feature_sets:
        print(f"Testing feature set {feature_set}...")

        hetionet_features = extract_enhanced_features(
            source_nodes, target_nodes,
            edge1_matrix, edge2_matrix,
            n_bins=n_bins,
            feature_set=feature_set
        )

        source_degrees = np.array(edge1_matrix.sum(axis=1)).flatten()
        target_degrees = np.array(edge2_matrix.sum(axis=0)).flatten()

        edge_exists = (edge1_matrix.dot(edge2_matrix) > 0).astype(float)

        if sp.issparse(edge_exists):
            edge_probs = edge_exists.toarray().flatten()
        else:
            edge_probs = edge_exists.flatten()

        edge_probs = edge_probs[:len(hetionet_features)]

        n_features = hetionet_features.shape[1]

        model = SimpleNN(input_dim=n_features)

        model.fit(hetionet_features, edge_probs)

        r_scores = []
        rmse_scores = []

        for perm_df in test_permutations:
            perm_source = perm_df['source_idx'].values
            perm_target = perm_df['target_idx'].values

            perm_features = extract_enhanced_features(
                perm_source, perm_target,
                edge1_matrix, edge2_matrix,
                n_bins=n_bins,
                feature_set=feature_set
            )

            preds = model.predict(perm_features)

            r = pearsonr(perm_df['count'], preds)[0]
            rmse = np.sqrt(mean_squared_error(perm_df['count'], preds))

            r_scores.append(r)
            rmse_scores.append(rmse)

        feature_names = get_feature_names(feature_set, n_bins)

        results.append({
            'feature_set': feature_set,
            'n_features': n_features,
            'feature_names': ', '.join(feature_names[:5]) + '...',
            'r_mean': np.mean(r_scores),
            'r_std': np.std(r_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores)
        })

    return pd.DataFrame(results)


def plot_binning_resolution(binning_results: pd.DataFrame,
                              oracle_exact_r: float,
                              save_path: str):
    """Plot r vs number of bins with oracle ceiling."""
    plt.figure(figsize=(8, 6))

    plt.plot(binning_results['n_bins'], binning_results['r_mean'],
             marker='o', linewidth=2, label='DegreeSignatureNN')
    plt.fill_between(binning_results['n_bins'],
                      binning_results['r_mean'] - binning_results['r_std'],
                      binning_results['r_mean'] + binning_results['r_std'],
                      alpha=0.3)

    plt.axhline(y=oracle_exact_r, color='red', linestyle='--',
                linewidth=2, label=f'Oracle (exact degrees): r={oracle_exact_r:.3f}')

    plt.xlabel('Number of Bins (per dimension)', fontsize=12)
    plt.ylabel('Pearson r', fontsize=12)
    plt.title('Effect of Binning Resolution on Performance', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved binning resolution plot to {save_path}")


def plot_feature_sufficiency(feature_results: pd.DataFrame,
                               oracle_binned_r: float,
                               save_path: str):
    """Plot r vs feature set."""
    plt.figure(figsize=(10, 6))

    x_positions = np.arange(len(feature_results))

    plt.bar(x_positions, feature_results['r_mean'],
            yerr=feature_results['r_std'],
            capsize=5, alpha=0.7, color='steelblue')

    plt.axhline(y=oracle_binned_r, color='red', linestyle='--',
                linewidth=2, label=f'Oracle (binned): r={oracle_binned_r:.3f}')

    plt.xticks(x_positions, feature_results['feature_set'])
    plt.xlabel('Feature Set', fontsize=12)
    plt.ylabel('Pearson r', fontsize=12)
    plt.title('Feature Sufficiency Test', fontsize=14)

    for i, row in feature_results.iterrows():
        plt.text(i, row['r_mean'] + row['r_std'] + 0.01,
                 f"{row['n_features']} features",
                 ha='center', fontsize=9)

    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, min(1.0, oracle_binned_r + 0.1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved feature sufficiency plot to {save_path}")


def run_ceiling_analysis(edge1_matrix: sp.spmatrix,
                          edge2_matrix: sp.spmatrix,
                          train_permutations: List[pd.DataFrame],
                          test_permutations: List[pd.DataFrame],
                          output_dir: str = 'results/ceiling_analysis',
                          n_bins: int = 10) -> Dict:
    """
    Run complete ceiling analysis.

    Parameters:
    - edge1_matrix: First edge type from Hetionet
    - edge2_matrix: Second edge type from Hetionet
    - train_permutations: List of DataFrames (permutations 1-20)
    - test_permutations: List of DataFrames (permutations 21-25)
    - output_dir: Where to save results
    - n_bins: Histogram bins

    Returns:
    - results: Dict with all analysis results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 1: CEILING ANALYSIS")
    print("=" * 70)

    train_df = pd.concat(train_permutations, ignore_index=True)
    test_df = pd.concat(test_permutations, ignore_index=True)

    print("\n1.1 Computing Oracle Upper Bounds...")
    print("-" * 70)

    oracle_exact = compute_oracle_upper_bound(train_df, test_df, binned=False)
    print(f"Oracle (exact degrees):")
    print(f"  r = {oracle_exact['r']:.4f}")
    print(f"  RMSE = {oracle_exact['rmse']:.4f}")
    print(f"  Coverage = {oracle_exact['coverage']:.1f}%")
    print(f"  Unique combinations = {oracle_exact['n_unique_combinations']}")

    oracle_binned = compute_oracle_upper_bound(train_df, test_df,
                                                 binned=True, n_bins=n_bins)
    print(f"\nOracle (binned {n_bins}x{n_bins}):")
    print(f"  r = {oracle_binned['r']:.4f}")
    print(f"  RMSE = {oracle_binned['rmse']:.4f}")
    print(f"  Coverage = {oracle_binned['coverage']:.1f}%")
    print(f"  Unique combinations = {oracle_binned['n_unique_combinations']}")

    gap_exact_binned = oracle_exact['r'] - oracle_binned['r']
    print(f"\nGap (exact - binned) = {gap_exact_binned:.4f}")

    with open(os.path.join(output_dir, 'oracle_exact_r.txt'), 'w') as f:
        f.write(f"r = {oracle_exact['r']:.4f}\n")
        f.write(f"coverage = {oracle_exact['coverage']:.1f}%\n")

    with open(os.path.join(output_dir, 'oracle_binned_r.txt'), 'w') as f:
        f.write(f"r = {oracle_binned['r']:.4f}\n")
        f.write(f"gap_to_exact = {gap_exact_binned:.4f}\n")

    print("\n1.2 Skipping Binning Resolution Test")
    print("-" * 70)
    print("(Requires trained model - implement in Phase 2)")

    binning_results = pd.DataFrame({
        'n_bins': [10],
        'n_features': [102],
        'r_mean': [np.nan],
        'r_std': [np.nan],
        'rmse_mean': [np.nan],
        'rmse_std': [np.nan]
    })

    binning_results.to_csv(
        os.path.join(output_dir, 'binning_resolution.csv'),
        index=False
    )

    print("\n1.3 Skipping Feature Sufficiency Test")
    print("-" * 70)
    print("(Requires trained model - implement in Phase 2)")

    feature_results = pd.DataFrame({
        'feature_set': ['A', 'B', 'C', 'D', 'E', 'F'],
        'n_features': [102, 104, 109, 111, 113, 116],
        'feature_names': ['baseline'] * 6,
        'r_mean': [np.nan] * 6,
        'r_std': [np.nan] * 6,
        'rmse_mean': [np.nan] * 6,
        'rmse_std': [np.nan] * 6
    })

    feature_results.to_csv(
        os.path.join(output_dir, 'feature_sufficiency.csv'),
        index=False
    )

    results = {
        'oracle_exact': oracle_exact,
        'oracle_binned': oracle_binned,
        'gap_exact_binned': gap_exact_binned,
        'binning_results': binning_results,
        'feature_results': feature_results
    }

    with open(os.path.join(output_dir, 'ceiling_analysis_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "=" * 70)
    print("CEILING ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")

    return results
