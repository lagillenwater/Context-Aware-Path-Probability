"""
Validate mean and variance prediction for pair-level null distribution modeling.

This script re-validates the Nov 3 approach with proper methodology:
1. Train on mean and std computed from K permutations (K = 2,3,4,5,7,9)
2. Validate on perms 10-14 (hyperparameter tuning only)
3. Test on perms 15-20 (individual evaluation, z-score calibration)

The goal is to find minimum K needed for accurate mean prediction and
well-calibrated z-scores for anomaly detection.

Usage:
    python test_src/validate_mean_variance_prediction.py CbGpPW

References:
    - docs/2025-11-11_MEAN_VARIANCE_VALIDATION_PLAN.md
    - docs/GROUND_TRUTH_SUMMARY_2025-11-11.md
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))


def load_edge_matrices(edge1_type, edge2_type, data_dir):
    """
    Load edge matrices from data directory.

    Args:
        edge1_type: First edge type (e.g., 'CbG')
        edge2_type: Second edge type (e.g., 'GpPW')
        data_dir: Path to data directory

    Returns:
        Tuple of (edge1_matrix, edge2_matrix)
    """
    edge1_file = data_dir / 'edges' / f'{edge1_type}.sparse.npz'
    edge2_file = data_dir / 'edges' / f'{edge2_type}.sparse.npz'

    if not edge1_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge1_file}")
    if not edge2_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge2_file}")

    edge1 = sp.load_npz(str(edge1_file))
    edge2 = sp.load_npz(str(edge2_file))

    # Convert boolean to int32 for pathway counting
    if edge1.dtype == bool:
        edge1 = edge1.astype(np.int32)
    if edge2.dtype == bool:
        edge2 = edge2.astype(np.int32)

    return edge1, edge2


def load_permuted_edge_matrices(edge1_type, edge2_type, perm_num, data_dir):
    """
    Load edge matrices from a specific permutation.

    Args:
        edge1_type: First edge type (e.g., 'CbG')
        edge2_type: Second edge type (e.g., 'GpPW')
        perm_num: Permutation number (0-20)
        data_dir: Path to data directory

    Returns:
        Tuple of (edge1_matrix, edge2_matrix) for the permutation
    """
    perm_dir = data_dir / 'permutations' / f'{perm_num:03d}.hetmat' / 'edges'

    edge1_file = perm_dir / f'{edge1_type}.sparse.npz'
    edge2_file = perm_dir / f'{edge2_type}.sparse.npz'

    if not edge1_file.exists():
        raise FileNotFoundError(f"Permutation edge file not found: {edge1_file}")
    if not edge2_file.exists():
        raise FileNotFoundError(f"Permutation edge file not found: {edge2_file}")

    edge1 = sp.load_npz(str(edge1_file))
    edge2 = sp.load_npz(str(edge2_file))

    # Convert boolean to int32
    if edge1.dtype == bool:
        edge1 = edge1.astype(np.int32)
    if edge2.dtype == bool:
        edge2 = edge2.astype(np.int32)

    return edge1, edge2


def sample_pairs(edge1, edge2, n_samples=10000, random_state=42):
    """
    Sample node pairs for training and evaluation.

    Strategy: 50% pairs with pathways, 50% random pairs

    Args:
        edge1: First edge sparse matrix
        edge2: Second edge sparse matrix
        n_samples: Total number of pairs to sample
        random_state: Random seed

    Returns:
        Array of (source_idx, target_idx) pairs, shape (n_samples, 2)
    """
    np.random.seed(random_state)

    # Compute pathway matrix
    pathway = edge1 @ edge2

    # Pairs with pathways
    pathway_coo = pathway.tocoo()
    pathway_pairs = np.column_stack([pathway_coo.row, pathway_coo.col])

    # Sample 50% from pairs with pathways
    n_with_pathways = n_samples // 2
    if len(pathway_pairs) > n_with_pathways:
        idx = np.random.choice(len(pathway_pairs), n_with_pathways, replace=False)
        sampled_with_pathways = pathway_pairs[idx]
    else:
        sampled_with_pathways = pathway_pairs

    # Sample 50% random pairs
    n_random = n_samples - len(sampled_with_pathways)
    n_sources = edge1.shape[0]
    n_targets = edge2.shape[1]

    random_sources = np.random.randint(0, n_sources, n_random)
    random_targets = np.random.randint(0, n_targets, n_random)
    random_pairs = np.column_stack([random_sources, random_targets])

    # Combine
    all_pairs = np.vstack([sampled_with_pathways, random_pairs])

    # Shuffle
    np.random.shuffle(all_pairs)

    return all_pairs


def extract_degree_features(pairs, edge1, edge2):
    """
    Extract 5 degree features for each pair.

    Features:
        - deg_source
        - deg_target
        - deg_source * deg_target
        - deg_source ** 2
        - deg_target ** 2

    Args:
        pairs: Array of (source_idx, target_idx), shape (n, 2)
        edge1: First edge sparse matrix
        edge2: Second edge sparse matrix

    Returns:
        Feature matrix X, shape (n, 5)
    """
    # Compute degrees
    source_degrees = np.asarray(edge1.sum(axis=1)).ravel()
    target_degrees = np.asarray(edge2.sum(axis=0)).ravel()

    # Extract degrees for sampled pairs
    deg_src = source_degrees[pairs[:, 0]]
    deg_tgt = target_degrees[pairs[:, 1]]

    # Construct feature matrix
    X = np.column_stack([
        deg_src,
        deg_tgt,
        deg_src * deg_tgt,
        deg_src ** 2,
        deg_tgt ** 2
    ])

    return X


def compute_pathway_counts(pairs, edge1, edge2):
    """
    Compute pathway counts for sampled pairs.

    Args:
        pairs: Array of (source_idx, target_idx), shape (n, 2)
        edge1: First edge sparse matrix
        edge2: Second edge sparse matrix

    Returns:
        Array of pathway counts, shape (n,)
    """
    pathway = edge1 @ edge2

    counts = np.zeros(len(pairs))
    for i, (src, tgt) in enumerate(pairs):
        counts[i] = pathway[src, tgt]

    return counts


def train_models(X, mu_train, sigma_train):
    """
    Train linear regression models for mean and std prediction.

    Args:
        X: Feature matrix, shape (n, 5)
        mu_train: Training mean targets, shape (n,)
        sigma_train: Training std targets, shape (n,)

    Returns:
        Tuple of (model_mean, model_std)
    """
    model_mean = LinearRegression()
    model_std = LinearRegression()

    model_mean.fit(X, mu_train)
    model_std.fit(X, sigma_train)

    return model_mean, model_std


def evaluate_z_scores(z):
    """
    Compute z-score calibration metrics.

    Args:
        z: Z-scores, shape (n,)

    Returns:
        Dictionary with calibration metrics
    """
    return {
        'z_mean': np.mean(np.abs(z)),
        'z_std': np.std(z),
        'z_outliers': np.mean(np.abs(z) > 3),
        'qq_corr': stats.probplot(z)[1][2]
    }


def run_experiment(metapath, edge1_type, edge2_type, K_values, data_dir,
                   n_samples=10000, random_state=42):
    """
    Run full validation experiment for given metapath and K values.

    Args:
        metapath: Metapath name (e.g., 'CbGpPW')
        edge1_type: First edge type
        edge2_type: Second edge type
        K_values: List of K values to test
        data_dir: Path to data directory
        n_samples: Number of pairs to sample
        random_state: Random seed

    Returns:
        DataFrame with results
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {metapath}")
    print(f"{'='*70}\n")

    # Load original graph (perm 0)
    print(f"Loading edges for {metapath}...")
    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(
        edge1_type, edge2_type, 0, data_dir
    )

    # Sample pairs from perm 0
    print(f"Sampling {n_samples} node pairs...")
    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples, random_state)
    print(f"  Sampled pairs: {len(pairs)}")

    # Extract features (degrees from perm 0)
    print(f"Extracting degree features...")
    X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)
    print(f"  Feature matrix: {X.shape}")

    results = []

    for K in K_values:
        print(f"\n{'='*50}")
        print(f"K = {K} (training on perms 0-{K-1})")
        print(f"{'='*50}")

        # Compute pathway counts for training perms
        print(f"  Computing training targets from perms 0-{K-1}...")
        perm_counts_train = []
        for perm_i in range(K):
            edge1, edge2 = load_permuted_edge_matrices(
                edge1_type, edge2_type, perm_i, data_dir
            )
            counts = compute_pathway_counts(pairs, edge1, edge2)
            perm_counts_train.append(counts)
            print(f"    Perm {perm_i}: mean={counts.mean():.2f}, std={counts.std():.2f}")

        # Compute training targets
        mu_train = np.mean(perm_counts_train, axis=0)
        sigma_train = np.std(perm_counts_train, axis=0)
        print(f"  Training targets: mu_mean={mu_train.mean():.2f}, sigma_mean={sigma_train.mean():.2f}")

        # Train models
        print(f"  Training models...")
        model_mean, model_std = train_models(X, mu_train, sigma_train)

        # Predict
        mu_pred = model_mean.predict(X)
        sigma_pred = model_std.predict(X)
        print(f"  Predictions: mu_mean={mu_pred.mean():.2f}, sigma_mean={sigma_pred.mean():.2f}")

        # Evaluate on test perms 15-20
        print(f"  Evaluating on test perms 15-20...")
        for test_perm in range(15, 21):
            edge1, edge2 = load_permuted_edge_matrices(
                edge1_type, edge2_type, test_perm, data_dir
            )
            counts_test = compute_pathway_counts(pairs, edge1, edge2)

            # Mean prediction quality
            r_mean = np.corrcoef(mu_pred, counts_test)[0, 1]
            mae_mean = mean_absolute_error(mu_pred, counts_test)

            # Z-score calibration
            z = (counts_test - mu_pred) / (sigma_pred + 1e-6)  # Add epsilon to avoid division by zero
            z_metrics = evaluate_z_scores(z)

            result = {
                'K': K,
                'test_perm': test_perm,
                'r_mean': r_mean,
                'mae_mean': mae_mean,
                **z_metrics
            }
            results.append(result)

            print(f"    Perm {test_perm}: r={r_mean:.3f}, z_mean={z_metrics['z_mean']:.3f}, z_std={z_metrics['z_std']:.3f}")

    return pd.DataFrame(results)


def create_summary(df):
    """
    Create summary statistics grouped by K.

    Args:
        df: Results DataFrame

    Returns:
        Summary DataFrame
    """
    summary = df.groupby('K').agg({
        'r_mean': ['mean', 'std'],
        'mae_mean': ['mean', 'std'],
        'z_mean': ['mean', 'std'],
        'z_std': ['mean', 'std'],
        'z_outliers': ['mean', 'std'],
        'qq_corr': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    return summary


def plot_results(summary, output_dir):
    """
    Create visualization of K vs performance metrics.

    Args:
        summary: Summary DataFrame
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    K = summary['K']

    # Panel 1: Mean prediction quality
    ax = axes[0, 0]
    ax.errorbar(K, summary['r_mean_mean'], yerr=summary['r_mean_std'],
                marker='o', capsize=5, label='Mean r')
    ax.axhline(0.80, color='red', linestyle='--', label='Target (0.80)')
    ax.set_xlabel('K (training permutations)')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Mean Prediction Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Z-score mean
    ax = axes[0, 1]
    ax.errorbar(K, summary['z_mean_mean'], yerr=summary['z_mean_std'],
                marker='o', capsize=5, label='Z mean')
    ax.axhline(0.8, color='red', linestyle='--', label='Target (0.8)')
    ax.axhspan(0.7, 0.9, alpha=0.2, color='green', label='Acceptable')
    ax.set_xlabel('K (training permutations)')
    ax.set_ylabel('mean(|z|)')
    ax.set_title('Z-Score Mean (Bias)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Z-score std
    ax = axes[1, 0]
    ax.errorbar(K, summary['z_std_mean'], yerr=summary['z_std_std'],
                marker='o', capsize=5, label='Z std')
    ax.axhline(1.0, color='red', linestyle='--', label='Target (1.0)')
    ax.axhspan(0.9, 1.1, alpha=0.2, color='green', label='Acceptable')
    ax.set_xlabel('K (training permutations)')
    ax.set_ylabel('std(z)')
    ax.set_title('Z-Score Std (Calibration)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Z-score outliers
    ax = axes[1, 1]
    ax.errorbar(K, summary['z_outliers_mean'], yerr=summary['z_outliers_std'],
                marker='o', capsize=5, label='Outlier rate')
    ax.axhline(0.003, color='red', linestyle='--', label='Target (0.003)')
    ax.set_xlabel('K (training permutations)')
    ax.set_ylabel('P(|z| > 3)')
    ax.set_title('Z-Score Outlier Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'K_vs_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {output_file}")


def generate_diagnostic_plots(metapath, edge1_type, edge2_type, data_dir,
                                output_dir, K=5, n_samples=10000, random_state=42):
    """
    Generate comprehensive distribution diagnostic plots for K=5.

    Args:
        metapath: Metapath name
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory
        output_dir: Directory to save plots
        K: Number of training permutations
        n_samples: Number of pairs to sample
        random_state: Random seed
    """
    print(f"\n{'='*70}")
    print(f"GENERATING DIAGNOSTIC PLOTS FOR K={K}")
    print(f"{'='*70}\n")

    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(
        edge1_type, edge2_type, 0, data_dir
    )

    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples, random_state)
    X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)

    print(f"Computing training targets from perms 0-{K-1}...")
    perm_counts_train = []
    for perm_i in range(K):
        edge1, edge2 = load_permuted_edge_matrices(
            edge1_type, edge2_type, perm_i, data_dir
        )
        counts = compute_pathway_counts(pairs, edge1, edge2)
        perm_counts_train.append(counts)

    mu_train = np.mean(perm_counts_train, axis=0)
    sigma_train = np.std(perm_counts_train, axis=0)

    print(f"Training models...")
    model_mean, model_std = train_models(X, mu_train, sigma_train)

    mu_pred = model_mean.predict(X)
    sigma_pred = model_std.predict(X)
    sigma_pred = np.maximum(sigma_pred, 0.01)

    test_perms = list(range(15, 21))
    print(f"Computing test statistics on perms {test_perms}...")

    counts_test = []
    for test_perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(
            edge1_type, edge2_type, test_perm, data_dir
        )
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)

    counts_test = np.array(counts_test).T

    # Compute z-scores and flatten data for INDIVIDUAL (pair, perm) combinations
    z_scores = []
    counts_flat = []
    mu_pred_expanded = []
    sigma_pred_expanded = []

    for i in range(len(pairs)):
        for j in range(counts_test.shape[1]):
            count = counts_test[i, j]
            z = (count - mu_pred[i]) / sigma_pred[i]
            z_scores.append(z)
            counts_flat.append(count)
            mu_pred_expanded.append(mu_pred[i])
            sigma_pred_expanded.append(sigma_pred[i])

    z_scores = np.array(z_scores)
    counts_flat = np.array(counts_flat)
    mu_pred_expanded = np.array(mu_pred_expanded)
    sigma_pred_expanded = np.array(sigma_pred_expanded)

    print(f"Creating diagnostic visualization...")
    create_diagnostic_plot(
        z_scores, mu_pred, sigma_pred, counts_test, counts_flat,
        mu_pred_expanded, sigma_pred_expanded, X, metapath, output_dir
    )


def create_diagnostic_plot(z_scores, mu_pred, sigma_pred, counts_test, counts_flat,
                            mu_pred_expanded, sigma_pred_expanded, X, metapath, output_dir):
    """
    Create 6-panel diagnostic visualization.

    Args:
        z_scores: All z-scores (n_pairs * n_test_perms,)
        mu_pred: Predicted means per pair (n_pairs,)
        sigma_pred: Predicted stds per pair (n_pairs,)
        counts_test: Test counts (n_pairs, n_test_perms)
        counts_flat: Flattened counts (n_pairs * n_test_perms,)
        mu_pred_expanded: Predicted means expanded (n_pairs * n_test_perms,)
        sigma_pred_expanded: Predicted stds expanded (n_pairs * n_test_perms,)
        X: Feature matrix (n_pairs, 5)
        metapath: Metapath name
        output_dir: Directory to save plot
    """
    fig = plt.figure(figsize=(16, 10))

    # Panel 1: Z-score histogram
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(z_scores, bins=50, density=True, alpha=0.7,
             label=f'Empirical (n={len(z_scores)})')
    x = np.linspace(-4, 4, 100)
    ax1.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2,
             label='N(0,1)')
    ax1.set_xlabel('Z-score')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Z-Score Distribution\n' +
                  f'mean={np.mean(np.abs(z_scores)):.3f}, ' +
                  f'std={np.std(z_scores):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Q-Q plot
    ax2 = plt.subplot(2, 3, 2)
    stats.probplot(z_scores, dist="norm", plot=ax2)
    qq_corr = stats.probplot(z_scores)[1][2]
    ax2.set_title(f'Q-Q Plot\nr={qq_corr:.4f}')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Mean calibration (INDIVIDUAL observations)
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(counts_flat, mu_pred_expanded, alpha=0.1, s=5)
    lim_max = max(counts_flat.max(), mu_pred_expanded.max())
    ax3.plot([0, lim_max], [0, lim_max], 'r--', linewidth=2,
             label='Perfect calibration')
    r_mean = np.corrcoef(counts_flat, mu_pred_expanded)[0, 1]
    ax3.set_xlabel('Observed count (individual perms)')
    ax3.set_ylabel('Predicted mean')
    ax3.set_title(f'Mean Calibration\nr={r_mean:.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Variance calibration (absolute residuals)
    ax4 = plt.subplot(2, 3, 4)
    abs_residuals = np.abs(counts_flat - mu_pred_expanded)
    ax4.scatter(sigma_pred_expanded, abs_residuals, alpha=0.1, s=5)
    lim_max = max(sigma_pred_expanded.max(), abs_residuals.max())
    ax4.plot([0, lim_max], [0, lim_max], 'r--', linewidth=2,
             label='Perfect calibration')
    r_std = np.corrcoef(sigma_pred_expanded, abs_residuals)[0, 1]
    ax4.set_xlabel('Predicted std')
    ax4.set_ylabel('|Observed - Predicted mean|')
    ax4.set_title(f'Std Calibration\nr={r_std:.4f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: Example pair distributions
    ax5 = plt.subplot(2, 3, 5)

    deg_source = X[:, 0]
    deg_target = X[:, 1]

    low_idx = np.argmin(deg_source * deg_target)
    high_idx = np.argmax(deg_source * deg_target)
    mid_idx = np.argsort(deg_source * deg_target)[len(mu_pred) // 2]

    example_indices = [low_idx, mid_idx, high_idx]
    colors = ['blue', 'green', 'red']
    labels = ['Low degree', 'Med degree', 'High degree']

    for idx, color, label in zip(example_indices, colors, labels):
        counts = counts_test[idx, :]
        mu = mu_pred[idx]
        sigma = sigma_pred[idx]

        x = np.linspace(max(0, mu - 3*sigma), mu + 3*sigma, 100)
        pdf = stats.norm.pdf(x, mu, sigma)

        ax5.hist(counts, bins=10, density=True, alpha=0.3,
                 color=color, label=f'{label}: N({mu:.1f},{sigma:.1f})')
        ax5.plot(x, pdf, color=color, linewidth=2)

    ax5.set_xlabel('Path count')
    ax5.set_ylabel('Density')
    ax5.set_title('Example Pair Distributions')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Residual analysis (standardized residuals = z-scores)
    ax6 = plt.subplot(2, 3, 6)

    ax6.scatter(mu_pred_expanded, z_scores, alpha=0.1, s=5)
    ax6.axhline(0, color='r', linestyle='--', linewidth=2)
    ax6.axhline(2, color='gray', linestyle=':', linewidth=1)
    ax6.axhline(-2, color='gray', linestyle=':', linewidth=1)
    ax6.set_xlabel('Predicted mean')
    ax6.set_ylabel('Standardized residual (z-score)')
    ax6.set_title(f'Residual Analysis\n' +
                  f'mean={np.mean(z_scores):.3f}, ' +
                  f'std={np.std(z_scores):.3f}')
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'Distribution Diagnostics: {metapath} (K=5, baseline)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = output_dir / f'{metapath}_distribution_diagnostics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved diagnostic plot: {output_file}")


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python validate_mean_variance_prediction.py METAPATH")
        print("Example: python validate_mean_variance_prediction.py CbGpPW")
        sys.exit(1)

    metapath = sys.argv[1]

    # Metapath to edge types mapping
    metapath_config = {
        'CbGpPW': ('CbG', 'GpPW'),
        'CtDaG': ('CtD', 'DaG'),
        'CrCbG': ('CrC', 'CbG'),
        'CbGaD': ('CbG', 'GaD'),
        'CpDaG': ('CpD', 'DaG')
    }

    if metapath not in metapath_config:
        print(f"Error: Unknown metapath '{metapath}'")
        print(f"Available: {list(metapath_config.keys())}")
        sys.exit(1)

    edge1_type, edge2_type = metapath_config[metapath]

    # Configuration
    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'mean_variance_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    K_values = [2, 3, 4, 5, 7, 9]
    n_samples = 10000
    random_state = 42

    print(f"Configuration:")
    print(f"  Metapath: {metapath}")
    print(f"  Edge 1: {edge1_type}")
    print(f"  Edge 2: {edge2_type}")
    print(f"  K values: {K_values}")
    print(f"  Samples: {n_samples}")
    print(f"  Output: {output_dir}")

    # Run experiment
    results_df = run_experiment(
        metapath, edge1_type, edge2_type, K_values, data_dir,
        n_samples, random_state
    )

    # Save detailed results
    output_file = output_dir / f'{metapath}_K_comparison.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results: {output_file}")

    # Create and save summary
    summary = create_summary(results_df)
    summary_file = output_dir / f'{metapath}_K_comparison_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")

    print("\nSummary Statistics:")
    print(summary.to_string(index=False))

    # Create plots
    plot_results(summary, output_dir)

    # Generate diagnostic plots for K=5 (optimal)
    generate_diagnostic_plots(
        metapath, edge1_type, edge2_type, data_dir, output_dir,
        K=5, n_samples=n_samples, random_state=random_state
    )

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
