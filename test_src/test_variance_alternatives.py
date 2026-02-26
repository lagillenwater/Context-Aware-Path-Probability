"""
Test alternative variance modeling approaches for pathway count prediction.

Compares three variance models:
1. Baseline: Linear regression on 5 degree features
2. Degree-stratified: Separate linear models for low/medium/high degree ranges
3. Quantile regression: Predict percentile spread, convert to variance

Goal: Determine if alternatives improve Q-Q calibration (baseline Q-Q corr = 0.714)

Usage:
    python test_src/test_variance_alternatives.py CbGpPW --model baseline
    python test_src/test_variance_alternatives.py CbGpPW --model stratified
    python test_src/test_variance_alternatives.py CbGpPW --model quantile
    python test_src/test_variance_alternatives.py CbGpPW --all

References:
    - docs/2025-11-11_IMPROVEMENTS_RESULTS.md (baseline results)
    - docs/2025-11-11_MEAN_VARIANCE_VALIDATION_RESULTS.md (K=5 validation)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as sp_stats
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

from test_src.validate_mean_variance_prediction import (
    load_permuted_edge_matrices,
    sample_pairs,
    extract_degree_features,
    compute_pathway_counts,
    evaluate_z_scores
)


def train_linear_variance(X, mu_train, sigma_train):
    """
    Baseline: Linear regression for variance prediction.

    Args:
        X: Feature matrix, shape (n, d)
        mu_train: Training mean targets, shape (n,)
        sigma_train: Training std targets, shape (n,)

    Returns:
        Tuple of (model_mean, model_std, model_type='linear')
    """
    model_mean = LinearRegression()
    model_std = LinearRegression()

    model_mean.fit(X, mu_train)
    model_std.fit(X, sigma_train)

    return model_mean, model_std, 'linear'


def train_stratified_variance(X, mu_train, sigma_train, n_strata=3):
    """
    Degree-stratified variance models.

    Partitions feature space by source and target degree, trains separate
    variance models for each stratum to address heteroscedasticity.

    Args:
        X: Feature matrix with columns [deg_src, deg_tgt, ...], shape (n, d)
        mu_train: Training mean targets, shape (n,)
        sigma_train: Training std targets, shape (n,)
        n_strata: Number of strata per dimension (default 3: low/medium/high)

    Returns:
        Tuple of (model_mean, stratified_variance_models, model_type='stratified')
    """
    model_mean = LinearRegression()
    model_mean.fit(X, mu_train)

    deg_src = X[:, 0]
    deg_tgt = X[:, 1]

    src_bins = np.percentile(deg_src, np.linspace(0, 100, n_strata + 1))
    tgt_bins = np.percentile(deg_tgt, np.linspace(0, 100, n_strata + 1))

    src_strata = np.digitize(deg_src, src_bins[1:-1])
    tgt_strata = np.digitize(deg_tgt, tgt_bins[1:-1])

    stratified_models = {}
    for i in range(n_strata):
        for j in range(n_strata):
            mask = (src_strata == i) & (tgt_strata == j)
            if mask.sum() < 10:
                continue

            X_stratum = X[mask]
            sigma_stratum = sigma_train[mask]

            model_std = LinearRegression()
            model_std.fit(X_stratum, sigma_stratum)
            stratified_models[(i, j)] = model_std

    metadata = {
        'src_bins': src_bins,
        'tgt_bins': tgt_bins,
        'n_strata': n_strata,
        'models': stratified_models
    }

    return model_mean, metadata, 'stratified'


def train_quantile_variance(X, mu_train, counts_train, quantiles=(0.05, 0.95)):
    """
    Quantile regression variance estimation.

    Predicts low and high quantiles of count distribution, uses spread
    to estimate variance.

    Args:
        X: Feature matrix, shape (n, d)
        mu_train: Training mean targets, shape (n,)
        counts_train: Full count matrix, shape (n, K) for K training perms
        quantiles: Tuple of (low, high) quantiles (default 5th, 95th percentile)

    Returns:
        Tuple of (model_mean, quantile_models, model_type='quantile')
    """
    model_mean = LinearRegression()
    model_mean.fit(X, mu_train)

    q_low, q_high = quantiles

    model_q_low = QuantileRegressor(quantile=q_low, alpha=0.1, solver='highs')
    model_q_high = QuantileRegressor(quantile=q_high, alpha=0.1, solver='highs')

    counts_flat = counts_train.ravel()
    X_expanded = np.repeat(X, counts_train.shape[1], axis=0)

    model_q_low.fit(X_expanded, counts_flat)
    model_q_high.fit(X_expanded, counts_flat)

    metadata = {
        'model_low': model_q_low,
        'model_high': model_q_high,
        'q_low': q_low,
        'q_high': q_high
    }

    return model_mean, metadata, 'quantile'


def predict_linear_variance(model_mean, model_std, X):
    """
    Predict with baseline linear variance model.

    Args:
        model_mean: Trained mean model
        model_std: Trained std model
        X: Feature matrix, shape (n, d)

    Returns:
        Tuple of (mu_pred, sigma_pred)
    """
    mu_pred = model_mean.predict(X)
    sigma_pred = model_std.predict(X)
    sigma_pred = np.maximum(sigma_pred, 0.1)

    return mu_pred, sigma_pred


def predict_stratified_variance(model_mean, metadata, X):
    """
    Predict with degree-stratified variance models.

    Args:
        model_mean: Trained mean model
        metadata: Dict with stratified models and bin edges
        X: Feature matrix, shape (n, d)

    Returns:
        Tuple of (mu_pred, sigma_pred)
    """
    mu_pred = model_mean.predict(X)

    deg_src = X[:, 0]
    deg_tgt = X[:, 1]

    src_bins = metadata['src_bins']
    tgt_bins = metadata['tgt_bins']
    n_strata = metadata['n_strata']
    stratified_models = metadata['models']

    src_strata = np.digitize(deg_src, src_bins[1:-1])
    tgt_strata = np.digitize(deg_tgt, tgt_bins[1:-1])

    sigma_pred = np.zeros(len(X))

    global_model = LinearRegression()
    all_X = []
    all_sigma = []
    for models in stratified_models.values():
        if hasattr(models, 'coef_'):
            all_X.append(models.coef_)

    for i in range(len(X)):
        stratum = (src_strata[i], tgt_strata[i])
        if stratum in stratified_models:
            sigma_pred[i] = stratified_models[stratum].predict(X[i:i+1])[0]
        else:
            sigma_pred[i] = np.mean([m.predict(X[i:i+1])[0]
                                    for m in stratified_models.values()])

    sigma_pred = np.maximum(sigma_pred, 0.1)

    return mu_pred, sigma_pred


def predict_quantile_variance(model_mean, metadata, X):
    """
    Predict with quantile regression variance.

    Args:
        model_mean: Trained mean model
        metadata: Dict with quantile models and parameters
        X: Feature matrix, shape (n, d)

    Returns:
        Tuple of (mu_pred, sigma_pred)
    """
    mu_pred = model_mean.predict(X)

    model_low = metadata['model_low']
    model_high = metadata['model_high']
    q_low = metadata['q_low']
    q_high = metadata['q_high']

    pred_low = model_low.predict(X)
    pred_high = model_high.predict(X)

    z_low = sp_stats.norm.ppf(q_low)
    z_high = sp_stats.norm.ppf(q_high)

    sigma_pred = (pred_high - pred_low) / (z_high - z_low)
    sigma_pred = np.maximum(sigma_pred, 0.1)

    return mu_pred, sigma_pred


def compute_test_statistics(pairs, edge1_type, edge2_type, data_dir,
                            test_perms, model_mean, model_variance,
                            model_type, X):
    """
    Compute test statistics on individual permutations.

    Args:
        pairs: Array of (source_idx, target_idx)
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory
        test_perms: List of test permutation indices
        model_mean: Trained mean model
        model_variance: Trained variance model (or metadata dict)
        model_type: 'linear', 'stratified', or 'quantile'
        X: Feature matrix for test pairs

    Returns:
        Dict with statistics
    """
    if model_type == 'linear':
        mu_pred, sigma_pred = predict_linear_variance(model_mean, model_variance, X)
    elif model_type == 'stratified':
        mu_pred, sigma_pred = predict_stratified_variance(model_mean, model_variance, X)
    elif model_type == 'quantile':
        mu_pred, sigma_pred = predict_quantile_variance(model_mean, model_variance, X)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    counts_test = []
    for perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)
    counts_test = np.column_stack(counts_test)

    n_test = len(test_perms)
    z_scores = []
    counts_flat = []
    mu_pred_expanded = []
    sigma_pred_expanded = []

    for i in range(len(pairs)):
        for j in range(n_test):
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

    return {
        'mu_pred_per_pair': mu_pred,
        'sigma_pred_per_pair': sigma_pred,
        'mu_pred': mu_pred_expanded,
        'sigma_pred': sigma_pred_expanded,
        'counts_test': counts_test,
        'counts_flat': counts_flat,
        'z_scores': z_scores
    }


def create_diagnostic_plot(stats_dict, metapath, model_type, output_dir):
    """
    Create 6-panel diagnostic visualization.

    Args:
        stats_dict: Dict from compute_test_statistics
        metapath: Metapath name
        model_type: Model type string
        output_dir: Output directory path
    """
    z_scores = stats_dict['z_scores']
    mu_pred = stats_dict['mu_pred']
    sigma_pred = stats_dict['sigma_pred']
    counts_flat = stats_dict['counts_flat']
    mu_pred_per_pair = stats_dict['mu_pred_per_pair']
    sigma_pred_per_pair = stats_dict['sigma_pred_per_pair']
    counts_test = stats_dict['counts_test']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Distribution Diagnostics: {metapath}, {model_type} variance model',
                 fontsize=16, y=0.995)

    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    z_mean = np.mean(np.abs(z_scores))
    z_std = np.std(z_scores)

    x = np.linspace(-4, 4, 100)
    ax1.hist(z_scores, bins=50, density=True, alpha=0.7,
             label=f'Empirical (n={len(z_scores)})')
    ax1.plot(x, sp_stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Z-score')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Z-Score Distribution\nmean={z_mean:.3f}, std={z_std:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    qq_result = sp_stats.probplot(z_scores, dist="norm", plot=ax2)
    qq_corr = np.corrcoef(qq_result[0][0], qq_result[0][1])[0, 1]
    ax2.set_title(f'Q-Q Plot\nr={qq_corr:.4f}')
    ax2.grid(True, alpha=0.3)

    ax3.scatter(counts_flat, mu_pred, alpha=0.1, s=5)
    r_mean = np.corrcoef(counts_flat, mu_pred)[0, 1]
    max_val = max(counts_flat.max(), mu_pred.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect calibration')
    ax3.set_xlabel('Observed count (individual perms)')
    ax3.set_ylabel('Predicted mean')
    ax3.set_title(f'Mean Calibration\nr={r_mean:.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    abs_residuals = np.abs(counts_flat - mu_pred)
    ax4.scatter(sigma_pred, abs_residuals, alpha=0.1, s=5)
    r_std = np.corrcoef(sigma_pred, abs_residuals)[0, 1]
    max_sigma = sigma_pred.max()
    ax4.plot([0, max_sigma], [0, max_sigma], 'r--', linewidth=2,
             label='Perfect calibration')
    ax4.set_xlabel('Predicted std')
    ax4.set_ylabel('|Observed - Predicted mean|')
    ax4.set_title(f'Std Calibration\nr={r_std:.4f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    example_indices = [0, len(counts_test) // 2, len(counts_test) - 1]
    for idx in example_indices:
        counts = counts_test[idx]
        mu = mu_pred_per_pair[idx]
        sigma = sigma_pred_per_pair[idx]

        x_range = np.linspace(max(0, mu - 3*sigma), mu + 3*sigma, 100)
        ax5.plot(x_range, sp_stats.norm.pdf(x_range, mu, sigma),
                label=f'N({mu:.2f},{sigma:.2f})', linewidth=2)
        ax5.hist(counts, bins=20, alpha=0.3, density=True)

    ax5.set_xlabel('Pathway count')
    ax5.set_ylabel('Density')
    ax5.set_title('Example Pair Distributions')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    residuals = (counts_flat - mu_pred) / sigma_pred
    ax6.scatter(mu_pred, residuals, alpha=0.1, s=5)
    ax6.axhline(0, color='r', linestyle='--', linewidth=2)
    ax6.axhline(2, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax6.axhline(-2, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Predicted mean')
    ax6.set_ylabel('Standardized residual (z-score)')
    ax6.set_title(f'Residual Analysis\nmean={np.mean(residuals):.3f}, std={np.std(residuals):.3f}')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f'distribution_diagnostics_{model_type}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved diagnostic plot: {output_path}")

    print(f"\nDiagnostic Summary ({model_type}):")
    print(f"  Z-score mean: {z_mean:.4f} (target: 0.8)")
    print(f"  Z-score std: {z_std:.4f} (target: 1.0)")
    print(f"  Mean r (individual perms): {r_mean:.4f}")
    print(f"  Std r (absolute residuals): {r_std:.4f}")
    print(f"  Q-Q corr: {qq_corr:.4f} (target: >0.95)")


def run_experiment(metapath, edge1_type, edge2_type, data_dir, output_dir,
                   model_type='linear', K=5, n_samples=10000, random_state=42):
    """
    Run variance model experiment.

    Args:
        metapath: Metapath name
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory
        output_dir: Path to output directory
        model_type: 'linear', 'stratified', or 'quantile'
        K: Number of training permutations
        n_samples: Number of pairs to sample
        random_state: Random seed

    Returns:
        DataFrame with per-permutation results
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {metapath} - {model_type} variance")
    print(f"{'='*70}\n")

    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(
        edge1_type, edge2_type, 0, data_dir
    )

    print(f"Sampling {n_samples} node pairs...")
    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples, random_state)
    print(f"  Sampled pairs: {len(pairs)}")

    print(f"Extracting features...")
    X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)
    print(f"  Feature shape: {X.shape}")

    print(f"Computing training targets from {K} permutations...")
    train_perms = list(range(K))
    counts_train = []
    for perm in train_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_train.append(counts)
    counts_train = np.column_stack(counts_train)

    mu_train = np.mean(counts_train, axis=1)
    sigma_train = np.std(counts_train, axis=1, ddof=1)

    print(f"Training {model_type} variance model...")
    if model_type == 'linear':
        model_mean, model_variance, _ = train_linear_variance(X, mu_train, sigma_train)
    elif model_type == 'stratified':
        model_mean, model_variance, _ = train_stratified_variance(X, mu_train, sigma_train)
    elif model_type == 'quantile':
        model_mean, model_variance, _ = train_quantile_variance(X, mu_train, counts_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Computing test statistics (perms 15-20)...")
    test_perms = list(range(15, 21))
    stats_dict = compute_test_statistics(
        pairs, edge1_type, edge2_type, data_dir, test_perms,
        model_mean, model_variance, model_type, X
    )

    print(f"Generating diagnostic plots...")
    create_diagnostic_plot(stats_dict, metapath, model_type, output_dir)

    results = []
    counts_test = stats_dict['counts_test']
    mu_pred_per_pair = stats_dict['mu_pred_per_pair']
    sigma_pred_per_pair = stats_dict['sigma_pred_per_pair']

    for idx, perm in enumerate(test_perms):
        counts = counts_test[:, idx]
        z_scores = (counts - mu_pred_per_pair) / sigma_pred_per_pair

        r_mean = np.corrcoef(counts, mu_pred_per_pair)[0, 1]
        z_metrics = evaluate_z_scores(z_scores)

        results.append({
            'test_perm': perm,
            'model_type': model_type,
            'r_mean': r_mean,
            'z_mean': z_metrics['z_mean'],
            'z_std': z_metrics['z_std'],
            'z_outliers': z_metrics['z_outliers']
        })

    results_df = pd.DataFrame(results)

    output_file = output_dir / f'{metapath}_{model_type}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results: {output_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Test alternative variance models')
    parser.add_argument('metapath', help='Metapath name (e.g., CbGpPW)')
    parser.add_argument('--model', choices=['linear', 'stratified', 'quantile'],
                       help='Variance model type')
    parser.add_argument('--all', action='store_true',
                       help='Run all three models')
    parser.add_argument('--K', type=int, default=5,
                       help='Number of training permutations')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of node pairs to sample')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    metapath_configs = {
        'CbGpPW': ('CbG', 'GpPW')
    }

    if args.metapath not in metapath_configs:
        print(f"Error: Unknown metapath {args.metapath}")
        print(f"Available: {list(metapath_configs.keys())}")
        sys.exit(1)

    edge1_type, edge2_type = metapath_configs[args.metapath]

    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'variance_alternatives'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        models = ['linear', 'stratified', 'quantile']
    elif args.model:
        models = [args.model]
    else:
        print("Error: Must specify --model or --all")
        sys.exit(1)

    all_results = []
    for model_type in models:
        results_df = run_experiment(
            args.metapath, edge1_type, edge2_type, data_dir, output_dir,
            model_type=model_type, K=args.K, n_samples=args.n_samples,
            random_state=args.random_state
        )
        all_results.append(results_df)

    if len(all_results) > 1:
        comparison_df = pd.concat(all_results, ignore_index=True)
        comparison_file = output_dir / f'{args.metapath}_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nSaved comparison: {comparison_file}")

        print("\nComparison Summary:")
        print(comparison_df.groupby('model_type')[['r_mean', 'z_mean', 'z_std', 'z_outliers']].mean())


if __name__ == '__main__':
    main()
