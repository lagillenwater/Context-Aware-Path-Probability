"""
Validate feature and model improvements for pair-level prediction.

Tests three improvements:
1. Enhanced features: Add expected intermediate degree statistics (7 features)
2. Negative binomial variance: Model variance using negative binomial
3. Combined: Both improvements together

Usage:
    python test_src/validate_improvements.py CbGpPW --features enhanced --variance negbin

References:
    - docs/2025-11-11_MEAN_VARIANCE_VALIDATION_RESULTS.md (baseline r=0.787)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt

# Add src to path
repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

# Import functions from base validation script
from validate_mean_variance_prediction import (
    load_permuted_edge_matrices,
    sample_pairs,
    extract_degree_features,
    compute_pathway_counts,
    evaluate_z_scores
)


def compute_expected_intermediate_features(pairs, edge1, edge2):
    """
    Compute expected statistics of intermediate nodes using degree distributions.

    This is topology-invariant: uses probabilistic expectations instead of
    enumerating actual intermediates from a specific permutation.

    For metapath source->intermediate->target:
    - P(node i is intermediate) ∝ P(source connects to i) * P(i connects to target)
    - For degree-based estimate: ∝ deg_in(i) * deg_out(i)

    Args:
        pairs: Array of (source_idx, target_idx), shape (n, 2)
        edge1: First edge matrix (source -> intermediate)
        edge2: Second edge matrix (intermediate -> target)

    Returns:
        Feature matrix with 7 columns:
            - expected_n_intermediates
            - expected_mean_deg_in
            - expected_mean_deg_out
            - expected_std_deg
            - expected_min_deg
            - expected_max_deg
            - deg_source * expected_mean_deg_total
    """
    n_pairs = len(pairs)
    n_intermediates = edge1.shape[1]

    # Compute degrees
    source_degrees = np.asarray(edge1.sum(axis=1)).ravel()
    target_degrees = np.asarray(edge2.sum(axis=0)).ravel()
    intermediate_deg_in = np.asarray(edge1.sum(axis=0)).ravel()  # In-degree from edge1
    intermediate_deg_out = np.asarray(edge2.sum(axis=1)).ravel()  # Out-degree from edge2
    intermediate_deg_total = intermediate_deg_in + intermediate_deg_out

    # Initialize feature matrix
    features = np.zeros((n_pairs, 7))

    for idx, (src, tgt) in enumerate(pairs):
        deg_src = source_degrees[src]
        deg_tgt = target_degrees[tgt]

        if deg_src == 0 or deg_tgt == 0:
            # No paths possible
            features[idx, :] = 0
            continue

        # Probability that node i is an intermediate
        # Proportional to: (i connected to src) * (i connected to tgt)
        # Approximate: deg_in(i) * deg_out(i) / (total_edges^2)
        # Simplified: deg_in(i) * deg_out(i)

        prob_intermediate = intermediate_deg_in * intermediate_deg_out
        prob_intermediate = prob_intermediate / (prob_intermediate.sum() + 1e-10)

        # Expected number of intermediates
        # For a random bipartite graph: E[n_intermediates] ≈ deg_src * deg_tgt / n_nodes
        # More accurate: use actual degree distributions
        expected_n = np.sum(prob_intermediate > 0)
        features[idx, 0] = expected_n

        # Expected mean in-degree
        expected_mean_deg_in = np.sum(prob_intermediate * intermediate_deg_in)
        features[idx, 1] = expected_mean_deg_in

        # Expected mean out-degree
        expected_mean_deg_out = np.sum(prob_intermediate * intermediate_deg_out)
        features[idx, 2] = expected_mean_deg_out

        # Expected std of total degree
        expected_mean_deg = np.sum(prob_intermediate * intermediate_deg_total)
        expected_var_deg = np.sum(prob_intermediate * (intermediate_deg_total - expected_mean_deg)**2)
        features[idx, 3] = np.sqrt(expected_var_deg)

        # Expected min degree
        # Approximate as percentile of degree distribution weighted by prob
        sorted_idx = np.argsort(intermediate_deg_total)
        cumsum_prob = np.cumsum(prob_intermediate[sorted_idx])
        min_idx = sorted_idx[cumsum_prob >= 0.1][0] if np.any(cumsum_prob >= 0.1) else 0
        features[idx, 4] = intermediate_deg_total[min_idx]

        # Expected max degree
        max_idx = sorted_idx[cumsum_prob >= 0.9][0] if np.any(cumsum_prob >= 0.9) else -1
        features[idx, 5] = intermediate_deg_total[max_idx]

        # Interaction: source degree * expected mean intermediate degree
        features[idx, 6] = deg_src * expected_mean_deg

    return features


def extract_degree_features_enhanced(pairs, edge1, edge2):
    """
    Extract 12 features: 5 degree + 7 expected intermediate.

    Args:
        pairs: Array of (source_idx, target_idx), shape (n, 2)
        edge1: First edge sparse matrix
        edge2: Second edge sparse matrix

    Returns:
        Feature matrix X, shape (n, 12)
    """
    # Base degree features (5)
    X_base = extract_degree_features(pairs, edge1, edge2)

    # Expected intermediate features (7)
    X_intermediate = compute_expected_intermediate_features(pairs, edge1, edge2)

    # Concatenate
    X_enhanced = np.column_stack([X_base, X_intermediate])

    return X_enhanced


def train_models_linear(X, mu_train, sigma_train):
    """
    Train linear models for mean and std prediction (baseline).

    Args:
        X: Feature matrix, shape (n, d)
        mu_train: Training mean targets, shape (n,)
        sigma_train: Training std targets, shape (n,)

    Returns:
        Tuple of (model_mean, model_std, variance_type='linear')
    """
    model_mean = LinearRegression()
    model_std = LinearRegression()

    model_mean.fit(X, mu_train)
    model_std.fit(X, sigma_train)

    return model_mean, model_std, 'linear'


def train_models_negbin(X, mu_train, sigma_train):
    """
    Train mean model + negative binomial dispersion model.

    Negative binomial variance: var = mu + mu^2 / r
    Model dispersion parameter r as function of features.

    Args:
        X: Feature matrix, shape (n, d)
        mu_train: Training mean targets, shape (n,)
        sigma_train: Training std targets, shape (n,)

    Returns:
        Tuple of (model_mean, model_dispersion, variance_type='negbin')
    """
    # Fit mean model
    model_mean = LinearRegression()
    model_mean.fit(X, mu_train)

    # Compute empirical dispersion parameter
    empirical_var = sigma_train ** 2
    # Negative binomial: var = mu + mu^2/r
    # Solve for r: r = mu^2 / (var - mu)
    empirical_r = mu_train ** 2 / (empirical_var - mu_train + 1e-6)
    # Clip to reasonable range
    empirical_r = np.clip(empirical_r, 0.1, 100)

    # Fit log-linear model for dispersion
    model_dispersion = LinearRegression()
    model_dispersion.fit(X, np.log(empirical_r + 1e-6))

    return model_mean, model_dispersion, 'negbin'


def predict_with_linear(model_mean, model_std, X):
    """
    Predict mean and std using linear models (baseline).

    Args:
        model_mean: Trained mean model
        model_std: Trained std model
        X: Feature matrix, shape (n, d)

    Returns:
        Tuple of (mu_pred, sigma_pred)
    """
    mu_pred = model_mean.predict(X)
    sigma_pred = model_std.predict(X)
    sigma_pred = np.maximum(sigma_pred, 0.1)  # Ensure positive

    return mu_pred, sigma_pred


def predict_with_negbin(model_mean, model_dispersion, X):
    """
    Predict mean and std using negative binomial variance.

    Args:
        model_mean: Trained mean model
        model_dispersion: Trained dispersion model
        X: Feature matrix, shape (n, d)

    Returns:
        Tuple of (mu_pred, sigma_pred)
    """
    mu_pred = model_mean.predict(X)
    mu_pred = np.maximum(mu_pred, 0)  # Ensure non-negative

    log_r_pred = model_dispersion.predict(X)
    r_pred = np.exp(log_r_pred)
    r_pred = np.clip(r_pred, 0.1, 100)

    # Negative binomial variance: var = mu + mu^2/r
    var_pred = mu_pred + mu_pred ** 2 / (r_pred + 1e-6)
    sigma_pred = np.sqrt(var_pred)
    sigma_pred = np.maximum(sigma_pred, 0.1)  # Ensure positive

    return mu_pred, sigma_pred


def run_experiment(metapath, edge1_type, edge2_type, data_dir,
                   feature_type='base', variance_type='linear',
                   K=5, n_samples=10000, random_state=42):
    """
    Run validation experiment with specified features and variance model.

    Args:
        metapath: Metapath name
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory
        feature_type: 'base' (5 features) or 'enhanced' (12 features)
        variance_type: 'linear' or 'negbin'
        K: Number of training permutations
        n_samples: Number of pairs to sample
        random_state: Random seed

    Returns:
        DataFrame with results
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {metapath}")
    print(f"Features: {feature_type} ({5 if feature_type=='base' else 12} features)")
    print(f"Variance: {variance_type}")
    print(f"{'='*70}\n")

    # Load perm 0
    print(f"Loading edges for {metapath}...")
    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(
        edge1_type, edge2_type, 0, data_dir
    )

    # Sample pairs
    print(f"Sampling {n_samples} node pairs...")
    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples, random_state)
    print(f"  Sampled pairs: {len(pairs)}")

    # Extract features
    print(f"Extracting features...")
    if feature_type == 'base':
        X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)
    else:  # enhanced
        X = extract_degree_features_enhanced(pairs, edge1_perm0, edge2_perm0)
    print(f"  Feature matrix: {X.shape}")

    # Compute training targets from perms 0 to K-1
    print(f"Computing training targets from perms 0-{K-1}...")
    perm_counts_train = []
    for perm_i in range(K):
        edge1, edge2 = load_permuted_edge_matrices(
            edge1_type, edge2_type, perm_i, data_dir
        )
        counts = compute_pathway_counts(pairs, edge1, edge2)
        perm_counts_train.append(counts)
        print(f"  Perm {perm_i}: mean={counts.mean():.2f}, std={counts.std():.2f}")

    mu_train = np.mean(perm_counts_train, axis=0)
    sigma_train = np.std(perm_counts_train, axis=0)
    print(f"Training targets: mu_mean={mu_train.mean():.2f}, sigma_mean={sigma_train.mean():.2f}")

    # Train models
    print(f"Training models...")
    if variance_type == 'linear':
        model_mean, model_var, var_type = train_models_linear(X, mu_train, sigma_train)
    else:  # negbin
        model_mean, model_var, var_type = train_models_negbin(X, mu_train, sigma_train)

    # Predict
    if variance_type == 'linear':
        mu_pred, sigma_pred = predict_with_linear(model_mean, model_var, X)
    else:  # negbin
        mu_pred, sigma_pred = predict_with_negbin(model_mean, model_var, X)
    print(f"Predictions: mu_mean={mu_pred.mean():.2f}, sigma_mean={sigma_pred.mean():.2f}")

    # Evaluate on test perms 15-20
    print(f"Evaluating on test perms 15-20...")
    results = []
    for test_perm in range(15, 21):
        edge1, edge2 = load_permuted_edge_matrices(
            edge1_type, edge2_type, test_perm, data_dir
        )
        counts_test = compute_pathway_counts(pairs, edge1, edge2)

        # Mean prediction quality
        r_mean = np.corrcoef(mu_pred, counts_test)[0, 1]
        mae_mean = mean_absolute_error(mu_pred, counts_test)

        # Z-score calibration
        z = (counts_test - mu_pred) / (sigma_pred + 1e-6)
        z_metrics = evaluate_z_scores(z)

        result = {
            'feature_type': feature_type,
            'variance_type': variance_type,
            'test_perm': test_perm,
            'r_mean': r_mean,
            'mae_mean': mae_mean,
            **z_metrics
        }
        results.append(result)

        print(f"  Perm {test_perm}: r={r_mean:.3f}, z_mean={z_metrics['z_mean']:.3f}, z_std={z_metrics['z_std']:.3f}")

    return pd.DataFrame(results)


def create_summary(df):
    """
    Create summary statistics grouped by experiment configuration.

    Args:
        df: Results DataFrame

    Returns:
        Summary DataFrame
    """
    summary = df.groupby(['feature_type', 'variance_type']).agg({
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


def plot_comparison(summary, output_dir):
    """
    Create comparison plot across all experiments.

    Args:
        summary: Summary DataFrame
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Create labels
    labels = []
    for _, row in summary.iterrows():
        feat = 'Base (5)' if row['feature_type'] == 'base' else 'Enhanced (12)'
        var = 'Linear' if row['variance_type'] == 'linear' else 'NegBin'
        labels.append(f"{feat}\n{var}")

    x = np.arange(len(labels))

    # Panel 1: Mean prediction quality
    ax = axes[0, 0]
    ax.bar(x, summary['r_mean_mean'], yerr=summary['r_mean_std'], capsize=5)
    ax.axhline(0.80, color='red', linestyle='--', label='Target (0.80)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Mean Prediction Quality')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Z-score mean
    ax = axes[0, 1]
    ax.bar(x, summary['z_mean_mean'], yerr=summary['z_mean_std'], capsize=5)
    ax.axhline(0.8, color='red', linestyle='--', label='Target (0.8)')
    ax.axhspan(0.7, 0.9, alpha=0.2, color='green', label='Acceptable')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('mean(|z|)')
    ax.set_title('Z-Score Mean (Bias)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Z-score std
    ax = axes[1, 0]
    ax.bar(x, summary['z_std_mean'], yerr=summary['z_std_std'], capsize=5)
    ax.axhline(1.0, color='red', linestyle='--', label='Target (1.0)')
    ax.axhspan(0.9, 1.1, alpha=0.2, color='green', label='Acceptable')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('std(z)')
    ax.set_title('Z-Score Std (Calibration)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Z-score outliers
    ax = axes[1, 1]
    ax.bar(x, summary['z_outliers_mean'], yerr=summary['z_outliers_std'], capsize=5)
    ax.axhline(0.003, color='red', linestyle='--', label='Target (0.003)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('P(|z| > 3)')
    ax.set_title('Z-Score Outlier Rate')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / 'comparison_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {output_file}")


def generate_diagnostic_plots(metapath, edge1_type, edge2_type, data_dir,
                                output_dir, feature_type='base',
                                variance_type='linear', K=5,
                                n_samples=10000, random_state=42):
    """
    Generate comprehensive distribution diagnostic plots.

    Args:
        metapath: Metapath name
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory
        output_dir: Directory to save plots
        feature_type: 'base' or 'enhanced'
        variance_type: 'linear' or 'negbin'
        K: Number of training permutations
        n_samples: Number of pairs to sample
        random_state: Random seed
    """
    print(f"\n{'='*70}")
    print(f"GENERATING DIAGNOSTIC PLOTS")
    print(f"  Feature type: {feature_type}")
    print(f"  Variance type: {variance_type}")
    print(f"  K: {K}")
    print(f"{'='*70}\n")

    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(
        edge1_type, edge2_type, 0, data_dir
    )

    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples, random_state)

    if feature_type == 'enhanced':
        X = extract_degree_features_enhanced(pairs, edge1_perm0, edge2_perm0)
    else:
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
    if variance_type == 'negbin':
        model_mean, model_std, _ = train_models_negbin(X, mu_train, sigma_train)
    else:
        model_mean, model_std = train_models(X, mu_train, sigma_train)

    mu_pred = model_mean.predict(X)

    if variance_type == 'negbin':
        log_r_pred = model_std.predict(X)
        r_pred = np.exp(log_r_pred)
        r_pred = np.clip(r_pred, 0.1, 100)
        var_pred = mu_pred + mu_pred**2 / r_pred
        sigma_pred = np.sqrt(var_pred)
    else:
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
    create_diagnostic_plot_improvements(
        z_scores, mu_pred, sigma_pred, counts_test, counts_flat,
        mu_pred_expanded, sigma_pred_expanded, X, metapath,
        feature_type, variance_type, output_dir
    )


def create_diagnostic_plot_improvements(z_scores, mu_pred, sigma_pred,
                                         counts_test, counts_flat,
                                         mu_pred_expanded, sigma_pred_expanded,
                                         X, metapath, feature_type,
                                         variance_type, output_dir):
    """
    Create 6-panel diagnostic visualization for improvement experiments.

    Args:
        z_scores: All z-scores (n_pairs * n_test_perms,)
        mu_pred: Predicted means per pair (n_pairs,)
        sigma_pred: Predicted stds per pair (n_pairs,)
        counts_test: Test counts (n_pairs, n_test_perms)
        counts_flat: Flattened counts (n_pairs * n_test_perms,)
        mu_pred_expanded: Predicted means expanded (n_pairs * n_test_perms,)
        sigma_pred_expanded: Predicted stds expanded (n_pairs * n_test_perms,)
        X: Feature matrix (n_pairs, 5 or 12)
        metapath: Metapath name
        feature_type: 'base' or 'enhanced'
        variance_type: 'linear' or 'negbin'
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

    feat_label = 'base' if feature_type == 'base' else 'enhanced'
    var_label = 'linear' if variance_type == 'linear' else 'negbin'
    plt.suptitle(f'Distribution Diagnostics: {metapath} ' +
                 f'({feat_label} features, {var_label} variance)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = output_dir / f'{metapath}_diagnostics_{feature_type}_{variance_type}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved diagnostic plot: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Validate feature and model improvements'
    )
    parser.add_argument('metapath', type=str, help='Metapath (e.g., CbGpPW)')
    parser.add_argument('--features', type=str, choices=['base', 'enhanced'],
                        default='base', help='Feature type')
    parser.add_argument('--variance', type=str, choices=['linear', 'negbin'],
                        default='linear', help='Variance model type')
    parser.add_argument('--all', action='store_true',
                        help='Run all 4 experiments (baseline + 3 improvements)')

    args = parser.parse_args()

    metapath = args.metapath

    # Metapath configuration
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
    output_dir = repo_dir / 'results' / 'improvements'
    output_dir.mkdir(parents=True, exist_ok=True)

    K = 5
    n_samples = 10000
    random_state = 42

    if args.all:
        # Run all 4 experiments
        experiments = [
            ('base', 'linear'),      # Baseline
            ('enhanced', 'linear'),  # Exp 1: Enhanced features
            ('base', 'negbin'),      # Exp 2: Negative binomial
            ('enhanced', 'negbin')   # Exp 3: Combined
        ]
    else:
        # Run single experiment
        experiments = [(args.features, args.variance)]

    all_results = []

    for feat_type, var_type in experiments:
        results_df = run_experiment(
            metapath, edge1_type, edge2_type, data_dir,
            feature_type=feat_type, variance_type=var_type,
            K=K, n_samples=n_samples, random_state=random_state
        )
        all_results.append(results_df)

        # Save individual experiment results
        output_file = output_dir / f'{metapath}_{feat_type}_{var_type}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved results: {output_file}")

        # Generate diagnostic plots for this experiment
        generate_diagnostic_plots(
            metapath, edge1_type, edge2_type, data_dir, output_dir,
            feature_type=feat_type, variance_type=var_type,
            K=K, n_samples=n_samples, random_state=random_state
        )

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Create summary
    summary = create_summary(combined_df)
    summary_file = output_dir / f'{metapath}_comparison_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")

    print("\nSummary Statistics:")
    print(summary.to_string(index=False))

    # Create comparison plot if multiple experiments
    if len(experiments) > 1:
        plot_comparison(summary, output_dir)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
