"""
Visualize distribution diagnostics for null distribution prediction.

Creates comprehensive diagnostic plots showing how well predicted distributions
match empirical test data:
1. Z-score histogram vs. N(0,1)
2. Q-Q plot
3. Mean calibration scatter
4. Variance calibration scatter
5. Example pair distributions
6. Residual analysis

Usage:
    python visualize_distribution_diagnostics.py CbGpPW --experiment baseline
    python visualize_distribution_diagnostics.py CbGpPW --experiment enhanced_linear
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
from sklearn.linear_model import LinearRegression
from pathlib import Path
import argparse
import sys


def load_permuted_edge_matrices(edge1_type, edge2_type, perm_num, data_dir):
    """
    Load edge matrices from a specific permutation.

    Args:
        edge1_type: First edge type (e.g., 'CbG')
        edge2_type: Second edge type (e.g., 'GpPW')
        perm_num: Permutation number (0-20)
        data_dir: Path to data directory

    Returns:
        Tuple of (edge1_matrix, edge2_matrix) as scipy sparse matrices
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

    # Convert boolean to int32 if needed
    if edge1.dtype == bool:
        edge1 = edge1.astype(np.int32)
    if edge2.dtype == bool:
        edge2 = edge2.astype(np.int32)

    return edge1, edge2


def count_2hop_paths(edge1, edge2, source_idx, target_idx):
    """
    Count 2-hop paths from source to target through intermediates.

    Args:
        edge1: First edge matrix (source -> intermediate)
        edge2: Second edge matrix (intermediate -> target)
        source_idx: Source node index
        target_idx: Target node index

    Returns:
        Number of paths
    """
    intermediates = edge1[source_idx, :].nonzero()[1]
    if len(intermediates) == 0:
        return 0

    count = 0
    for intermediate in intermediates:
        if edge2[intermediate, target_idx]:
            count += 1

    return count


def sample_pairs(edge1, edge2, n_samples=10000, random_state=42):
    """
    Sample 50% pairs with pathways, 50% random pairs.

    Args:
        edge1: First edge matrix
        edge2: Second edge matrix
        n_samples: Number of pairs to sample
        random_state: Random seed

    Returns:
        Array of (source_idx, target_idx) pairs
    """
    np.random.seed(random_state)

    n_sources = edge1.shape[0]
    n_targets = edge2.shape[1]

    n_positive = n_samples // 2
    n_random = n_samples - n_positive

    positive_pairs = []
    attempts = 0
    max_attempts = n_positive * 10

    while len(positive_pairs) < n_positive and attempts < max_attempts:
        s = np.random.randint(0, n_sources)
        t = np.random.randint(0, n_targets)

        if count_2hop_paths(edge1, edge2, s, t) > 0:
            positive_pairs.append((s, t))

        attempts += 1

    random_pairs = [
        (np.random.randint(0, n_sources), np.random.randint(0, n_targets))
        for _ in range(n_random)
    ]

    all_pairs = positive_pairs + random_pairs
    return np.array(all_pairs)


def extract_degree_features(pairs, edge1, edge2):
    """
    Extract 5 degree features for each pair.

    Features:
    - deg_source: Source node degree
    - deg_target: Target node degree
    - deg_source * deg_target: Interaction
    - deg_source^2: Source degree squared
    - deg_target^2: Target degree squared

    Args:
        pairs: Array of (source_idx, target_idx) pairs
        edge1: First edge matrix
        edge2: Second edge matrix

    Returns:
        Feature matrix (n, 5)
    """
    deg_source = np.array(edge1.sum(axis=1)).flatten()
    deg_target = np.array(edge2.sum(axis=0)).flatten()

    features = []
    for s, t in pairs:
        ds = deg_source[s]
        dt = deg_target[t]
        features.append([
            ds,
            dt,
            ds * dt,
            ds * ds,
            dt * dt
        ])

    return np.array(features)


def compute_expected_intermediate_features(pairs, edge1, edge2):
    """
    Compute 7 expected intermediate degree features.

    Args:
        pairs: Array of (source_idx, target_idx) pairs
        edge1: First edge matrix
        edge2: Second edge matrix

    Returns:
        Feature matrix (n, 7)
    """
    deg_in = np.array(edge1.sum(axis=0)).flatten()
    deg_out = np.array(edge2.sum(axis=1)).flatten()

    features = []
    for s, t in pairs:
        source_neighbors = edge1[s, :].nonzero()[1]
        target_neighbors = edge2[:, t].nonzero()[0]

        possible_intermediates = np.intersect1d(source_neighbors, target_neighbors)

        if len(possible_intermediates) == 0:
            features.append([0, 0, 0, 0, 0, 0, 0])
            continue

        deg_in_inter = deg_in[possible_intermediates]
        deg_out_inter = deg_out[possible_intermediates]

        n_inter = len(possible_intermediates)
        mean_deg_in = np.mean(deg_in_inter)
        mean_deg_out = np.mean(deg_out_inter)
        std_deg = np.std(np.concatenate([deg_in_inter, deg_out_inter]))
        min_deg = np.min(np.concatenate([deg_in_inter, deg_out_inter]))
        max_deg = np.max(np.concatenate([deg_in_inter, deg_out_inter]))
        interaction = mean_deg_in * mean_deg_out

        features.append([
            n_inter,
            mean_deg_in,
            mean_deg_out,
            std_deg,
            min_deg,
            max_deg,
            interaction
        ])

    return np.array(features)


def extract_degree_features_enhanced(pairs, edge1, edge2):
    """
    Extract 12 features: 5 degree + 7 expected intermediate.

    Args:
        pairs: Array of (source_idx, target_idx) pairs
        edge1: First edge matrix
        edge2: Second edge matrix

    Returns:
        Feature matrix (n, 12)
    """
    X_base = extract_degree_features(pairs, edge1, edge2)
    X_intermediate = compute_expected_intermediate_features(pairs, edge1, edge2)
    return np.column_stack([X_base, X_intermediate])


def load_or_train_models(metapath, edge1_type, edge2_type, data_dir,
                          feature_type='base', variance_type='linear',
                          K=5, n_samples=10000, random_state=42):
    """
    Load or train models for mean and variance prediction.

    Args:
        metapath: Metapath name
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory
        feature_type: 'base' or 'enhanced'
        variance_type: 'linear' or 'negbin'
        K: Number of training permutations
        n_samples: Number of pairs to sample
        random_state: Random seed

    Returns:
        Tuple of (model_mean, model_std, pairs, X, variance_model_type)
    """
    print(f"\nTraining models...")
    print(f"  Feature type: {feature_type}")
    print(f"  Variance type: {variance_type}")
    print(f"  K: {K}")

    edge1_list = []
    edge2_list = []
    for perm in range(K):
        edge1, edge2 = load_permuted_edge_matrices(
            edge1_type, edge2_type, perm, data_dir
        )
        edge1_list.append(edge1)
        edge2_list.append(edge2)

    edge1_train = edge1_list[0]
    edge2_train = edge2_list[0]

    pairs = sample_pairs(edge1_train, edge2_train, n_samples, random_state)

    if feature_type == 'enhanced':
        X = extract_degree_features_enhanced(pairs, edge1_train, edge2_train)
    else:
        X = extract_degree_features(pairs, edge1_train, edge2_train)

    counts_train = np.zeros((len(pairs), K))
    for k in range(K):
        for i, (s, t) in enumerate(pairs):
            counts_train[i, k] = count_2hop_paths(
                edge1_list[k], edge2_list[k], s, t
            )

    mu_train = np.mean(counts_train, axis=1)
    sigma_train = np.std(counts_train, axis=1, ddof=1)

    model_mean = LinearRegression()
    model_mean.fit(X, mu_train)

    if variance_type == 'negbin':
        empirical_r = mu_train**2 / (sigma_train**2 - mu_train + 1e-6)
        empirical_r = np.clip(empirical_r, 0.1, 100)

        model_std = LinearRegression()
        model_std.fit(X, np.log(empirical_r + 1e-6))
        variance_model_type = 'negbin'
    else:
        model_std = LinearRegression()
        model_std.fit(X, sigma_train)
        variance_model_type = 'linear'

    print(f"  Models trained on {len(pairs)} pairs")

    return model_mean, model_std, pairs, X, variance_model_type


def compute_test_statistics(pairs, edge1_type, edge2_type, data_dir,
                             test_perms, model_mean, model_std, X,
                             variance_model_type='linear'):
    """
    Compute predictions and empirical statistics on test permutations.

    Args:
        pairs: Array of (source_idx, target_idx) pairs
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory
        test_perms: List of test permutation numbers
        model_mean: Trained mean model
        model_std: Trained variance model
        X: Feature matrix
        variance_model_type: 'linear' or 'negbin'

    Returns:
        Dictionary with:
            - mu_pred: Predicted means
            - sigma_pred: Predicted standard deviations
            - mu_emp: Empirical means (from test perms)
            - sigma_emp: Empirical stds (from test perms)
            - counts_test: Actual counts (n_pairs, n_test_perms)
            - z_scores: Z-scores for all (pair, perm) combinations
    """
    print(f"\nComputing test statistics...")

    n_test = len(test_perms)
    counts_test = np.zeros((len(pairs), n_test))

    for i, perm in enumerate(test_perms):
        edge1, edge2 = load_permuted_edge_matrices(
            edge1_type, edge2_type, perm, data_dir
        )
        for j, (s, t) in enumerate(pairs):
            counts_test[j, i] = count_2hop_paths(edge1, edge2, s, t)

    mu_pred = model_mean.predict(X)

    if variance_model_type == 'negbin':
        log_r_pred = model_std.predict(X)
        r_pred = np.exp(log_r_pred)
        r_pred = np.clip(r_pred, 0.1, 100)
        var_pred = mu_pred + mu_pred**2 / r_pred
        sigma_pred = np.sqrt(var_pred)
    else:
        sigma_pred = model_std.predict(X)

    sigma_pred = np.maximum(sigma_pred, 0.01)

    # Compute z-scores for INDIVIDUAL (pair, perm) combinations
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

    print(f"  Computed statistics for {len(pairs)} pairs x {n_test} test perms")

    return {
        'mu_pred_per_pair': mu_pred,
        'sigma_pred_per_pair': sigma_pred,
        'mu_pred': mu_pred_expanded,
        'sigma_pred': sigma_pred_expanded,
        'counts_test': counts_test,
        'counts_flat': counts_flat,
        'z_scores': z_scores
    }


def create_diagnostic_plot(stats, feature_type, variance_type,
                            pairs, X, output_file):
    """
    Create 6-panel diagnostic visualization.

    Args:
        stats: Dictionary from compute_test_statistics
        feature_type: 'base' or 'enhanced'
        variance_type: 'linear' or 'negbin'
        pairs: Array of (source_idx, target_idx) pairs
        X: Feature matrix
        output_file: Path to save plot
    """
    fig = plt.figure(figsize=(16, 10))

    z_scores = stats['z_scores']
    mu_pred_expanded = stats['mu_pred']
    sigma_pred_expanded = stats['sigma_pred']
    counts_flat = stats['counts_flat']
    counts_test = stats['counts_test']

    # Panel 1: Z-score histogram
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(z_scores, bins=50, density=True, alpha=0.7,
             label=f'Empirical (n={len(z_scores)})')
    x = np.linspace(-4, 4, 100)
    ax1.plot(x, sp_stats.norm.pdf(x, 0, 1), 'r-', linewidth=2,
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
    sp_stats.probplot(z_scores, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot\nr={np.corrcoef(np.sort(z_scores), sp_stats.norm.ppf((np.arange(len(z_scores)) + 0.5) / len(z_scores)))[0,1]:.4f}')
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

    mu_pred_per_pair = stats['mu_pred_per_pair']
    sigma_pred_per_pair = stats['sigma_pred_per_pair']

    deg_source = X[:, 0]
    deg_target = X[:, 1]

    low_idx = np.argmin(deg_source * deg_target)
    high_idx = np.argmax(deg_source * deg_target)
    mid_idx = np.argsort(deg_source * deg_target)[len(mu_pred_per_pair) // 2]

    example_indices = [low_idx, mid_idx, high_idx]
    colors = ['blue', 'green', 'red']
    labels = ['Low degree', 'Med degree', 'High degree']

    for idx, color, label in zip(example_indices, colors, labels):
        counts = counts_test[idx, :]
        mu = mu_pred_per_pair[idx]
        sigma = sigma_pred_per_pair[idx]

        x = np.linspace(max(0, mu - 3*sigma), mu + 3*sigma, 100)
        pdf = sp_stats.norm.pdf(x, mu, sigma)

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

    plt.suptitle(f'Distribution Diagnostics: {feature_type} features, ' +
                 f'{variance_type} variance',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved diagnostic plot: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Visualize distribution diagnostics'
    )
    parser.add_argument('metapath', type=str, help='Metapath (e.g., CbGpPW)')
    parser.add_argument('--experiment', type=str,
                        default='baseline',
                        help='Experiment name (baseline, enhanced_linear, ' +
                             'base_negbin, enhanced_negbin)')
    parser.add_argument('--K', type=int, default=5,
                        help='Number of training permutations')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of pairs to sample')

    args = parser.parse_args()

    metapath = args.metapath
    experiment = args.experiment
    K = args.K
    n_samples = args.n_samples

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

    experiment_config = {
        'baseline': ('base', 'linear'),
        'enhanced_linear': ('enhanced', 'linear'),
        'base_negbin': ('base', 'negbin'),
        'enhanced_negbin': ('enhanced', 'negbin')
    }

    if experiment not in experiment_config:
        print(f"Error: Unknown experiment '{experiment}'")
        print(f"Available: {list(experiment_config.keys())}")
        sys.exit(1)

    feature_type, variance_type = experiment_config[experiment]

    repo_dir = Path(__file__).parent.parent
    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'improvements'
    output_dir.mkdir(parents=True, exist_ok=True)

    test_perms = list(range(15, 21))
    random_state = 42

    print(f"Configuration:")
    print(f"  Metapath: {metapath}")
    print(f"  Edge 1: {edge1_type}")
    print(f"  Edge 2: {edge2_type}")
    print(f"  Experiment: {experiment}")
    print(f"  Feature type: {feature_type}")
    print(f"  Variance type: {variance_type}")
    print(f"  K: {K}")
    print(f"  Test perms: {test_perms}")
    print(f"  Samples: {n_samples}")

    model_mean, model_std, pairs, X, variance_model_type = load_or_train_models(
        metapath, edge1_type, edge2_type, data_dir,
        feature_type, variance_type, K, n_samples, random_state
    )

    stats = compute_test_statistics(
        pairs, edge1_type, edge2_type, data_dir, test_perms,
        model_mean, model_std, X, variance_model_type
    )

    output_file = output_dir / f'distribution_diagnostics_{experiment}.png'
    create_diagnostic_plot(
        stats, feature_type, variance_type, pairs, X, output_file
    )

    print("\nDiagnostic Summary:")
    print(f"  Z-score mean: {np.mean(np.abs(stats['z_scores'])):.4f} (target: 0.8)")
    print(f"  Z-score std: {np.std(stats['z_scores']):.4f} (target: 1.0)")
    print(f"  Mean r (individual perms): {np.corrcoef(stats['counts_flat'], stats['mu_pred'])[0,1]:.4f}")
    print(f"  Std r (absolute residuals): {np.corrcoef(stats['sigma_pred'], np.abs(stats['counts_flat'] - stats['mu_pred']))[0,1]:.4f}")
    print(f"  Q-Q corr: {np.corrcoef(np.sort(stats['z_scores']), sp_stats.norm.ppf((np.arange(len(stats['z_scores'])) + 0.5) / len(stats['z_scores'])))[0,1]:.4f} (target: >0.95)")


if __name__ == '__main__':
    main()
