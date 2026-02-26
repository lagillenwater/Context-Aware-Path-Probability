#!/usr/bin/env python3
"""
Test Phase 5b: Bias Pattern Diagnostics.

This script analyzes whether prediction bias varies systematically with
pathway count magnitude (heteroscedasticity) or is constant across the range.

Usage:
    python test_src/test_phase5b_bias_diagnostics.py
    python test_src/test_phase5b_bias_diagnostics.py --metapath CtDaG
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import argparse
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import extract_features_from_original
from pathway_evaluation_v2 import validate_on_permutations


def parse_metapath(metapath_str):
    """
    Parse metapath string into edge types.

    Parameters
    ----------
    metapath_str : str
        Metapath code

    Returns
    -------
    tuple
        (edge1_type, edge2_type)
    """
    metapath_map = {
        'CbGpPW': ('CbG', 'GpPW'),
        'GiGiG': ('GiG', 'GiG'),
        'CtDaG': ('CtD', 'DaG'),
        'CrCbG': ('CrC', 'CbG'),
        'CbGiG': ('CbG', 'GiG'),
        'CpDaG': ('CpD', 'DaG'),
        'CbGpBP': ('CbG', 'GpBP'),
    }

    if metapath_str in metapath_map:
        return metapath_map[metapath_str]
    else:
        raise ValueError(f"Unknown metapath: {metapath_str}")


def plot_bias_diagnostics(y_true, y_pred, output_file):
    """
    Create diagnostic plots for bias pattern analysis.

    Parameters
    ----------
    y_true : np.ndarray
        True values (permutation averages)
    y_pred : np.ndarray
        Predicted values
    output_file : Path
        Output file path
    """
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.axhline(np.mean(residuals), color='blue', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(residuals):.4f}')
    ax1.set_xlabel('Predicted Value', fontsize=11)
    ax1.set_ylabel('Residual (Pred - True)', fontsize=11)
    ax1.set_title('Residuals vs Predicted Values', fontsize=12,
                  fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    ax2.scatter(y_true, residuals, alpha=0.6, s=50, color='orange')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.axhline(np.mean(residuals), color='blue', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(residuals):.4f}')
    ax2.set_xlabel('True Value', fontsize=11)
    ax2.set_ylabel('Residual (Pred - True)', fontsize=11)
    ax2.set_title('Residuals vs True Values', fontsize=12,
                  fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = axes[0, 2]
    ax3.scatter(y_true, abs_residuals, alpha=0.6, s=50, color='purple')

    z = np.polyfit(y_true, abs_residuals, 1)
    p = np.poly1d(z)
    y_true_sorted = np.sort(y_true)
    ax3.plot(y_true_sorted, p(y_true_sorted), 'r--', linewidth=2,
             label=f'Linear fit: slope={z[0]:.4f}')

    ax3.set_xlabel('True Value', fontsize=11)
    ax3.set_ylabel('Absolute Residual', fontsize=11)
    ax3.set_title('Heteroscedasticity Check', fontsize=12,
                  fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = axes[1, 0]
    ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.axvline(np.mean(residuals), color='blue', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(residuals):.4f}')
    ax4.set_xlabel('Residual', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')

    ax5 = axes[1, 1]
    ax5.scatter(y_true, y_pred, alpha=0.6, s=50, color='green')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect prediction')

    r, p_val = pearsonr(y_true, y_pred)
    ax5.text(0.05, 0.95, f'r = {r:.4f}\np = {p_val:.2e}',
             transform=ax5.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax5.set_xlabel('True Value', fontsize=11)
    ax5.set_ylabel('Predicted Value', fontsize=11)
    ax5.set_title('Predicted vs True Values', fontsize=12,
                  fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    ax6 = axes[1, 2]
    n_quartiles = 4
    quartiles = np.quantile(y_true, np.linspace(0, 1, n_quartiles + 1))
    quartile_labels = []
    quartile_biases = []
    quartile_abs_biases = []

    for i in range(n_quartiles):
        mask = (y_true >= quartiles[i]) & (y_true < quartiles[i + 1])
        if i == n_quartiles - 1:
            mask = (y_true >= quartiles[i]) & (y_true <= quartiles[i + 1])

        quartile_labels.append(f'Q{i+1}')
        quartile_biases.append(np.mean(residuals[mask]))
        quartile_abs_biases.append(np.mean(abs_residuals[mask]))

    x_pos = np.arange(n_quartiles)
    ax6.bar(x_pos, quartile_biases, alpha=0.7, label='Mean bias')
    ax6.bar(x_pos, quartile_abs_biases, alpha=0.7, label='Mean |bias|')
    ax6.axhline(0, color='black', linewidth=0.5)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(quartile_labels)
    ax6.set_xlabel('Quartile', fontsize=11)
    ax6.set_ylabel('Bias', fontsize=11)
    ax6.set_title('Bias by Pathway Count Quartile', fontsize=12,
                  fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_bias_pattern(y_true, y_pred):
    """
    Compute bias pattern statistics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Bias pattern statistics
    """
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)

    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    hetero_corr, hetero_p = pearsonr(y_true, abs_residuals)

    n_quartiles = 4
    quartiles = np.quantile(y_true, np.linspace(0, 1, n_quartiles + 1))
    quartile_biases = []
    quartile_abs_biases = []
    quartile_ranges = []

    for i in range(n_quartiles):
        mask = (y_true >= quartiles[i]) & (y_true < quartiles[i + 1])
        if i == n_quartiles - 1:
            mask = (y_true >= quartiles[i]) & (y_true <= quartiles[i + 1])

        quartile_ranges.append(f'[{quartiles[i]:.4f}, {quartiles[i+1]:.4f}]')
        quartile_biases.append(float(np.mean(residuals[mask])))
        quartile_abs_biases.append(float(np.mean(abs_residuals[mask])))

    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'mean_bias': float(np.mean(residuals)),
        'std_bias': float(np.std(residuals)),
        'mean_abs_bias': float(np.mean(abs_residuals)),
        'rmse': float(np.sqrt(np.mean(residuals ** 2))),
        'heteroscedasticity_corr': float(hetero_corr),
        'heteroscedasticity_p': float(hetero_p),
        'quartile_ranges': quartile_ranges,
        'quartile_biases': quartile_biases,
        'quartile_abs_biases': quartile_abs_biases,
        'bias_range': float(max(quartile_biases) - min(quartile_biases))
    }


def main():
    """
    Main Phase 5b diagnostic function.
    """
    parser = argparse.ArgumentParser(
        description='Phase 5b: Bias Pattern Diagnostics'
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
        help='First permutation ID for validation (default: 0)'
    )
    parser.add_argument(
        '--perm-end',
        type=int,
        default=19,
        help='Last permutation ID for validation (default: 19)'
    )

    args = parser.parse_args()

    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results' / 'phase5b_bias_diagnostics'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 5b: Bias Pattern Diagnostics")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Bins: {args.n_bins} x {args.n_bins}")
    print(f"Feature Set: E (polynomial terms)")
    print(f"Validation permutations: {args.perm_start:03d}-{args.perm_end:03d}")
    print(f"Results directory: {results_dir}")
    print()

    edge1_type, edge2_type = parse_metapath(args.metapath)
    print(f"Edge types: {edge1_type} -> {edge2_type}")
    print()

    print("=" * 80)
    print("Step 1: Training model")
    print("=" * 80)

    X_train, y_train, metadata = extract_features_from_original(
        edge1_type, edge2_type, data_dir, args.n_bins,
        feature_set='E'
    )

    print(f"Features: {metadata['n_features']}")
    print(f"Training samples: {metadata['n_samples']}")
    print()

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(f"Model trained")
    print()

    print("=" * 80)
    print("Step 2: Getting predictions and true values")
    print("=" * 80)

    y_train_pred = lr.predict(X_train)

    class LRWrapper:
        """Wrapper for sklearn LinearRegression."""
        def __init__(self, model):
            self.model = model
            self.device = 'cpu'

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            pass

        def __call__(self, X):
            if isinstance(X, torch.Tensor):
                X_np = X.cpu().numpy()
            else:
                X_np = X
            predictions = self.model.predict(X_np).reshape(-1, 1)
            return torch.FloatTensor(predictions)

    perm_ids = list(range(args.perm_start, args.perm_end + 1))
    validation_results = validate_on_permutations(
        LRWrapper(lr), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    y_true = validation_results['y_true']
    y_pred = validation_results['y_pred']

    print(f"Predictions obtained for {len(y_true)} bins")
    print(f"Validation r: {validation_results['validation_r']:.4f}")
    print(f"Mean bias: {validation_results['mean_error']:.4f}")
    print()

    print("=" * 80)
    print("Step 3: Analyzing bias pattern")
    print("=" * 80)

    stats = analyze_bias_pattern(y_true, y_pred)

    print(f"Overall Statistics:")
    print(f"  Pearson r: {stats['pearson_r']:.4f} (p = {stats['pearson_p']:.2e})")
    print(f"  Spearman r: {stats['spearman_r']:.4f} (p = {stats['spearman_p']:.2e})")
    print(f"  Mean bias: {stats['mean_bias']:.4f}")
    print(f"  Std bias: {stats['std_bias']:.4f}")
    print(f"  Mean |bias|: {stats['mean_abs_bias']:.4f}")
    print(f"  RMSE: {stats['rmse']:.4f}")
    print()

    print(f"Heteroscedasticity Test:")
    print(f"  Correlation (true value vs |residual|): {stats['heteroscedasticity_corr']:.4f}")
    print(f"  p-value: {stats['heteroscedasticity_p']:.2e}")

    if stats['heteroscedasticity_p'] < 0.05:
        if stats['heteroscedasticity_corr'] > 0:
            print(f"  Result: HETEROSCEDASTIC (bias increases with pathway count)")
        else:
            print(f"  Result: HETEROSCEDASTIC (bias decreases with pathway count)")
    else:
        print(f"  Result: HOMOSCEDASTIC (constant bias)")
    print()

    print(f"Bias by Quartile:")
    for i, (range_str, bias, abs_bias) in enumerate(zip(
        stats['quartile_ranges'],
        stats['quartile_biases'],
        stats['quartile_abs_biases']
    )):
        print(f"  Q{i+1} {range_str}: bias = {bias:+.4f}, |bias| = {abs_bias:.4f}")

    print(f"\n  Bias range across quartiles: {stats['bias_range']:.4f}")

    if stats['bias_range'] > 0.01:
        print(f"  Result: VARIABLE BIAS (simple offset may be insufficient)")
    else:
        print(f"  Result: CONSTANT BIAS (simple offset is appropriate)")
    print()

    print("=" * 80)
    print("Step 4: Saving results")
    print("=" * 80)

    stats_file = results_dir / f'{args.metapath}_bias_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved: {stats_file}")

    plot_file = results_dir / f'{args.metapath}_bias_diagnostics.png'
    plot_bias_diagnostics(y_true, y_pred, plot_file)
    print(f"Diagnostic plots saved: {plot_file}")
    print()

    print("=" * 80)
    print("Phase 5b diagnostics complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
