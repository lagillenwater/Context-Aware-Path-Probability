#!/usr/bin/env python3
"""
Test Phase 5c: Repeated Random Train/Test Splits.

This script tests regularization using repeated random 80/20 splits instead
of K-Fold CV to get a different perspective on overfitting.

Usage:
    python test_src/test_phase5c_repeated_splits.py
    python test_src/test_phase5c_repeated_splits.py --metapath CtDaG
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

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


def run_repeated_splits(model, X, y, perm_ids, edge1_type, edge2_type,
                         data_dir, n_bins, n_splits=5):
    """
    Run repeated random train/test splits.

    Parameters
    ----------
    model : sklearn estimator
        Model to test
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    perm_ids : list
        Permutation IDs for validation
    edge1_type : str
        First edge type
    edge2_type : str
        Second edge type
    data_dir : Path
        Data directory
    n_bins : int
        Number of bins
    n_splits : int
        Number of random splits

    Returns
    -------
    dict
        Split results
    """
    split_results = []

    for split_idx in range(n_splits):
        random_state = 42 + split_idx

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        # Train model
        model.fit(X_train, y_train)

        # Training performance
        y_train_pred = model.predict(X_train)
        train_r, _ = pearsonr(y_train_pred, y_train)
        train_bias = np.mean(y_train_pred - y_train)

        # Test performance (held-out bins)
        y_test_pred = model.predict(X_test)
        test_r, _ = pearsonr(y_test_pred, y_test)
        test_bias = np.mean(y_test_pred - y_test)

        split_results.append({
            'split': split_idx + 1,
            'train_r': train_r,
            'test_r': test_r,
            'train_bias': train_bias,
            'test_bias': test_bias
        })

    # Train on all data and validate on permutations
    model.fit(X, y)

    class ModelWrapper:
        """Wrapper for sklearn models."""
        def __init__(self, model):
            self.model = model
            self.device = 'cpu'

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            pass

        def __call__(self, X_torch):
            import torch
            if isinstance(X_torch, torch.Tensor):
                X_np = X_torch.cpu().numpy()
            else:
                X_np = X_torch
            predictions = self.model.predict(X_np).reshape(-1, 1)
            return torch.FloatTensor(predictions)

    perm_results = validate_on_permutations(
        ModelWrapper(model), edge1_type, edge2_type, perm_ids,
        data_dir, n_bins, 'cpu',
        feature_set='E'
    )

    return {
        'split_results': split_results,
        'perm_validation_r': perm_results['validation_r'],
        'perm_mean_error': perm_results['mean_error'],
        'perm_rmse': perm_results['rmse']
    }


def plot_comparison(results_df, output_file):
    """
    Plot comparison of regularization methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results for each method
    output_file : Path
        Output file path
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    methods = results_df['method'].values
    x_pos = np.arange(len(methods))

    ax1 = axes[0, 0]
    ax1.bar(x_pos, results_df['mean_train_r'], color='steelblue', alpha=0.7,
            label='Training r')
    ax1.bar(x_pos, results_df['perm_r'], color='coral', alpha=0.7,
            label='Permutation r')
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=0.5,
                label='Perfect fit')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Correlation r', fontsize=11)
    ax1.set_title('Training vs Permutation r', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')

    ax2 = axes[0, 1]
    train_std = results_df['std_train_r']
    test_std = results_df['std_test_r']
    width = 0.35
    ax2.bar(x_pos - width/2, results_df['mean_train_r'], width,
            yerr=train_std, label='Training r', color='steelblue', capsize=3)
    ax2.bar(x_pos + width/2, results_df['mean_test_r'], width,
            yerr=test_std, label='Test r', color='coral', capsize=3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Correlation r', fontsize=11)
    ax2.set_title('Train vs Test r (Repeated Splits)', fontsize=12,
                  fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    ax3 = axes[0, 2]
    ax3.bar(x_pos, results_df['perm_r'], color='coral')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('Permutation r', fontsize=11)
    ax3.set_title('Permutation Validation r', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    ax4 = axes[1, 0]
    ax4.bar(x_pos, results_df['perm_bias'], color='crimson')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.axhline(-0.01, color='green', linestyle='--',
                label='Target: |bias| < 0.01')
    ax4.axhline(0.01, color='green', linestyle='--')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel('Permutation Bias', fontsize=11)
    ax4.set_title('Prediction Bias', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')

    ax5 = axes[1, 1]
    ax5.bar(x_pos, results_df['perm_rmse'], color='orange')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(methods, rotation=45, ha='right')
    ax5.set_ylabel('RMSE', fontsize=11)
    ax5.set_title('Permutation RMSE', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')

    ax6 = axes[1, 2]
    overfitting_gap = results_df['mean_train_r'] - results_df['mean_test_r']
    colors = ['red' if gap > 0.1 else 'green' for gap in overfitting_gap]
    ax6.bar(x_pos, overfitting_gap, color=colors)
    ax6.axhline(0, color='black', linewidth=0.5)
    ax6.axhline(0.1, color='orange', linestyle='--',
                label='Overfitting threshold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(methods, rotation=45, ha='right')
    ax6.set_ylabel('Training r - Test r', fontsize=11)
    ax6.set_title('Overfitting Gap (Within Graph)', fontsize=12,
                  fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main repeated splits testing function.
    """
    parser = argparse.ArgumentParser(
        description='Phase 5c: Repeated Random Train/Test Splits'
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
        '--n-splits',
        type=int,
        default=5,
        help='Number of random splits (default: 5)'
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
    results_dir = repo_dir / 'results' / 'phase5c_regularization'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 5c: Repeated Random Train/Test Splits")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Bins: {args.n_bins} x {args.n_bins}")
    print(f"Feature Set: E (polynomial terms)")
    print(f"Number of splits: {args.n_splits}")
    print(f"Train/test split: 80/20")
    print(f"Validation permutations: {args.perm_start:03d}-{args.perm_end:03d}")
    print(f"Results directory: {results_dir}")
    print()

    edge1_type, edge2_type = parse_metapath(args.metapath)
    print(f"Edge types: {edge1_type} -> {edge2_type}")
    print()

    print("=" * 80)
    print("Step 1: Extracting features")
    print("=" * 80)

    X_train, y_train, metadata = extract_features_from_original(
        edge1_type, edge2_type, data_dir, args.n_bins,
        feature_set='E'
    )

    print(f"Features: {metadata['n_features']}")
    print(f"Training samples: {metadata['n_samples']}")
    print()

    perm_ids = list(range(args.perm_start, args.perm_end + 1))

    models_to_test = [
        ('LinearRegression', LinearRegression()),
        ('Ridge(α=0.1)', Ridge(alpha=0.1)),
        ('Ridge(α=1.0)', Ridge(alpha=1.0)),
        ('Ridge(α=10.0)', Ridge(alpha=10.0)),
    ]

    results = []

    print("=" * 80)
    print("Step 2: Testing with Repeated Random Splits")
    print("=" * 80)
    print()

    for method_name, model in models_to_test:
        print(f"Testing: {method_name}")
        print("-" * 80)

        split_results = run_repeated_splits(
            model, X_train, y_train, perm_ids,
            edge1_type, edge2_type, data_dir, args.n_bins,
            n_splits=args.n_splits
        )

        split_df = pd.DataFrame(split_results['split_results'])

        mean_train_r = split_df['train_r'].mean()
        std_train_r = split_df['train_r'].std()
        mean_test_r = split_df['test_r'].mean()
        std_test_r = split_df['test_r'].std()

        print(f"  Train r: {mean_train_r:.4f} ± {std_train_r:.4f}")
        print(f"  Test r: {mean_test_r:.4f} ± {std_test_r:.4f}")
        print(f"  Overfitting gap (train - test): {mean_train_r - mean_test_r:.4f}")
        print(f"  Permutation r: {split_results['perm_validation_r']:.4f}")
        print(f"  Permutation bias: {split_results['perm_mean_error']:.4f}")
        print(f"  Permutation RMSE: {split_results['perm_rmse']:.4f}")
        print()

        results.append({
            'method': method_name,
            'mean_train_r': float(mean_train_r),
            'std_train_r': float(std_train_r),
            'mean_test_r': float(mean_test_r),
            'std_test_r': float(std_test_r),
            'perm_r': float(split_results['perm_validation_r']),
            'perm_bias': float(split_results['perm_mean_error']),
            'perm_rmse': float(split_results['perm_rmse'])
        })

    print("=" * 80)
    print("REPEATED SPLITS COMPARISON")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df['overfitting_gap'] = (results_df['mean_train_r'] -
                                      results_df['mean_test_r'])

    print(results_df.to_string(index=False))
    print()

    print("Key Observations:")
    print("-" * 80)
    lr_idx = results_df[results_df['method'] == 'LinearRegression'].index[0]
    lr_gap = results_df.loc[lr_idx, 'overfitting_gap']
    lr_perm_r = results_df.loc[lr_idx, 'perm_r']

    print(f"LinearRegression:")
    print(f"  Overfitting gap (within graph): {lr_gap:.4f}")
    print(f"  Permutation r (actual task): {lr_perm_r:.4f}")
    print()

    best_gap_idx = results_df['overfitting_gap'].idxmin()
    print(f"Lowest overfitting gap: {results_df.iloc[best_gap_idx]['method']}")
    print(f"  Gap: {results_df.iloc[best_gap_idx]['overfitting_gap']:.4f}")
    print(f"  But permutation r: {results_df.iloc[best_gap_idx]['perm_r']:.4f}")
    print()

    best_perm_idx = results_df['perm_r'].idxmax()
    print(f"Best permutation r: {results_df.iloc[best_perm_idx]['method']}")
    print(f"  Permutation r: {results_df.iloc[best_perm_idx]['perm_r']:.4f}")
    print(f"  Overfitting gap: {results_df.iloc[best_perm_idx]['overfitting_gap']:.4f}")
    print()

    csv_file = results_dir / f'{args.metapath}_repeated_splits.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"Results saved: {csv_file}")

    plot_file = results_dir / f'{args.metapath}_repeated_splits.png'
    plot_comparison(results_df, plot_file)
    print(f"Plot saved: {plot_file}")
    print()

    print("=" * 80)
    print("Phase 5c repeated splits complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
