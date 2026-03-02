#!/usr/bin/env python3
"""
Regularization study: ridge/lasso to address overfitting.

This script tests whether regularized linear models (Ridge, Lasso) reduce
the overfitting observed in linear-model CV results (training r = 1.0, perm r = 0.887).

Usage:
    python test_src/regularization_study.py
    python test_src/regularization_study.py --metapath CtDaG
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold

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


def run_cv_with_model(model, X, y, perm_ids, edge1_type, edge2_type,
                       data_dir, n_bins):
    """
    Run K-Fold CV with given model.

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

    Returns
    -------
    dict
        CV results
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model.fit(X_train, y_train)

        # Training performance
        y_train_pred = model.predict(X_train)
        train_r, _ = pearsonr(y_train_pred, y_train)
        train_bias = np.mean(y_train_pred - y_train)

        # Validation performance (held-out bins)
        y_val_pred = model.predict(X_val)
        val_r, _ = pearsonr(y_val_pred, y_val)
        val_bias = np.mean(y_val_pred - y_val)

        fold_results.append({
            'fold': fold_idx + 1,
            'train_r': train_r,
            'val_r': val_r,
            'train_bias': train_bias,
            'val_bias': val_bias
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
        'fold_results': fold_results,
        'perm_validation_r': perm_results['validation_r'],
        'perm_mean_error': perm_results['mean_error'],
        'perm_rmse': perm_results['rmse']
    }


def plot_regularization_comparison(results_df, output_file):
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
    ax2.bar(x_pos, results_df['mean_train_r'], yerr=train_std,
            color='steelblue', capsize=5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Training r', fontsize=11)
    ax2.set_title('Training r Across Folds', fontsize=12, fontweight='bold')
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
    overfitting_gap = results_df['mean_train_r'] - results_df['perm_r']
    colors = ['red' if gap > 0.1 else 'green' for gap in overfitting_gap]
    ax6.bar(x_pos, overfitting_gap, color=colors)
    ax6.axhline(0, color='black', linewidth=0.5)
    ax6.axhline(0.1, color='orange', linestyle='--',
                label='Overfitting threshold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(methods, rotation=45, ha='right')
    ax6.set_ylabel('Training r - Perm r', fontsize=11)
    ax6.set_title('Overfitting Gap', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main regularization-study function.
    """
    parser = argparse.ArgumentParser(
        description='Regularization study: ridge/lasso'
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
    results_dir = repo_dir / 'results' / 'phase5c_regularization'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Regularization Study: Ridge/Lasso for Overfitting")
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
        ('Ridge(α=0.001)', Ridge(alpha=0.001)),
        ('Ridge(α=0.01)', Ridge(alpha=0.01)),
        ('Ridge(α=0.1)', Ridge(alpha=0.1)),
        ('Ridge(α=1.0)', Ridge(alpha=1.0)),
        ('Ridge(α=10.0)', Ridge(alpha=10.0)),
        ('Lasso(α=0.001)', Lasso(alpha=0.001, max_iter=10000)),
        ('Lasso(α=0.01)', Lasso(alpha=0.01, max_iter=10000)),
        ('Lasso(α=0.1)', Lasso(alpha=0.1, max_iter=10000)),
    ]

    results = []

    print("=" * 80)
    print("Step 2: Testing Regularization Methods")
    print("=" * 80)
    print()

    for method_name, model in models_to_test:
        print(f"Testing: {method_name}")
        print("-" * 80)

        cv_results = run_cv_with_model(
            model, X_train, y_train, perm_ids,
            edge1_type, edge2_type, data_dir, args.n_bins
        )

        fold_df = pd.DataFrame(cv_results['fold_results'])

        mean_train_r = fold_df['train_r'].mean()
        std_train_r = fold_df['train_r'].std()
        mean_val_r = fold_df['val_r'].mean()

        print(f"  CV Training r: {mean_train_r:.4f} ± {std_train_r:.4f}")
        print(f"  CV Validation r: {mean_val_r:.4f}")
        print(f"  Permutation r: {cv_results['perm_validation_r']:.4f}")
        print(f"  Permutation bias: {cv_results['perm_mean_error']:.4f}")
        print(f"  Permutation RMSE: {cv_results['perm_rmse']:.4f}")
        print(f"  Overfitting gap: {mean_train_r - cv_results['perm_validation_r']:.4f}")
        print()

        results.append({
            'method': method_name,
            'mean_train_r': float(mean_train_r),
            'std_train_r': float(std_train_r),
            'mean_val_r': float(mean_val_r),
            'perm_r': float(cv_results['perm_validation_r']),
            'perm_bias': float(cv_results['perm_mean_error']),
            'perm_rmse': float(cv_results['perm_rmse'])
        })

    print("=" * 80)
    print("REGULARIZATION METHOD COMPARISON")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df['overfitting_gap'] = results_df['mean_train_r'] - results_df['perm_r']

    results_df = results_df.sort_values('overfitting_gap')

    print(results_df.to_string(index=False))
    print()

    best_idx = results_df['overfitting_gap'].idxmin()
    print(f"Best method (lowest overfitting): {results_df.iloc[best_idx]['method']}")
    print(f"  Training r: {results_df.iloc[best_idx]['mean_train_r']:.4f}")
    print(f"  Permutation r: {results_df.iloc[best_idx]['perm_r']:.4f}")
    print(f"  Overfitting gap: {results_df.iloc[best_idx]['overfitting_gap']:.4f}")
    print(f"  Permutation bias: {results_df.iloc[best_idx]['perm_bias']:.4f}")
    print()

    csv_file = results_dir / f'{args.metapath}_regularization_comparison.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"Results saved: {csv_file}")

    plot_file = results_dir / f'{args.metapath}_regularization_comparison.png'
    plot_regularization_comparison(results_df, plot_file)
    print(f"Plot saved: {plot_file}")
    print()

    print("=" * 80)
    print("Regularization study complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
