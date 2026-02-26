#!/usr/bin/env python3
"""
Test Phase 5: Bias Correction and Cross-Validation for Linear Regression.

This script implements K-Fold CV on bins and tests bias-aware loss functions
to eliminate systematic underprediction bias while maintaining high accuracy.

Usage:
    python test_src/test_phase5_linear_regression_cv.py
    python test_src/test_phase5_linear_regression_cv.py --metapath CtDaG
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import argparse
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import time

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import extract_features_from_original
from pathway_evaluation_v2 import validate_on_permutations


class BiasAwareLinearRegression:
    """
    Linear Regression with bias-aware loss function.

    Minimizes MSE while penalizing systematic bias (mean error).
    """

    def __init__(self, bias_penalty=1.0):
        """
        Initialize bias-aware linear regression.

        Parameters
        ----------
        bias_penalty : float
            Weight for bias penalty term (lambda)
        """
        self.bias_penalty = bias_penalty
        self.coef_ = None
        self.intercept_ = None

    def _loss(self, params, X, y):
        """
        Compute loss: MSE + lambda * (mean_error)^2.

        Parameters
        ----------
        params : np.ndarray
            Model parameters [intercept, coef1, coef2, ...]
        X : np.ndarray
            Features
        y : np.ndarray
            Targets

        Returns
        -------
        float
            Loss value
        """
        intercept = params[0]
        coef = params[1:]

        y_pred = X @ coef + intercept
        residuals = y_pred - y

        mse = np.mean(residuals ** 2)
        bias = np.mean(residuals)

        return mse + self.bias_penalty * (bias ** 2)

    def fit(self, X, y):
        """
        Fit model by minimizing bias-aware loss.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets

        Returns
        -------
        self
        """
        n_features = X.shape[1]
        init_params = np.zeros(n_features + 1)

        result = minimize(
            self._loss,
            init_params,
            args=(X, y),
            method='L-BFGS-B'
        )

        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]

        return self

    def predict(self, X):
        """
        Predict using fitted model.

        Parameters
        ----------
        X : np.ndarray
            Test features

        Returns
        -------
        np.ndarray
            Predictions
        """
        return X @ self.coef_ + self.intercept_


class AsymmetricLinearRegression:
    """
    Linear Regression with asymmetric loss (penalizes underprediction more).

    Loss: mean((y_pred - y_true)^2 * (1 + alpha * I[y_pred < y_true]))
    """

    def __init__(self, asymmetry=2.0):
        """
        Initialize asymmetric linear regression.

        Parameters
        ----------
        asymmetry : float
            Penalty multiplier for underprediction (alpha)
        """
        self.asymmetry = asymmetry
        self.coef_ = None
        self.intercept_ = None

    def _loss(self, params, X, y):
        """
        Compute asymmetric loss.

        Parameters
        ----------
        params : np.ndarray
            Model parameters [intercept, coef1, coef2, ...]
        X : np.ndarray
            Features
        y : np.ndarray
            Targets

        Returns
        -------
        float
            Loss value
        """
        intercept = params[0]
        coef = params[1:]

        y_pred = X @ coef + intercept
        residuals = y_pred - y
        squared_errors = residuals ** 2

        weights = np.where(
            y_pred < y,
            1 + self.asymmetry,
            1.0
        )

        return np.mean(weights * squared_errors)

    def fit(self, X, y):
        """
        Fit model by minimizing asymmetric loss.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets

        Returns
        -------
        self
        """
        n_features = X.shape[1]
        init_params = np.zeros(n_features + 1)

        result = minimize(
            self._loss,
            init_params,
            args=(X, y),
            method='L-BFGS-B'
        )

        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]

        return self

    def predict(self, X):
        """
        Predict using fitted model.

        Parameters
        ----------
        X : np.ndarray
            Test features

        Returns
        -------
        np.ndarray
            Predictions
        """
        return X @ self.coef_ + self.intercept_


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
        'CbGaD': ('CbG', 'GaD'),
        'CrCbG': ('CrC', 'CbG'),
        'CbGiG': ('CbG', 'GiG'),
        'CpDaG': ('CpD', 'DaG'),
        'CbGpBP': ('CbG', 'GpBP'),
    }

    if metapath_str in metapath_map:
        return metapath_map[metapath_str]
    else:
        raise ValueError(f"Unknown metapath: {metapath_str}")


def plot_cv_results(cv_results, output_file):
    """
    Plot cross-validation results.

    Parameters
    ----------
    cv_results : list of dict
        CV fold results
    output_file : Path
        Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    folds = [r['fold'] for r in cv_results]
    x_pos = np.arange(len(folds))

    ax1 = axes[0, 0]
    train_r = [r['train_r'] for r in cv_results]
    val_r = [r['val_r'] for r in cv_results]
    ax1.plot(x_pos, train_r, 'o-', label='Training', linewidth=2, markersize=8)
    ax1.plot(x_pos, val_r, 's-', label='Validation (bins)',
             linewidth=2, markersize=8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(folds)
    ax1.set_xlabel('Fold', fontsize=11)
    ax1.set_ylabel('Correlation r', fontsize=11)
    ax1.set_title('CV Performance by Fold', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    perm_r = [r['perm_r'] for r in cv_results]
    ax2.bar(x_pos, perm_r, color='steelblue')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(folds)
    ax2.set_xlabel('Fold', fontsize=11)
    ax2.set_ylabel('Permutation Validation r', fontsize=11)
    ax2.set_title('Permutation Validation by Fold', fontsize=12,
                  fontweight='bold')
    ax2.axhline(np.mean(perm_r), color='red', linestyle='--',
                label=f'Mean: {np.mean(perm_r):.4f}')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    ax3 = axes[1, 0]
    train_bias = [r['train_bias'] for r in cv_results]
    val_bias = [r['val_bias'] for r in cv_results]
    ax3.plot(x_pos, train_bias, 'o-', label='Training',
             linewidth=2, markersize=8, color='orange')
    ax3.plot(x_pos, val_bias, 's-', label='Validation (bins)',
             linewidth=2, markersize=8, color='coral')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(folds)
    ax3.set_xlabel('Fold', fontsize=11)
    ax3.set_ylabel('Mean Error (Bias)', fontsize=11)
    ax3.set_title('Bias by Fold', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    perm_bias = [r['perm_bias'] for r in cv_results]
    ax4.bar(x_pos, perm_bias, color='crimson')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(folds)
    ax4.set_xlabel('Fold', fontsize=11)
    ax4.set_ylabel('Permutation Mean Error', fontsize=11)
    ax4.set_title('Permutation Bias by Fold', fontsize=12,
                  fontweight='bold')
    ax4.axhline(np.mean(perm_bias), color='blue', linestyle='--',
                label=f'Mean: {np.mean(perm_bias):.4f}')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_bias_comparison(bias_results, output_file):
    """
    Plot comparison of bias correction methods.

    Parameters
    ----------
    bias_results : pd.DataFrame
        Results for each bias correction method
    output_file : Path
        Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = bias_results['method'].values
    x_pos = np.arange(len(methods))

    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, bias_results['validation_r'], color='steelblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Validation r', fontsize=11)
    ax1.set_title('Correlation by Bias Correction Method', fontsize=12,
                  fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, bias_results['mean_error'], color='crimson')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(-0.01, color='green', linestyle='--',
                label='Target: |bias| < 0.01')
    ax2.axhline(0.01, color='green', linestyle='--')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Mean Error (Bias)', fontsize=11)
    ax2.set_title('Bias by Method', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    ax3 = axes[1, 0]
    bars = ax3.bar(x_pos, bias_results['rmse'], color='coral')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('Prediction Error by Method', fontsize=12,
                  fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    abs_bias = np.abs(bias_results['mean_error'])
    combined_score = bias_results['validation_r'] * (1 - abs_bias / 0.1)
    bars = ax4.bar(x_pos, combined_score, color='teal')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel('Combined Score', fontsize=11)
    ax4.set_title('Accuracy-Bias Trade-off', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main Phase 5 function: CV and bias correction.
    """
    parser = argparse.ArgumentParser(
        description='Phase 5: Bias Correction and Cross-Validation'
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
        '--n-folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
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
    results_dir = repo_dir / 'results' / 'phase5_bias_correction'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 5: Bias Correction and Cross-Validation")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Bins: {args.n_bins} x {args.n_bins}")
    print(f"CV Folds: {args.n_folds}")
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

    X_all, y_all, metadata = extract_features_from_original(
        edge1_type, edge2_type, data_dir, args.n_bins,
        feature_set='E'
    )

    print(f"Total bins: {metadata['n_samples']}")
    print(f"Features: {metadata['n_features']}")
    print()

    print("=" * 80)
    print("Step 2: K-Fold Cross-Validation")
    print("=" * 80)
    print()

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    cv_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_all)):
        print(f"Fold {fold_idx + 1}/{args.n_folds}")
        print("-" * 80)

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        print(f"  Train bins: {len(train_idx)}")
        print(f"  Val bins: {len(val_idx)}")

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_train_pred = lr.predict(X_train)
        y_val_pred = lr.predict(X_val)

        train_r, _ = pearsonr(y_train, y_train_pred)
        val_r, _ = pearsonr(y_val, y_val_pred)

        train_bias = np.mean(y_train_pred - y_train)
        val_bias = np.mean(y_val_pred - y_val)

        print(f"  Train r: {train_r:.4f}, bias: {train_bias:.4f}")
        print(f"  Val r (bins): {val_r:.4f}, bias: {val_bias:.4f}")

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
        perm_results = validate_on_permutations(
            LRWrapper(lr), edge1_type, edge2_type, perm_ids,
            data_dir, args.n_bins, 'cpu',
            feature_set='E'
        )

        print(f"  Val r (perms): {perm_results['validation_r']:.4f}, "
              f"bias: {perm_results['mean_error']:.4f}")
        print()

        cv_results.append({
            'fold': fold_idx + 1,
            'train_r': float(train_r),
            'val_r': float(val_r),
            'perm_r': float(perm_results['validation_r']),
            'train_bias': float(train_bias),
            'val_bias': float(val_bias),
            'perm_bias': float(perm_results['mean_error']),
            'perm_rmse': float(perm_results['rmse'])
        })

    cv_df = pd.DataFrame(cv_results)
    print("=" * 80)
    print("CV SUMMARY")
    print("=" * 80)
    print(cv_df.to_string(index=False))
    print()
    print(f"Mean permutation r: {cv_df['perm_r'].mean():.4f} "
          f"+/- {cv_df['perm_r'].std():.4f}")
    print(f"Mean permutation bias: {cv_df['perm_bias'].mean():.4f} "
          f"+/- {cv_df['perm_bias'].std():.4f}")
    print()

    cv_file = results_dir / f'{args.metapath}_cv_results.csv'
    cv_df.to_csv(cv_file, index=False)
    print(f"CV results saved: {cv_file}")

    plot_file = results_dir / f'{args.metapath}_cv_results.png'
    plot_cv_results(cv_results, plot_file)
    print(f"CV plot saved: {plot_file}")
    print()

    print("=" * 80)
    print("Step 3: Bias Correction Methods")
    print("=" * 80)
    print()

    bias_results = []

    print("Method 1: Baseline Linear Regression (no correction)")
    print("-" * 80)
    lr_baseline = LinearRegression()
    lr_baseline.fit(X_all, y_all)

    baseline_results = validate_on_permutations(
        LRWrapper(lr_baseline), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {baseline_results['validation_r']:.4f}")
    print(f"  Bias: {baseline_results['mean_error']:.4f}")
    print(f"  RMSE: {baseline_results['rmse']:.4f}")
    print()

    bias_results.append({
        'method': 'Baseline',
        'validation_r': float(baseline_results['validation_r']),
        'mean_error': float(baseline_results['mean_error']),
        'rmse': float(baseline_results['rmse'])
    })

    print("Method 2: Simple Offset Correction")
    print("-" * 80)
    offset = baseline_results['mean_error']
    print(f"  Offset: {offset:.4f}")

    class OffsetLR:
        """Linear Regression with additive offset."""
        def __init__(self, model, offset):
            self.model = model
            self.offset = offset
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
            predictions = self.model.predict(X_np) - self.offset
            return torch.FloatTensor(predictions.reshape(-1, 1))

    offset_results = validate_on_permutations(
        OffsetLR(lr_baseline, offset), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {offset_results['validation_r']:.4f}")
    print(f"  Bias: {offset_results['mean_error']:.4f}")
    print(f"  RMSE: {offset_results['rmse']:.4f}")
    print()

    bias_results.append({
        'method': 'Offset',
        'validation_r': float(offset_results['validation_r']),
        'mean_error': float(offset_results['mean_error']),
        'rmse': float(offset_results['rmse'])
    })

    print("Method 3: Bias-Aware Loss (MSE + bias penalty)")
    print("-" * 80)
    for penalty in [0.1, 1.0, 10.0]:
        print(f"  Testing penalty lambda = {penalty}")

        lr_bias_aware = BiasAwareLinearRegression(bias_penalty=penalty)
        lr_bias_aware.fit(X_all, y_all)

        class BiasAwareWrapper:
            """Wrapper for BiasAwareLinearRegression."""
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

        bias_aware_results = validate_on_permutations(
            BiasAwareWrapper(lr_bias_aware), edge1_type, edge2_type, perm_ids,
            data_dir, args.n_bins, 'cpu',
            feature_set='E'
        )

        print(f"    Validation r: {bias_aware_results['validation_r']:.4f}")
        print(f"    Bias: {bias_aware_results['mean_error']:.4f}")
        print(f"    RMSE: {bias_aware_results['rmse']:.4f}")

        bias_results.append({
            'method': f'BiasAware(λ={penalty})',
            'validation_r': float(bias_aware_results['validation_r']),
            'mean_error': float(bias_aware_results['mean_error']),
            'rmse': float(bias_aware_results['rmse'])
        })

    print()

    print("Method 4: Asymmetric Loss")
    print("-" * 80)
    for asymmetry in [1.0, 2.0, 5.0]:
        print(f"  Testing asymmetry alpha = {asymmetry}")

        lr_asymmetric = AsymmetricLinearRegression(asymmetry=asymmetry)
        lr_asymmetric.fit(X_all, y_all)

        asymmetric_results = validate_on_permutations(
            BiasAwareWrapper(lr_asymmetric), edge1_type, edge2_type, perm_ids,
            data_dir, args.n_bins, 'cpu',
            feature_set='E'
        )

        print(f"    Validation r: {asymmetric_results['validation_r']:.4f}")
        print(f"    Bias: {asymmetric_results['mean_error']:.4f}")
        print(f"    RMSE: {asymmetric_results['rmse']:.4f}")

        bias_results.append({
            'method': f'Asymmetric(α={asymmetry})',
            'validation_r': float(asymmetric_results['validation_r']),
            'mean_error': float(asymmetric_results['mean_error']),
            'rmse': float(asymmetric_results['rmse'])
        })

    print()

    print("=" * 80)
    print("BIAS CORRECTION SUMMARY")
    print("=" * 80)

    bias_df = pd.DataFrame(bias_results)
    print(bias_df.to_string(index=False))
    print()

    best_idx = bias_df['mean_error'].abs().idxmin()
    print(f"Best method (lowest |bias|): {bias_df.iloc[best_idx]['method']}")
    print(f"  Validation r: {bias_df.iloc[best_idx]['validation_r']:.4f}")
    print(f"  Bias: {bias_df.iloc[best_idx]['mean_error']:.4f}")
    print()

    bias_file = results_dir / f'{args.metapath}_bias_comparison.csv'
    bias_df.to_csv(bias_file, index=False)
    print(f"Bias comparison saved: {bias_file}")

    bias_plot = results_dir / f'{args.metapath}_bias_comparison.png'
    plot_bias_comparison(bias_df, bias_plot)
    print(f"Bias plot saved: {bias_plot}")
    print()

    print("=" * 80)
    print("Phase 5 complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
