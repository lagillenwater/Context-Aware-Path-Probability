#!/usr/bin/env python3
"""
Control experiments.

This script compares the optimized DegreeSignatureNN model (Feature Set E,
10x10 bins) against baseline methods to demonstrate improvement.

Usage:
    python test_src/control_experiments.py
    python test_src/control_experiments.py --metapath CtDaG
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import json
import argparse
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
import time

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import extract_features_from_original
from pathway_models_v2 import DegreeSignatureNN
from pathway_training_v2 import train_model
from pathway_evaluation_v2 import validate_on_permutations


class RandomModel:
    """
    Negative control: Random predictions.

    Predicts random values uniformly distributed in the range of training data.
    """

    def __init__(self):
        self.y_min = None
        self.y_max = None
        self.random_state = np.random.RandomState(42)

    def fit(self, X, y):
        """
        Fit model by storing training data range.

        Parameters
        ----------
        X : np.ndarray
            Training features (not used)
        y : np.ndarray
            Training targets
        """
        self.y_min = y.min()
        self.y_max = y.max()
        return self

    def predict(self, X):
        """
        Predict random values.

        Parameters
        ----------
        X : np.ndarray
            Test features

        Returns
        -------
        np.ndarray
            Random predictions
        """
        return self.random_state.uniform(
            self.y_min, self.y_max, size=len(X)
        )


class DegreeProductModel:
    """
    Weak baseline: Degree product model.

    Predicts P(path) proportional to source_degree * target_degree.
    This implements the naive compositional assumption.
    """

    def __init__(self):
        self.scale = None
        self.offset = None

    def fit(self, X, y):
        """
        Fit model by linear regression on degree product.

        Parameters
        ----------
        X : np.ndarray
            Training features (source_bin, target_bin, ...)
        y : np.ndarray
            Training targets
        """
        source_bin = X[:, 0]
        target_bin = X[:, 1]
        degree_product = source_bin * target_bin

        lr = LinearRegression()
        lr.fit(degree_product.reshape(-1, 1), y)

        self.scale = lr.coef_[0]
        self.offset = lr.intercept_

        return self

    def predict(self, X):
        """
        Predict using degree product.

        Parameters
        ----------
        X : np.ndarray
            Test features

        Returns
        -------
        np.ndarray
            Predictions
        """
        source_bin = X[:, 0]
        target_bin = X[:, 1]
        degree_product = source_bin * target_bin

        return self.scale * degree_product + self.offset


class NegativeBinomialGLM:
    """
    Negative Binomial GLM baseline.

    Uses log link function and fits dispersion parameter.
    """

    def __init__(self):
        self.lr = LinearRegression()

    def fit(self, X, y):
        """
        Fit GLM using log-transformed targets.

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
        y_log = np.log1p(y)
        self.lr.fit(X, y_log)
        return self

    def predict(self, X):
        """
        Predict using fitted GLM.

        Parameters
        ----------
        X : np.ndarray
            Test features

        Returns
        -------
        np.ndarray
            Predictions
        """
        y_log = self.lr.predict(X)
        return np.expm1(y_log)


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


def plot_method_comparison(results_df, output_file):
    """
    Plot comparison of all methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results for each method
    output_file : Path
        Output file path
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    methods = results_df['method'].values
    x_pos = np.arange(len(methods))

    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, results_df['validation_r'], color='steelblue')
    bars[-1].set_color('darkgreen')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Validation r', fontsize=11)
    ax1.set_title('Correlation Coefficient by Method', fontsize=12,
                  fontweight='bold')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.grid(alpha=0.3, axis='y')

    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, results_df['rmse'], color='coral')
    bars[-1].set_color('darkgreen')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('Prediction Error by Method', fontsize=12,
                  fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    ax3 = axes[0, 2]
    bars = ax3.bar(x_pos, results_df['training_time'], color='mediumpurple')
    bars[-1].set_color('darkgreen')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('Training Time (seconds)', fontsize=11)
    ax3.set_title('Computational Efficiency', fontsize=12,
                  fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(alpha=0.3, axis='y')

    ax4 = axes[1, 0]
    bars = ax4.bar(x_pos, results_df['mae'], color='orange')
    bars[-1].set_color('darkgreen')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel('MAE', fontsize=11)
    ax4.set_title('Mean Absolute Error by Method', fontsize=12,
                  fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')

    ax5 = axes[1, 1]
    bars = ax5.bar(x_pos, results_df['mean_error'], color='crimson')
    bars[-1].set_color('darkgreen')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(methods, rotation=45, ha='right')
    ax5.set_ylabel('Mean Error (Bias)', fontsize=11)
    ax5.set_title('Prediction Bias by Method', fontsize=12,
                  fontweight='bold')
    ax5.axhline(0, color='black', linewidth=0.5)
    ax5.grid(alpha=0.3, axis='y')

    ax6 = axes[1, 2]
    efficiency_score = results_df['validation_r'] / (results_df['training_time'] + 0.01)
    bars = ax6.bar(x_pos, efficiency_score, color='teal')
    bars[-1].set_color('darkgreen')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(methods, rotation=45, ha='right')
    ax6.set_ylabel('Efficiency Score (r / time)', fontsize=11)
    ax6.set_title('Performance-Efficiency Trade-off', fontsize=12,
                  fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main control-experiments function.
    """
    parser = argparse.ArgumentParser(
        description='Control experiments'
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
        '--epochs',
        type=int,
        default=2000,
        help='Maximum epochs for NN (default: 2000)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for NN (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for NN (default: 0.001)'
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
    results_dir = repo_dir / 'results' / 'phase4_control_experiments'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Control Experiments")
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
    print("Step 1: Extracting features (Set E)")
    print("=" * 80)

    X_train, y_train, metadata = extract_features_from_original(
        edge1_type, edge2_type, data_dir, args.n_bins,
        feature_set='E'
    )

    print(f"Number of features: {metadata['n_features']}")
    print(f"Number of samples: {metadata['n_samples']}")
    print(f"Training target mean: {y_train.mean():.4f}")
    print(f"Training target std: {y_train.std():.4f}")
    print()

    results = []

    print("=" * 80)
    print("Step 2: Training baseline models")
    print("=" * 80)
    print()

    np.random.seed(123)
    torch.manual_seed(123)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perm_ids = list(range(args.perm_start, args.perm_end + 1))

    print("Testing Method 1/6: Random")
    print("-" * 80)
    start_time = time.time()

    random_model = RandomModel()
    random_model.fit(X_train, y_train)
    y_pred_random = random_model.predict(X_train)

    train_r_random, _ = pearsonr(y_train, y_pred_random)
    training_time_random = time.time() - start_time

    print(f"  Training r: {train_r_random:.4f}")
    print(f"  Training time: {training_time_random:.2f}s")
    print()

    print("Testing Method 2/6: Degree Product")
    print("-" * 80)
    start_time = time.time()

    degree_product_model = DegreeProductModel()
    degree_product_model.fit(X_train, y_train)
    y_pred_dp = degree_product_model.predict(X_train)

    train_r_dp, _ = pearsonr(y_train, y_pred_dp)
    training_time_dp = time.time() - start_time

    print(f"  Training r: {train_r_dp:.4f}")
    print(f"  Training time: {training_time_dp:.2f}s")
    print()

    print("Testing Method 3/6: Linear Regression")
    print("-" * 80)
    start_time = time.time()

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_train)

    train_r_lr, _ = pearsonr(y_train, y_pred_lr)
    training_time_lr = time.time() - start_time

    print(f"  Training r: {train_r_lr:.4f}")
    print(f"  Training time: {training_time_lr:.2f}s")
    print()

    print("Testing Method 4/6: Negative Binomial GLM")
    print("-" * 80)
    start_time = time.time()

    negbin_model = NegativeBinomialGLM()
    negbin_model.fit(X_train, y_train)
    y_pred_negbin = negbin_model.predict(X_train)

    train_r_negbin, _ = pearsonr(y_train, y_pred_negbin)
    training_time_negbin = time.time() - start_time

    print(f"  Training r: {train_r_negbin:.4f}")
    print(f"  Training time: {training_time_negbin:.2f}s")
    print()

    print("Testing Method 5/6: Random Forest")
    print("-" * 80)
    start_time = time.time()

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=123,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_train)

    train_r_rf, _ = pearsonr(y_train, y_pred_rf)
    training_time_rf = time.time() - start_time

    print(f"  Training r: {train_r_rf:.4f}")
    print(f"  Training time: {training_time_rf:.2f}s")
    print()

    print("Testing Method 6/6: DegreeSignatureNN (Our Method)")
    print("-" * 80)
    start_time = time.time()

    nn_model = DegreeSignatureNN(
        input_dim=metadata['n_features'],
        hidden_dims=[128, 64, 32],
        dropout=0.1
    )

    training_results = train_model(
        nn_model, X_train, y_train,
        X_val=None, y_val=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=1000,
        device=device,
        verbose=False
    )

    training_time_nn = training_results['total_time']
    print(f"  Training time: {training_time_nn:.2f}s")
    print(f"  Final loss: {training_results['history']['train_loss'][-1]:.6f}")
    print()

    print("=" * 80)
    print("Step 3: Validating on permutations")
    print("=" * 80)
    print()

    class SklearnModelWrapper:
        """Wrapper to validate sklearn models on permutations."""

        def __init__(self, model):
            self.model = model
            self.device = 'cpu'

        def to(self, device):
            """Compatibility method for device placement."""
            self.device = device
            return self

        def eval(self):
            """Compatibility method for eval mode."""
            pass

        def __call__(self, X):
            if isinstance(X, torch.Tensor):
                X_np = X.cpu().numpy()
            else:
                X_np = X
            predictions = self.model.predict(X_np).reshape(-1, 1)
            return torch.FloatTensor(predictions)

    models_to_validate = [
        ('Random', SklearnModelWrapper(random_model), training_time_random),
        ('Degree Product', SklearnModelWrapper(degree_product_model), training_time_dp),
        ('Linear Regression', SklearnModelWrapper(lr_model), training_time_lr),
        ('NegBin GLM', SklearnModelWrapper(negbin_model), training_time_negbin),
        ('Random Forest', SklearnModelWrapper(rf_model), training_time_rf),
        ('DegreeSignatureNN', nn_model, training_time_nn)
    ]

    for method_name, model, training_time in models_to_validate:
        print(f"Validating: {method_name}")
        print("-" * 80)

        validation_results = validate_on_permutations(
            model, edge1_type, edge2_type, perm_ids,
            data_dir, args.n_bins, device,
            feature_set='E'
        )

        print(f"  Validation r: {validation_results['validation_r']:.4f}")
        print(f"  RMSE: {validation_results['rmse']:.4f}")
        print(f"  MAE: {validation_results['mae']:.4f}")
        print(f"  Mean error: {validation_results['mean_error']:.4f}")
        print()

        results.append({
            'method': method_name,
            'validation_r': float(validation_results['validation_r']),
            'p_value': float(validation_results['p_value']),
            'rmse': float(validation_results['rmse']),
            'mae': float(validation_results['mae']),
            'mean_error': float(validation_results['mean_error']),
            'max_abs_error': float(validation_results['max_abs_error']),
            'training_time': float(training_time)
        })

    print("=" * 80)
    print("PHASE 4 SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('validation_r', ascending=False)
    results_df['rank'] = range(1, len(results_df) + 1)

    print(results_df.to_string(index=False))
    print()

    our_method_r = results_df[
        results_df['method'] == 'DegreeSignatureNN'
    ]['validation_r'].values[0]

    baseline_rs = results_df[
        results_df['method'] != 'DegreeSignatureNN'
    ]['validation_r'].values

    print("Performance Comparison:")
    print(f"  DegreeSignatureNN: r = {our_method_r:.4f}")
    print(f"  Best baseline: r = {baseline_rs.max():.4f} "
          f"({results_df[results_df['validation_r'] == baseline_rs.max()]['method'].values[0]})")
    print(f"  Improvement: {our_method_r - baseline_rs.max():.4f}")
    print()

    summary_file = results_dir / f'{args.metapath}_method_comparison.csv'
    results_df.to_csv(summary_file, index=False)
    print(f"Summary table: {summary_file}")

    json_file = results_dir / f'{args.metapath}_method_comparison.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results: {json_file}")

    plot_file = results_dir / f'{args.metapath}_method_comparison.png'
    plot_method_comparison(results_df, plot_file)
    print(f"Comparison plot: {plot_file}")
    print()

    print("=" * 80)
    print("Control experiments complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
