#!/usr/bin/env python3
"""
Test Phase 5b: Theoretical Correction Formulas.

This script tests various theoretically-motivated correction formulas
to address heteroscedastic bias in pathway count predictions.

Usage:
    python test_src/test_phase5b_theoretical_correction.py
    python test_src/test_phase5b_theoretical_correction.py --metapath CtDaG
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
from sklearn.linear_model import LinearRegression

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import extract_features_from_original
from pathway_evaluation_v2 import validate_on_permutations
from theoretical_correction import (
    MultiplicativeCorrectionModel,
    RatioCorrectionModel,
    QuantileCorrectionModel
)


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


def plot_correction_comparison(results_df, output_file):
    """
    Plot comparison of correction methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results for each method
    output_file : Path
        Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = results_df['method'].values
    x_pos = np.arange(len(methods))

    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, results_df['validation_r'], color='steelblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Validation r', fontsize=11)
    ax1.set_title('Correlation by Correction Method', fontsize=12,
                  fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, results_df['mean_error'], color='crimson')
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
    bars = ax3.bar(x_pos, results_df['rmse'], color='coral')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('Prediction Error by Method', fontsize=12,
                  fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    abs_bias = np.abs(results_df['mean_error'])
    combined_score = results_df['validation_r'] * (1 - abs_bias / 0.1)
    bars = ax4.bar(x_pos, combined_score, color='teal')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel('Combined Score (r × bias penalty)', fontsize=11)
    ax4.set_title('Accuracy-Bias Trade-off', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main Phase 5b theoretical correction function.
    """
    parser = argparse.ArgumentParser(
        description='Phase 5b: Theoretical Correction Formulas'
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
    results_dir = repo_dir / 'results' / 'phase5b_theoretical_correction'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 5b: Theoretical Correction Formulas")
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

        def __call__(self, X):
            if isinstance(X, torch.Tensor):
                X_np = X.cpu().numpy()
            else:
                X_np = X
            predictions = self.model.predict(X_np).reshape(-1, 1)
            return torch.FloatTensor(predictions)

    results = []

    print("=" * 80)
    print("Step 2: Testing Correction Methods")
    print("=" * 80)
    print()

    print("Method 1: Baseline (No Correction)")
    print("-" * 80)
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)

    baseline_results = validate_on_permutations(
        ModelWrapper(baseline), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {baseline_results['validation_r']:.4f}")
    print(f"  Bias: {baseline_results['mean_error']:.4f}")
    print(f"  RMSE: {baseline_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'Baseline',
        'validation_r': float(baseline_results['validation_r']),
        'mean_error': float(baseline_results['mean_error']),
        'rmse': float(baseline_results['rmse'])
    })

    print("Method 2: Simple Offset")
    print("-" * 80)
    offset = baseline_results['mean_error']
    print(f"  Offset: {offset:.4f}")

    class OffsetModel:
        """Baseline + offset."""
        def __init__(self, model, offset):
            self.model = model
            self.offset = offset

        def predict(self, X):
            return self.model.predict(X) - self.offset

    offset_model = OffsetModel(baseline, offset)

    offset_results = validate_on_permutations(
        ModelWrapper(offset_model), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {offset_results['validation_r']:.4f}")
    print(f"  Bias: {offset_results['mean_error']:.4f}")
    print(f"  RMSE: {offset_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'Offset',
        'validation_r': float(offset_results['validation_r']),
        'mean_error': float(offset_results['mean_error']),
        'rmse': float(offset_results['rmse'])
    })

    print("Method 3: Multiplicative Correction")
    print("-" * 80)
    mult_model = MultiplicativeCorrectionModel(base_model=LinearRegression())
    mult_model.fit(X_train, y_train)

    print(f"  Learned parameters: alpha={mult_model.alpha_:.4f}, "
          f"beta={mult_model.beta_:.4f}")

    mult_results = validate_on_permutations(
        ModelWrapper(mult_model), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {mult_results['validation_r']:.4f}")
    print(f"  Bias: {mult_results['mean_error']:.4f}")
    print(f"  RMSE: {mult_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'Multiplicative',
        'validation_r': float(mult_results['validation_r']),
        'mean_error': float(mult_results['mean_error']),
        'rmse': float(mult_results['rmse'])
    })

    print("Method 4: Ratio Correction (Polynomial deg=2)")
    print("-" * 80)
    ratio_poly2 = RatioCorrectionModel(
        base_model=LinearRegression(),
        method='polynomial',
        degree=2
    )
    ratio_poly2.fit(X_train, y_train)

    print(f"  Polynomial coefficients: {ratio_poly2.correction_params_}")

    ratio_poly2_results = validate_on_permutations(
        ModelWrapper(ratio_poly2), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {ratio_poly2_results['validation_r']:.4f}")
    print(f"  Bias: {ratio_poly2_results['mean_error']:.4f}")
    print(f"  RMSE: {ratio_poly2_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'Ratio-Poly2',
        'validation_r': float(ratio_poly2_results['validation_r']),
        'mean_error': float(ratio_poly2_results['mean_error']),
        'rmse': float(ratio_poly2_results['rmse'])
    })

    print("Method 5: Ratio Correction (Polynomial deg=3)")
    print("-" * 80)
    ratio_poly3 = RatioCorrectionModel(
        base_model=LinearRegression(),
        method='polynomial',
        degree=3
    )
    ratio_poly3.fit(X_train, y_train)

    ratio_poly3_results = validate_on_permutations(
        ModelWrapper(ratio_poly3), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {ratio_poly3_results['validation_r']:.4f}")
    print(f"  Bias: {ratio_poly3_results['mean_error']:.4f}")
    print(f"  RMSE: {ratio_poly3_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'Ratio-Poly3',
        'validation_r': float(ratio_poly3_results['validation_r']),
        'mean_error': float(ratio_poly3_results['mean_error']),
        'rmse': float(ratio_poly3_results['rmse'])
    })

    print("Method 6: Quantile Correction (4 quantiles)")
    print("-" * 80)
    quantile4 = QuantileCorrectionModel(
        base_model=LinearRegression(),
        n_quantiles=4
    )
    quantile4.fit(X_train, y_train)

    print(f"  Quantile corrections: {quantile4.quantile_corrections_}")

    quantile4_results = validate_on_permutations(
        ModelWrapper(quantile4), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {quantile4_results['validation_r']:.4f}")
    print(f"  Bias: {quantile4_results['mean_error']:.4f}")
    print(f"  RMSE: {quantile4_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'Quantile-4',
        'validation_r': float(quantile4_results['validation_r']),
        'mean_error': float(quantile4_results['mean_error']),
        'rmse': float(quantile4_results['rmse'])
    })

    print("Method 7: Quantile Correction (10 quantiles)")
    print("-" * 80)
    quantile10 = QuantileCorrectionModel(
        base_model=LinearRegression(),
        n_quantiles=10
    )
    quantile10.fit(X_train, y_train)

    quantile10_results = validate_on_permutations(
        ModelWrapper(quantile10), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {quantile10_results['validation_r']:.4f}")
    print(f"  Bias: {quantile10_results['mean_error']:.4f}")
    print(f"  RMSE: {quantile10_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'Quantile-10',
        'validation_r': float(quantile10_results['validation_r']),
        'mean_error': float(quantile10_results['mean_error']),
        'rmse': float(quantile10_results['rmse'])
    })

    print("=" * 80)
    print("CORRECTION METHOD COMPARISON")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_error', key=lambda x: np.abs(x))

    print(results_df.to_string(index=False))
    print()

    best_idx = results_df['mean_error'].abs().idxmin()
    print(f"Best method (lowest |bias|): {results_df.iloc[best_idx]['method']}")
    print(f"  Validation r: {results_df.iloc[best_idx]['validation_r']:.4f}")
    print(f"  Bias: {results_df.iloc[best_idx]['mean_error']:.4f}")
    print(f"  RMSE: {results_df.iloc[best_idx]['rmse']:.4f}")
    print()

    csv_file = results_dir / f'{args.metapath}_correction_comparison.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"Results saved: {csv_file}")

    plot_file = results_dir / f'{args.metapath}_correction_comparison.png'
    plot_correction_comparison(results_df, plot_file)
    print(f"Plot saved: {plot_file}")
    print()

    print("=" * 80)
    print("Phase 5b theoretical correction complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
