#!/usr/bin/env python3
"""
Degree-aware correction evaluation.

This script implements correction models that learn from the difference between
original graph and permutation 0 pathway counts, accounting for both degree
features and pathway count magnitude.

Usage:
    python test_src/degree_aware_correction_eval.py
    python test_src/degree_aware_correction_eval.py --metapath CtDaG
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
from sklearn.linear_model import LinearRegression, Ridge

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import (
    extract_features_from_original,
    extract_features_from_permutation
)
from pathway_evaluation_v2 import validate_on_permutations
from degree_aware_correction import (
    DegreeAwareCorrectionModel,
    AdaptiveDegreeAwareCorrectionModel,
    MultiplicativeDegreeAwareCorrectionModel
)


def parse_metapath(metapath_str):
    """Parse metapath string into edge types."""
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
    """Plot comparison of correction methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = results_df['method'].values
    x_pos = np.arange(len(methods))

    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, results_df['validation_r'], color='steelblue')
    best_idx = results_df['validation_r'].idxmax()
    bars[best_idx].set_color('darkgreen')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Validation r', fontsize=11)
    ax1.set_title('Correlation by Correction Method', fontsize=12,
                  fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, results_df['mean_error'], color='crimson')
    best_idx = results_df['mean_error'].abs().idxmin()
    bars[best_idx].set_color('darkgreen')
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
    best_idx = results_df['rmse'].idxmin()
    bars[best_idx].set_color('darkgreen')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('Prediction Error by Method', fontsize=12,
                  fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    abs_bias = np.abs(results_df['mean_error'])
    combined_score = results_df['validation_r'] - abs_bias * 2
    bars = ax4.bar(x_pos, combined_score, color='teal')
    best_idx = combined_score.idxmax()
    bars[best_idx].set_color('darkgreen')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel('Combined Score', fontsize=11)
    ax4.set_title('Overall Performance (r - 2×|bias|)', fontsize=12,
                  fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main degree-aware-correction-eval function."""
    parser = argparse.ArgumentParser(
        description='Degree-aware correction evaluation'
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
        default=1,
        help='First permutation ID for validation (default: 1)'
    )
    parser.add_argument(
        '--perm-end',
        type=int,
        default=19,
        help='Last permutation ID for validation (default: 19)'
    )

    args = parser.parse_args()

    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results' / 'phase5b_degree_aware_correction'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Degree-Aware Correction Evaluation")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Bins: {args.n_bins} x {args.n_bins}")
    print(f"Feature Set: E (polynomial terms)")
    print(f"Training: Original graph + Permutation 0")
    print(f"Validation: Permutations {args.perm_start:03d}-{args.perm_end:03d}")
    print(f"Results directory: {results_dir}")
    print()

    edge1_type, edge2_type = parse_metapath(args.metapath)
    print(f"Edge types: {edge1_type} -> {edge2_type}")
    print()

    print("=" * 80)
    print("Step 1: Extracting pathway counts from original graph and perm 0")
    print("=" * 80)

    X_train, y_original, metadata = extract_features_from_original(
        edge1_type, edge2_type, data_dir, args.n_bins,
        feature_set='E'
    )

    print(f"Features: {metadata['n_features']}")
    print(f"Training samples: {metadata['n_samples']}")
    print(f"Original graph mean pathway count: {y_original.mean():.4f}")
    print()

    _, y_perm0, _ = extract_features_from_permutation(
        edge1_type, edge2_type, 0, data_dir, args.n_bins,
        feature_set='E'
    )

    print(f"Permutation 0 mean pathway count: {y_perm0.mean():.4f}")
    print(f"Mean difference (perm0 - original): {(y_perm0 - y_original).mean():.4f}")
    print(f"Difference std: {(y_perm0 - y_original).std():.4f}")
    print()

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

    perm_ids = list(range(args.perm_start, args.perm_end + 1))
    results = []

    print("=" * 80)
    print("Step 2: Testing Correction Methods")
    print("=" * 80)
    print()

    print("Method 1: Baseline (Train on original, no correction)")
    print("-" * 80)
    baseline = LinearRegression()
    baseline.fit(X_train, y_original)

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

    print("Method 3: Degree-Aware Correction (Ridge α=0.1)")
    print("-" * 80)
    deg_aware_01 = DegreeAwareCorrectionModel(
        base_model=LinearRegression(),
        alpha=0.1,
        use_interaction=True
    )
    deg_aware_01.fit(X_train, y_original, y_perm0)

    deg_aware_01_results = validate_on_permutations(
        ModelWrapper(deg_aware_01), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {deg_aware_01_results['validation_r']:.4f}")
    print(f"  Bias: {deg_aware_01_results['mean_error']:.4f}")
    print(f"  RMSE: {deg_aware_01_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'DegAware(α=0.1)',
        'validation_r': float(deg_aware_01_results['validation_r']),
        'mean_error': float(deg_aware_01_results['mean_error']),
        'rmse': float(deg_aware_01_results['rmse'])
    })

    print("Method 4: Degree-Aware Correction (Ridge α=1.0)")
    print("-" * 80)
    deg_aware_1 = DegreeAwareCorrectionModel(
        base_model=LinearRegression(),
        alpha=1.0,
        use_interaction=True
    )
    deg_aware_1.fit(X_train, y_original, y_perm0)

    deg_aware_1_results = validate_on_permutations(
        ModelWrapper(deg_aware_1), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {deg_aware_1_results['validation_r']:.4f}")
    print(f"  Bias: {deg_aware_1_results['mean_error']:.4f}")
    print(f"  RMSE: {deg_aware_1_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'DegAware(α=1.0)',
        'validation_r': float(deg_aware_1_results['validation_r']),
        'mean_error': float(deg_aware_1_results['mean_error']),
        'rmse': float(deg_aware_1_results['rmse'])
    })

    print("Method 5: Degree-Aware Correction (Ridge α=10.0)")
    print("-" * 80)
    deg_aware_10 = DegreeAwareCorrectionModel(
        base_model=LinearRegression(),
        alpha=10.0,
        use_interaction=True
    )
    deg_aware_10.fit(X_train, y_original, y_perm0)

    deg_aware_10_results = validate_on_permutations(
        ModelWrapper(deg_aware_10), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {deg_aware_10_results['validation_r']:.4f}")
    print(f"  Bias: {deg_aware_10_results['mean_error']:.4f}")
    print(f"  RMSE: {deg_aware_10_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'DegAware(α=10.0)',
        'validation_r': float(deg_aware_10_results['validation_r']),
        'mean_error': float(deg_aware_10_results['mean_error']),
        'rmse': float(deg_aware_10_results['rmse'])
    })

    print("Method 6: Adaptive Degree-Aware (4 regions)")
    print("-" * 80)
    adaptive_4 = AdaptiveDegreeAwareCorrectionModel(
        base_model=LinearRegression(),
        n_regions=4,
        alpha=1.0
    )
    adaptive_4.fit(X_train, y_original, y_perm0)

    adaptive_4_results = validate_on_permutations(
        ModelWrapper(adaptive_4), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {adaptive_4_results['validation_r']:.4f}")
    print(f"  Bias: {adaptive_4_results['mean_error']:.4f}")
    print(f"  RMSE: {adaptive_4_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'Adaptive-4',
        'validation_r': float(adaptive_4_results['validation_r']),
        'mean_error': float(adaptive_4_results['mean_error']),
        'rmse': float(adaptive_4_results['rmse'])
    })

    print("Method 7: Multiplicative Degree-Aware")
    print("-" * 80)
    mult_deg_aware = MultiplicativeDegreeAwareCorrectionModel(
        base_model=LinearRegression(),
        alpha=1.0
    )
    mult_deg_aware.fit(X_train, y_original, y_perm0)

    mult_deg_aware_results = validate_on_permutations(
        ModelWrapper(mult_deg_aware), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu',
        feature_set='E'
    )

    print(f"  Validation r: {mult_deg_aware_results['validation_r']:.4f}")
    print(f"  Bias: {mult_deg_aware_results['mean_error']:.4f}")
    print(f"  RMSE: {mult_deg_aware_results['rmse']:.4f}")
    print()

    results.append({
        'method': 'Multiplicative',
        'validation_r': float(mult_deg_aware_results['validation_r']),
        'mean_error': float(mult_deg_aware_results['mean_error']),
        'rmse': float(mult_deg_aware_results['rmse'])
    })

    print("=" * 80)
    print("CORRECTION METHOD COMPARISON")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df['abs_bias'] = results_df['mean_error'].abs()
    results_df = results_df.sort_values('abs_bias')

    print(results_df[['method', 'validation_r', 'mean_error', 'rmse']].to_string(index=False))
    print()

    best_idx = results_df['abs_bias'].idxmin()
    print(f"Best method (lowest |bias|): {results_df.iloc[best_idx]['method']}")
    print(f"  Validation r: {results_df.iloc[best_idx]['validation_r']:.4f}")
    print(f"  Bias: {results_df.iloc[best_idx]['mean_error']:.4f}")
    print(f"  RMSE: {results_df.iloc[best_idx]['rmse']:.4f}")
    print()

    csv_file = results_dir / f'{args.metapath}_degree_aware_comparison.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"Results saved: {csv_file}")

    plot_file = results_dir / f'{args.metapath}_degree_aware_comparison.png'
    plot_correction_comparison(results_df, plot_file)
    print(f"Plot saved: {plot_file}")
    print()

    print("=" * 80)
    print("Degree-aware correction evaluation complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
