#!/usr/bin/env python3
"""
Feature comparison: compare feature sets A-F on original graph.

This script trains DegreeSignatureNN models with different feature sets
to determine which features improve performance and reduce underprediction bias.

Usage:
    python test_src/feature_comparison.py
    python test_src/feature_comparison.py --metapath CtDaG
    python test_src/feature_comparison.py --epochs 1000
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import argparse
import pandas as pd

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import extract_features_from_original
from pathway_models_v2 import DegreeSignatureNN
from pathway_training_v2 import train_model
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


def plot_training_curves(history, feature_set, output_file):
    """
    Plot training loss curves.

    Parameters
    ----------
    history : dict
        Training history with 'train_loss'
    feature_set : str
        Feature set name
    output_file : Path
        Output file path
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(history['train_loss'], linewidth=2, label='Training Loss', color='blue')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (MSE)', fontsize=11)
    ax.set_title(f'Feature Set {feature_set}: Training Loss',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(results, feature_set, output_file):
    """
    Plot predicted vs actual pathway counts.

    Parameters
    ----------
    results : dict
        Validation results
    feature_set : str
        Feature set name
    output_file : Path
        Output file path
    """
    y_true = results['y_true']
    y_pred = results['y_pred']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--',
             linewidth=2, label='Perfect prediction')

    ax1.set_xlabel('Permutation Average Pathway Count', fontsize=11)
    ax1.set_ylabel('Model Prediction', fontsize=11)
    ax1.set_title(
        f'Feature Set {feature_set}: Predictions vs Permutation Average\n'
        f'r = {results["validation_r"]:.4f}, '
        f'RMSE = {results["rmse"]:.3f}',
        fontsize=12, fontweight='bold'
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Residual plot
    ax2 = axes[1]
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)
    ax2.axhline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Permutation Average Pathway Count', fontsize=11)
    ax2.set_ylabel('Residual (Predicted - Actual)', fontsize=11)
    ax2.set_title(
        f'Residual Analysis\n'
        f'MAE = {results["mae"]:.3f}, '
        f'Max Error = {results["max_abs_error"]:.3f}',
        fontsize=12, fontweight='bold'
    )
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_set_comparison(results_df, output_file):
    """
    Plot comparison of feature sets.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results for each feature set
    output_file : Path
        Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Correlation coefficient
    ax1 = axes[0, 0]
    ax1.bar(results_df['feature_set'], results_df['validation_r'])
    ax1.axhline(0.88, color='r', linestyle='--', label='Target (0.88)')
    ax1.set_xlabel('Feature Set', fontsize=11)
    ax1.set_ylabel('Validation r', fontsize=11)
    ax1.set_title('Correlation by Feature Set', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')

    # RMSE
    ax2 = axes[0, 1]
    ax2.bar(results_df['feature_set'], results_df['rmse'])
    ax2.set_xlabel('Feature Set', fontsize=11)
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    # Mean Error (bias)
    ax3 = axes[1, 0]
    colors = ['red' if x < 0 else 'green' for x in results_df['mean_error']]
    ax3.bar(results_df['feature_set'], results_df['mean_error'], color=colors)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Feature Set', fontsize=11)
    ax3.set_ylabel('Mean Error', fontsize=11)
    ax3.set_title('Prediction Bias (negative = underprediction)',
                  fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    # Number of features
    ax4 = axes[1, 1]
    ax4.bar(results_df['feature_set'], results_df['n_features'])
    ax4.set_xlabel('Feature Set', fontsize=11)
    ax4.set_ylabel('Number of Features', fontsize=11)
    ax4.set_title('Feature Dimensionality', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main feature-comparison function.
    """
    parser = argparse.ArgumentParser(
        description='Feature comparison: compare feature sets A-F'
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
        default=500,
        help='Maximum epochs (default: 500)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
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

    # Setup paths
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results' / 'phase3_feature_comparison'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Feature Set Comparison")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Number of bins: {args.n_bins}")
    print(f"Validation permutations: {args.perm_start:03d}-{args.perm_end:03d}")
    print(f"Results directory: {results_dir}")
    print()

    # Parse metapath
    edge1_type, edge2_type = parse_metapath(args.metapath)
    print(f"Edge types: {edge1_type} -> {edge2_type}")
    print()

    # Test all feature sets
    feature_sets = ['A', 'B', 'C', 'D', 'E', 'F']
    results = []

    for feature_set in feature_sets:
        print("=" * 80)
        print(f"Testing Feature Set {feature_set}")
        print("=" * 80)

        # Extract features
        print(f"Step 1: Extracting feature set {feature_set}...")
        X_train, y_train, metadata = extract_features_from_original(
            edge1_type, edge2_type, data_dir, args.n_bins,
            feature_set=feature_set
        )

        print(f"  Number of features: {metadata['n_features']}")
        print(f"  Number of samples: {metadata['n_samples']}")
        print()

        # Create model
        print("Step 2: Creating model...")
        # Use seed 123 for better convergence across all feature sets
        np.random.seed(123)
        torch.manual_seed(123)

        model = DegreeSignatureNN(
            input_dim=metadata['n_features'],
            hidden_dims=[128, 64, 32],
            dropout=0.1
        )
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
        print()

        # Train model
        print("Step 3: Training model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        training_results = train_model(
            model, X_train, y_train,
            X_val=None, y_val=None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=1000,
            device=device,
            verbose=False
        )

        print(f"  Training time: {training_results['total_time']:.2f}s")
        print(f"  Final loss: {training_results['history']['train_loss'][-1]:.6f}")
        print()

        # Validate on permutations
        print("Step 4: Validating on permutations...")
        perm_ids = list(range(args.perm_start, args.perm_end + 1))

        validation_results = validate_on_permutations(
            model, edge1_type, edge2_type, perm_ids,
            data_dir, args.n_bins, device,
            feature_set=feature_set
        )

        print(f"  Validation r: {validation_results['validation_r']:.4f}")
        print(f"  RMSE: {validation_results['rmse']:.4f}")
        print(f"  MAE: {validation_results['mae']:.4f}")
        print(f"  Mean error: {validation_results['mean_error']:.4f}")
        print()

        # Compute inference time
        print("Step 5: Computing inference time...")
        import time
        X_tensor = torch.FloatTensor(X_train).to(device)
        model.eval()
        start = time.time()
        with torch.no_grad():
            _ = model(X_tensor)
        inference_time = (time.time() - start) * 1000  # ms
        print(f"  Inference time: {inference_time:.2f} ms")
        print()

        # Save individual plots
        print("Step 6: Saving visualizations...")
        training_plot = results_dir / f'set_{feature_set}_training_curve.png'
        plot_training_curves(training_results['history'], feature_set, training_plot)
        print(f"  Training curve: {training_plot}")

        predictions_plot = results_dir / f'set_{feature_set}_predictions.png'
        plot_predictions(validation_results, feature_set, predictions_plot)
        print(f"  Predictions plot: {predictions_plot}")
        print()

        # Count parameters
        n_parameters = sum(p.numel() for p in model.parameters())

        # Store results
        results.append({
            'feature_set': feature_set,
            'n_features': metadata['n_features'],
            'n_parameters': n_parameters,
            'validation_r': float(validation_results['validation_r']),
            'p_value': float(validation_results['p_value']),
            'rmse': float(validation_results['rmse']),
            'mae': float(validation_results['mae']),
            'mean_error': float(validation_results['mean_error']),
            'max_abs_error': float(validation_results['max_abs_error']),
            'training_time': training_results['total_time'],
            'inference_time': inference_time,
            'final_loss': float(training_results['history']['train_loss'][-1])
        })

    # Create summary
    print("=" * 80)
    print("PHASE 3 SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()

    # Identify best feature set
    best_r_idx = results_df['validation_r'].idxmax()
    best_r_set = results_df.iloc[best_r_idx]['feature_set']

    least_bias_idx = results_df['mean_error'].abs().idxmin()
    least_bias_set = results_df.iloc[least_bias_idx]['feature_set']

    fastest_train_idx = results_df['training_time'].idxmin()
    fastest_train_set = results_df.iloc[fastest_train_idx]['feature_set']

    fastest_inference_idx = results_df['inference_time'].idxmin()
    fastest_inference_set = results_df.iloc[fastest_inference_idx]['feature_set']

    print(f"Best correlation: Feature Set {best_r_set} "
          f"(r = {results_df.iloc[best_r_idx]['validation_r']:.4f})")
    print(f"Least bias: Feature Set {least_bias_set} "
          f"(mean error = {results_df.iloc[least_bias_idx]['mean_error']:.4f})")
    print(f"Fastest training: Feature Set {fastest_train_set} "
          f"({results_df.iloc[fastest_train_idx]['training_time']:.2f}s)")
    print(f"Fastest inference: Feature Set {fastest_inference_set} "
          f"({results_df.iloc[fastest_inference_idx]['inference_time']:.2f} ms)")
    print()

    # Check if bias improved
    baseline_bias = results_df[results_df['feature_set'] == 'A']['mean_error'].values[0]
    print(f"Baseline (Set A) bias: {baseline_bias:.4f}")

    improvements = []
    for _, row in results_df.iterrows():
        if row['feature_set'] != 'A':
            bias_change = row['mean_error'] - baseline_bias
            if abs(row['mean_error']) < abs(baseline_bias):
                improvements.append(
                    f"  Set {row['feature_set']}: {row['mean_error']:.4f} "
                    f"(improved by {abs(bias_change):.4f})"
                )

    if improvements:
        print("Bias improvements:")
        for imp in improvements:
            print(imp)
    else:
        print("No feature set reduced bias below Set A")
    print()

    # Save results
    print("Step 5: Saving results...")

    # Save summary table
    summary_file = results_dir / f'{args.metapath}_feature_comparison.csv'
    results_df.to_csv(summary_file, index=False)
    print(f"  Summary table: {summary_file}")

    # Save detailed JSON
    json_file = results_dir / f'{args.metapath}_feature_comparison.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Detailed results: {json_file}")

    # Plot comparison
    plot_file = results_dir / f'{args.metapath}_feature_comparison.png'
    plot_feature_set_comparison(results_df, plot_file)
    print(f"  Comparison plot: {plot_file}")
    print()

    print("=" * 80)
    print("Feature comparison complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
