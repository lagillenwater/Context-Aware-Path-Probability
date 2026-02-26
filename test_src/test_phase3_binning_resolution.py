#!/usr/bin/env python3
"""
Test Phase 3: Binning Resolution Optimization.

This script tests different bin sizes with Feature Set E (best from Phase 2)
to find the optimal trade-off between resolution and sample size.

Usage:
    python test_src/test_phase3_binning_resolution.py
    python test_src/test_phase3_binning_resolution.py --metapath CtDaG
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


def plot_binning_comparison(results_df, output_file):
    """
    Plot comparison of binning resolutions.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results for each bin size
    output_file : Path
        Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Correlation coefficient
    ax1 = axes[0, 0]
    ax1.plot(results_df['n_bins'], results_df['validation_r'],
             marker='o', linewidth=2, markersize=8)
    ax1.axhline(0.88, color='r', linestyle='--', label='Target (0.88)')
    ax1.set_xlabel('Number of Bins (per dimension)', fontsize=11)
    ax1.set_ylabel('Validation r', fontsize=11)
    ax1.set_title('Correlation vs Bin Resolution', fontsize=12,
                  fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # RMSE
    ax2 = axes[0, 1]
    ax2.plot(results_df['n_bins'], results_df['rmse'],
             marker='o', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Bins (per dimension)', fontsize=11)
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('Prediction Error vs Bin Resolution', fontsize=12,
                  fontweight='bold')
    ax2.grid(alpha=0.3)

    # Mean Error (bias)
    ax3 = axes[1, 0]
    ax3.plot(results_df['n_bins'], results_df['mean_error'],
             marker='o', linewidth=2, markersize=8, color='red')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Number of Bins (per dimension)', fontsize=11)
    ax3.set_ylabel('Mean Error', fontsize=11)
    ax3.set_title('Prediction Bias vs Bin Resolution', fontsize=12,
                  fontweight='bold')
    ax3.grid(alpha=0.3)

    # Training samples
    ax4 = axes[1, 1]
    ax4.plot(results_df['n_bins'], results_df['n_samples'],
             marker='o', linewidth=2, markersize=8, color='green')
    ax4.set_xlabel('Number of Bins (per dimension)', fontsize=11)
    ax4.set_ylabel('Number of Training Samples', fontsize=11)
    ax4.set_title('Sample Size vs Bin Resolution', fontsize=12,
                  fontweight='bold')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main Phase 3 function: test binning resolutions.
    """
    parser = argparse.ArgumentParser(
        description='Phase 3: Binning Resolution Optimization'
    )
    parser.add_argument(
        '--metapath',
        default='CbGpPW',
        help='Metapath to test (default: CbGpPW)'
    )
    parser.add_argument(
        '--bin-sizes',
        nargs='+',
        type=int,
        default=[5, 8, 10, 12, 15, 20],
        help='Bin sizes to test (default: 5 8 10 12 15 20)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2000,
        help='Maximum epochs (default: 2000)'
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
    results_dir = repo_dir / 'results' / 'phase3_binning_resolution'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 3: Binning Resolution Optimization")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Bin sizes to test: {args.bin_sizes}")
    print(f"Feature Set: E (polynomial terms)")
    print(f"Validation permutations: {args.perm_start:03d}-{args.perm_end:03d}")
    print(f"Results directory: {results_dir}")
    print()

    # Parse metapath
    edge1_type, edge2_type = parse_metapath(args.metapath)
    print(f"Edge types: {edge1_type} -> {edge2_type}")
    print()

    # Test all bin sizes
    results = []

    for n_bins in args.bin_sizes:
        print("=" * 80)
        print(f"Testing Bin Size: {n_bins} x {n_bins} = {n_bins**2} bins")
        print("=" * 80)

        # Extract features with Feature Set E
        print(f"Step 1: Extracting features (Set E, n_bins={n_bins})...")
        X_train, y_train, metadata = extract_features_from_original(
            edge1_type, edge2_type, data_dir, n_bins,
            feature_set='E'
        )

        print(f"  Number of features: {metadata['n_features']}")
        print(f"  Number of samples: {metadata['n_samples']}")
        print(f"  Training target mean: {y_train.mean():.4f}")
        print(f"  Training target std: {y_train.std():.4f}")
        print()

        # Create model
        print("Step 2: Creating model...")
        np.random.seed(123)
        torch.manual_seed(123)

        model = DegreeSignatureNN(
            input_dim=metadata['n_features'],
            hidden_dims=[128, 64, 32],
            dropout=0.1
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params}")
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
            data_dir, n_bins, device,
            feature_set='E'
        )

        print(f"  Validation r: {validation_results['validation_r']:.4f}")
        print(f"  RMSE: {validation_results['rmse']:.4f}")
        print(f"  MAE: {validation_results['mae']:.4f}")
        print(f"  Mean error: {validation_results['mean_error']:.4f}")
        print()

        # Store results
        results.append({
            'n_bins': n_bins,
            'n_samples': metadata['n_samples'],
            'n_features': metadata['n_features'],
            'n_parameters': n_params,
            'validation_r': float(validation_results['validation_r']),
            'p_value': float(validation_results['p_value']),
            'rmse': float(validation_results['rmse']),
            'mae': float(validation_results['mae']),
            'mean_error': float(validation_results['mean_error']),
            'max_abs_error': float(validation_results['max_abs_error']),
            'training_time': training_results['total_time'],
            'final_loss': float(training_results['history']['train_loss'][-1])
        })

    # Create summary
    print("=" * 80)
    print("PHASE 3 SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()

    # Identify best bin size
    best_r_idx = results_df['validation_r'].idxmax()
    best_r_bins = results_df.iloc[best_r_idx]['n_bins']

    least_bias_idx = results_df['mean_error'].abs().idxmin()
    least_bias_bins = results_df.iloc[least_bias_idx]['n_bins']

    print(f"Best correlation: {int(best_r_bins)} bins "
          f"(r = {results_df.iloc[best_r_idx]['validation_r']:.4f})")
    print(f"Least bias: {int(least_bias_bins)} bins "
          f"(mean error = {results_df.iloc[least_bias_idx]['mean_error']:.4f})")
    print()

    # Check for trade-offs
    print("Resolution vs Performance Trade-offs:")
    for _, row in results_df.iterrows():
        print(f"  {int(row['n_bins'])} bins: "
              f"r={row['validation_r']:.4f}, "
              f"samples={int(row['n_samples'])}, "
              f"bias={row['mean_error']:.4f}")
    print()

    # Save results
    print("Step 5: Saving results...")

    # Save summary table
    summary_file = results_dir / f'{args.metapath}_binning_comparison.csv'
    results_df.to_csv(summary_file, index=False)
    print(f"  Summary table: {summary_file}")

    # Save detailed JSON
    json_file = results_dir / f'{args.metapath}_binning_comparison.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Detailed results: {json_file}")

    # Plot comparison
    plot_file = results_dir / f'{args.metapath}_binning_comparison.png'
    plot_binning_comparison(results_df, plot_file)
    print(f"  Comparison plot: {plot_file}")
    print()

    print("=" * 80)
    print("Phase 3 complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
