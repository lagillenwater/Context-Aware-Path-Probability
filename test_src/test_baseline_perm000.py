#!/usr/bin/env python3
"""
Test Phase 1b baseline: Train DegreeSignatureNN on permutation 000.

This script trains the baseline DegreeSignatureNN model on permutation 000
and validates on the average of permutations 001-020.

Usage:
    python test_src/test_baseline_perm000.py
    python test_src/test_baseline_perm000.py --metapath CtDaG
    python test_src/test_baseline_perm000.py --n-bins 15 --epochs 1000
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import argparse

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import extract_features_from_permutation
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


def plot_training_curves(history, output_file):
    """
    Plot training and test loss curves.

    Parameters
    ----------
    history : dict
        Training history with 'train_loss' and optionally 'val_loss'
    output_file : Path
        Output file path
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(history['train_loss'], linewidth=2, label='Training Loss')

    if 'val_loss' in history and history['val_loss']:
        ax.plot(history['val_loss'], linewidth=2, linestyle='--',
                label='Test Loss')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (MSE)', fontsize=11)
    ax.set_title('Training and Test Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(results, metapath, output_file):
    """
    Plot predicted vs actual pathway counts.

    Parameters
    ----------
    results : dict
        Validation results
    metapath : str
        Metapath code
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
        f'{metapath}: Model Predictions vs Permutation Average\\n'
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
        f'Residual Analysis\\n'
        f'MAE = {results["mae"]:.3f}, '
        f'Max Error = {results["max_abs_error"]:.3f}',
        fontsize=12, fontweight='bold'
    )
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main baseline training function.
    """
    parser = argparse.ArgumentParser(
        description='Train baseline DegreeSignatureNN on permutation 000'
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
        default=1,
        help='First permutation ID for validation (default: 1)'
    )
    parser.add_argument(
        '--perm-end',
        type=int,
        default=20,
        help='Last permutation ID for validation (default: 20)'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results' / 'baseline_perm000'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 1b: Baseline Establishment on Permutation 000")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Number of bins: {args.n_bins}")
    print(f"Training: permutation 000")
    print(f"Validation permutations: {args.perm_start:03d}-{args.perm_end:03d}")
    print(f"Results directory: {results_dir}")
    print()

    # Parse metapath
    edge1_type, edge2_type = parse_metapath(args.metapath)
    print(f"Edge types: {edge1_type} -> {edge2_type}")
    print()

    # Step 1: Extract features from permutation 000
    print("Step 1: Extracting features from permutation 000...")
    print("  Loading from: data/permutations/000.hetmat/edges/")
    X_train, y_train, metadata = extract_features_from_permutation(
        edge1_type, edge2_type, 0, data_dir, args.n_bins, feature_set='A'
    )

    print(f"  Edge1 shape: {metadata['edge1_shape']}")
    print(f"  Edge2 shape: {metadata['edge2_shape']}")
    print(f"  Number of features: {metadata['n_features']}")
    print(f"  Number of training samples (bins): {metadata['n_samples']}")
    print(f"  Training target mean: {y_train.mean():.4f}")
    print(f"  Training target std: {y_train.std():.4f}")
    print()

    # Split into train/test for validation during training
    from sklearn.model_selection import train_test_split
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=789
    )
    print(f"  Split into {len(X_train_split)} train, {len(X_test_split)} test")
    print()

    # Step 2: Create model
    print("Step 2: Creating DegreeSignatureNN model...")

    # Set random seed for reproducibility
    # Note: seed 789 gives good convergence for this problem
    np.random.seed(789)
    torch.manual_seed(789)

    model = DegreeSignatureNN(
        input_dim=metadata['n_features'],
        hidden_dims=[128, 64, 32],
        dropout=0.1
    )
    print(model)
    print()

    # Step 3: Train model
    print("Step 3: Training model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print()

    training_results = train_model(
        model, X_train_split, y_train_split,
        X_val=X_test_split, y_val=y_test_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=50,
        device=device,
        verbose=True
    )

    print(f"  Training time: {training_results['total_time']:.2f}s")
    print(f"  Final epoch: {training_results['final_epoch']}")
    print(f"  Final training loss: "
          f"{training_results['history']['train_loss'][-1]:.6f}")
    print()

    # Step 4: Validate on permutations
    print(f"Step 4: Validating on permutations "
          f"{args.perm_start:03d}-{args.perm_end:03d}...")
    print("  Using permutation 000 features for validation")
    perm_ids = list(range(args.perm_start, args.perm_end + 1))

    validation_results = validate_on_permutations(
        model, edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, device,
        feature_source='permutation',
        feature_perm_id=0
    )

    print(f"  Validation r: {validation_results['validation_r']:.4f}")
    print(f"  P-value: {validation_results['p_value']:.4e}")
    print(f"  RMSE: {validation_results['rmse']:.4f}")
    print(f"  MAE: {validation_results['mae']:.4f}")
    print(f"  Mean error: {validation_results['mean_error']:.4f}")
    print(f"  Max absolute error: {validation_results['max_abs_error']:.4f}")
    print()

    # Step 5: Save results
    print("Step 5: Saving results...")

    # Save model
    model_file = results_dir / f'{args.metapath}_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': metadata['n_features'],
            'hidden_dims': [128, 64, 32],
            'dropout': 0.1
        },
        'metadata': metadata
    }, model_file)
    print(f"  Model saved: {model_file}")

    # Save metrics
    metrics = {
        'metapath': args.metapath,
        'edge1_type': edge1_type,
        'edge2_type': edge2_type,
        'training_source': 'permutation_000',
        'n_bins': args.n_bins,
        'n_features': metadata['n_features'],
        'n_training_samples': metadata['n_samples'],
        'training_time': training_results['total_time'],
        'final_epoch': training_results['final_epoch'],
        'validation_r': float(validation_results['validation_r']),
        'validation_p_value': float(validation_results['p_value']),
        'validation_rmse': float(validation_results['rmse']),
        'validation_mae': float(validation_results['mae']),
        'validation_mean_error': float(validation_results['mean_error']),
        'validation_max_abs_error': float(
            validation_results['max_abs_error']
        ),
        'n_validation_permutations': validation_results['n_permutations']
    }

    metrics_file = results_dir / f'{args.metapath}_baseline_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_file}")

    # Save training curves
    training_curve_file = results_dir / f'{args.metapath}_training_curve.png'
    plot_training_curves(training_results['history'], training_curve_file)
    print(f"  Training curves saved: {training_curve_file}")

    # Save predictions plot
    predictions_file = results_dir / f'{args.metapath}_predictions.png'
    plot_predictions(validation_results, args.metapath, predictions_file)
    print(f"  Predictions plot saved: {predictions_file}")
    print()

    # Final summary
    print("=" * 80)
    print("PHASE 1b BASELINE RESULTS")
    print("=" * 80)
    print(f"Metapath: {args.metapath}")
    print(f"Training: permutation 000")
    print(f"Validation r: {validation_results['validation_r']:.4f}")
    print(f"Target: r >= 0.85")
    print()

    if validation_results['validation_r'] >= 0.88:
        print("SUCCESS: Baseline performance REPRODUCED")
        print(f"  Achieved r = {validation_results['validation_r']:.4f} "
              f"(target: 0.88)")
    elif validation_results['validation_r'] >= 0.85:
        print("ACCEPTABLE: Baseline performance achieved")
        print(f"  Achieved r = {validation_results['validation_r']:.4f} "
              f"(threshold: 0.85)")
    else:
        print("BELOW TARGET: Further investigation needed")
        print(f"  Achieved r = {validation_results['validation_r']:.4f} "
              f"(threshold: 0.85)")

    print("=" * 80)
    print()
    print("Phase 1b complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
