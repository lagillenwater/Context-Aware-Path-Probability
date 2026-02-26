"""
Prepare real pathway training data and test loss functions.

This script:
1. Generates degree-binned training data from Hetionet (notebook 18a logic)
2. Tests all loss functions on REAL pathway data
3. Analyzes residual patterns
4. Reports honest performance metrics

Usage:
    python prepare_and_test_pathway_data.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

from src.intermediate_signatures import (
    compute_intermediate_signature,
    create_degree_bins,
    extract_training_features,
    assign_to_bins
)
from src.pathway_losses import HuberLoss, LogScaleMSE, QuantileLoss
from src.models.degree_signature_nn import DegreeSignatureNN
from src.models.random_baseline import RandomBaseline
from src.models.degree_product_baseline import DegreeProductBaseline
from src.baseline_framework import BaselineFramework
import torch
import torch.nn as nn


def prepare_pathway_training_data(metapath='CbGpPW',
                                  edge1_type='CbG',
                                  edge2_type='GpPW',
                                  n_degree_bins=10,
                                  n_inter_bins=10):
    """
    Generate degree-binned training data from Hetionet.

    Implements logic from notebook 18a.

    Parameters
    ----------
    metapath : str
        Metapath name
    edge1_type : str
        First edge type
    edge2_type : str
        Second edge type
    n_degree_bins : int
        Number of degree bins
    n_inter_bins : int
        Number of bins for intermediate signature

    Returns
    -------
    df : pd.DataFrame
        Training data with features and targets
    """
    print("=" * 80)
    print(f"PREPARING PATHWAY TRAINING DATA: {metapath}")
    print("=" * 80)

    repo_dir = Path.cwd()
    data_dir = repo_dir / 'data'

    print(f"\nLoading edge matrices...")

    edge1_file = data_dir / 'edges' / f'{edge1_type}.sparse.npz'
    if not edge1_file.exists():
        edge1_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / \
                     f'{edge1_type}.sparse.npz'

    edge2_file = data_dir / 'edges' / f'{edge2_type}.sparse.npz'
    if not edge2_file.exists():
        edge2_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / \
                     f'{edge2_type}.sparse.npz'

    if not edge1_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge1_file}")
    if not edge2_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge2_file}")

    edge1_matrix = sp.load_npz(str(edge1_file))
    edge2_matrix = sp.load_npz(str(edge2_file))

    print(f"  Edge1 ({edge1_type}): {edge1_matrix.shape}, "
          f"{edge1_matrix.nnz:,} edges")
    print(f"  Edge2 ({edge2_type}): {edge2_matrix.shape}, "
          f"{edge2_matrix.nnz:,} edges")

    if edge1_matrix.dtype == bool or edge1_matrix.dtype == np.bool_:
        print(f"  Converting Edge1 from boolean to int32")
        edge1_matrix = edge1_matrix.astype(np.int32)
    if edge2_matrix.dtype == bool or edge2_matrix.dtype == np.bool_:
        print(f"  Converting Edge2 from boolean to int32")
        edge2_matrix = edge2_matrix.astype(np.int32)

    print(f"\nComputing node degrees...")
    source_degrees = np.asarray(edge1_matrix.sum(axis=1)).ravel()
    target_degrees = np.asarray(edge2_matrix.sum(axis=1)).ravel()

    print(f"  Source nodes: {len(source_degrees)}, "
          f"degree range: {source_degrees[source_degrees>0].min():.0f} - "
          f"{source_degrees.max():.0f}")
    print(f"  Target nodes: {len(target_degrees)}, "
          f"degree range: {target_degrees[target_degrees>0].min():.0f} - "
          f"{target_degrees.max():.0f}")

    print(f"\nCreating degree bins...")
    source_bins = create_degree_bins(source_degrees, n_degree_bins)
    target_bins = create_degree_bins(target_degrees, n_degree_bins)

    print(f"  Source bins: {source_bins}")
    print(f"  Target bins: {target_bins}")
    print(f"  Total bin combinations: "
          f"{(len(source_bins)-1) * (len(target_bins)-1)}")

    print(f"\nComputing intermediate signatures...")
    signatures = compute_intermediate_signature(
        edge1_matrix=edge1_matrix,
        edge2_matrix=edge2_matrix,
        source_degrees=source_degrees,
        target_degrees=target_degrees,
        source_bins=source_bins,
        target_bins=target_bins,
        n_intermediate_bins=n_inter_bins
    )

    print(f"  Computed signatures for {len(signatures)} degree bin pairs")

    print(f"\nComputing pathway counts...")
    pathway_matrix = edge1_matrix @ edge2_matrix

    print(f"  Pathway matrix: {pathway_matrix.shape}, "
          f"{pathway_matrix.nnz:,} non-zero pathways")
    print(f"  Value range: {pathway_matrix.data.min()} - "
          f"{pathway_matrix.data.max()}")

    pathway_coo = pathway_matrix.tocoo()
    pathway_dict = {(i, j): v for i, j, v in
                    zip(pathway_coo.row, pathway_coo.col, pathway_coo.data)}

    source_bin_assignments = assign_to_bins(source_degrees, source_bins)
    target_bin_assignments = assign_to_bins(target_degrees, target_bins)

    bin_pathway_counts = {}
    for (i, j), count in pathway_dict.items():
        src_bin = source_bin_assignments[i]
        tgt_bin = target_bin_assignments[j]
        key = (src_bin, tgt_bin)
        if key not in bin_pathway_counts:
            bin_pathway_counts[key] = []
        bin_pathway_counts[key].append(count)

    print(f"  Aggregated for {len(bin_pathway_counts)} bins")

    print(f"\nCreating training dataset...")
    X_signatures, bin_pairs = extract_training_features(
        signatures, normalize=True
    )

    training_data = []
    for idx, (src_bin, tgt_bin) in enumerate(bin_pairs):
        sig_features = X_signatures[idx]
        counts = bin_pathway_counts.get((src_bin, tgt_bin), [0])

        row = {
            'source_bin': src_bin,
            'target_bin': tgt_bin,
            'pathway_count_mean': np.mean(counts),
            'pathway_count_std': np.std(counts),
            'pathway_count_median': np.median(counts),
            'pathway_count_q25': np.percentile(counts, 25),
            'pathway_count_q75': np.percentile(counts, 75),
            'n_pairs_in_bin': len(counts)
        }

        for i, val in enumerate(sig_features):
            row[f'inter_sig_{i}'] = val

        training_data.append(row)

    df = pd.DataFrame(training_data)

    print(f"\n  Dataset shape: {df.shape}")
    print(f"  Pathway count range: {df['pathway_count_mean'].min():.1f} - "
          f"{df['pathway_count_mean'].max():.1f}")
    print(f"  Pathway count mean: {df['pathway_count_mean'].mean():.1f}")

    output_dir = repo_dir / 'results' / 'pathway_nn' / 'training_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{metapath}_training_data.csv'
    df.to_csv(output_file, index=False)

    print(f"\n  Saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

    return df


def test_loss_functions_on_real_data(df):
    """
    Test all loss functions on real pathway data.

    Parameters
    ----------
    df : pd.DataFrame
        Training data from prepare_pathway_training_data

    Returns
    -------
    summary_df : pd.DataFrame
        Performance summary
    comparison_df : pd.DataFrame
        Baseline comparisons
    """
    print("\n" + "=" * 80)
    print("TESTING LOSS FUNCTIONS ON REAL PATHWAY DATA")
    print("=" * 80)

    feature_cols = ['source_bin', 'target_bin'] + \
                   [f'inter_sig_{i}' for i in range(100)]

    X = df[feature_cols].values
    y = df['pathway_count_mean'].values

    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"  Target mean: {y.mean():.1f}")
    print(f"  Target std: {y.std():.1f}")

    print(f"\n" + "-" * 80)
    print("CREATING MODELS")
    print("-" * 80)

    common_params = {
        'hidden_dims': (128, 64, 32),
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'n_epochs': 500,
        'early_stopping_patience': 50,
        'random_state': 42,
        'device': 'cpu'
    }

    models = {
        'mse': DegreeSignatureNN(**common_params, loss_fn=nn.MSELoss()),
        'huber_1.0': DegreeSignatureNN(**common_params,
                                       loss_fn=HuberLoss(delta=1.0)),
        'huber_0.5': DegreeSignatureNN(**common_params,
                                       loss_fn=HuberLoss(delta=0.5)),
        'huber_0.3': DegreeSignatureNN(**common_params,
                                       loss_fn=HuberLoss(delta=0.3)),
        'log_mse': DegreeSignatureNN(**common_params, loss_fn=LogScaleMSE()),
        'quantile_0.5': DegreeSignatureNN(**common_params,
                                         loss_fn=QuantileLoss(0.5)),
        'random': RandomBaseline(random_state=42),
        'degree_product': DegreeProductBaseline()
    }

    print(f"Created {len(models)} models")

    print("\n" + "-" * 80)
    print("RUNNING 5-FOLD CROSS-VALIDATION")
    print("-" * 80)

    framework = BaselineFramework(n_splits=5, random_state=42)

    summary_df, comparison_df = framework.evaluate_with_baselines(
        models=models,
        X=X,
        y=y,
        baseline_names=['random', 'degree_product', 'mse'],
        fit_params=None
    )

    print("\n" + "=" * 80)
    print("RESULTS ON REAL PATHWAY DATA")
    print("=" * 80)

    framework.print_report(summary_df, comparison_df, target_r=0.95)

    return summary_df, comparison_df, models, X, y


def analyze_residuals(summary_df, models, X, y):
    """
    Detailed residual analysis comparing best loss to MSE.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Performance summary
    models : dict
        Trained models
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    """
    print("\n" + "=" * 80)
    print("DETAILED RESIDUAL ANALYSIS")
    print("=" * 80)

    best_model_name = summary_df.iloc[0]['model']
    mse_model_name = 'mse'

    if best_model_name == mse_model_name:
        print("\nMSE is the best model - no improvement from alternative losses")
        return

    print(f"\nComparing: {best_model_name} vs {mse_model_name}")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model = models[best_model_name]
    mse_model = models[mse_model_name]

    print(f"\nTraining models on 80% of data...")
    best_model.fit(X_train, y_train)
    mse_model.fit(X_train, y_train)

    y_pred_best = best_model.predict(X_test)
    y_pred_mse = mse_model.predict(X_test)

    residuals_best = y_pred_best - y_test
    residuals_mse = y_pred_mse - y_test

    print(f"\n" + "-" * 80)
    print(f"RESIDUAL STATISTICS")
    print("-" * 80)

    print(f"\n{best_model_name}:")
    print(f"  Mean residual: {np.mean(residuals_best):+.4f}")
    print(f"  Median residual: {np.median(residuals_best):+.4f}")
    print(f"  Std residual: {np.std(residuals_best):.4f}")
    print(f"  MAE: {np.mean(np.abs(residuals_best)):.4f}")
    print(f"  % Overpredicting: {100 * np.mean(residuals_best > 0):.1f}%")
    print(f"  % Underpredicting: {100 * np.mean(residuals_best < 0):.1f}%")

    print(f"\n{mse_model_name}:")
    print(f"  Mean residual: {np.mean(residuals_mse):+.4f}")
    print(f"  Median residual: {np.median(residuals_mse):+.4f}")
    print(f"  Std residual: {np.std(residuals_mse):.4f}")
    print(f"  MAE: {np.mean(np.abs(residuals_mse)):.4f}")
    print(f"  % Overpredicting: {100 * np.mean(residuals_mse > 0):.1f}%")
    print(f"  % Underpredicting: {100 * np.mean(residuals_mse < 0):.1f}%")

    low_mask = y_test < np.percentile(y_test, 33)
    mid_mask = (y_test >= np.percentile(y_test, 33)) & \
               (y_test < np.percentile(y_test, 67))
    high_mask = y_test >= np.percentile(y_test, 67)

    print(f"\n" + "-" * 80)
    print(f"BIAS BY COUNT RANGE")
    print("-" * 80)

    print(f"\n{best_model_name}:")
    print(f"  Low (0-33%):    {np.mean(residuals_best[low_mask]):+.4f}")
    print(f"  Mid (33-67%):   {np.mean(residuals_best[mid_mask]):+.4f}")
    print(f"  High (67-100%): {np.mean(residuals_best[high_mask]):+.4f}")

    print(f"\n{mse_model_name}:")
    print(f"  Low (0-33%):    {np.mean(residuals_mse[low_mask]):+.4f}")
    print(f"  Mid (33-67%):   {np.mean(residuals_mse[mid_mask]):+.4f}")
    print(f"  High (67-100%): {np.mean(residuals_mse[high_mask]):+.4f}")

    overpred_improvement = np.abs(np.mean(residuals_best)) < \
                          np.abs(np.mean(residuals_mse))

    print(f"\n" + "-" * 80)
    print(f"OVERPREDICTION ANALYSIS")
    print("-" * 80)

    if overpred_improvement:
        print(f"\n{best_model_name} reduces overprediction bias:")
        print(f"  MSE bias: {np.mean(residuals_mse):+.4f}")
        print(f"  {best_model_name} bias: {np.mean(residuals_best):+.4f}")
        print(f"  Improvement: {np.abs(np.mean(residuals_best)) - np.abs(np.mean(residuals_mse)):+.4f}")
    else:
        print(f"\n{best_model_name} does NOT reduce overprediction bias")
        print(f"  MSE bias: {np.mean(residuals_mse):+.4f}")
        print(f"  {best_model_name} bias: {np.mean(residuals_best):+.4f}")
        print(f"  Change: {np.abs(np.mean(residuals_best)) - np.abs(np.mean(residuals_mse)):+.4f}")


def main():
    """Main execution."""
    print("=" * 80)
    print("REAL PATHWAY DATA: LOSS FUNCTION EVALUATION")
    print("=" * 80)
    print("\nThis script:")
    print("  1. Generates real CbGpPW training data from Hetionet")
    print("  2. Tests all loss functions on REAL data")
    print("  3. Analyzes residual patterns")
    print("  4. Provides honest assessment vs r > 0.95 target")

    repo_dir = Path.cwd()

    training_data_file = repo_dir / 'results' / 'pathway_nn' / \
                        'training_data' / 'CbGpPW_training_data.csv'

    if training_data_file.exists():
        print(f"\nFound existing training data: {training_data_file}")
        response = input("Use existing data? (y/n): ").lower()
        if response == 'y':
            df = pd.read_csv(training_data_file)
            print(f"Loaded {len(df)} samples")
        else:
            df = prepare_pathway_training_data()
    else:
        df = prepare_pathway_training_data()

    summary_df, comparison_df, models, X, y = test_loss_functions_on_real_data(df)

    analyze_residuals(summary_df, models, X, y)

    output_dir = repo_dir / 'results' / 'loss_function_comparison_real'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        output_dir / 'loss_comparison_summary_real_data.csv',
        index=False
    )
    if not comparison_df.empty:
        comparison_df.to_csv(
            output_dir / 'loss_comparison_vs_baselines_real_data.csv',
            index=False
        )

    print(f"\n\nResults saved to: {output_dir}")

    print("\n" + "=" * 80)
    print("HONEST ASSESSMENT")
    print("=" * 80)

    best_r = summary_df.iloc[0]['pearson_r_mean']
    best_name = summary_df.iloc[0]['model']

    print(f"\nBest model: {best_name}")
    print(f"Performance: r = {best_r:.4f}")
    print(f"Target: r > 0.95")

    if best_r > 0.95:
        print(f"\nSTATUS: SUCCESS - Target achieved!")
        print(f"Exceeded target by: {best_r - 0.95:.4f}")
    else:
        print(f"\nSTATUS: NOT MET - Further work needed")
        print(f"Gap to target: {0.95 - best_r:.4f}")
        print(f"\nPossible next steps:")
        print(f"  1. Try combined losses (Huber + log-MSE)")
        print(f"  2. Increase model capacity (more hidden units)")
        print(f"  3. Proceed to PathwayTransformer architecture")
        print(f"  4. Add regularization (L2, dropout)")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
