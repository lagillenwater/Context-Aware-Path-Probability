"""
Test alternative loss functions for pathway frequency prediction.

This script compares MSE, Huber, log-MSE, quantile, and negative binomial
losses on the CbGpPW metapath to determine which best addresses the
overprediction bias observed with standard MSE loss.

Usage:
    python test_loss_functions.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

from src.pathway_losses import (
    HuberLoss, LogScaleMSE, QuantileLoss, NegativeBinomialLoss
)
from src.models.degree_signature_nn import DegreeSignatureNN
from src.models.random_baseline import RandomBaseline
from src.models.degree_product_baseline import DegreeProductBaseline
from src.baseline_framework import BaselineFramework, EvaluationMetrics


def load_or_create_data(data_path: str = None) -> tuple:
    """
    Load CbGpPW training data or create synthetic data.

    Parameters
    ----------
    data_path : str, optional
        Path to training data CSV. If None, creates synthetic data.

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, 102)
    y : np.ndarray
        Target pathway counts (n_samples,)
    """
    if data_path and Path(data_path).exists():
        print(f"Loading training data from: {data_path}")
        df = pd.read_csv(data_path)

        feature_cols = ['source_bin', 'target_bin'] + \
                      [f'inter_sig_{i}' for i in range(100)]

        X = df[feature_cols].values
        y = df['pathway_count_mean'].values

        print(f"  Loaded {len(X)} degree bin combinations")
        print(f"  Feature shape: {X.shape}")
        print(f"  Target range: [{y.min():.1f}, {y.max():.1f}]")
        print(f"  Target mean: {y.mean():.1f}")

        return X, y

    else:
        print("Creating synthetic data for testing...")

        np.random.seed(42)

        n_samples = 100

        source_bins = np.random.randint(0, 10, n_samples)
        target_bins = np.random.randint(0, 10, n_samples)

        intermediate_sigs = np.random.dirichlet(
            np.ones(100), size=n_samples
        )

        X = np.column_stack([
            source_bins,
            target_bins,
            intermediate_sigs
        ])

        degree_product = (source_bins + 1) * (target_bins + 1)
        sig_complexity = np.sum(intermediate_sigs[:, :10], axis=1)

        y = (
            5 * degree_product +
            20 * sig_complexity +
            10 * np.log1p(degree_product) +
            np.random.lognormal(mean=2, sigma=1, size=n_samples)
        )

        print(f"  Created {n_samples} synthetic samples")
        print(f"  Feature shape: {X.shape}")
        print(f"  Target range: [{y.min():.1f}, {y.max():.1f}]")
        print(f"  Target mean: {y.mean():.1f}")

        return X, y


def create_models_with_losses(random_state: int = 42) -> dict:
    """
    Create DegreeSignatureNN models with different loss functions.

    Parameters
    ----------
    random_state : int
        Random seed

    Returns
    -------
    models : dict
        Dictionary of model_name to model instance
    """
    models = {}

    common_params = {
        'hidden_dims': (128, 64, 32),
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'n_epochs': 500,
        'early_stopping_patience': 50,
        'random_state': random_state,
        'device': 'cpu'
    }

    models['mse'] = DegreeSignatureNN(
        **common_params,
        loss_fn=nn.MSELoss()
    )

    models['huber_1.0'] = DegreeSignatureNN(
        **common_params,
        loss_fn=HuberLoss(delta=1.0)
    )

    models['huber_0.5'] = DegreeSignatureNN(
        **common_params,
        loss_fn=HuberLoss(delta=0.5)
    )

    models['log_mse'] = DegreeSignatureNN(
        **common_params,
        loss_fn=LogScaleMSE()
    )

    models['quantile_0.5'] = DegreeSignatureNN(
        **common_params,
        loss_fn=QuantileLoss(quantile=0.5)
    )

    models['quantile_0.6'] = DegreeSignatureNN(
        **common_params,
        loss_fn=QuantileLoss(quantile=0.6)
    )

    models['negbin'] = DegreeSignatureNN(
        **common_params,
        loss_fn=NegativeBinomialLoss()
    )

    models['random'] = RandomBaseline(random_state=random_state)

    models['degree_product'] = DegreeProductBaseline()

    return models


def main():
    """Main evaluation script."""
    print("=" * 80)
    print("PATHWAY LOSS FUNCTION COMPARISON")
    print("=" * 80)
    print("\nGoal: Fix overprediction bias in current DegreeSignatureNN (MSE loss)")
    print("Target: r > 0.95 with balanced residuals\n")

    repo_dir = Path.cwd()
    data_dir = repo_dir / 'results' / 'pathway_nn' / 'training_data'

    possible_data_paths = [
        data_dir / 'CbGpPW_training_data.csv',
    ]

    data_path = None
    for path in possible_data_paths:
        if path.exists():
            data_path = str(path)
            break

    print("Note: Using synthetic data for testing loss functions")
    print("Run notebook 18a to generate real CbGpPW training data\n")

    X, y = load_or_create_data(None)

    print("\n" + "-" * 80)
    print("CREATING MODELS")
    print("-" * 80)

    models = create_models_with_losses(random_state=42)

    print(f"Created {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")

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
    print("RESULTS")
    print("=" * 80)

    framework.print_report(summary_df, comparison_df, target_r=0.95)

    output_dir = repo_dir / 'results' / 'loss_function_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / 'loss_comparison_summary.csv', index=False)
    if not comparison_df.empty:
        comparison_df.to_csv(
            output_dir / 'loss_comparison_vs_baselines.csv',
            index=False
        )

    print(f"\n\nResults saved to: {output_dir}")

    print("\n" + "=" * 80)
    print("RESIDUAL ANALYSIS")
    print("=" * 80)

    best_model_name = summary_df.iloc[0]['model']
    mse_model_name = 'mse'

    if best_model_name != mse_model_name and best_model_name in models:
        print(f"\nComparing residuals: {best_model_name} vs {mse_model_name}")

        best_model = models[best_model_name]
        mse_model = models[mse_model_name]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        best_model.fit(X_train, y_train)
        mse_model.fit(X_train, y_train)

        y_pred_best = best_model.predict(X_test)
        y_pred_mse = mse_model.predict(X_test)

        residuals_best = y_pred_best - y_test
        residuals_mse = y_pred_mse - y_test

        print(f"\n{best_model_name} residuals:")
        print(f"  Mean: {np.mean(residuals_best):+.4f}")
        print(f"  Median: {np.median(residuals_best):+.4f}")
        print(f"  Std: {np.std(residuals_best):.4f}")
        print(f"  % Positive: {100 * np.mean(residuals_best > 0):.1f}%")

        print(f"\n{mse_model_name} residuals:")
        print(f"  Mean: {np.mean(residuals_mse):+.4f}")
        print(f"  Median: {np.median(residuals_mse):+.4f}")
        print(f"  Std: {np.std(residuals_mse):.4f}")
        print(f"  % Positive: {100 * np.mean(residuals_mse > 0):.1f}%")

        low_mask = y_test < np.percentile(y_test, 33)
        mid_mask = (y_test >= np.percentile(y_test, 33)) & \
                   (y_test < np.percentile(y_test, 67))
        high_mask = y_test >= np.percentile(y_test, 67)

        print(f"\nBias by count range ({best_model_name}):")
        print(f"  Low (0-33%):  {np.mean(residuals_best[low_mask]):+.4f}")
        print(f"  Mid (33-67%): {np.mean(residuals_best[mid_mask]):+.4f}")
        print(f"  High (67-100%): {np.mean(residuals_best[high_mask]):+.4f}")

        print(f"\nBias by count range ({mse_model_name}):")
        print(f"  Low (0-33%):  {np.mean(residuals_mse[low_mask]):+.4f}")
        print(f"  Mid (33-67%): {np.mean(residuals_mse[mid_mask]):+.4f}")
        print(f"  High (67-100%): {np.mean(residuals_mse[high_mask]):+.4f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_r = summary_df.iloc[0]['pearson_r_mean']
    best_name = summary_df.iloc[0]['model']

    print(f"\nBest performing loss: {best_name} (r = {best_r:.4f})")

    if best_r > 0.95:
        print("STATUS: Target r > 0.95 ACHIEVED!")
        print(f"Improvement needed for Phase 3 Transformer: None")
    elif best_r > 0.90:
        print("STATUS: Good performance (r > 0.90)")
        print(f"Improvement needed for Phase 3 Transformer: {0.95 - best_r:.4f}")
    elif best_r > 0.88:
        print("STATUS: Modest improvement over baseline")
        print(f"Improvement needed for Phase 3 Transformer: {0.95 - best_r:.4f}")
    else:
        print("STATUS: Minimal improvement")
        print("Recommendation: Proceed to PathwayTransformer for better architecture")

    mse_row = summary_df[summary_df['model'] == 'mse']
    if not mse_row.empty and best_name != 'mse':
        mse_r = mse_row.iloc[0]['pearson_r_mean']
        improvement = best_r - mse_r
        print(f"\nImprovement over MSE: +{improvement:.4f}")

        if improvement > 0.01:
            print(f"RECOMMENDATION: Use {best_name} loss for all future training")
        else:
            print("RECOMMENDATION: MSE performance is comparable, "
                  "either loss is acceptable")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
