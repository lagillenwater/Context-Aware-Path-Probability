#!/usr/bin/env python3
"""
Test if Ridge regularization is necessary for correction model.

Compare LinearRegression vs Ridge for the correction model.
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression, Ridge

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pathway_features_v2 import extract_features_from_original, extract_features_from_permutation
from pathway_evaluation_v2 import validate_on_permutations
from degree_aware_correction import DegreeAwareCorrectionModel


def parse_metapath(metapath_str):
    """Parse metapath string into edge types."""
    metapath_map = {
        'CbGpPW': ('CbG', 'GpPW'),
        'CtDaG': ('CtD', 'DaG'),
        'CrCbG': ('CrC', 'CbG'),
    }
    return metapath_map[metapath_str]


def main():
    parser = argparse.ArgumentParser(description='Test Ridge necessity')
    parser.add_argument('--metapath', default='CbGpPW', help='Metapath to test')
    parser.add_argument('--n-bins', type=int, default=10, help='Number of bins')
    args = parser.parse_args()

    data_dir = repo_dir / 'data'
    edge1_type, edge2_type = parse_metapath(args.metapath)

    print(f"Testing Ridge necessity for {args.metapath}")
    print("=" * 80)

    # Extract features
    X_train, y_original, metadata = extract_features_from_original(
        edge1_type, edge2_type, data_dir, args.n_bins, feature_set='E'
    )

    _, y_perm0, _ = extract_features_from_permutation(
        edge1_type, edge2_type, 0, data_dir, args.n_bins, feature_set='E'
    )

    n_samples = metadata['n_samples']
    n_features = metadata['n_features']

    print(f"Training samples: {n_samples}")
    print(f"Base model features: {n_features}")
    print(f"Correction model features: 15")
    print(f"Samples per feature (correction): {n_samples / 15:.2f}")
    print()

    perm_ids = list(range(1, 20))

    # Test 1: LinearRegression for correction
    print("Test 1: LinearRegression for correction model")
    print("-" * 80)

    model_lr = DegreeAwareCorrectionModel(
        base_model=LinearRegression(),
        correction_model=LinearRegression(),
        use_interaction=True
    )

    try:
        model_lr.fit(X_train, y_original, y_perm0)

        class ModelWrapper:
            def __init__(self, model):
                self.model = model
                self.device = 'cpu'
            def to(self, device):
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

        results_lr = validate_on_permutations(
            ModelWrapper(model_lr), edge1_type, edge2_type, perm_ids,
            data_dir, args.n_bins, 'cpu', feature_set='E'
        )

        print(f"  Validation r: {results_lr['validation_r']:.4f}")
        print(f"  Bias: {results_lr['mean_error']:.4f}")
        print(f"  RMSE: {results_lr['rmse']:.4f}")
    except Exception as e:
        print(f"  FAILED: {e}")
        results_lr = None
    print()

    # Test 2: Ridge(α=0.1) for correction
    print("Test 2: Ridge(α=0.1) for correction model")
    print("-" * 80)

    model_ridge = DegreeAwareCorrectionModel(
        base_model=LinearRegression(),
        correction_model=Ridge(alpha=0.1),
        use_interaction=True
    )

    model_ridge.fit(X_train, y_original, y_perm0)

    results_ridge = validate_on_permutations(
        ModelWrapper(model_ridge), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu', feature_set='E'
    )

    print(f"  Validation r: {results_ridge['validation_r']:.4f}")
    print(f"  Bias: {results_ridge['mean_error']:.4f}")
    print(f"  RMSE: {results_ridge['rmse']:.4f}")
    print()

    # Test 3: Ridge(α=1.0) for correction
    print("Test 3: Ridge(α=1.0) for correction model")
    print("-" * 80)

    model_ridge1 = DegreeAwareCorrectionModel(
        base_model=LinearRegression(),
        correction_model=Ridge(alpha=1.0),
        use_interaction=True
    )

    model_ridge1.fit(X_train, y_original, y_perm0)

    results_ridge1 = validate_on_permutations(
        ModelWrapper(model_ridge1), edge1_type, edge2_type, perm_ids,
        data_dir, args.n_bins, 'cpu', feature_set='E'
    )

    print(f"  Validation r: {results_ridge1['validation_r']:.4f}")
    print(f"  Bias: {results_ridge1['mean_error']:.4f}")
    print(f"  RMSE: {results_ridge1['rmse']:.4f}")
    print()

    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if results_lr:
        print(f"LinearRegression:  r={results_lr['validation_r']:.4f}, bias={results_lr['mean_error']:.4f}, RMSE={results_lr['rmse']:.4f}")
    else:
        print("LinearRegression:  FAILED")

    print(f"Ridge(α=0.1):      r={results_ridge['validation_r']:.4f}, bias={results_ridge['mean_error']:.4f}, RMSE={results_ridge['rmse']:.4f}")
    print(f"Ridge(α=1.0):      r={results_ridge1['validation_r']:.4f}, bias={results_ridge1['mean_error']:.4f}, RMSE={results_ridge1['rmse']:.4f}")
    print()

    if results_lr:
        diff = results_ridge['validation_r'] - results_lr['validation_r']
        print(f"Ridge improvement: {diff:+.4f} ({diff/results_lr['validation_r']*100:+.2f}%)")

    print()
    print("CONCLUSION:")
    if n_samples < 20:
        print(f"  With only {n_samples} samples and 15 features, regularization is CRITICAL")
    elif results_lr and abs(results_ridge['validation_r'] - results_lr['validation_r']) < 0.001:
        print(f"  Ridge and LinearRegression perform similarly (diff < 0.001)")
        print("  Regularization not strictly necessary but doesn't hurt")
    elif results_lr:
        print(f"  Ridge shows meaningful improvement")
        print("  Regularization is beneficial")
    else:
        print("  LinearRegression failed - Ridge is NECESSARY")

    return 0


if __name__ == '__main__':
    sys.exit(main())
