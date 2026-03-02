#!/usr/bin/env python3
"""
Phase 0: Validate Pair-Level Hypothesis

Test whether pair-specific features can predict pathway null distributions.

Decision criterion: If r > 0.85, proceed with full pair-level pipeline.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import argparse

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pair_level_features import extract_features_for_pairs, extract_pair_targets
from pair_level_sampling import load_and_sample_pairs


def compute_average_across_permutations(
    edge1_type: str,
    edge2_type: str,
    pair_indices: np.ndarray,
    perm_ids: list,
    data_dir: Path
) -> np.ndarray:
    """
    Compute average pathway count for each pair across permutations.

    Parameters
    ----------
    edge1_type : str
        First edge type
    edge2_type : str
        Second edge type
    pair_indices : np.ndarray
        Array of (source_idx, target_idx) pairs
    perm_ids : list
        List of permutation IDs
    data_dir : Path
        Data directory

    Returns
    -------
    np.ndarray
        Average pathway count for each pair across permutations
    """
    counts = []
    for perm_id in perm_ids:
        y_perm = extract_pair_targets(
            edge1_type, edge2_type, pair_indices, perm_id, data_dir
        )
        counts.append(y_perm)

    counts = np.array(counts)
    return np.mean(counts, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 0: Validate pair-level hypothesis'
    )
    parser.add_argument(
        '--metapath',
        default='CbGpPW',
        help='Metapath to test (CbGpPW, CtDaG, CrCbG)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='Number of pairs to sample'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=10,
        help='Number of bins for intermediate signature'
    )
    parser.add_argument(
        '--feature-set',
        default='E',
        help='Feature set (A, B, C, D, E)'
    )
    parser.add_argument(
        '--sampling-strategy',
        default='random',
        help='Sampling strategy (random, pathway_stratified, degree_stratified)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=20,
        help='Number of permutations to average'
    )
    args = parser.parse_args()

    # Parse metapath
    metapath_map = {
        'CbGpPW': ('CbG', 'GpPW'),
        'CtDaG': ('CtD', 'DaG'),
        'CrCbG': ('CrC', 'CbG'),
    }
    edge1_type, edge2_type = metapath_map[args.metapath]

    data_dir = repo_dir / 'data'

    print("=" * 80)
    print("Phase 0: Validate Pair-Level Hypothesis")
    print("=" * 80)
    print()
    print(f"Metapath: {args.metapath} ({edge1_type} -> {edge2_type})")
    print(f"Sampling strategy: {args.sampling_strategy}")
    print(f"Number of pairs: {args.n_samples:,}")
    print(f"Feature set: {args.feature_set}")
    print(f"Intermediate signature bins: {args.n_bins}")
    print(f"Permutations to average: {args.n_permutations}")
    print()

    # Step 1: Sample pairs
    print("Step 1: Sampling pairs...")
    pair_indices = load_and_sample_pairs(
        edge1_type, edge2_type, data_dir,
        args.n_samples, args.sampling_strategy, random_state=42
    )
    print(f"  Sampled {len(pair_indices):,} pairs")
    print()

    # Step 2: Extract features
    print("Step 2: Extracting features...")
    X, metadata = extract_features_for_pairs(
        edge1_type, edge2_type, pair_indices, data_dir,
        args.n_bins, args.feature_set
    )
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features: {metadata['n_features']}")
    print()

    # Step 3: Compute targets (average pathway counts across permutations)
    print("Step 3: Computing targets (averaging across permutations)...")
    perm_ids = list(range(1, args.n_permutations + 1))
    y = compute_average_across_permutations(
        edge1_type, edge2_type, pair_indices, perm_ids, data_dir
    )
    print(f"  Target statistics:")
    print(f"    Mean: {np.mean(y):.4f}")
    print(f"    Std: {np.std(y):.4f}")
    print(f"    Min: {np.min(y):.4f}")
    print(f"    Max: {np.max(y):.4f}")
    print(f"    % zero: {np.sum(y == 0) / len(y) * 100:.1f}%")
    print()

    # Step 4: Split data
    print("Step 4: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print()

    # Step 5: Train model
    print("Step 5: Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  Training complete")
    print()

    # Step 6: Evaluate
    print("Step 6: Evaluating...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training metrics
    train_r = np.corrcoef(y_train, y_train_pred)[0, 1]
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    train_bias = np.mean(y_train_pred - y_train)

    # Test metrics
    test_r = np.corrcoef(y_test, y_test_pred)[0, 1]
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    test_bias = np.mean(y_test_pred - y_test)

    print(f"  Training metrics:")
    print(f"    r = {train_r:.4f}")
    print(f"    RMSE = {train_rmse:.4f}")
    print(f"    Bias = {train_bias:.4f}")
    print()
    print(f"  Test metrics:")
    print(f"    r = {test_r:.4f}")
    print(f"    RMSE = {test_rmse:.4f}")
    print(f"    Bias = {test_bias:.4f}")
    print()

    # Decision
    print("=" * 80)
    print("DECISION")
    print("=" * 80)
    print()
    print(f"Test correlation: r = {test_r:.4f}")
    print(f"Decision criterion: r > 0.85")
    print()

    if test_r > 0.85:
        print("✓ HYPOTHESIS VALIDATED")
        print("  Pair-level features can predict pathway null distributions.")
        print("  Recommendation: Proceed with full pair-level pipeline (Phases 1-6)")
    elif test_r > 0.70:
        print("⚠ MARGINAL PERFORMANCE")
        print("  Pair-level features show promise but below target.")
        print("  Recommendations:")
        print("    - Try feature set 'E' if not already used")
        print("    - Increase n_bins for intermediate signature")
        print("    - Try degree_stratified sampling")
        print("    - Increase n_samples to 50,000")
    else:
        print("✗ HYPOTHESIS REJECTED")
        print("  Pair-level features do not predict well enough.")
        print("  Recommendations:")
        print("    - Investigate feature engineering")
        print("    - Consider alternative approaches")
        print("    - May need to use bin-level approach with caveats")

    print()

    # Additional diagnostics
    print("=" * 80)
    print("DIAGNOSTICS")
    print("=" * 80)
    print()

    # Feature importance (top 5)
    feature_importance = np.abs(model.coef_)
    top_features = np.argsort(feature_importance)[-5:][::-1]
    print("Top 5 most important features (by |coefficient|):")
    for i, feat_idx in enumerate(top_features, 1):
        print(f"  {i}. Feature {feat_idx}: coef = {model.coef_[feat_idx]:.6f}")
    print()

    # Prediction distribution
    print("Prediction distribution:")
    print(f"  Mean predicted: {np.mean(y_test_pred):.4f}")
    print(f"  Mean actual: {np.mean(y_test):.4f}")
    print(f"  Std predicted: {np.std(y_test_pred):.4f}")
    print(f"  Std actual: {np.std(y_test):.4f}")
    print()

    # Residual analysis
    residuals = y_test - y_test_pred
    print("Residual statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std: {np.std(residuals):.4f}")
    print(f"  |Residual| / |Actual| (mean): {np.mean(np.abs(residuals) / (y_test + 1e-10)):.4f}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
