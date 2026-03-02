#!/usr/bin/env python3
"""
Baseline pair-level model.

Train production model on 100,000 stratified pairs.
Target: r > 0.85, |bias| < 0.01
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import argparse
import pickle
import time

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

from pair_level_features import extract_features_for_pairs, extract_pair_targets
from pair_level_sampling import load_and_sample_pairs


def list_available_permutation_ids(data_dir: Path) -> list[int]:
    """List available local permutation IDs from data/permutations."""
    perm_dir = data_dir / 'permutations'
    if not perm_dir.exists():
        return []
    perm_ids = []
    for child in perm_dir.glob('*.hetmat'):
        try:
            perm_ids.append(int(child.stem))
        except ValueError:
            continue
    return sorted(set(perm_ids))


def compute_average_across_permutations(
    edge1_type: str,
    edge2_type: str,
    pair_indices: np.ndarray,
    perm_ids: list,
    data_dir: Path
) -> tuple:
    """
    Compute average and variance of pathway counts across permutations.

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
    mean : np.ndarray
        Average pathway count for each pair
    variance : np.ndarray
        Variance of pathway counts for each pair
    """
    counts = []
    print(f"  Computing pathway counts across {len(perm_ids)} permutations...")
    for i, perm_id in enumerate(perm_ids, 1):
        if i % 5 == 0:
            print(f"    Permutation {i}/{len(perm_ids)}")
        y_perm = extract_pair_targets(
            edge1_type, edge2_type, pair_indices, perm_id, data_dir
        )
        counts.append(y_perm)

    counts = np.array(counts)
    mean = np.mean(counts, axis=0)
    variance = np.var(counts, axis=0)
    return mean, variance


def main():
    parser = argparse.ArgumentParser(
        description='Baseline pair-level model'
    )
    parser.add_argument(
        '--metapath',
        default='CbGpPW',
        help='Metapath to test (CbGpPW, CtDaG, CrCbG)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100000,
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
        default='B',
        help='Feature set (A, B, C, D, E)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=20,
        help='Number of permutations to average'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained model to disk'
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
    results_dir = repo_dir / 'results' / 'pair_level_models'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Baseline Pair-Level Model")
    print("=" * 80)
    print()
    print(f"Metapath: {args.metapath} ({edge1_type} -> {edge2_type})")
    print(f"Sampling strategy: pathway_stratified")
    print(f"Number of pairs: {args.n_samples:,}")
    print(f"Feature set: {args.feature_set}")
    print(f"Permutations to average: {args.n_permutations}")
    print()

    start_time = time.time()

    # Step 1: Sample pairs (stratified by pathway count)
    print("Step 1: Sampling pairs (stratified by pathway count)...")
    pair_indices = load_and_sample_pairs(
        edge1_type, edge2_type, data_dir,
        args.n_samples, 'pathway_stratified', random_state=42
    )
    print(f"  Sampled {len(pair_indices):,} pairs")
    print(f"  Elapsed: {time.time() - start_time:.1f}s")
    print()

    # Step 2: Extract features
    print("Step 2: Extracting features...")
    step_start = time.time()
    X, metadata = extract_features_for_pairs(
        edge1_type, edge2_type, pair_indices, data_dir,
        args.n_bins, args.feature_set
    )
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Elapsed: {time.time() - step_start:.1f}s")
    print()

    # Step 3: Compute targets (average pathway counts across permutations)
    print("Step 3: Computing targets...")
    step_start = time.time()
    available_perm_ids = [perm_id for perm_id in list_available_permutation_ids(data_dir) if perm_id > 0]
    if not available_perm_ids:
        raise FileNotFoundError(
            f"No non-zero permutations found in {data_dir / 'permutations'}; "
            "phase 1 requires at least one null permutation."
        )
    if args.n_permutations > len(available_perm_ids):
        print(
            f"  Requested {args.n_permutations} permutations, but only "
            f"{len(available_perm_ids)} are available. Using available set."
        )
    perm_ids = available_perm_ids[: min(args.n_permutations, len(available_perm_ids))]
    print(f"  Using permutation IDs: {perm_ids}")
    y, y_variance = compute_average_across_permutations(
        edge1_type, edge2_type, pair_indices, perm_ids, data_dir
    )
    print(f"  Target statistics:")
    print(f"    Mean: {np.mean(y):.6f}")
    print(f"    Std: {np.std(y):.6f}")
    print(f"    Min: {np.min(y):.6f}")
    print(f"    Max: {np.max(y):.6f}")
    print(f"    % zero: {np.sum(y == 0) / len(y) * 100:.1f}%")
    print(f"  Variance statistics:")
    print(f"    Mean: {np.mean(y_variance):.6f}")
    print(f"    Std: {np.std(y_variance):.6f}")
    print(f"  Elapsed: {time.time() - step_start:.1f}s")
    print()

    # Step 4: Split data
    print("Step 4: Splitting data...")
    X_train, X_test, y_train, y_test, var_train, var_test = train_test_split(
        X, y, y_variance, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print()

    # Step 5: Train model
    print("Step 5: Training Linear Regression...")
    step_start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_time = time.time() - step_start
    print(f"  Training complete")
    print(f"  Training time: {train_time:.2f}s")
    print()

    # Step 6: Evaluate
    print("Step 6: Evaluating...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training metrics
    train_r = np.corrcoef(y_train, y_train_pred)[0, 1]
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    train_bias = np.mean(y_train_pred - y_train)

    # Test metrics
    test_r = np.corrcoef(y_test, y_test_pred)[0, 1]
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    test_bias = np.mean(y_test_pred - y_test)

    print(f"  Training metrics:")
    print(f"    r = {train_r:.4f}")
    print(f"    RMSE = {train_rmse:.6f}")
    print(f"    MAE = {train_mae:.6f}")
    print(f"    Bias = {train_bias:.6f}")
    print()
    print(f"  Test metrics:")
    print(f"    r = {test_r:.4f}")
    print(f"    RMSE = {test_rmse:.6f}")
    print(f"    MAE = {test_mae:.6f}")
    print(f"    Bias = {test_bias:.6f}")
    print()

    # Step 7: Additional diagnostics
    print("Step 7: Additional diagnostics...")

    # Stratified performance (by pathway count)
    test_pathway_bins = np.digitize(y_test, bins=[0, 0.01, 0.1, 1.0, np.inf])
    bin_labels = ['Zero', 'Low (0-0.1)', 'Med (0.1-1)', 'High (>1)']

    print("  Performance by pathway count stratum:")
    for bin_idx, label in enumerate(bin_labels, 1):
        mask = test_pathway_bins == bin_idx
        if np.sum(mask) == 0:
            continue
        r_bin = np.corrcoef(y_test[mask], y_test_pred[mask])[0, 1] if np.sum(mask) > 1 else np.nan
        bias_bin = np.mean(y_test_pred[mask] - y_test[mask])
        print(f"    {label}: n={np.sum(mask):,}, r={r_bin:.4f}, bias={bias_bin:.6f}")
    print()

    # Feature importance
    feature_importance = np.abs(model.coef_)
    top_features = np.argsort(feature_importance)[-5:][::-1]
    print("  Top 5 most important features (by |coefficient|):")
    for i, feat_idx in enumerate(top_features, 1):
        print(f"    {i}. Feature {feat_idx}: coef = {model.coef_[feat_idx]:.6f}")
    print()

    # Prediction distribution comparison
    print("  Prediction vs actual distribution:")
    print(f"    Mean predicted: {np.mean(y_test_pred):.6f}")
    print(f"    Mean actual: {np.mean(y_test):.6f}")
    print(f"    Std predicted: {np.std(y_test_pred):.6f}")
    print(f"    Std actual: {np.std(y_test):.6f}")
    print()

    # Residual analysis
    residuals = y_test - y_test_pred
    print("  Residual statistics:")
    print(f"    Mean: {np.mean(residuals):.6f}")
    print(f"    Std: {np.std(residuals):.6f}")
    print(f"    |Residual| / |Actual| (median): {np.median(np.abs(residuals) / (np.abs(y_test) + 1e-10)):.4f}")
    print()

    # Total time
    total_time = time.time() - start_time
    print(f"Total elapsed time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()

    # Step 8: Save model
    if args.save_model:
        print("Step 8: Saving model...")
        model_path = results_dir / f'phase1_{args.metapath}_model.pkl'
        model_data = {
            'model': model,
            'metadata': metadata,
            'metapath': args.metapath,
            'edge1_type': edge1_type,
            'edge2_type': edge2_type,
            'feature_set': args.feature_set,
            'n_samples': args.n_samples,
            'n_permutations': len(perm_ids),
            'permutation_ids': perm_ids,
            'train_r': train_r,
            'test_r': test_r,
            'test_rmse': test_rmse,
            'test_bias': test_bias,
            'train_time': train_time
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  Model saved to: {model_path}")
        print()

    # Decision
    print("=" * 80)
    print("PHASE 1 RESULTS")
    print("=" * 80)
    print()
    print(f"Test correlation: r = {test_r:.4f}")
    print(f"Test bias: {test_bias:.6f}")
    print(f"Target: r > 0.85, |bias| < 0.01")
    print()

    if test_r > 0.90 and abs(test_bias) < 0.01:
        print("✓✓ EXCELLENT - Exceeds target (r > 0.90)")
        print("  Ready for pair-level degree correction")
    elif test_r > 0.85 and abs(test_bias) < 0.01:
        print("✓ PASS - Meets target")
        print("  Ready for pair-level degree correction")
    elif test_r > 0.85:
        print("⚠ MARGINAL - Good correlation but bias too high")
        print(f"  Bias = {test_bias:.6f} exceeds threshold (0.01)")
    elif abs(test_bias) < 0.01:
        print("⚠ MARGINAL - Low bias but correlation below target")
        print(f"  r = {test_r:.4f} below threshold (0.85)")
    else:
        print("✗ FAIL - Does not meet target")
        print("  Recommendations:")
        print("    - Try larger sample size (200k pairs)")
        print("    - Try different feature sets")
        print("    - Check for data quality issues")

    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
