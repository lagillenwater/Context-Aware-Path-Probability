"""
Test Hetionet → Permutation Translation approach.

Key insight: Instead of predicting null counts directly, learn the systematic
difference between original Hetionet counts and permutation counts.

Approach:
1. Enumerate exact counts in original Hetionet
2. Compute mean counts across perms 0-4
3. Learn correction: null_count = het_count - f(degrees, topology_features)
4. Apply to predict null counts for new pairs

Advantages:
- Uses original Hetionet (which we have)
- Learns systematic topology loss (assortativity, clustering)
- Should scale to longer paths without more permutations

Usage:
    python test_src/test_hetionet_translation.py CbGpPW

References:
    - Baseline linear: r=0.787
    - Current best: RF r=0.778, Q-Q=0.815
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys
import argparse

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

from test_src.validate_mean_variance_prediction import (
    load_permuted_edge_matrices,
    sample_pairs,
    extract_degree_features,
    compute_pathway_counts,
    evaluate_z_scores
)


def enumerate_hetionet_counts(pairs, edge1, edge2):
    """
    Enumerate exact pathway counts in original Hetionet.

    For 2-hop metapath A-e1-B-e2-C, count paths by matrix multiplication:
    counts[i,j] = (edge1[i,:] * edge2[:,j]).sum()

    Args:
        pairs: Node pairs (n_pairs, 2)
        edge1: First edge matrix
        edge2: Second edge matrix

    Returns:
        counts: Exact pathway counts (n_pairs,)
    """
    counts = compute_pathway_counts(pairs, edge1, edge2)
    return counts


def train_translation_model(X, het_counts, null_counts_mean, model_type='rf'):
    """
    Train model to predict correction from Hetionet to null.

    correction = het_counts - null_counts_mean
    model: correction = f(degrees, het_counts)

    Args:
        X: Degree features (n_pairs, 5)
        het_counts: Counts in original Hetionet (n_pairs,)
        null_counts_mean: Mean counts across perms 0-4 (n_pairs,)
        model_type: 'linear' or 'rf'

    Returns:
        model: Trained correction model
    """
    correction = het_counts - null_counts_mean

    features = np.column_stack([X, het_counts.reshape(-1, 1)])

    print(f"Training {model_type} correction model...")
    print(f"  Correction statistics:")
    print(f"    Mean: {correction.mean():.2f}")
    print(f"    Std: {correction.std():.2f}")
    print(f"    Range: [{correction.min():.2f}, {correction.max():.2f}]")

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(features, correction)

    correction_pred = model.predict(features)
    r_correction = np.corrcoef(correction, correction_pred)[0, 1]
    print(f"  Correction model r: {r_correction:.4f}")

    return model


def predict_with_translation(model, X, het_counts):
    """
    Predict null counts using translation model.

    Args:
        model: Trained correction model
        X: Degree features (n_pairs, 5)
        het_counts: Hetionet counts (n_pairs,)

    Returns:
        null_pred: Predicted null counts (n_pairs,)
    """
    features = np.column_stack([X, het_counts.reshape(-1, 1)])
    correction_pred = model.predict(features)
    null_pred = het_counts - correction_pred

    null_pred = np.maximum(null_pred, 0)

    return null_pred


def main():
    parser = argparse.ArgumentParser(description='Test Hetionet translation model')
    parser.add_argument('metapath', help='Metapath name (e.g., CbGpPW)')
    parser.add_argument('--model', choices=['linear', 'rf'], default='rf',
                       help='Correction model type')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of node pairs to sample')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    metapath_configs = {
        'CbGpPW': ('CbG', 'GpPW')
    }

    if args.metapath not in metapath_configs:
        print(f"Error: Unknown metapath {args.metapath}")
        print(f"Available: {list(metapath_configs.keys())}")
        sys.exit(1)

    edge1_type, edge2_type = metapath_configs[args.metapath]

    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'hetionet_translation'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"Hetionet → Permutation Translation: {args.metapath}")
    print("="*70)

    print(f"\nSampling {args.n_samples} node pairs...")
    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(edge1_type, edge2_type, 0, data_dir)
    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples=args.n_samples,
                        random_state=args.random_state)
    print(f"  Sampled pairs: {len(pairs)}")

    X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)

    print("\nEnumerating counts in original Hetionet (perm 0)...")
    het_counts = enumerate_hetionet_counts(pairs, edge1_perm0, edge2_perm0)
    print(f"  Hetionet count statistics:")
    print(f"    Mean: {het_counts.mean():.2f}")
    print(f"    Std: {het_counts.std():.2f}")
    print(f"    Non-zero: {(het_counts > 0).sum()} / {len(het_counts)}")

    train_perms = list(range(5))
    test_perms = list(range(15, 21))

    print("\nComputing null counts (perms 0-4)...")
    counts_train = []
    for perm in train_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_train.append(counts)
    counts_train = np.column_stack(counts_train)
    null_counts_mean = counts_train.mean(axis=1)

    print(f"  Null count statistics:")
    print(f"    Mean: {null_counts_mean.mean():.2f}")
    print(f"    Std: {null_counts_mean.std():.2f}")

    translation_model = train_translation_model(
        X, het_counts, null_counts_mean, model_type=args.model
    )

    print("\nEvaluating on test permutations...")
    null_pred = predict_with_translation(translation_model, X, het_counts)

    counts_test = []
    for perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)
    counts_test = np.column_stack(counts_test)

    results = []
    for idx, perm in enumerate(test_perms):
        counts = counts_test[:, idx]
        r_mean = np.corrcoef(counts, null_pred)[0, 1]

        results.append({
            'test_perm': perm,
            'model_type': f'hetionet_translation_{args.model}',
            'r_mean': r_mean
        })

    results_df = pd.DataFrame(results)

    mean_r = results_df['r_mean'].mean()
    print(f"\nTest Results (Hetionet Translation):")
    print(f"  Mean r across test perms: {mean_r:.4f}")
    print(f"  Comparison to baseline linear: r=0.787")
    print(f"  Comparison to best (RF): r=0.778")

    output_file = output_dir / f'{args.metapath}_hetionet_translation_{args.model}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results: {output_file}")


if __name__ == '__main__':
    main()
