"""
Prepare training data for Phase 2 feature ablation study.

For each metapath and feature set:
1. Bin nodes by degree (10×10 quantile bins)
2. Extract features for each bin combination
3. Compute pathway counts (mean, std, percentiles)
4. Save to CSV

Output: results/phase2_training_data/{metapath}_features_{set}.csv
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.enhanced_features import extract_enhanced_features, get_feature_names


def load_edge_matrix(data_dir, edge_abbrev):
    """Load edge adjacency matrix."""
    edge_path = os.path.join(data_dir, 'edges', f'{edge_abbrev}.sparse.npz')
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge file not found: {edge_path}")
    matrix = sp.load_npz(edge_path)

    # Convert bool to int32 for pathway counting
    if matrix.dtype == bool:
        matrix = matrix.astype(np.int32)

    return matrix


def prepare_training_data_single(metapath_name, edge1_abbrev, edge2_abbrev,
                                   feature_set='A', n_bins=10, output_dir='results/phase2_training_data'):
    """
    Prepare training data for a single metapath and feature set.

    Returns:
    - df: DataFrame with columns [feature_1, feature_2, ..., pathway_count_mean, ...]
    """
    print(f"Preparing: {metapath_name}, Feature Set {feature_set}")

    # Load edges from original Hetionet
    edge1 = load_edge_matrix('data', edge1_abbrev)
    edge2 = load_edge_matrix('data', edge2_abbrev)

    # Compute pathway counts
    pathway_matrix = edge1.dot(edge2)
    if sp.issparse(pathway_matrix):
        pathway_matrix = pathway_matrix.toarray()

    # Get degrees
    source_degrees = np.array(edge1.sum(axis=1)).flatten()
    target_degrees = np.array(edge2.sum(axis=0)).flatten()

    # Create degree bins
    source_bins, source_bin_edges = pd.qcut(source_degrees, q=n_bins,
                                              labels=False, retbins=True,
                                              duplicates='drop')
    target_bins, target_bin_edges = pd.qcut(target_degrees, q=n_bins,
                                              labels=False, retbins=True,
                                              duplicates='drop')

    n_source_bins = len(np.unique(source_bins))
    n_target_bins = len(np.unique(target_bins))

    print(f"  Bins: {n_source_bins} source × {n_target_bins} target = {n_source_bins * n_target_bins} combinations")

    # For each bin combination, extract features and compute pathway statistics
    rows = []

    for src_bin in range(n_source_bins):
        for tgt_bin in range(n_target_bins):
            # Find nodes in this bin combination
            src_mask = (source_bins == src_bin)
            tgt_mask = (target_bins == tgt_bin)

            src_nodes = np.where(src_mask)[0]
            tgt_nodes = np.where(tgt_mask)[0]

            if len(src_nodes) == 0 or len(tgt_nodes) == 0:
                continue

            # Extract features for representative pair (first node in each bin)
            # Note: Features should be the same for all pairs in this bin combo
            features = extract_enhanced_features(
                np.array([src_nodes[0]]),
                np.array([tgt_nodes[0]]),
                edge1, edge2,
                n_bins=n_bins,
                feature_set=feature_set
            )

            feature_vector = features[0, :]  # Single pair

            # Get pathway counts for all pairs in this bin combination
            pathway_counts_bin = []
            for i in src_nodes:
                for j in tgt_nodes:
                    pathway_counts_bin.append(pathway_matrix[i, j])

            pathway_counts_bin = np.array(pathway_counts_bin)

            # Compute statistics
            row_data = {
                'source_bin': src_bin,
                'target_bin': tgt_bin,
                'n_source_nodes': len(src_nodes),
                'n_target_nodes': len(tgt_nodes),
                'n_pairs': len(pathway_counts_bin),
                'pathway_count_mean': pathway_counts_bin.mean(),
                'pathway_count_std': pathway_counts_bin.std(),
                'pathway_count_median': np.median(pathway_counts_bin),
                'pathway_count_p25': np.percentile(pathway_counts_bin, 25),
                'pathway_count_p75': np.percentile(pathway_counts_bin, 75),
                'pathway_count_min': pathway_counts_bin.min(),
                'pathway_count_max': pathway_counts_bin.max()
            }

            # Add features
            feature_names = get_feature_names(feature_set, n_bins=n_bins)
            for feat_name, feat_val in zip(feature_names, feature_vector):
                row_data[f'feat_{feat_name}'] = feat_val

            rows.append(row_data)

    df = pd.DataFrame(rows)

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{metapath_name}_features_{feature_set}.csv')
    df.to_csv(output_path, index=False)

    print(f"  Saved: {output_path} ({len(df)} bin combinations)")
    print(f"  Pathway count range: [{df['pathway_count_mean'].min():.2f}, {df['pathway_count_mean'].max():.2f}]")

    return df


def prepare_all_training_data():
    """Prepare training data for all metapaths and feature sets."""
    metapaths = [
        ('CbGpPW', 'CbG', 'GpPW'),
        ('GiGiG', 'GiG', 'GiG'),
        ('CtDaG', 'CtD', 'DaG'),
        ('AdGpBP', 'AdG', 'GpBP'),
        ('CrCbG', 'CrC', 'CbG')
    ]

    feature_sets = ['A', 'B', 'C', 'D', 'E', 'F']

    print("=" * 70)
    print("PREPARING TRAINING DATA FOR PHASE 2")
    print("=" * 70)
    print()

    total = len(metapaths) * len(feature_sets)
    completed = 0

    for metapath_name, edge1_abbrev, edge2_abbrev in metapaths:
        print(f"\nMetapath: {metapath_name}")
        print("-" * 70)

        for feature_set in feature_sets:
            try:
                df = prepare_training_data_single(
                    metapath_name, edge1_abbrev, edge2_abbrev,
                    feature_set=feature_set, n_bins=10
                )
                completed += 1
                print(f"  Progress: {completed}/{total}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        print()

    print("=" * 70)
    print(f"COMPLETED: {completed}/{total} datasets prepared")
    print("=" * 70)

    if completed == total:
        print("All training data prepared successfully.")
        print("Ready for Step 4: Hybrid evaluation")
        return True
    else:
        print(f"WARNING: {total - completed} datasets failed")
        return False


if __name__ == "__main__":
    success = prepare_all_training_data()
    sys.exit(0 if success else 1)
