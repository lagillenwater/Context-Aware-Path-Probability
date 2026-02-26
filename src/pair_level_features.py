#!/usr/bin/env python3
"""
Pair-level feature extraction for pathway null prediction.

This module extracts features for specific (source, target) node pairs,
in contrast to bin-level aggregation.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict


def compute_pair_intermediate_signature(
    edge1: sp.csr_matrix,
    edge2: sp.csr_matrix,
    source_idx: int,
    target_idx: int,
    n_bins: int = 10
) -> np.ndarray:
    """
    Compute intermediate node degree signature for a specific pair.

    For paths source -> intermediate -> target, compute histogram of
    intermediate node degrees.

    Parameters
    ----------
    edge1 : sp.csr_matrix
        First edge matrix (source -> intermediate)
    edge2 : sp.csr_matrix
        Second edge matrix (intermediate -> target)
    source_idx : int
        Source node index
    target_idx : int
        Target node index
    n_bins : int
        Number of bins for degree histogram

    Returns
    -------
    np.ndarray
        Histogram of intermediate node degrees (n_bins bins)
    """
    # Get intermediate nodes on paths from source to target
    source_neighbors = edge1.getrow(source_idx).indices
    target_neighbors = edge2.getcol(target_idx).indices

    # Find intersection (nodes on paths)
    intermediate_nodes = np.intersect1d(source_neighbors, target_neighbors)

    if len(intermediate_nodes) == 0:
        return np.zeros(n_bins)

    # Compute degrees of intermediate nodes
    intermediate_degrees = np.array(
        edge1[:, intermediate_nodes].sum(axis=0) +
        edge2[intermediate_nodes, :].sum(axis=1).T
    ).flatten()

    # Create histogram
    hist, _ = np.histogram(
        intermediate_degrees,
        bins=n_bins,
        range=(0, intermediate_degrees.max() + 1)
    )

    # Normalize
    hist = hist.astype(float) / len(intermediate_nodes)

    return hist


def extract_pair_features(
    edge1: sp.csr_matrix,
    edge2: sp.csr_matrix,
    source_idx: int,
    target_idx: int,
    n_bins: int = 10,
    feature_set: str = 'E'
) -> np.ndarray:
    """
    Extract features for a specific (source, target) pair.

    Parameters
    ----------
    edge1 : sp.csr_matrix
        First edge matrix (source -> intermediate)
    edge2 : sp.csr_matrix
        Second edge matrix (intermediate -> target)
    source_idx : int
        Source node index
    target_idx : int
        Target node index
    n_bins : int
        Number of bins for intermediate signature
    feature_set : str
        Feature set to use ('A', 'B', 'C', 'D', 'E')

    Returns
    -------
    np.ndarray
        Feature vector for this pair
    """
    # Basic degree features
    deg_source = edge1.getrow(source_idx).sum()
    deg_target = edge2.getcol(target_idx).sum()

    features = [deg_source, deg_target]

    if feature_set in ['B', 'C', 'D', 'E']:
        # Degree interactions
        features.extend([
            deg_source * deg_target,
            deg_source ** 2,
            deg_target ** 2
        ])

    if feature_set in ['C', 'D', 'E']:
        # Intermediate signature
        intermediate_sig = compute_pair_intermediate_signature(
            edge1, edge2, source_idx, target_idx, n_bins
        )
        features.extend(intermediate_sig.tolist())

    if feature_set in ['D', 'E']:
        # Cross-terms with intermediate signature
        cross_terms = []
        intermediate_sig = compute_pair_intermediate_signature(
            edge1, edge2, source_idx, target_idx, n_bins
        )
        for feat in intermediate_sig:
            cross_terms.extend([
                feat * deg_source,
                feat * deg_target
            ])
        features.extend(cross_terms)

    if feature_set == 'E':
        # Additional cross-terms
        intermediate_sig = compute_pair_intermediate_signature(
            edge1, edge2, source_idx, target_idx, n_bins
        )
        additional_terms = []
        for feat in intermediate_sig:
            additional_terms.extend([
                feat * deg_source * deg_target,
                feat * (deg_source ** 2),
                feat * (deg_target ** 2)
            ])
        features.extend(additional_terms)

    return np.array(features)


def extract_features_for_pairs(
    edge1_type: str,
    edge2_type: str,
    pair_indices: np.ndarray,
    data_dir: str,
    n_bins: int = 10,
    feature_set: str = 'E'
) -> Tuple[np.ndarray, Dict]:
    """
    Extract features for multiple pairs.

    Parameters
    ----------
    edge1_type : str
        First edge type
    edge2_type : str
        Second edge type
    pair_indices : np.ndarray
        Array of (source_idx, target_idx) pairs, shape (n_pairs, 2)
    data_dir : str
        Data directory
    n_bins : int
        Number of bins for intermediate signature
    feature_set : str
        Feature set to use

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n_pairs, n_features)
    metadata : dict
        Metadata about extraction
    """
    from pathlib import Path

    data_dir = Path(data_dir)
    edges_dir = data_dir / 'edges'

    # Load edges
    edge1 = sp.load_npz(edges_dir / f'{edge1_type}.sparse.npz')
    edge2 = sp.load_npz(edges_dir / f'{edge2_type}.sparse.npz')

    # Extract features for each pair
    feature_list = []
    for source_idx, target_idx in pair_indices:
        features = extract_pair_features(
            edge1, edge2, source_idx, target_idx, n_bins, feature_set
        )
        feature_list.append(features)

    X = np.array(feature_list)

    metadata = {
        'n_pairs': len(pair_indices),
        'n_features': X.shape[1],
        'edge1_type': edge1_type,
        'edge2_type': edge2_type,
        'n_bins': n_bins,
        'feature_set': feature_set
    }

    return X, metadata


def extract_pair_targets(
    edge1_type: str,
    edge2_type: str,
    pair_indices: np.ndarray,
    perm_id: int,
    data_dir: str
) -> np.ndarray:
    """
    Extract target pathway counts for pairs from a permutation.

    Parameters
    ----------
    edge1_type : str
        First edge type
    edge2_type : str
        Second edge type
    pair_indices : np.ndarray
        Array of (source_idx, target_idx) pairs
    perm_id : int
        Permutation ID (0 for original, 1-200 for permutations)
    data_dir : str
        Data directory

    Returns
    -------
    y : np.ndarray
        Pathway counts for each pair
    """
    from pathlib import Path

    data_dir = Path(data_dir)
    edges_dir = data_dir / 'edges'

    # Load edges
    if perm_id == 0:
        edge1 = sp.load_npz(edges_dir / f'{edge1_type}.sparse.npz')
        edge2 = sp.load_npz(edges_dir / f'{edge2_type}.sparse.npz')
    else:
        perm_dir = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges'
        edge1 = sp.load_npz(perm_dir / f'{edge1_type}.sparse.npz')
        edge2 = sp.load_npz(perm_dir / f'{edge2_type}.sparse.npz')

    # Compute pathway matrix
    pathway_matrix = edge1 @ edge2

    # Extract counts for specified pairs
    y = np.zeros(len(pair_indices))
    for i, (source_idx, target_idx) in enumerate(pair_indices):
        y[i] = pathway_matrix[source_idx, target_idx]

    return y
