"""
Feature extraction for pathway prediction (v2).

This module extracts features from the original Hetionet graph to train
models that predict average pathway counts across degree-preserving
permutations.

Functions
---------
extract_features_from_original
    Main function to extract features and targets from original graph
extract_features_from_permutation
    Extract features and targets from a single permutation
compute_intermediate_signature
    Compute histogram of intermediate node degrees
extract_features_setA
    Baseline features (degree bins + intermediate signature)
extract_features_setB
    Set A + log transforms
extract_features_setC
    Set B + summary statistics
extract_features_setD
    Set C + neighbor context
extract_features_setE
    Set D + polynomial terms
extract_features_setF
    Set E + interaction terms
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd


def compute_degree_bins(degrees, n_bins=10):
    """
    Compute degree bin assignments using quantile-based binning.

    Parameters
    ----------
    degrees : np.ndarray
        Array of node degrees
    n_bins : int
        Number of bins

    Returns
    -------
    np.ndarray
        Bin assignments (0 to n_bins-1)
    np.ndarray
        Bin edges
    """
    if len(degrees) == 0:
        return np.array([]), np.array([])

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(degrees, quantiles)
    bin_edges = np.unique(bin_edges)

    bins = np.digitize(degrees, bin_edges[1:], right=False)

    return bins, bin_edges


def compute_intermediate_signature(
    edge1_matrix,
    edge2_matrix,
    source_indices,
    target_indices,
    n_bins=10
):
    """
    Compute intermediate node degree signature for a bin pair.

    For paths source -> intermediate -> target, computes a histogram
    of (in_degree, out_degree) of intermediate nodes.

    Parameters
    ----------
    edge1_matrix : scipy.sparse matrix
        First edge type (source -> intermediate)
    edge2_matrix : scipy.sparse matrix
        Second edge type (intermediate -> target)
    source_indices : np.ndarray
        Indices of source nodes in this bin
    target_indices : np.ndarray
        Indices of target nodes in this bin
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    np.ndarray
        Flattened 10x10 histogram (100 features)
    """
    # Get submatrices
    edge1_sub = edge1_matrix[source_indices, :]
    edge2_sub = edge2_matrix[:, target_indices]

    # Find intermediate nodes involved in any path
    intermediate_from_source = np.array(edge1_sub.sum(axis=0)).flatten() > 0
    intermediate_to_target = np.array(edge2_sub.sum(axis=1)).flatten() > 0
    intermediate_in_paths = intermediate_from_source & intermediate_to_target

    if intermediate_in_paths.sum() == 0:
        return np.zeros(n_bins * n_bins)

    intermediate_indices = np.where(intermediate_in_paths)[0]

    # Compute in-degrees and out-degrees of intermediate nodes
    in_degrees = np.array(edge1_matrix[:, intermediate_indices].sum(
        axis=0)).flatten()
    out_degrees = np.array(edge2_matrix[intermediate_indices, :].sum(
        axis=1)).flatten()

    # Bin the degrees
    in_bins, _ = compute_degree_bins(in_degrees, n_bins)
    out_bins, _ = compute_degree_bins(out_degrees, n_bins)

    # Create 2D histogram
    histogram = np.zeros((n_bins, n_bins))
    for i_bin, o_bin in zip(in_bins, out_bins):
        if i_bin < n_bins and o_bin < n_bins:
            histogram[i_bin, o_bin] += 1

    # Normalize to sum to 1
    total = histogram.sum()
    if total > 0:
        histogram = histogram / total

    return histogram.flatten()


def extract_features_setA(
    edge1_matrix,
    edge2_matrix,
    source_bin,
    target_bin,
    source_indices,
    target_indices,
    n_bins=10
):
    """
    Extract baseline feature set A.

    Features:
    - source_degree_bin (1 feature)
    - target_degree_bin (1 feature)
    - intermediate_signature (100 features)

    Total: 102 features

    Parameters
    ----------
    edge1_matrix : scipy.sparse matrix
        First edge type matrix
    edge2_matrix : scipy.sparse matrix
        Second edge type matrix
    source_bin : int
        Source degree bin
    target_bin : int
        Target degree bin
    source_indices : np.ndarray
        Indices of source nodes
    target_indices : np.ndarray
        Indices of target nodes
    n_bins : int
        Number of bins

    Returns
    -------
    np.ndarray
        Feature vector (102 dims)
    """
    features = []

    # Degree bins
    features.append(source_bin)
    features.append(target_bin)

    # Intermediate signature
    signature = compute_intermediate_signature(
        edge1_matrix, edge2_matrix,
        source_indices, target_indices,
        n_bins
    )
    features.extend(signature)

    return np.array(features)


def extract_features_setB(
    edge1_matrix,
    edge2_matrix,
    source_bin,
    target_bin,
    source_indices,
    target_indices,
    n_bins=10
):
    """
    Extract feature set B: Set A + log transforms.

    Features:
    - All features from Set A (102)
    - log(source_bin + 1) (1 feature)
    - log(target_bin + 1) (1 feature)
    - log(intermediate_signature + 1) (100 features)

    Total: 204 features

    Parameters
    ----------
    Same as extract_features_setA

    Returns
    -------
    np.ndarray
        Feature vector (204 dims)
    """
    # Get Set A features
    features_a = extract_features_setA(
        edge1_matrix, edge2_matrix,
        source_bin, target_bin,
        source_indices, target_indices,
        n_bins
    )

    features = list(features_a)

    # Add log transforms
    features.append(np.log1p(source_bin))
    features.append(np.log1p(target_bin))

    # Log transform of intermediate signature (last 100 features of Set A)
    signature = features_a[2:]
    features.extend(np.log1p(signature))

    return np.array(features)


def extract_features_setC(
    edge1_matrix,
    edge2_matrix,
    source_bin,
    target_bin,
    source_indices,
    target_indices,
    n_bins=10
):
    """
    Extract feature set C: Set B + summary statistics.

    Features:
    - All features from Set B (204)
    - mean(intermediate_signature) (1 feature)
    - std(intermediate_signature) (1 feature)
    - min(intermediate_signature) (1 feature)
    - max(intermediate_signature) (1 feature)

    Total: 208 features

    Parameters
    ----------
    Same as extract_features_setA

    Returns
    -------
    np.ndarray
        Feature vector (208 dims)
    """
    # Get Set B features
    features_b = extract_features_setB(
        edge1_matrix, edge2_matrix,
        source_bin, target_bin,
        source_indices, target_indices,
        n_bins
    )

    features = list(features_b)

    # Get intermediate signature (features 2:102 from original Set A)
    signature = features_b[2:102]

    # Add summary statistics
    features.append(np.mean(signature))
    features.append(np.std(signature))
    features.append(np.min(signature))
    features.append(np.max(signature))

    return np.array(features)


def extract_features_setD(
    edge1_matrix,
    edge2_matrix,
    source_bin,
    target_bin,
    source_indices,
    target_indices,
    n_bins=10
):
    """
    Extract feature set D: Set C + neighbor context.

    Features:
    - All features from Set C (208)
    - source_nodes_count (1 feature)
    - target_nodes_count (1 feature)
    - source_edge_density (1 feature)
    - target_edge_density (1 feature)

    Total: 212 features

    Parameters
    ----------
    Same as extract_features_setA

    Returns
    -------
    np.ndarray
        Feature vector (212 dims)
    """
    # Get Set C features
    features_c = extract_features_setC(
        edge1_matrix, edge2_matrix,
        source_bin, target_bin,
        source_indices, target_indices,
        n_bins
    )

    features = list(features_c)

    # Add neighbor context
    n_source = len(source_indices)
    n_target = len(target_indices)
    features.append(n_source)
    features.append(n_target)

    # Edge densities
    if n_source > 0:
        source_edges = edge1_matrix[source_indices, :].sum()
        source_density = source_edges / (n_source * edge1_matrix.shape[1])
    else:
        source_density = 0.0

    if n_target > 0:
        target_edges = edge2_matrix[:, target_indices].sum()
        target_density = target_edges / (edge2_matrix.shape[0] * n_target)
    else:
        target_density = 0.0

    features.append(source_density)
    features.append(target_density)

    return np.array(features)


def extract_features_setE(
    edge1_matrix,
    edge2_matrix,
    source_bin,
    target_bin,
    source_indices,
    target_indices,
    n_bins=10
):
    """
    Extract feature set E: Set D + polynomial terms.

    Features:
    - All features from Set D (212)
    - source_bin^2 (1 feature)
    - target_bin^2 (1 feature)
    - sqrt(source_bin) (1 feature)
    - sqrt(target_bin) (1 feature)

    Total: 216 features

    Parameters
    ----------
    Same as extract_features_setA

    Returns
    -------
    np.ndarray
        Feature vector (216 dims)
    """
    # Get Set D features
    features_d = extract_features_setD(
        edge1_matrix, edge2_matrix,
        source_bin, target_bin,
        source_indices, target_indices,
        n_bins
    )

    features = list(features_d)

    # Add polynomial terms
    features.append(source_bin ** 2)
    features.append(target_bin ** 2)
    features.append(np.sqrt(source_bin + 1))
    features.append(np.sqrt(target_bin + 1))

    return np.array(features)


def extract_features_setF(
    edge1_matrix,
    edge2_matrix,
    source_bin,
    target_bin,
    source_indices,
    target_indices,
    n_bins=10
):
    """
    Extract feature set F: Set E + interaction terms.

    Features:
    - All features from Set E (216)
    - source_bin * target_bin (1 feature)
    - source_bin / (target_bin + 1) (1 feature)
    - target_bin / (source_bin + 1) (1 feature)

    Total: 219 features

    Parameters
    ----------
    Same as extract_features_setA

    Returns
    -------
    np.ndarray
        Feature vector (219 dims)
    """
    # Get Set E features
    features_e = extract_features_setE(
        edge1_matrix, edge2_matrix,
        source_bin, target_bin,
        source_indices, target_indices,
        n_bins
    )

    features = list(features_e)

    # Add interaction terms
    features.append(source_bin * target_bin)
    features.append(source_bin / (target_bin + 1))
    features.append(target_bin / (source_bin + 1))

    return np.array(features)


def extract_features_from_original(
    edge1_type,
    edge2_type,
    data_dir,
    n_bins=10,
    feature_set='A'
):
    """
    Extract features from original Hetionet graph.

    Parameters
    ----------
    edge1_type : str
        First edge type code (e.g., 'CbG')
    edge2_type : str
        Second edge type code (e.g., 'GpPW')
    data_dir : Path or str
        Path to data directory
    n_bins : int
        Number of bins for degree discretization
    feature_set : str
        Feature set to extract ('A', 'B', 'C', 'D', 'E', 'F')

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_bins*n_bins, n_features)
    y : np.ndarray
        Target values (bin-level mean pathway counts)
    metadata : dict
        Metadata about bins, edges, etc.
    """
    data_dir = Path(data_dir)

    # Load edge matrices
    edges_dir = data_dir / 'edges'
    edge1_file = edges_dir / f'{edge1_type}.sparse.npz'
    edge2_file = edges_dir / f'{edge2_type}.sparse.npz'

    edge1 = sp.load_npz(edge1_file)
    edge2 = sp.load_npz(edge2_file)

    # Compute pathway matrix
    pathway_matrix = edge1 @ edge2

    # Compute degrees
    source_degrees = np.array(edge1.sum(axis=1)).flatten()
    target_degrees = np.array(edge2.sum(axis=0)).flatten()

    # Compute bins
    source_bins, source_edges = compute_degree_bins(source_degrees, n_bins)
    target_bins, target_edges = compute_degree_bins(target_degrees, n_bins)

    # Extract features and targets for each bin pair
    X_list = []
    y_list = []
    metadata_list = []

    for src_bin in range(n_bins):
        src_mask = source_bins == src_bin
        src_indices = np.where(src_mask)[0]

        if len(src_indices) == 0:
            continue

        for tgt_bin in range(n_bins):
            tgt_mask = target_bins == tgt_bin
            tgt_indices = np.where(tgt_mask)[0]

            if len(tgt_indices) == 0:
                continue

            # Extract features based on feature_set
            if feature_set == 'A':
                features = extract_features_setA(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'B':
                features = extract_features_setB(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'C':
                features = extract_features_setC(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'D':
                features = extract_features_setD(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'E':
                features = extract_features_setE(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'F':
                features = extract_features_setF(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            else:
                raise ValueError(
                    f"Feature set must be 'A', 'B', 'C', 'D', 'E', or 'F', "
                    f"got '{feature_set}'"
                )

            # Compute target (mean pathway count for this bin)
            submatrix = pathway_matrix[np.ix_(src_indices, tgt_indices)]
            if isinstance(submatrix, sp.spmatrix):
                counts = submatrix.toarray().flatten()
            else:
                counts = submatrix.flatten()

            mean_count = counts.mean()

            X_list.append(features)
            y_list.append(mean_count)
            metadata_list.append({
                'source_bin': src_bin,
                'target_bin': tgt_bin,
                'n_pairs': len(counts),
                'n_source_nodes': len(src_indices),
                'n_target_nodes': len(tgt_indices)
            })

    X = np.array(X_list)
    y = np.array(y_list)

    metadata = {
        'edge1_type': edge1_type,
        'edge2_type': edge2_type,
        'n_bins': n_bins,
        'feature_set': feature_set,
        'n_features': X.shape[1] if len(X) > 0 else 0,
        'n_samples': len(X),
        'source_edges': source_edges,
        'target_edges': target_edges,
        'bin_metadata': metadata_list,
        'edge1_shape': edge1.shape,
        'edge2_shape': edge2.shape,
        'edge1_nnz': edge1.nnz,
        'edge2_nnz': edge2.nnz
    }

    return X, y, metadata


def extract_features_from_permutation(
    edge1_type,
    edge2_type,
    perm_id,
    data_dir,
    n_bins=10,
    feature_set='A'
):
    """
    Extract features from a single permutation.

    Parameters
    ----------
    edge1_type : str
        First edge type code (e.g., 'CbG')
    edge2_type : str
        Second edge type code (e.g., 'GpPW')
    perm_id : int
        Permutation ID (e.g., 0 for perm 000)
    data_dir : Path or str
        Path to data directory
    n_bins : int
        Number of bins for degree discretization
    feature_set : str
        Feature set to extract ('A', 'B', 'C', 'D', 'E', 'F')

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_bins*n_bins, n_features)
    y : np.ndarray
        Target values (bin-level mean pathway counts)
    metadata : dict
        Metadata about bins, edges, etc.
    """
    data_dir = Path(data_dir)

    # Load edge matrices from permutation
    perm_dir = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges'
    edge1_file = perm_dir / f'{edge1_type}.sparse.npz'
    edge2_file = perm_dir / f'{edge2_type}.sparse.npz'

    if not edge1_file.exists():
        raise FileNotFoundError(f"Edge1 file not found: {edge1_file}")
    if not edge2_file.exists():
        raise FileNotFoundError(f"Edge2 file not found: {edge2_file}")

    edge1 = sp.load_npz(edge1_file)
    edge2 = sp.load_npz(edge2_file)

    # Compute pathway matrix
    pathway_matrix = edge1 @ edge2

    # Compute degrees
    source_degrees = np.array(edge1.sum(axis=1)).flatten()
    target_degrees = np.array(edge2.sum(axis=0)).flatten()

    # Compute bins
    source_bins, source_edges = compute_degree_bins(source_degrees, n_bins)
    target_bins, target_edges = compute_degree_bins(target_degrees, n_bins)

    # Extract features and targets for each bin pair
    X_list = []
    y_list = []
    metadata_list = []

    for src_bin in range(n_bins):
        src_mask = source_bins == src_bin
        src_indices = np.where(src_mask)[0]

        if len(src_indices) == 0:
            continue

        for tgt_bin in range(n_bins):
            tgt_mask = target_bins == tgt_bin
            tgt_indices = np.where(tgt_mask)[0]

            if len(tgt_indices) == 0:
                continue

            # Extract features based on feature_set
            if feature_set == 'A':
                features = extract_features_setA(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'B':
                features = extract_features_setB(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'C':
                features = extract_features_setC(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'D':
                features = extract_features_setD(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'E':
                features = extract_features_setE(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            elif feature_set == 'F':
                features = extract_features_setF(
                    edge1, edge2,
                    src_bin, tgt_bin,
                    src_indices, tgt_indices,
                    n_bins
                )
            else:
                raise ValueError(
                    f"Feature set must be 'A', 'B', 'C', 'D', 'E', or 'F', "
                    f"got '{feature_set}'"
                )

            # Compute target (mean pathway count for this bin)
            submatrix = pathway_matrix[np.ix_(src_indices, tgt_indices)]
            if isinstance(submatrix, sp.spmatrix):
                counts = submatrix.toarray().flatten()
            else:
                counts = submatrix.flatten()

            mean_count = counts.mean()

            X_list.append(features)
            y_list.append(mean_count)
            metadata_list.append({
                'source_bin': src_bin,
                'target_bin': tgt_bin,
                'n_pairs': len(counts),
                'n_source_nodes': len(src_indices),
                'n_target_nodes': len(tgt_indices)
            })

    X = np.array(X_list)
    y = np.array(y_list)

    metadata = {
        'edge1_type': edge1_type,
        'edge2_type': edge2_type,
        'perm_id': perm_id,
        'n_bins': n_bins,
        'feature_set': feature_set,
        'n_features': X.shape[1] if len(X) > 0 else 0,
        'n_samples': len(X),
        'source_edges': source_edges,
        'target_edges': target_edges,
        'bin_metadata': metadata_list,
        'edge1_shape': edge1.shape,
        'edge2_shape': edge2.shape,
        'edge1_nnz': edge1.nnz,
        'edge2_nnz': edge2.nnz
    }

    return X, y, metadata
