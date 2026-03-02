"""
Enhanced feature extraction for pathway prediction.

Extracts comprehensive degree-based features from Hetionet graph structure.
All features are valid for null distribution learning (no node-specific IDs).

Features extracted (116 total):
- Endpoint features: source/target degree bins (2)
- Intermediate histogram: 10x10 degree distribution (100)
- Summary statistics: mean, std, min, max, median (5)
- Log transforms: log(source_deg), log(target_deg) (2)
- Neighbor context: 2nd-order degree statistics (2)
- Polynomial terms: degree^2 (2)
- Interaction terms: degree products (3)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional
import hetmatpy.hetmat


def compute_mean_from_histogram(hist_2d: np.ndarray,
                                  bin_edges_x: np.ndarray,
                                  bin_edges_y: np.ndarray) -> float:
    """
    Compute mean from 2D histogram.

    Parameters:
    - hist_2d: 2D histogram array (e.g., 10x10)
    - bin_edges_x: Bin edges for x-axis (in_degree)
    - bin_edges_y: Bin edges for y-axis (out_degree)

    Returns:
    - mean_degree: Weighted mean degree
    """
    bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
    bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

    total_count = hist_2d.sum()
    if total_count == 0:
        return 0.0

    mean_x = np.sum(hist_2d.sum(axis=1) * bin_centers_x) / total_count
    mean_y = np.sum(hist_2d.sum(axis=0) * bin_centers_y) / total_count

    return (mean_x + mean_y) / 2


def compute_std_from_histogram(hist_2d: np.ndarray,
                                 bin_edges_x: np.ndarray,
                                 bin_edges_y: np.ndarray) -> float:
    """Compute standard deviation from 2D histogram."""
    bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
    bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

    total_count = hist_2d.sum()
    if total_count == 0:
        return 0.0

    mean_x = np.sum(hist_2d.sum(axis=1) * bin_centers_x) / total_count
    mean_y = np.sum(hist_2d.sum(axis=0) * bin_centers_y) / total_count

    var_x = np.sum(hist_2d.sum(axis=1) * (bin_centers_x - mean_x)**2) / total_count
    var_y = np.sum(hist_2d.sum(axis=0) * (bin_centers_y - mean_y)**2) / total_count

    return np.sqrt((var_x + var_y) / 2)


def compute_min_from_histogram(bin_edges_x: np.ndarray,
                                 bin_edges_y: np.ndarray) -> float:
    """Min degree is the start of first bin."""
    return min(bin_edges_x[0], bin_edges_y[0])


def compute_max_from_histogram(bin_edges_x: np.ndarray,
                                 bin_edges_y: np.ndarray) -> float:
    """Max degree is the end of last bin."""
    return max(bin_edges_x[-1], bin_edges_y[-1])


def compute_median_from_histogram(hist_2d: np.ndarray,
                                    bin_edges_x: np.ndarray,
                                    bin_edges_y: np.ndarray) -> float:
    """
    Approximate median from 2D histogram.

    Uses cumulative distribution to find 50th percentile.
    """
    flat_hist = hist_2d.flatten()
    all_bin_centers = []

    for i, edge_x in enumerate(bin_edges_x[:-1]):
        for j, edge_y in enumerate(bin_edges_y[:-1]):
            center_x = (bin_edges_x[i] + bin_edges_x[i+1]) / 2
            center_y = (bin_edges_y[j] + bin_edges_y[j+1]) / 2
            mean_center = (center_x + center_y) / 2
            all_bin_centers.append(mean_center)

    all_bin_centers = np.array(all_bin_centers)
    total_count = flat_hist.sum()

    if total_count == 0:
        return 0.0

    sorted_indices = np.argsort(all_bin_centers)
    cumsum = np.cumsum(flat_hist[sorted_indices])
    median_idx = np.searchsorted(cumsum, total_count / 2)

    return all_bin_centers[sorted_indices[median_idx]]


def compute_intermediate_signature(source_nodes: np.ndarray,
                                     target_nodes: np.ndarray,
                                     edge1_matrix: sp.spmatrix,
                                     edge2_matrix: sp.spmatrix,
                                     n_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    Compute intermediate node degree signature for 2-hop paths.

    Parameters:
    - source_nodes: Array of source node indices
    - target_nodes: Array of target node indices
    - edge1_matrix: Sparse adjacency matrix for first edge type
    - edge2_matrix: Sparse adjacency matrix for second edge type
    - n_bins: Number of bins for histogram (default 10)

    Returns:
    - signature: Dict with 'histogram' (100-dim), 'bin_edges_in', 'bin_edges_out'
    """
    all_intermediate_in_degs = []
    all_intermediate_out_degs = []

    for source_idx in source_nodes:
        intermediates = edge1_matrix[source_idx].nonzero()[1]

        if len(intermediates) == 0:
            continue

        in_degrees = np.array(edge1_matrix[:, intermediates].sum(axis=0)).flatten()
        out_degrees = np.array(edge2_matrix[intermediates, :].sum(axis=1)).flatten()

        all_intermediate_in_degs.extend(in_degrees)
        all_intermediate_out_degs.extend(out_degrees)

    if len(all_intermediate_in_degs) == 0:
        return {
            'histogram': np.zeros(n_bins * n_bins),
            'bin_edges_in': np.linspace(0, 1, n_bins + 1),
            'bin_edges_out': np.linspace(0, 1, n_bins + 1)
        }

    all_intermediate_in_degs = np.array(all_intermediate_in_degs)
    all_intermediate_out_degs = np.array(all_intermediate_out_degs)

    hist_2d, bin_edges_in, bin_edges_out = np.histogram2d(
        all_intermediate_in_degs,
        all_intermediate_out_degs,
        bins=n_bins
    )

    return {
        'histogram': hist_2d.flatten(),
        'bin_edges_in': bin_edges_in,
        'bin_edges_out': bin_edges_out
    }


def extract_enhanced_features(source_nodes: np.ndarray,
                                target_nodes: np.ndarray,
                                edge1_matrix: sp.spmatrix,
                                edge2_matrix: sp.spmatrix,
                                n_bins: int = 10,
                                feature_set: str = 'F') -> np.ndarray:
    """
    Extract enhanced features for pathway prediction.

    Parameters:
    - source_nodes: Array of source node indices
    - target_nodes: Array of target node indices
    - edge1_matrix: Sparse adjacency for first edge
    - edge2_matrix: Sparse adjacency for second edge
    - n_bins: Histogram bins (default 10)
    - feature_set: Which feature set to extract ('A', 'B', 'C', 'D', 'E', 'F')

    Returns:
    - features: Array of shape (n_samples, n_features)
      - Set A: 102 features (baseline)
      - Set B: 104 features (+ log transforms)
      - Set C: 109 features (+ summary stats)
      - Set D: 111 features (+ neighbor context)
      - Set E: 113 features (+ polynomial)
      - Set F: 116 features (+ interactions)
    """
    n_samples = len(source_nodes)

    source_degrees = np.array(edge1_matrix.sum(axis=1)).flatten()[source_nodes]
    target_degrees = np.array(edge2_matrix.sum(axis=0)).flatten()[target_nodes]

    source_bins = pd.qcut(source_degrees, q=n_bins, labels=False,
                           duplicates='drop')
    target_bins = pd.qcut(target_degrees, q=n_bins, labels=False,
                           duplicates='drop')

    sig = compute_intermediate_signature(source_nodes, target_nodes,
                                          edge1_matrix, edge2_matrix, n_bins)

    histogram = sig['histogram']
    bin_edges_in = sig['bin_edges_in']
    bin_edges_out = sig['bin_edges_out']

    hist_2d = histogram.reshape(n_bins, n_bins)

    feature_dict = {
        'source_bin': source_bins,
        'target_bin': target_bins,
        'histogram': np.tile(histogram, (n_samples, 1))
    }

    if feature_set in ['B', 'C', 'D', 'E', 'F']:
        feature_dict['log_source_deg'] = np.log1p(source_degrees)
        feature_dict['log_target_deg'] = np.log1p(target_degrees)

    if feature_set in ['C', 'D', 'E', 'F']:
        mean_deg = compute_mean_from_histogram(hist_2d, bin_edges_in, bin_edges_out)
        std_deg = compute_std_from_histogram(hist_2d, bin_edges_in, bin_edges_out)
        min_deg = compute_min_from_histogram(bin_edges_in, bin_edges_out)
        max_deg = compute_max_from_histogram(bin_edges_in, bin_edges_out)
        median_deg = compute_median_from_histogram(hist_2d, bin_edges_in, bin_edges_out)

        feature_dict['mean_int_deg'] = np.full(n_samples, mean_deg)
        feature_dict['std_int_deg'] = np.full(n_samples, std_deg)
        feature_dict['min_int_deg'] = np.full(n_samples, min_deg)
        feature_dict['max_int_deg'] = np.full(n_samples, max_deg)
        feature_dict['median_int_deg'] = np.full(n_samples, median_deg)

    if feature_set in ['D', 'E', 'F']:
        neighbor_in_degs = []
        neighbor_out_degs = []

        for source_idx in source_nodes:
            intermediates = edge1_matrix[source_idx].nonzero()[1]
            if len(intermediates) > 0:
                for inter_idx in intermediates:
                    neighbors_of_inter = edge2_matrix[inter_idx].nonzero()[1]
                    neighbor_degs = np.array(edge2_matrix[:, neighbors_of_inter].sum(axis=0)).flatten()
                    neighbor_in_degs.extend(neighbor_degs)

        if len(neighbor_in_degs) > 0:
            neighbor_mean = np.mean(neighbor_in_degs)
            neighbor_std = np.std(neighbor_in_degs)
        else:
            neighbor_mean = 0.0
            neighbor_std = 0.0

        feature_dict['neighbor_deg_mean'] = np.full(n_samples, neighbor_mean)
        feature_dict['neighbor_deg_std'] = np.full(n_samples, neighbor_std)

    if feature_set in ['E', 'F']:
        feature_dict['source_deg_sq'] = source_degrees ** 2
        feature_dict['target_deg_sq'] = target_degrees ** 2

    if feature_set == 'F':
        feature_dict['source_x_target'] = source_degrees * target_degrees

        if 'mean_int_deg' in feature_dict:
            mean_int = feature_dict['mean_int_deg'][0]
            feature_dict['source_x_mean_int'] = source_degrees * mean_int
            feature_dict['target_x_mean_int'] = target_degrees * mean_int
        else:
            feature_dict['source_x_mean_int'] = np.zeros(n_samples)
            feature_dict['target_x_mean_int'] = np.zeros(n_samples)

    feature_list = []
    for key in ['source_bin', 'target_bin']:
        feature_list.append(feature_dict[key].reshape(-1, 1))

    feature_list.append(feature_dict['histogram'])

    for key in ['log_source_deg', 'log_target_deg']:
        if key in feature_dict:
            feature_list.append(feature_dict[key].reshape(-1, 1))

    for key in ['mean_int_deg', 'std_int_deg', 'min_int_deg', 'max_int_deg', 'median_int_deg']:
        if key in feature_dict:
            feature_list.append(feature_dict[key].reshape(-1, 1))

    for key in ['neighbor_deg_mean', 'neighbor_deg_std']:
        if key in feature_dict:
            feature_list.append(feature_dict[key].reshape(-1, 1))

    for key in ['source_deg_sq', 'target_deg_sq']:
        if key in feature_dict:
            feature_list.append(feature_dict[key].reshape(-1, 1))

    for key in ['source_x_target', 'source_x_mean_int', 'target_x_mean_int']:
        if key in feature_dict:
            feature_list.append(feature_dict[key].reshape(-1, 1))

    features = np.hstack(feature_list)

    return features


def get_feature_names(feature_set: str = 'F', n_bins: int = 10) -> List[str]:
    """
    Get feature names for a given feature set.

    Parameters:
    - feature_set: Which set ('A', 'B', 'C', 'D', 'E', 'F')
    - n_bins: Number of histogram bins

    Returns:
    - names: List of feature names
    """
    names = ['source_bin', 'target_bin']

    for i in range(n_bins):
        for j in range(n_bins):
            names.append(f'hist_in{i}_out{j}')

    if feature_set in ['B', 'C', 'D', 'E', 'F']:
        names.extend(['log_source_deg', 'log_target_deg'])

    if feature_set in ['C', 'D', 'E', 'F']:
        names.extend(['mean_int_deg', 'std_int_deg', 'min_int_deg',
                      'max_int_deg', 'median_int_deg'])

    if feature_set in ['D', 'E', 'F']:
        names.extend(['neighbor_deg_mean', 'neighbor_deg_std'])

    if feature_set in ['E', 'F']:
        names.extend(['source_deg_sq', 'target_deg_sq'])

    if feature_set == 'F':
        names.extend(['source_x_target', 'source_x_mean_int', 'target_x_mean_int'])

    return names
