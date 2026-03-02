"""
Permutation validation module for hypothesis testing.

This module tests the core hypothesis that training on the original Hetionet
graph can predict average pathway counts across degree-preserving permutations.

Functions
---------
extract_pathway_bins_from_graph
    Extract pathway counts by degree bin from a single graph
extract_pathway_bins_from_single_permutation
    Extract pathway counts from a single permutation
extract_pathway_bins_from_permutations
    Extract pathway counts averaged across permutations
test_original_vs_permutation_correlation
    Compute correlation between original and permutation averages
compute_degree_bins
    Compute degree bin assignments for nodes
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, List, Dict
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_degree_bins(degrees, n_bins=10):
    """
    Compute degree bin assignments using quantile-based binning.

    Parameters
    ----------
    degrees : np.ndarray
        Array of node degrees
    n_bins : int
        Number of bins (default: 10)

    Returns
    -------
    np.ndarray
        Bin assignments (0 to n_bins-1)
    np.ndarray
        Bin edges
    """
    if len(degrees) == 0:
        return np.array([]), np.array([])

    # Use quantile-based binning to ensure roughly equal samples per bin
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(degrees, quantiles)

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)

    # Assign bins (rightmost edge is inclusive)
    bins = np.digitize(degrees, bin_edges[1:], right=False)

    return bins, bin_edges


def extract_pathway_bins_from_graph(
    edge1_matrix,
    edge2_matrix,
    n_bins=10
):
    """
    Extract pathway counts by source/target degree bin from a single graph.

    For a 2-hop path (source -> intermediate -> target), this function:
    1. Computes the pathway matrix (edge1 @ edge2)
    2. Bins source and target nodes by their degrees
    3. Computes mean pathway count for each (source_bin, target_bin) pair

    Parameters
    ----------
    edge1_matrix : scipy.sparse matrix
        First edge type matrix (source -> intermediate)
    edge2_matrix : scipy.sparse matrix
        Second edge type matrix (intermediate -> target)
    n_bins : int
        Number of bins for degree discretization (default: 10)

    Returns
    -------
    pd.DataFrame
        Columns: source_bin, target_bin, pathway_count, n_pairs
    """
    # Compute pathway matrix
    pathway_matrix = edge1_matrix @ edge2_matrix

    # Compute degrees
    source_degrees = np.array(edge1_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge2_matrix.sum(axis=0)).flatten()

    # Compute bins
    source_bins, source_edges = compute_degree_bins(source_degrees, n_bins)
    target_bins, target_edges = compute_degree_bins(target_degrees, n_bins)

    # Extract pathway counts for each bin pair
    bin_data = []

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

            # Extract submatrix for this bin pair
            submatrix = pathway_matrix[np.ix_(src_indices, tgt_indices)]

            # Compute mean pathway count
            if isinstance(submatrix, sp.spmatrix):
                counts = submatrix.toarray().flatten()
            else:
                counts = submatrix.flatten()

            mean_count = counts.mean()
            n_pairs = len(counts)

            bin_data.append({
                'source_bin': src_bin,
                'target_bin': tgt_bin,
                'pathway_count': mean_count,
                'n_pairs': n_pairs
            })

    return pd.DataFrame(bin_data)


def extract_pathway_bins_from_single_permutation(
    edge1_type,
    edge2_type,
    perm_id,
    data_dir,
    n_bins=10
):
    """
    Extract pathway counts from a single permutation.

    Parameters
    ----------
    edge1_type : str
        First edge type code (e.g., 'CbG')
    edge2_type : str
        Second edge type code (e.g., 'GpPW')
    perm_id : int
        Permutation ID (e.g., 0 for perm 000)
    data_dir : Path or str
        Path to data directory containing permutations
    n_bins : int
        Number of bins for degree discretization (default: 10)

    Returns
    -------
    pd.DataFrame
        Columns: source_bin, target_bin, pathway_count, n_pairs
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

    # Extract bins using existing function
    return extract_pathway_bins_from_graph(edge1, edge2, n_bins)


def extract_pathway_bins_from_permutations(
    edge1_type,
    edge2_type,
    perm_ids,
    data_dir,
    n_bins=10
):
    """
    Extract pathway counts averaged across permutations.

    Parameters
    ----------
    edge1_type : str
        First edge type code (e.g., 'CbG')
    edge2_type : str
        Second edge type code (e.g., 'GpPW')
    perm_ids : list of int
        Permutation IDs to average over
    data_dir : Path or str
        Path to data directory containing permutations
    n_bins : int
        Number of bins for degree discretization (default: 10)

    Returns
    -------
    pd.DataFrame
        Columns: source_bin, target_bin, mean_pathway_count,
                 std_pathway_count, n_permutations
    """
    data_dir = Path(data_dir)

    # Collect pathway counts from each permutation
    all_perm_data = []

    for perm_id in perm_ids:
        # Load edge matrices
        perm_dir = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges'
        edge1_file = perm_dir / f'{edge1_type}.sparse.npz'
        edge2_file = perm_dir / f'{edge2_type}.sparse.npz'

        if not edge1_file.exists() or not edge2_file.exists():
            print(f"Warning: Missing files for perm {perm_id:03d}, skipping")
            continue

        edge1 = sp.load_npz(edge1_file)
        edge2 = sp.load_npz(edge2_file)

        # Extract bins for this permutation
        perm_bins = extract_pathway_bins_from_graph(edge1, edge2, n_bins)
        perm_bins['perm_id'] = perm_id

        all_perm_data.append(perm_bins)

    # Combine all permutations
    combined = pd.concat(all_perm_data, ignore_index=True)

    # Compute mean and std across permutations
    aggregated = combined.groupby(['source_bin', 'target_bin']).agg({
        'pathway_count': ['mean', 'std', 'count']
    }).reset_index()

    # Flatten column names
    aggregated.columns = [
        'source_bin', 'target_bin',
        'mean_pathway_count', 'std_pathway_count', 'n_permutations'
    ]

    return aggregated


def test_original_vs_permutation_correlation(
    original_bins,
    permutation_bins
):
    """
    Compute correlation between original and permutation average bins.

    Parameters
    ----------
    original_bins : pd.DataFrame
        Pathway counts from original graph
        Columns: source_bin, target_bin, pathway_count
    permutation_bins : pd.DataFrame
        Average pathway counts from permutations
        Columns: source_bin, target_bin, mean_pathway_count

    Returns
    -------
    dict
        {
            'correlation': float,
            'p_value': float,
            'rmse': float,
            'mae': float,
            'n_bins': int,
            'hypothesis_valid': bool (r > 0.85)
        }
    """
    # Merge on bin indices
    merged = original_bins.merge(
        permutation_bins,
        on=['source_bin', 'target_bin'],
        how='inner',
        suffixes=('_original', '_perm')
    )

    if len(merged) == 0:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'n_bins': 0,
            'hypothesis_valid': False,
            'error': 'No matching bins'
        }

    # Extract values
    original_vals = merged['pathway_count'].values
    perm_vals = merged['mean_pathway_count'].values

    # Compute correlation
    r, p_value = pearsonr(original_vals, perm_vals)

    # Compute error metrics
    rmse = np.sqrt(mean_squared_error(perm_vals, original_vals))
    mae = mean_absolute_error(perm_vals, original_vals)

    return {
        'correlation': r,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'n_bins': len(merged),
        'hypothesis_valid': r > 0.85,
        'merged_data': merged
    }


def analyze_residuals(original_bins, permutation_bins):
    """
    Analyze residuals between original and permutation predictions.

    Parameters
    ----------
    original_bins : pd.DataFrame
        Pathway counts from original graph
    permutation_bins : pd.DataFrame
        Average pathway counts from permutations

    Returns
    -------
    dict
        Residual statistics and patterns
    """
    # Merge on bin indices
    merged = original_bins.merge(
        permutation_bins,
        on=['source_bin', 'target_bin'],
        how='inner',
        suffixes=('_original', '_perm')
    )

    # Compute residuals
    merged['residual'] = (
        merged['pathway_count'] - merged['mean_pathway_count']
    )
    merged['abs_residual'] = merged['residual'].abs()
    merged['relative_residual'] = (
        merged['residual'] / (merged['mean_pathway_count'] + 1e-10)
    )

    # Compute statistics
    stats = {
        'mean_residual': merged['residual'].mean(),
        'std_residual': merged['residual'].std(),
        'mean_abs_residual': merged['abs_residual'].mean(),
        'median_abs_residual': merged['abs_residual'].median(),
        'max_abs_residual': merged['abs_residual'].max(),
        'q95_abs_residual': merged['abs_residual'].quantile(0.95),
        'mean_relative_residual': merged['relative_residual'].mean(),
    }

    # Identify bins with largest residuals
    top_residuals = merged.nlargest(10, 'abs_residual')[
        ['source_bin', 'target_bin', 'pathway_count',
         'mean_pathway_count', 'residual', 'abs_residual']
    ]

    return {
        'statistics': stats,
        'top_residuals': top_residuals,
        'merged_data': merged
    }
