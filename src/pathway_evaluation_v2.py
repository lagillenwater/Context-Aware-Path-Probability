"""
Evaluation utilities for pathway prediction models (v2).

This module provides evaluation functions to validate models by comparing
predictions to permutation averages.

Functions
---------
validate_on_permutations
    Validate model predictions against permutation average bins
compute_metrics
    Compute evaluation metrics (r, RMSE, MAE)
"""

import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from typing import Dict


def validate_on_permutations(
    model,
    edge1_type,
    edge2_type,
    perm_ids,
    data_dir,
    n_bins=10,
    device='cpu',
    feature_source='original',
    feature_perm_id=None,
    feature_set='A'
):
    """
    Validate model by predicting permutation average bins.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    edge1_type : str
        First edge type code
    edge2_type : str
        Second edge type code
    perm_ids : list of int
        Permutation IDs for validation
    data_dir : Path or str
        Data directory
    n_bins : int
        Number of bins
    device : str
        Device to use
    feature_source : str
        Source for feature extraction ('original' or 'permutation')
    feature_perm_id : int or None
        Permutation ID to extract features from (if feature_source='permutation')
    feature_set : str
        Feature set to extract ('A', 'B', 'C', 'D', 'E', 'F')

    Returns
    -------
    dict
        Validation metrics and predictions
    """
    from pathway_features_v2 import (
        extract_features_from_original,
        extract_features_from_permutation,
        compute_degree_bins
    )

    data_dir = Path(data_dir)
    model = model.to(device)
    model.eval()

    # Extract features from specified source
    if feature_source == 'original':
        X_features, _, metadata = extract_features_from_original(
            edge1_type, edge2_type, data_dir, n_bins, feature_set=feature_set
        )
    elif feature_source == 'permutation':
        if feature_perm_id is None:
            raise ValueError(
                "feature_perm_id must be specified when "
                "feature_source='permutation'"
            )
        X_features, _, metadata = extract_features_from_permutation(
            edge1_type, edge2_type, feature_perm_id,
            data_dir, n_bins, feature_set=feature_set
        )
    else:
        raise ValueError(
            f"feature_source must be 'original' or 'permutation', "
            f"got '{feature_source}'"
        )

    # Get bin structure from metadata
    bin_metadata = metadata['bin_metadata']

    # Compute permutation averages for each bin
    permutation_averages = []

    for perm_id in perm_ids:
        perm_dir = (data_dir / 'permutations' /
                    f'{perm_id:03d}.hetmat' / 'edges')
        edge1_file = perm_dir / f'{edge1_type}.sparse.npz'
        edge2_file = perm_dir / f'{edge2_type}.sparse.npz'

        if not edge1_file.exists() or not edge2_file.exists():
            continue

        edge1 = sp.load_npz(edge1_file)
        edge2 = sp.load_npz(edge2_file)
        pathway_matrix = edge1 @ edge2

        # Compute degrees
        source_degrees = np.array(edge1.sum(axis=1)).flatten()
        target_degrees = np.array(edge2.sum(axis=0)).flatten()

        # Compute bins
        source_bins, _ = compute_degree_bins(source_degrees, n_bins)
        target_bins, _ = compute_degree_bins(target_degrees, n_bins)

        # Extract pathway counts for each bin
        perm_counts = []
        for bin_meta in bin_metadata:
            src_bin = bin_meta['source_bin']
            tgt_bin = bin_meta['target_bin']

            src_indices = np.where(source_bins == src_bin)[0]
            tgt_indices = np.where(target_bins == tgt_bin)[0]

            if len(src_indices) == 0 or len(tgt_indices) == 0:
                perm_counts.append(0.0)
                continue

            submatrix = pathway_matrix[np.ix_(src_indices, tgt_indices)]
            if isinstance(submatrix, sp.spmatrix):
                counts = submatrix.toarray().flatten()
            else:
                counts = submatrix.flatten()

            perm_counts.append(counts.mean())

        permutation_averages.append(perm_counts)

    # Average across permutations
    permutation_averages = np.array(permutation_averages)
    y_true = permutation_averages.mean(axis=0)
    y_std = permutation_averages.std(axis=0)

    # Get model predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_features).to(device)
        y_pred = model(X_tensor).cpu().numpy().flatten()

    # Compute metrics
    r, p_value = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Per-bin errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    results = {
        'validation_r': r,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'mean_error': errors.mean(),
        'std_error': errors.std(),
        'mean_abs_error': abs_errors.mean(),
        'median_abs_error': np.median(abs_errors),
        'max_abs_error': abs_errors.max(),
        'q95_abs_error': np.percentile(abs_errors, 95),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_std': y_std,
        'n_permutations': len(permutation_averages),
        'n_bins': len(y_true),
        'bin_metadata': bin_metadata
    }

    return results


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Metrics dictionary
    """
    r, p_value = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'r': r,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae
    }
