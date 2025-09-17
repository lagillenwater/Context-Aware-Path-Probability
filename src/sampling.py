"""
Representative Sampling Functions for Edge Prediction

This module provides functions for creating representative datasets with stratified
positive sampling and degree-matched negative sampling to reduce coefficient of 
variation and improve model stability.

MIGRATED FROM: notebooks/3_learn_null_edge.ipynb
This code was previously defined inline in the notebook and has been moved here
for better modularity, reusability, and maintainability.

Functions:
    - stratified_positive_sampling: Sample positive edges while preserving degree distribution
    - representative_negative_sampling: Generate negative edges that match positive edge characteristics  
    - create_representative_dataset: Complete pipeline for representative dataset creation

Usage:
    from src.sampling import create_representative_dataset
    
    X, y, report = create_representative_dataset(
        edges_matrix=your_matrix,
        degrees=your_degrees,  
        n_positive=1000,
        n_negative=1000,
        random_state=42
    )
"""

import numpy as np
import scipy.sparse
from typing import Dict, Tuple, Union, Optional


def stratified_positive_sampling(edges_matrix, degrees, n_samples, random_state=42):
    """
    Perform stratified sampling of positive edges based on degree distributions.
    
    Parameters:
    -----------
    edges_matrix : scipy.sparse matrix
        The adjacency matrix containing positive edges
    degrees : dict
        Dictionary with 'source' and 'target' degree arrays
    n_samples : int
        Number of positive edges to sample
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    sampled_edges : np.ndarray
        Array of sampled edge indices [source_idx, target_idx]
    weights : np.ndarray
        Sampling weights for each edge
    """
    np.random.seed(random_state)
    
    # Get all existing edges
    source_indices, target_indices = edges_matrix.nonzero()
    
    if len(source_indices) == 0:
        return np.array([]), np.array([])
    
    # Calculate source and target degrees for all edges
    edge_source_degrees = degrees['source'][source_indices]
    edge_target_degrees = degrees['target'][target_indices]
    
    # Create combined degree bins for stratification
    source_percentiles = np.percentile(edge_source_degrees, [20, 40, 60, 80])
    target_percentiles = np.percentile(edge_target_degrees, [20, 40, 60, 80])
    
    # Assign each edge to a stratum based on source and target degree percentiles
    source_bins = np.digitize(edge_source_degrees, source_percentiles)
    target_bins = np.digitize(edge_target_degrees, target_percentiles)
    
    # Create combined strata (source_bin, target_bin)
    strata = source_bins * 10 + target_bins
    unique_strata = np.unique(strata)
    
    print(f"Stratified sampling: Found {len(unique_strata)} degree-based strata")
    
    # Sample proportionally from each stratum
    sampled_indices = []
    
    for stratum in unique_strata:
        stratum_mask = (strata == stratum)
        stratum_indices = np.where(stratum_mask)[0]
        
        if len(stratum_indices) == 0:
            continue
            
        # Calculate proportion of total edges in this stratum
        stratum_size = len(stratum_indices)
        total_edges = len(source_indices)
        stratum_proportion = stratum_size / total_edges
        
        # Sample proportionally from this stratum
        stratum_sample_size = max(1, int(n_samples * stratum_proportion))
        stratum_sample_size = min(stratum_sample_size, len(stratum_indices))
        
        selected_in_stratum = np.random.choice(
            stratum_indices, stratum_sample_size, replace=False
        )
        sampled_indices.extend(selected_in_stratum)
    
    # If we haven't reached the target sample size, fill randomly
    if len(sampled_indices) < n_samples:
        remaining_needed = n_samples - len(sampled_indices)
        remaining_indices = np.setdiff1d(np.arange(len(source_indices)), sampled_indices)
        
        if len(remaining_indices) > 0:
            additional_samples = np.random.choice(
                remaining_indices, 
                min(remaining_needed, len(remaining_indices)), 
                replace=False
            )
            sampled_indices.extend(additional_samples)
    
    # If we have too many samples, randomly subsample
    if len(sampled_indices) > n_samples:
        sampled_indices = np.random.choice(sampled_indices, n_samples, replace=False)
    
    sampled_indices = np.array(sampled_indices)
    
    # Get the actual edge pairs
    sampled_edges = np.column_stack([
        source_indices[sampled_indices], 
        target_indices[sampled_indices]
    ])
    
    # Equal weights for all samples (could be modified for weighted sampling)
    weights = np.ones(len(sampled_edges))
    
    print(f"Sampled {len(sampled_edges)} positive edges using stratified sampling")
    
    return sampled_edges, weights


def representative_negative_sampling(edges_matrix, degrees, positive_edges, n_samples, method='degree_matched', random_state=42, verbose=True):
    """
    Generate representative negative edges that match positive edge characteristics.
    
    Parameters:
    -----------
    edges_matrix : scipy.sparse matrix
        The adjacency matrix (to avoid existing edges)
    degrees : dict
        Dictionary with 'source' and 'target' degree arrays
    positive_edges : np.ndarray
        Array of positive edges for matching characteristics
    n_samples : int
        Number of negative edges to generate
    method : str
        Sampling method: 'degree_matched', 'distribution_matched', or 'hybrid'
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    negative_edges : np.ndarray
        Array of generated negative edges [source_idx, target_idx]
    sampling_info : dict
        Information about the sampling process
    """
    np.random.seed(random_state)
    
    # Convert sparse matrix to set for faster lookup
    existing_edges = set(zip(*edges_matrix.nonzero()))
    
    # Get matrix dimensions
    n_source_nodes, n_target_nodes = edges_matrix.shape
    
    # Analyze positive edge characteristics
    pos_source_degrees = degrees['source'][positive_edges[:, 0]]
    pos_target_degrees = degrees['target'][positive_edges[:, 1]]
    
    if verbose:
        print(f"\nGenerating {n_samples} negative edges using {method} method...")
        print(f"Positive edge degree stats - Source: {pos_source_degrees.mean():.1f}±{pos_source_degrees.std():.1f}")
        print(f"Positive edge degree stats - Target: {pos_target_degrees.mean():.1f}±{pos_target_degrees.std():.1f}")
    
    negative_edges = []
    attempts = 0
    max_attempts = min(n_samples * 2, 50000)  # Much more aggressive limit for speed
    
    if method == 'degree_matched':
        # FAST BATCH APPROACH: Generate many candidates at once instead of one-by-one
        if verbose:
            print("Using fast batch sampling approach...")
        
        # Create degree-based probability weights for faster sampling
        source_probs = degrees['source'] / degrees['source'].sum()
        target_probs = degrees['target'] / degrees['target'].sum()
        
        # Generate candidates in large batches
        batch_size = min(n_samples * 10, 100000)  # Generate 10x candidates per batch
        while len(negative_edges) < n_samples and attempts < max_attempts:
            # Sample source and target nodes based on degree distributions
            source_batch = np.random.choice(n_source_nodes, size=batch_size, p=source_probs)
            target_batch = np.random.choice(n_target_nodes, size=batch_size, p=target_probs)
            
            # Create candidate edges
            candidate_edges = list(zip(source_batch, target_batch))
            
            # Filter out existing edges (vectorized check)
            valid_candidates = []
            for edge in candidate_edges:
                if edge not in existing_edges:
                    valid_candidates.append(list(edge))
                    if len(valid_candidates) + len(negative_edges) >= n_samples:
                        break
            
            # Add valid candidates to negative edges
            negative_edges.extend(valid_candidates)
            attempts += batch_size
            
            if verbose and len(negative_edges) % 10000 == 0:
                print(f"Generated {len(negative_edges)} negative edges so far...")
        
        # Trim to exact number needed
        negative_edges = negative_edges[:n_samples]
    
    # Convert to numpy array
    negative_edges = np.array(negative_edges)
    
    # Generate sampling info
    if len(negative_edges) > 0:
        neg_source_degrees = degrees['source'][negative_edges[:, 0]]
        neg_target_degrees = degrees['target'][negative_edges[:, 1]]
        
        sampling_info = {
            'method': method,
            'samples_generated': len(negative_edges),
            'attempts_made': attempts,
            'success_rate': len(negative_edges) / attempts if attempts > 0 else 0,
            'source_degree_stats': {
                'mean': neg_source_degrees.mean(),
                'std': neg_source_degrees.std(),
                'match_quality': 1 - abs(neg_source_degrees.mean() - pos_source_degrees.mean()) / pos_source_degrees.mean()
            },
            'target_degree_stats': {
                'mean': neg_target_degrees.mean(),
                'std': neg_target_degrees.std(),
                'match_quality': 1 - abs(neg_target_degrees.mean() - pos_target_degrees.mean()) / pos_target_degrees.mean()
            }
        }
    else:
        sampling_info = {
            'method': method,
            'samples_generated': 0,
            'attempts_made': attempts,
            'success_rate': 0,
            'source_degree_stats': None,
            'target_degree_stats': None
        }
    
    if verbose:
        print(f"Generated {len(negative_edges)} negative edges (success rate: {sampling_info['success_rate']:.3f})")
    
    return negative_edges, sampling_info


def create_representative_dataset(edges_matrix, degrees, n_positive, n_negative, pos_method='stratified', neg_method='degree_matched', random_state=42, balance_strategy='keep_all'):
    """
    Create a representative dataset with both positive and negative edges.
    
    Parameters:
    -----------
    edges_matrix : scipy.sparse matrix
        The adjacency matrix
    degrees : dict
        Dictionary with 'source' and 'target' degree arrays
    n_positive : int
        Number of positive edges to sample
    n_negative : int
        Number of negative edges to generate
    pos_method : str
        Method for positive sampling: 'stratified' or 'random'
    neg_method : str
        Method for negative sampling: 'degree_matched', 'distribution_matched', or 'hybrid'
    random_state : int
        Random seed for reproducibility
    balance_strategy : str
        Strategy for balancing dataset sizes:
        - 'keep_all': Keep all samples (may be imbalanced)
        - 'downsample_positive': Downsample positive edges to match negative count
        
    Returns:
    --------
    X : np.ndarray
        Feature matrix (source_degree, target_degree)
    y : np.ndarray
        Labels (1 for positive, 0 for negative)
    sampling_report : dict
        Detailed information about the sampling process
    """
    print(f"\nCreating representative dataset: {n_positive} positive + {n_negative} negative edges")
    print(f"Positive method: {pos_method}, Negative method: {neg_method}")
    
    # Sample positive edges
    if pos_method == 'stratified':
        positive_edges, pos_weights = stratified_positive_sampling(
            edges_matrix, degrees, n_positive, random_state
        )
    else:  # random sampling
        source_indices, target_indices = edges_matrix.nonzero()
        total_edges = len(source_indices)
        selected_idx = np.random.choice(total_edges, min(n_positive, total_edges), replace=False)
        positive_edges = np.column_stack([source_indices[selected_idx], target_indices[selected_idx]])
        pos_weights = np.ones(len(positive_edges))
    
    # Generate negative edges
    negative_edges, neg_info = representative_negative_sampling(
        edges_matrix, degrees, positive_edges, n_negative, neg_method, random_state, verbose=True
    )
    
    # Apply balancing strategy
    if balance_strategy == 'downsample_positive' and len(negative_edges) < len(positive_edges):
        print(f"\nApplying positive downsampling: {len(positive_edges)} -> {len(negative_edges)} positive edges")
        np.random.seed(random_state)
        downsample_idx = np.random.choice(len(positive_edges), len(negative_edges), replace=False)
        positive_edges = positive_edges[downsample_idx]
        if 'pos_weights' in locals():
            pos_weights = pos_weights[downsample_idx]
        print(f"Downsampled to {len(positive_edges)} positive edges to match {len(negative_edges)} negative edges")
    
    # Combine edges and create features
    all_edges = np.vstack([positive_edges, negative_edges])
    labels = np.hstack([np.ones(len(positive_edges)), np.zeros(len(negative_edges))])
    
    # Create features (source degree, target degree)
    source_degrees = degrees['source'][all_edges[:, 0]]
    target_degrees = degrees['target'][all_edges[:, 1]]
    features = np.column_stack([source_degrees, target_degrees])
    
    # Create sampling report
    pos_source_degs = degrees['source'][positive_edges[:, 0]]
    pos_target_degs = degrees['target'][positive_edges[:, 1]]
    
    if len(negative_edges) > 0:
        neg_source_degs = degrees['source'][negative_edges[:, 0]]
        neg_target_degs = degrees['target'][negative_edges[:, 1]]
        
        # Calculate degree balance quality
        source_balance = 1 - abs(neg_source_degs.mean() - pos_source_degs.mean()) / max(pos_source_degs.mean(), 1)
        target_balance = 1 - abs(neg_target_degs.mean() - pos_target_degs.mean()) / max(pos_target_degs.mean(), 1)
        degree_balance_quality = (source_balance + target_balance) / 2
    else:
        neg_source_degs = np.array([])
        neg_target_degs = np.array([])
        degree_balance_quality = 0
    
    sampling_report = {
        'positive_sampling': {
            'method': pos_method,
            'samples': len(positive_edges),
            'source_degree_stats': {
                'mean': pos_source_degs.mean(),
                'std': pos_source_degs.std(),
                'min': pos_source_degs.min(),
                'max': pos_source_degs.max()
            },
            'target_degree_stats': {
                'mean': pos_target_degs.mean(),
                'std': pos_target_degs.std(),
                'min': pos_target_degs.min(),
                'max': pos_target_degs.max()
            }
        },
        'negative_sampling': {
            'method': neg_method,
            'samples': len(negative_edges),
            'sampling_info': neg_info,
            'source_degree_stats': {
                'mean': neg_source_degs.mean() if len(neg_source_degs) > 0 else 0,
                'std': neg_source_degs.std() if len(neg_source_degs) > 0 else 0,
                'min': neg_source_degs.min() if len(neg_source_degs) > 0 else 0,
                'max': neg_source_degs.max() if len(neg_source_degs) > 0 else 0
            },
            'target_degree_stats': {
                'mean': neg_target_degs.mean() if len(neg_target_degs) > 0 else 0,
                'std': neg_target_degs.std() if len(neg_target_degs) > 0 else 0,
                'min': neg_target_degs.min() if len(neg_target_degs) > 0 else 0,
                'max': neg_target_degs.max() if len(neg_target_degs) > 0 else 0
            }
        },
        'sampling_quality': {
            'degree_balance_quality': degree_balance_quality
        },
        'dataset_stats': {
            'total_samples': len(all_edges),
            'positive_ratio': len(positive_edges) / len(all_edges),
            'feature_correlation': np.corrcoef(source_degrees, target_degrees)[0, 1]
        },
        'balancing': {
            'strategy': balance_strategy,
            'final_positive_count': len(positive_edges),
            'final_negative_count': len(negative_edges),
            'is_balanced': len(positive_edges) == len(negative_edges)
        }
    }
    
    print(f"\nDataset created successfully:")
    print(f"  Total samples: {len(all_edges)}")
    print(f"  Positive: {len(positive_edges)}, Negative: {len(negative_edges)}")
    print(f"  Feature correlation: {sampling_report['dataset_stats']['feature_correlation']:.3f}")
    
    return features, labels, sampling_report
