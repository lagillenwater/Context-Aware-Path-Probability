"""
Data Processing Utilities for DWPC Analysis

This module contains functions for preparing and processing data for neural network training,
including edge prediction data preparation and negative sampling.
"""

import numpy as np
import pandas as pd
import scipy.sparse
import pathlib


def prepare_edge_prediction_data(permutation_data, sample_negative_ratio=1.0):
    """
    Prepare training data for edge prediction based on source and target degrees.
    
    Parameters:
    -----------
    permutation_data : dict
        Dictionary containing source nodes, target nodes, and edge matrix
    sample_negative_ratio : float
        Ratio of negative to positive examples to sample
    
    Returns:
    --------
    features : numpy.ndarray
        Feature matrix with source and target degrees
    labels : numpy.ndarray
        Binary labels (1 for existing edges, 0 for non-existing)
    """
    # Use new parameterized keys, fall back to legacy keys for backwards compatibility
    source_nodes = permutation_data.get('source_nodes', permutation_data.get('anatomy_nodes'))
    target_nodes = permutation_data.get('target_nodes', permutation_data.get('gene_nodes'))
    edges = permutation_data.get('edges', permutation_data.get('aeg_edges'))
    
    # Get edge type information for logging
    edge_type = permutation_data.get('edge_type', 'AeG')
    source_type = permutation_data.get('source_node_type', 'Anatomy')
    target_type = permutation_data.get('target_node_type', 'Gene')
    
    print(f"Preparing {edge_type} edge prediction data ({source_type} -> {target_type})")
    
    # Get node degrees
    source_degrees = np.array(edges.sum(axis=1)).flatten()
    target_degrees = np.array(edges.sum(axis=0)).flatten()
    
    print(f"{source_type} degree range: {source_degrees.min()} - {source_degrees.max()}")
    print(f"{target_type} degree range: {target_degrees.min()} - {target_degrees.max()}")
    
    # Prepare positive examples (existing edges)
    rows, cols = edges.nonzero()
    positive_features = []
    positive_labels = []
    
    for source_idx, target_idx in zip(rows, cols):
        positive_features.append([source_degrees[source_idx], target_degrees[target_idx]])
        positive_labels.append(1)
    
    positive_features = np.array(positive_features)
    positive_labels = np.array(positive_labels)
    
    print(f"Number of positive examples (existing edges): {len(positive_labels)}")
    
    # Prepare negative examples (non-existing edges)
    num_negative = int(len(positive_labels) * sample_negative_ratio)
    negative_features = []
    negative_labels = []
    
    attempts = 0
    max_attempts = num_negative * 10  # Prevent infinite loops
    
    while len(negative_labels) < num_negative and attempts < max_attempts:
        # Random sample of source and target indices
        source_idx = np.random.randint(0, len(source_nodes))
        target_idx = np.random.randint(0, len(target_nodes))
        
        # Check if this pair doesn't have an edge
        if edges[source_idx, target_idx] == 0:
            negative_features.append([source_degrees[source_idx], target_degrees[target_idx]])
            negative_labels.append(0)
        
        attempts += 1
    
    negative_features = np.array(negative_features)
    negative_labels = np.array(negative_labels)
    
    print(f"Number of negative examples (non-existing edges): {len(negative_labels)}")
    
    # Combine positive and negative examples
    all_features = np.vstack([positive_features, negative_features])
    all_labels = np.concatenate([positive_labels, negative_labels])
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(all_labels))
    all_features = all_features[shuffle_idx]
    all_labels = all_labels[shuffle_idx]
    
    return all_features, all_labels


def load_permutation_data(permutation_name, permutations_dir, edge_type="AeG", source_node_type="Anatomy", target_node_type="Gene"):
    """
    Load edge matrix and node metadata for a specific permutation.
    
    Parameters:
    -----------
    permutation_name : str
        Name of the permutation directory (e.g., '001.hetmat')
    permutations_dir : pathlib.Path
        Path to the permutations directory
    edge_type : str
        Type of edge to load (e.g., 'AeG', 'CbG', 'DaG', etc.)
    source_node_type : str
        Type of source nodes (e.g., 'Anatomy', 'Compound', 'Disease', etc.)
    target_node_type : str
        Type of target nodes (e.g., 'Gene', 'Anatomy', 'Disease', etc.)
    
    Returns:
    --------
    dict : Dictionary containing loaded data with keys:
           - 'edges': scipy sparse matrix for specified edge type
           - 'source_nodes': pandas DataFrame of source nodes
           - 'target_nodes': pandas DataFrame of target nodes
           - 'permutation_path': pathlib.Path to the permutation directory
           - 'edge_type': str indicating the edge type loaded
           - 'source_node_type': str indicating source node type
           - 'target_node_type': str indicating target node type
    """
    # Set up paths for this permutation
    perm_dir = permutations_dir / permutation_name
    edges_dir = perm_dir / 'edges'
    nodes_dir = perm_dir / 'nodes'
    
    if not perm_dir.exists():
        raise FileNotFoundError(f"Permutation directory not found: {perm_dir}")
    
    print(f"Loading data from permutation: {permutation_name}")
    print(f"Permutation path: {perm_dir}")
    print(f"Edge type: {edge_type} ({source_node_type} -> {target_node_type})")
    
    # Load specified edge matrix
    edge_path = edges_dir / f'{edge_type}.sparse.npz'
    if not edge_path.exists():
        raise FileNotFoundError(f"{edge_type} edges file not found: {edge_path}")
    
    edges = scipy.sparse.load_npz(edge_path)
    print(f"Loaded {edge_type} edges: {edges.shape} matrix with {edges.nnz} non-zero entries")
    
    # Load source nodes
    source_path = nodes_dir / f'{source_node_type}.tsv'
    if not source_path.exists():
        raise FileNotFoundError(f"{source_node_type} nodes file not found: {source_path}")
    
    source_nodes = pd.read_csv(source_path, sep='\t')
    print(f"Loaded {source_node_type} nodes: {len(source_nodes)} nodes")
    print(f"{source_node_type} columns: {list(source_nodes.columns)}")
    
    # Load target nodes
    target_path = nodes_dir / f'{target_node_type}.tsv'
    if not target_path.exists():
        raise FileNotFoundError(f"{target_node_type} nodes file not found: {target_path}")
    
    target_nodes = pd.read_csv(target_path, sep='\t')
    print(f"Loaded {target_node_type} nodes: {len(target_nodes)} nodes")
    print(f"{target_node_type} columns: {list(target_nodes.columns)}")
    
    return {
        'edges': edges,
        'source_nodes': source_nodes,
        'target_nodes': target_nodes,
        'permutation_path': perm_dir,
        'edge_type': edge_type,
        'source_node_type': source_node_type,
        'target_node_type': target_node_type,
        # Backwards compatibility aliases
        'aeg_edges': edges,
        'anatomy_nodes': source_nodes,
        'gene_nodes': target_nodes
    }


def load_all_permutations(available_permutations, permutations_dir):
    """
    Load AeG edges, Anatomy nodes, and Gene nodes from all available permutations.
    
    Parameters:
    -----------
    available_permutations : list
        List of permutation directory names
    permutations_dir : pathlib.Path
        Path to the permutations directory
    
    Returns:
    --------
    dict : Dictionary with permutation names as keys and loaded data as values
    """
    all_permutations = {}
    
    for perm_name in available_permutations:
        try:
            print(f"\nLoading permutation: {perm_name}")
            perm_data = load_permutation_data(perm_name, permutations_dir)
            all_permutations[perm_name] = perm_data
            print(f"✓ Successfully loaded {perm_name}")
        except Exception as e:
            print(f"✗ Failed to load {perm_name}: {e}")
    
    return all_permutations
