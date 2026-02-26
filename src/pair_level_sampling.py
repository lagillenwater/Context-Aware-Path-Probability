#!/usr/bin/env python3
"""
Sampling strategies for pair-level pathway null prediction.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple


def sample_pairs_random(
    edge1: sp.csr_matrix,
    edge2: sp.csr_matrix,
    n_samples: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample random (source, target) pairs.

    Parameters
    ----------
    edge1 : sp.csr_matrix
        First edge matrix
    edge2 : sp.csr_matrix
        Second edge matrix
    n_samples : int
        Number of pairs to sample
    random_state : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of (source_idx, target_idx) pairs, shape (n_samples, 2)
    """
    rng = np.random.RandomState(random_state)

    n_source_nodes = edge1.shape[0]
    n_target_nodes = edge2.shape[1]

    source_indices = rng.randint(0, n_source_nodes, size=n_samples)
    target_indices = rng.randint(0, n_target_nodes, size=n_samples)

    return np.column_stack([source_indices, target_indices])


def sample_pairs_stratified_by_pathway_count(
    edge1: sp.csr_matrix,
    edge2: sp.csr_matrix,
    n_samples: int,
    pathway_matrix: sp.csr_matrix = None,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample pairs stratified by pathway count.

    Sample equally from:
    - Pairs with 0 pathways (no connection)
    - Pairs with low pathway counts (1-10 paths)
    - Pairs with medium pathway counts (11-100 paths)
    - Pairs with high pathway counts (>100 paths)

    Parameters
    ----------
    edge1 : sp.csr_matrix
        First edge matrix
    edge2 : sp.csr_matrix
        Second edge matrix
    n_samples : int
        Total number of pairs to sample
    pathway_matrix : sp.csr_matrix, optional
        Precomputed pathway matrix. If None, computed from edge1 @ edge2
    random_state : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of (source_idx, target_idx) pairs, shape (n_samples, 2)
    """
    rng = np.random.RandomState(random_state)

    if pathway_matrix is None:
        pathway_matrix = edge1 @ edge2

    n_source_nodes = edge1.shape[0]
    n_target_nodes = edge2.shape[1]

    # Define strata
    strata = []

    # Stratum 1: Zero pathways (sample from all pairs, exclude nonzero)
    nonzero_pairs = set(zip(*pathway_matrix.nonzero()))
    n_stratum_samples = n_samples // 4

    zero_pairs = []
    attempts = 0
    max_attempts = n_stratum_samples * 10
    while len(zero_pairs) < n_stratum_samples and attempts < max_attempts:
        src = rng.randint(0, n_source_nodes)
        tgt = rng.randint(0, n_target_nodes)
        if (src, tgt) not in nonzero_pairs:
            zero_pairs.append([src, tgt])
        attempts += 1

    strata.append(np.array(zero_pairs))

    # Stratum 2: Low pathway counts (1-10)
    pathway_coo = pathway_matrix.tocoo()
    low_mask = (pathway_coo.data > 0) & (pathway_coo.data <= 10)
    low_pairs = np.column_stack([pathway_coo.row[low_mask], pathway_coo.col[low_mask]])
    if len(low_pairs) > n_stratum_samples:
        low_pairs = low_pairs[rng.choice(len(low_pairs), n_stratum_samples, replace=False)]
    strata.append(low_pairs)

    # Stratum 3: Medium pathway counts (11-100)
    med_mask = (pathway_coo.data > 10) & (pathway_coo.data <= 100)
    med_pairs = np.column_stack([pathway_coo.row[med_mask], pathway_coo.col[med_mask]])
    if len(med_pairs) > n_stratum_samples:
        med_pairs = med_pairs[rng.choice(len(med_pairs), n_stratum_samples, replace=False)]
    strata.append(med_pairs)

    # Stratum 4: High pathway counts (>100)
    high_mask = pathway_coo.data > 100
    high_pairs = np.column_stack([pathway_coo.row[high_mask], pathway_coo.col[high_mask]])
    if len(high_pairs) > n_stratum_samples:
        high_pairs = high_pairs[rng.choice(len(high_pairs), n_stratum_samples, replace=False)]
    strata.append(high_pairs)

    # Combine all strata
    all_pairs = np.vstack([s for s in strata if len(s) > 0])

    # Shuffle
    rng.shuffle(all_pairs)

    return all_pairs[:n_samples]


def sample_pairs_stratified_by_degree(
    edge1: sp.csr_matrix,
    edge2: sp.csr_matrix,
    n_samples: int,
    n_degree_bins: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample pairs stratified by source and target degree.

    Sample equally from degree bins to ensure coverage of degree space.

    Parameters
    ----------
    edge1 : sp.csr_matrix
        First edge matrix
    edge2 : sp.csr_matrix
        Second edge matrix
    n_samples : int
        Total number of pairs to sample
    n_degree_bins : int
        Number of bins for each degree dimension
    random_state : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of (source_idx, target_idx) pairs, shape (n_samples, 2)
    """
    rng = np.random.RandomState(random_state)

    # Compute degrees
    source_degrees = np.array(edge1.sum(axis=1)).flatten()
    target_degrees = np.array(edge2.sum(axis=0)).flatten()

    # Bin degrees
    source_bins = np.searchsorted(
        np.percentile(source_degrees[source_degrees > 0],
                      np.linspace(0, 100, n_degree_bins + 1)[1:-1]),
        source_degrees
    )
    target_bins = np.searchsorted(
        np.percentile(target_degrees[target_degrees > 0],
                      np.linspace(0, 100, n_degree_bins + 1)[1:-1]),
        target_degrees
    )

    # Sample from each bin pair
    n_bin_pairs = n_degree_bins * n_degree_bins
    samples_per_bin = max(1, n_samples // n_bin_pairs)

    all_pairs = []
    for src_bin in range(n_degree_bins):
        src_indices = np.where(source_bins == src_bin)[0]
        if len(src_indices) == 0:
            continue

        for tgt_bin in range(n_degree_bins):
            tgt_indices = np.where(target_bins == tgt_bin)[0]
            if len(tgt_indices) == 0:
                continue

            # Sample pairs from this bin
            for _ in range(samples_per_bin):
                src = rng.choice(src_indices)
                tgt = rng.choice(tgt_indices)
                all_pairs.append([src, tgt])

    pairs = np.array(all_pairs)
    rng.shuffle(pairs)

    return pairs[:n_samples]


def load_and_sample_pairs(
    edge1_type: str,
    edge2_type: str,
    data_dir: str,
    n_samples: int,
    strategy: str = 'random',
    random_state: int = 42
) -> np.ndarray:
    """
    Load edges and sample pairs using specified strategy.

    Parameters
    ----------
    edge1_type : str
        First edge type
    edge2_type : str
        Second edge type
    data_dir : str
        Data directory
    n_samples : int
        Number of pairs to sample
    strategy : str
        Sampling strategy: 'random', 'pathway_stratified', 'degree_stratified'
    random_state : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of (source_idx, target_idx) pairs
    """
    data_dir = Path(data_dir)
    edges_dir = data_dir / 'edges'

    edge1 = sp.load_npz(edges_dir / f'{edge1_type}.sparse.npz')
    edge2 = sp.load_npz(edges_dir / f'{edge2_type}.sparse.npz')

    if strategy == 'random':
        return sample_pairs_random(edge1, edge2, n_samples, random_state)
    elif strategy == 'pathway_stratified':
        return sample_pairs_stratified_by_pathway_count(
            edge1, edge2, n_samples, None, random_state
        )
    elif strategy == 'degree_stratified':
        return sample_pairs_stratified_by_degree(
            edge1, edge2, n_samples, 5, random_state
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
