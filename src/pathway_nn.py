"""
Pathway Neural Network Module

Core classes and utilities for direct pathway count prediction.

Classes:
    PathwayNullPredictor: PyTorch neural network for pathway count prediction
    PerformanceTracker: Runtime and memory monitoring

Functions:
    compute_intermediate_degree_dist: Calculate intermediate node degree histogram
    stratified_sample_pairs: Sample pairs stratified by degree bins
    load_pathway_data: Load training data for a metapath
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.sparse as sp
import time
import psutil
import os
from pathlib import Path
from typing import Tuple, List, Dict


class PathwayNullPredictor(nn.Module):
    """
    Neural network for direct pathway count prediction.

    Predicts pathway counts from node degrees without assuming edge independence.

    Architecture:
        - Degree encoder: [source_deg, target_deg] → 64 features
        - Intermediate encoder: [inter_deg_histogram] → 64 features
        - Predictor: 128 features → pathway count (positive)

    Args:
        n_intermediate_bins: Number of bins for intermediate degree histogram
        dropout: Dropout probability for regularization
    """

    def __init__(self, n_intermediate_bins=20, dropout=0.2):
        super().__init__()

        # Encode source/target degrees
        self.degree_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Encode intermediate degree distribution
        self.intermediate_encoder = nn.Sequential(
            nn.Linear(n_intermediate_bins, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Combine and predict pathway counts
        self.predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensures positive outputs
        )

    def forward(self, source_deg, target_deg, inter_deg_dist):
        """
        Forward pass.

        Args:
            source_deg: Source node degrees (batch_size, 1)
            target_deg: Target node degrees (batch_size, 1)
            inter_deg_dist: Intermediate degree histogram (batch_size, n_bins)

        Returns:
            Predicted pathway counts (batch_size, 1)
        """
        # Encode degrees
        deg_features = self.degree_encoder(
            torch.cat([source_deg, target_deg], dim=1)
        )

        # Encode intermediate distribution
        inter_features = self.intermediate_encoder(inter_deg_dist)

        # Combine and predict
        combined = torch.cat([deg_features, inter_features], dim=1)
        return self.predictor(combined)


class PerformanceTracker:
    """
    Track runtime and memory usage for model training/inference.

    Usage:
        tracker = PerformanceTracker("ModelName")
        tracker.start()
        # ... train or predict ...
        elapsed, peak_memory = tracker.stop()
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.start_time = None
        self.peak_memory = 0
        self.process = psutil.Process(os.getpid())

    def start(self):
        """Start tracking."""
        self.start_time = time.time()
        self.peak_memory = self.process.memory_info().rss / 1024**3  # GB

    def stop(self) -> Tuple[float, float]:
        """
        Stop tracking and return results.

        Returns:
            (elapsed_time_sec, peak_memory_gb)
        """
        elapsed = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024**3
        self.peak_memory = max(self.peak_memory, current_memory)
        return elapsed, self.peak_memory


def compute_intermediate_degree_dist(
    edge1_matrix: sp.spmatrix,
    edge2_matrix: sp.spmatrix,
    source_idx: int,
    target_idx: int,
    n_bins: int = 20
) -> np.ndarray:
    """
    Compute intermediate node degree distribution for a pathway.

    For a 2-edge metapath (source → intermediate → target):
    - Find all intermediate nodes k where edge1[source, k] > 0 AND edge2[k, target] > 0
    - Compute histogram of degrees of these intermediate nodes

    Args:
        edge1_matrix: First edge matrix (sparse)
        edge2_matrix: Second edge matrix (sparse)
        source_idx: Source node index
        target_idx: Target node index
        n_bins: Number of histogram bins

    Returns:
        Normalized histogram (n_bins,)
    """
    # Get intermediate nodes connected to source
    edge1_row = edge1_matrix.getrow(source_idx)
    intermediate_from_source = edge1_row.nonzero()[1]

    if len(intermediate_from_source) == 0:
        return np.zeros(n_bins)

    # Get intermediate nodes connected to target
    edge2_col = edge2_matrix.getcol(target_idx)
    intermediate_to_target = edge2_col.nonzero()[0]

    if len(intermediate_to_target) == 0:
        return np.zeros(n_bins)

    # Find intersection (nodes in pathway)
    intermediate_in_pathway = np.intersect1d(
        intermediate_from_source,
        intermediate_to_target
    )

    if len(intermediate_in_pathway) == 0:
        return np.zeros(n_bins)

    # Compute degrees of intermediate nodes
    # Degree = out-degree from edge1 + in-degree from edge2
    degrees = (
        np.asarray(edge1_matrix[intermediate_in_pathway, :].sum(axis=1)).ravel() +
        np.asarray(edge2_matrix[:, intermediate_in_pathway].sum(axis=0)).ravel()
    )

    # Create histogram
    hist, _ = np.histogram(degrees, bins=n_bins, range=(0, degrees.max() + 1))

    # Normalize
    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist.astype(np.float32)


def stratified_sample_pairs(
    pathway_matrix: sp.spmatrix,
    source_degrees: np.ndarray,
    target_degrees: np.ndarray,
    n_samples: int,
    n_degree_bins: int = 10,
    random_state: int = 42
) -> List[Tuple[int, int]]:
    """
    Sample pairs from pathway matrix, stratified by source/target degree bins.

    Ensures coverage of degree space by sampling proportionally from each
    (source_degree_bin, target_degree_bin) combination.

    Args:
        pathway_matrix: Sparse pathway count matrix
        source_degrees: Degree of each source node
        target_degrees: Degree of each target node
        n_samples: Target number of samples
        n_degree_bins: Number of degree bins for stratification
        random_state: Random seed

    Returns:
        List of (source_idx, target_idx) tuples
    """
    np.random.seed(random_state)

    # Get all non-zero pathway locations
    pathway_coo = pathway_matrix.tocoo()
    all_pairs = list(zip(pathway_coo.row, pathway_coo.col))

    if len(all_pairs) == 0:
        return []

    # Create degree bins
    src_bins = pd.qcut(
        source_degrees[source_degrees > 0],
        q=n_degree_bins,
        labels=False,
        duplicates='drop'
    )
    tgt_bins = pd.qcut(
        target_degrees[target_degrees > 0],
        q=n_degree_bins,
        labels=False,
        duplicates='drop'
    )

    # Assign pairs to bins
    pairs_by_bin = {}
    for i, j in all_pairs:
        src_deg = source_degrees[i]
        tgt_deg = target_degrees[j]

        if src_deg == 0 or tgt_deg == 0:
            continue

        src_bin = np.searchsorted(src_bins, src_deg)
        tgt_bin = np.searchsorted(tgt_bins, tgt_deg)

        bin_key = (src_bin, tgt_bin)
        if bin_key not in pairs_by_bin:
            pairs_by_bin[bin_key] = []
        pairs_by_bin[bin_key].append((i, j))

    # Sample proportionally from each bin
    sampled_pairs = []
    samples_per_bin = max(1, n_samples // len(pairs_by_bin))

    for bin_key, pairs_in_bin in pairs_by_bin.items():
        n_sample = min(samples_per_bin, len(pairs_in_bin))
        sampled = np.random.choice(
            len(pairs_in_bin),
            size=n_sample,
            replace=False
        )
        sampled_pairs.extend([pairs_in_bin[idx] for idx in sampled])

    # If we sampled too few, add random samples
    if len(sampled_pairs) < n_samples:
        remaining = n_samples - len(sampled_pairs)
        extra = np.random.choice(
            len(all_pairs),
            size=min(remaining, len(all_pairs)),
            replace=False
        )
        sampled_pairs.extend([all_pairs[idx] for idx in extra])

    return sampled_pairs[:n_samples]


def load_pathway_data(
    metapath: str,
    results_dir: Path,
    train: bool = True
) -> pd.DataFrame:
    """
    Load training or validation data for a metapath.

    Args:
        metapath: Metapath abbreviation (e.g., 'CbGaD')
        results_dir: Base results directory
        train: If True, load training data; else validation data

    Returns:
        DataFrame with columns: source_deg, target_deg, inter_deg_bin_*, pathway_count
    """
    data_type = 'train' if train else 'validation'
    data_file = results_dir / 'training_data' / f'{metapath}_{data_type}.csv.gz'

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    return pd.read_csv(data_file, compression='gzip')
