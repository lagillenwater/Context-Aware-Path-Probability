"""
Intermediate Node Degree Signature Computation

This module computes degree signatures for intermediate nodes in metapaths.
Instead of tracking individual node pairs, we compute histograms of intermediate
node (in_degree, out_degree) joint distributions, aggregated by source-target
degree bins.

Key concepts:
- For a metapath like Compound -> Gene -> Disease (CbGaD):
  - Source: Compound nodes
  - Intermediate: Gene nodes
  - Target: Disease nodes

- For each (source_degree_bin, target_degree_bin) pair:
  - Collect all Gene nodes that participate in paths between those bins
  - Compute histogram of (Gene_in_degree, Gene_out_degree)
  - This becomes a fixed-length feature vector (e.g., 100-dim)

This allows models to learn how intermediate node degree structure affects
path probabilities without storing individual node IDs.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class IntermediateSignature:
    """
    Container for intermediate node degree signature.

    Attributes:
        source_bin: Source node degree bin index
        target_bin: Target node degree bin index
        in_degree_bins: Bin edges for intermediate in-degrees
        out_degree_bins: Bin edges for intermediate out-degrees
        histogram: 2D histogram of (in_degree, out_degree) counts
        n_paths: Total number of paths in this (source_bin, target_bin) pair
        n_intermediate_nodes: Number of unique intermediate nodes
    """
    source_bin: int
    target_bin: int
    in_degree_bins: np.ndarray
    out_degree_bins: np.ndarray
    histogram: np.ndarray
    n_paths: int
    n_intermediate_nodes: int

    def flatten(self) -> np.ndarray:
        """
        Flatten histogram to 1D feature vector.

        Returns:
            1D array of histogram values (row-major order)
        """
        return self.histogram.flatten()

    def normalize(self) -> np.ndarray:
        """
        Return normalized histogram (sums to 1).

        Returns:
            Normalized flattened histogram
        """
        flat = self.flatten()
        total = flat.sum()
        if total > 0:
            return flat / total
        return flat


def create_degree_bins(degrees: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Create quantile-based degree bins.

    Args:
        degrees: Array of node degrees
        n_bins: Number of bins to create

    Returns:
        Array of bin edges (length n_bins + 1), guaranteed to have exactly
        n_bins bins even if some percentiles coincide
    """
    # Filter out zero degrees
    nonzero_degrees = degrees[degrees > 0]

    if len(nonzero_degrees) == 0:
        return np.array([0, 1])

    # Create quantile-based bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(nonzero_degrees, percentiles)

    # CRITICAL: Ensure first bin starts at 0 to include zero-degree nodes
    # Do this BEFORE epsilon adjustment
    if bins[0] > 0:
        bins[0] = 0

    # CRITICAL: Ensure exactly n_bins bins by making duplicate edges unique
    # Add small epsilon to ensure strictly increasing sequence
    for i in range(1, len(bins)):
        if bins[i] <= bins[i-1]:
            bins[i] = bins[i-1] + 1e-10

    return bins


def assign_to_bins(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Assign values to bins.

    Args:
        values: Array of values to bin
        bins: Bin edges

    Returns:
        Array of bin indices (0 to len(bins)-2)
    """
    return np.digitize(values, bins, right=False) - 1


def compute_intermediate_signature(
    edge1_matrix: sp.spmatrix,
    edge2_matrix: sp.spmatrix,
    source_degrees: np.ndarray,
    target_degrees: np.ndarray,
    source_bins: np.ndarray,
    target_bins: np.ndarray,
    n_intermediate_bins: int = 10
) -> Dict[Tuple[int, int], IntermediateSignature]:
    """
    Compute intermediate node degree signatures for all degree bin pairs.

    For a metapath with edges:
        source --edge1--> intermediate --edge2--> target

    This function:
    1. Groups source-target pairs by their degree bins
    2. For each bin pair, identifies all intermediate nodes in paths
    3. Computes 2D histogram of intermediate (in_degree, out_degree)

    Args:
        edge1_matrix: Source x Intermediate adjacency matrix
        edge2_matrix: Intermediate x Target adjacency matrix
        source_degrees: Degree of each source node
        target_degrees: Degree of each target node
        source_bins: Bin edges for source degrees
        target_bins: Bin edges for target degrees
        n_intermediate_bins: Number of bins for intermediate degree histogram

    Returns:
        Dictionary mapping (source_bin, target_bin) -> IntermediateSignature
    """
    # Compute intermediate node degrees
    intermediate_in_degrees = np.asarray(edge1_matrix.sum(axis=0)).ravel()
    intermediate_out_degrees = np.asarray(edge2_matrix.sum(axis=1)).ravel()

    # Create bins for intermediate degrees
    in_degree_bin_edges = create_degree_bins(intermediate_in_degrees, n_intermediate_bins)
    out_degree_bin_edges = create_degree_bins(intermediate_out_degrees, n_intermediate_bins)

    # Convert matrices to COO for efficient iteration
    edge1_coo = edge1_matrix.tocoo()
    edge2_coo = edge2_matrix.tocoo()

    # Build intermediate -> sources and intermediate -> targets mappings
    intermediate_to_sources = {}  # intermediate_idx -> set of source indices
    intermediate_to_targets = {}  # intermediate_idx -> set of target indices

    for src, interm in zip(edge1_coo.row, edge1_coo.col):
        if interm not in intermediate_to_sources:
            intermediate_to_sources[interm] = set()
        intermediate_to_sources[interm].add(src)

    for interm, tgt in zip(edge2_coo.row, edge2_coo.col):
        if interm not in intermediate_to_targets:
            intermediate_to_targets[interm] = set()
        intermediate_to_targets[interm].add(tgt)

    # Assign source and target nodes to degree bins
    source_bin_assignments = assign_to_bins(source_degrees, source_bins)
    target_bin_assignments = assign_to_bins(target_degrees, target_bins)

    # Initialize result dictionary
    signatures = {}

    n_source_bins = len(source_bins) - 1
    n_target_bins = len(target_bins) - 1

    # Process each (source_bin, target_bin) pair
    for src_bin in range(n_source_bins):
        for tgt_bin in range(n_target_bins):
            # Find all intermediate nodes that connect this bin pair
            intermediate_nodes_in_paths = set()
            path_count = 0

            # Iterate through intermediate nodes
            for interm_idx in set(intermediate_to_sources.keys()) & set(intermediate_to_targets.keys()):
                sources = intermediate_to_sources[interm_idx]
                targets = intermediate_to_targets[interm_idx]

                # Check if this intermediate connects source_bin to target_bin
                sources_in_bin = [s for s in sources if source_bin_assignments[s] == src_bin]
                targets_in_bin = [t for t in targets if target_bin_assignments[t] == tgt_bin]

                if sources_in_bin and targets_in_bin:
                    intermediate_nodes_in_paths.add(interm_idx)
                    # Count paths through this intermediate
                    path_count += len(sources_in_bin) * len(targets_in_bin)

            # Compute histogram of intermediate degrees
            if intermediate_nodes_in_paths:
                interm_list = list(intermediate_nodes_in_paths)
                in_degs = intermediate_in_degrees[interm_list]
                out_degs = intermediate_out_degrees[interm_list]

                # Create 2D histogram
                histogram, _, _ = np.histogram2d(
                    in_degs,
                    out_degs,
                    bins=[in_degree_bin_edges, out_degree_bin_edges]
                )
            else:
                # No paths for this bin pair - create empty histogram
                histogram = np.zeros((len(in_degree_bin_edges) - 1,
                                     len(out_degree_bin_edges) - 1))

            # Store signature
            signatures[(src_bin, tgt_bin)] = IntermediateSignature(
                source_bin=src_bin,
                target_bin=tgt_bin,
                in_degree_bins=in_degree_bin_edges,
                out_degree_bins=out_degree_bin_edges,
                histogram=histogram,
                n_paths=path_count,
                n_intermediate_nodes=len(intermediate_nodes_in_paths)
            )

    return signatures


def extract_training_features(
    signatures: Dict[Tuple[int, int], IntermediateSignature],
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix from signatures for model training.

    Args:
        signatures: Dictionary of signatures from compute_intermediate_signature
        normalize: If True, normalize histograms to sum to 1

    Returns:
        Tuple of (X, bin_pairs):
            X: Feature matrix (n_bin_pairs, n_histogram_features)
            bin_pairs: Array of (source_bin, target_bin) for each row
    """
    bin_pairs = sorted(signatures.keys())

    if normalize:
        features = [signatures[bp].normalize() for bp in bin_pairs]
    else:
        features = [signatures[bp].flatten() for bp in bin_pairs]

    X = np.vstack(features)
    bin_pairs_array = np.array(bin_pairs)

    return X, bin_pairs_array


def get_signature_stats(
    signatures: Dict[Tuple[int, int], IntermediateSignature]
) -> Dict[str, any]:
    """
    Compute summary statistics for signatures.

    Args:
        signatures: Dictionary of signatures

    Returns:
        Dictionary of statistics
    """
    total_paths = sum(sig.n_paths for sig in signatures.values())
    total_intermediate = sum(sig.n_intermediate_nodes for sig in signatures.values())

    n_bins_with_paths = sum(1 for sig in signatures.values() if sig.n_paths > 0)
    n_total_bins = len(signatures)

    histogram_shape = next(iter(signatures.values())).histogram.shape

    return {
        'n_degree_bin_pairs': n_total_bins,
        'n_bins_with_paths': n_bins_with_paths,
        'sparsity': 1.0 - (n_bins_with_paths / n_total_bins),
        'total_paths': total_paths,
        'total_intermediate_nodes': total_intermediate,
        'histogram_shape': histogram_shape,
        'feature_dimension': histogram_shape[0] * histogram_shape[1]
    }


if __name__ == "__main__":
    # Example usage
    print("Intermediate Signature Module")
    print("=" * 70)
    print()
    print("This module computes degree signatures for intermediate nodes")
    print("in metapaths. It is designed to work with:")
    print()
    print("  - Sparse adjacency matrices (scipy.sparse)")
    print("  - Degree-binned aggregation for memory efficiency")
    print("  - Training models on original graph structure")
    print()
    print("See docstrings for detailed usage examples.")
