"""
Sampling module for DWPC P-Value Validation

Stratified sampling of node pairs by degree category.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

from . import config
from .data_loading import get_loader
from .utils import setup_logging

logger = setup_logging(__name__)


def compute_degree_bins(
    degrees: np.ndarray,
    quantiles: List[float]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute degree bins from quantiles.

    Parameters
    ----------
    degrees : np.ndarray
        Array of node degrees.
    quantiles : list of float
        Quantile boundaries (e.g., [0.0, 0.33, 0.67, 1.0]).

    Returns
    -------
    bins : np.ndarray
        Bin edges computed from quantiles.
    labels : list of str
        Bin labels (e.g., ['Low', 'Medium', 'High']).
    """
    bins = np.quantile(degrees, quantiles)
    labels = config.DEGREE_CATEGORY_NAMES[:len(quantiles) - 1]

    logger.debug(
        f"Computed degree bins: {bins} with labels {labels}"
    )

    return bins, labels


def categorize_node_pairs(
    source_degrees: np.ndarray,
    target_degrees: np.ndarray,
    source_bins: np.ndarray,
    target_bins: np.ndarray,
    category_names: List[str]
) -> List[Tuple[str, str]]:
    """
    Categorize node pairs by degree.

    Parameters
    ----------
    source_degrees : np.ndarray
        Source node degrees.
    target_degrees : np.ndarray
        Target node degrees.
    source_bins : np.ndarray
        Bin edges for source degrees.
    target_bins : np.ndarray
        Bin edges for target degrees.
    category_names : list of str
        Category names (e.g., ['Low', 'Medium', 'High']).

    Returns
    -------
    categories : list of tuple
        List of (source_category, target_category) for each pair.
    """
    # Digitize degrees into bins
    source_cats = np.digitize(source_degrees, source_bins, right=False) - 1
    target_cats = np.digitize(target_degrees, target_bins, right=False) - 1

    # Clip to valid range
    source_cats = np.clip(source_cats, 0, len(category_names) - 1)
    target_cats = np.clip(target_cats, 0, len(category_names) - 1)

    # Convert to category names
    categories = [
        (category_names[src_cat], category_names[tgt_cat])
        for src_cat, tgt_cat in zip(source_cats, target_cats)
    ]

    return categories


def sample_node_pairs(
    metaedge: str,
    source: str = "true",
    n_samples_per_category: int = 100,
    quantiles: Optional[List[float]] = None,
    category_names: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> List[Dict]:
    """
    Sample node pairs stratified by degree category.

    Parameters
    ----------
    metaedge : str
        Metaedge abbreviation (e.g., 'CbG').
    source : str
        Source of data: 'true', 'perm0', or 'permX'.
    n_samples_per_category : int
        Number of samples per degree category combination.
    quantiles : list of float, optional
        Quantile boundaries. Defaults to config.DEGREE_QUANTILES.
    category_names : list of str, optional
        Category names. Defaults to config.DEGREE_CATEGORY_NAMES.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    samples : list of dict
        List of sampled node pairs with metadata:
        - source_idx: Source node index
        - target_idx: Target node index
        - source_degree: Source node degree
        - target_degree: Target node degree
        - category: (source_category, target_category)
    """
    if quantiles is None:
        quantiles = config.DEGREE_QUANTILES
    if category_names is None:
        category_names = config.DEGREE_CATEGORY_NAMES

    rng = np.random.RandomState(random_state)
    loader = get_loader()

    # Get connected pairs
    source_nodes, target_nodes = loader.get_connected_node_pairs(metaedge, source)

    # Get degrees
    source_degrees = loader.get_node_degrees(metaedge, source, "source")
    target_degrees = loader.get_node_degrees(metaedge, source, "target")

    # Get degrees for connected pairs
    pair_source_degrees = source_degrees[source_nodes]
    pair_target_degrees = target_degrees[target_nodes]

    # Compute bins
    source_bins, _ = compute_degree_bins(source_degrees[source_degrees > 0], quantiles)
    target_bins, _ = compute_degree_bins(target_degrees[target_degrees > 0], quantiles)

    # Categorize pairs
    categories = categorize_node_pairs(
        pair_source_degrees,
        pair_target_degrees,
        source_bins,
        target_bins,
        category_names
    )

    # Group pairs by category
    category_pairs = defaultdict(list)
    for i, cat in enumerate(categories):
        category_pairs[cat].append(i)

    # Sample from each category
    samples = []
    for cat, indices in category_pairs.items():
        # Sample with replacement if needed
        n_available = len(indices)
        n_to_sample = min(n_samples_per_category, n_available)

        if n_to_sample < n_samples_per_category:
            logger.warning(
                f"{source} {metaedge} category {cat}: "
                f"Only {n_available} pairs available, requested {n_samples_per_category}"
            )

        sampled_indices = rng.choice(indices, size=n_to_sample, replace=False)

        for idx in sampled_indices:
            samples.append({
                'source_idx': int(source_nodes[idx]),
                'target_idx': int(target_nodes[idx]),
                'source_degree': int(pair_source_degrees[idx]),
                'target_degree': int(pair_target_degrees[idx]),
                'category': cat
            })

    logger.info(
        f"{source} {metaedge}: Sampled {len(samples)} pairs "
        f"across {len(category_pairs)} categories"
    )

    return samples


def sample_metapath_pairs(
    metapath: str,
    source: str = "true",
    n_samples_per_category: int = 100,
    quantiles: Optional[List[float]] = None,
    category_names: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> List[Dict]:
    """
    Sample source-target pairs for a metapath.

    For a metapath like CbGpPW (Compound->Gene->Pathway), this samples
    Compound-Pathway pairs stratified by the degree of the terminal nodes.

    Parameters
    ----------
    metapath : str
        Metapath abbreviation (e.g., 'CbGpPW').
    source : str
        Source of data: 'true', 'perm0', or 'permX'.
    n_samples_per_category : int
        Number of samples per degree category combination.
    quantiles : list of float, optional
        Quantile boundaries.
    category_names : list of str, optional
        Category names.
    random_state : int, optional
        Random seed.

    Returns
    -------
    samples : list of dict
        List of sampled source-target pairs with metadata:
        - source_idx: Source node index (first node in metapath)
        - target_idx: Target node index (last node in metapath)
        - source_degree: Source node degree
        - target_degree: Target node degree
        - category: (source_category, target_category)
        - metapath: Metapath abbreviation
    """
    if quantiles is None:
        quantiles = config.DEGREE_QUANTILES
    if category_names is None:
        category_names = config.DEGREE_CATEGORY_NAMES

    from .data_loading import get_metaedges_for_metapath

    # Get metaedges
    metaedges = get_metaedges_for_metapath(metapath)

    # For now, use the first and last metaedge to determine terminal node degrees
    first_metaedge = metaedges[0]
    last_metaedge = metaedges[-1]

    loader = get_loader()
    rng = np.random.RandomState(random_state)

    # Get degrees for source nodes (from first metaedge)
    source_degrees_all = loader.get_node_degrees(first_metaedge, source, "source")

    # Get degrees for target nodes (from last metaedge)
    target_degrees_all = loader.get_node_degrees(last_metaedge, source, "target")

    # For sampling, we need to find all valid source-target pairs
    # This is simplified: sample from all possible combinations
    # In a full implementation, we'd trace actual paths through the graph

    # Get nodes with non-zero degree
    source_nodes = np.where(source_degrees_all > 0)[0]
    target_nodes = np.where(target_degrees_all > 0)[0]

    source_degrees = source_degrees_all[source_nodes]
    target_degrees = target_degrees_all[target_nodes]

    # Compute bins
    source_bins, _ = compute_degree_bins(source_degrees, quantiles)
    target_bins, _ = compute_degree_bins(target_degrees, quantiles)

    # Vectorized categorization of all nodes
    source_bin_indices = np.digitize(source_degrees, source_bins, right=False) - 1
    source_bin_indices = np.clip(source_bin_indices, 0, len(category_names) - 1)
    source_categories = np.array([category_names[idx] for idx in source_bin_indices])

    target_bin_indices = np.digitize(target_degrees, target_bins, right=False) - 1
    target_bin_indices = np.clip(target_bin_indices, 0, len(category_names) - 1)
    target_categories = np.array([category_names[idx] for idx in target_bin_indices])

    # Sample from each category combination
    samples = []
    for src_cat in category_names:
        for tgt_cat in category_names:
            # Vectorized filtering
            src_mask = source_categories == src_cat
            tgt_mask = target_categories == tgt_cat

            src_nodes_in_cat = source_nodes[src_mask]
            src_degrees_in_cat = source_degrees[src_mask]
            tgt_nodes_in_cat = target_nodes[tgt_mask]
            tgt_degrees_in_cat = target_degrees[tgt_mask]

            # Sample pairs
            n_src = len(src_nodes_in_cat)
            n_tgt = len(tgt_nodes_in_cat)
            if n_src > 0 and n_tgt > 0:
                n_to_sample = min(n_samples_per_category, n_src, n_tgt)

                sampled_src_indices = rng.choice(n_src, size=n_to_sample, replace=False)
                sampled_tgt_indices = rng.choice(n_tgt, size=n_to_sample, replace=False)

                # Vectorized sample creation
                for src_idx, tgt_idx in zip(sampled_src_indices, sampled_tgt_indices):
                    samples.append({
                        'source_idx': int(src_nodes_in_cat[src_idx]),
                        'target_idx': int(tgt_nodes_in_cat[tgt_idx]),
                        'source_degree': int(src_degrees_in_cat[src_idx]),
                        'target_degree': int(tgt_degrees_in_cat[tgt_idx]),
                        'category': (src_cat, tgt_cat),
                        'metapath': metapath
                    })

    logger.info(
        f"{source} {metapath}: Sampled {len(samples)} source-target pairs"
    )

    return samples
