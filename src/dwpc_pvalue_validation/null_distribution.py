"""
Null distribution module for DWPC P-Value Validation

Builds null DWPC distributions from permuted networks stratified by degree.
"""

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

from . import config, sampling, dwpc_calculation
from .utils import setup_logging

logger = setup_logging(__name__)


def build_null_for_category(
    metapath: str,
    category: tuple,
    n_samples: int,
    perm_indices: List[int],
    damping_exponent: float = 0.5,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Build null DWPC distribution for a single degree category.

    Parameters
    ----------
    metapath : str
        Metapath abbreviation (e.g., 'CbGpPW').
    category : tuple
        Degree category (e.g., ('Low', 'High')).
    n_samples : int
        Number of node pairs to sample per permutation.
    perm_indices : list of int
        Permutation indices to use for null (e.g., [1, 2, 3, ..., 20]).
    damping_exponent : float
        DWPC damping exponent.
    random_state : int, optional
        Random seed.

    Returns
    -------
    null_dwpcs : np.ndarray
        Null DWPC values for this category across all permutations.
    """
    all_dwpcs = []

    for perm_idx in perm_indices:
        source = f"perm{perm_idx}"

        # Sample node pairs in this category
        samples = sampling.sample_metapath_pairs(
            metapath=metapath,
            source=source,
            n_samples_per_category=n_samples,
            quantiles=config.DEGREE_QUANTILES,
            category_names=config.DEGREE_CATEGORY_NAMES,
            random_state=random_state
        )

        # Filter to requested category
        samples_in_cat = [s for s in samples if s['category'] == category]

        if len(samples_in_cat) == 0:
            logger.warning(
                f"No samples found for {metapath} category {category} "
                f"in {source}"
            )
            continue

        # Calculate DWPCs
        dwpcs = dwpc_calculation.calculate_dwpc_for_samples(
            samples=samples_in_cat,
            metapath=metapath,
            source=source,
            damping_exponent=damping_exponent
        )

        all_dwpcs.extend(dwpcs)

    null_dwpcs = np.array(all_dwpcs)

    logger.info(
        f"{metapath} category {category}: "
        f"Built null with {len(null_dwpcs)} DWPCs from "
        f"{len(perm_indices)} permutations"
    )

    return null_dwpcs


def build_null_distributions(
    metapath: str,
    n_samples_per_category: int,
    perm_indices: List[int],
    damping_exponent: float = 0.5,
    quantiles: Optional[List[float]] = None,
    category_names: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> Dict[tuple, np.ndarray]:
    """
    Build null DWPC distributions for all degree categories.

    Parameters
    ----------
    metapath : str
        Metapath abbreviation.
    n_samples_per_category : int
        Samples per category per permutation.
    perm_indices : list of int
        Permutation indices for null.
    damping_exponent : float
        DWPC damping exponent.
    quantiles : list of float, optional
        Degree quantiles.
    category_names : list of str, optional
        Category names.
    random_state : int, optional
        Random seed.

    Returns
    -------
    null_by_category : dict
        Null DWPC distributions keyed by category tuple.
    """
    if quantiles is None:
        quantiles = config.DEGREE_QUANTILES
    if category_names is None:
        category_names = config.DEGREE_CATEGORY_NAMES

    logger.info(
        f"Building null distributions for {metapath} using "
        f"permutations {perm_indices[0]}-{perm_indices[-1]}"
    )

    # Get all category combinations
    categories = [
        (src_cat, tgt_cat)
        for src_cat in category_names
        for tgt_cat in category_names
    ]

    null_by_category = {}

    for category in categories:
        null_dwpcs = build_null_for_category(
            metapath=metapath,
            category=category,
            n_samples=n_samples_per_category,
            perm_indices=perm_indices,
            damping_exponent=damping_exponent,
            random_state=random_state
        )

        if len(null_dwpcs) > 0:
            null_by_category[category] = null_dwpcs

    logger.info(
        f"Built null distributions for {len(null_by_category)} categories"
    )

    return null_by_category


def compute_null_statistics(
    null_by_category: Dict[tuple, np.ndarray]
) -> Dict[tuple, Dict[str, float]]:
    """
    Compute summary statistics for null distributions.

    Parameters
    ----------
    null_by_category : dict
        Null DWPC distributions by category.

    Returns
    -------
    stats : dict
        Statistics for each category:
        - mean: Mean DWPC
        - std: Standard deviation
        - median: Median DWPC
        - n: Number of samples
        - n_zero: Number of zero DWPCs
        - proportion_zero: Proportion of zeros
    """
    stats = {}

    for category, dwpcs in null_by_category.items():
        n_zero = np.sum(dwpcs == 0)

        stats[category] = {
            'mean': float(np.mean(dwpcs)),
            'std': float(np.std(dwpcs)),
            'median': float(np.median(dwpcs)),
            'min': float(np.min(dwpcs)),
            'max': float(np.max(dwpcs)),
            'n': len(dwpcs),
            'n_zero': int(n_zero),
            'proportion_zero': float(n_zero / len(dwpcs)) if len(dwpcs) > 0 else 0.0
        }

    return stats


def save_null_distributions(
    null_by_category: Dict[tuple, np.ndarray],
    output_path: Path
):
    """
    Save null distributions to file.

    Parameters
    ----------
    null_by_category : dict
        Null distributions by category.
    output_path : Path
        Output file path (.npz format).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert category tuples to strings for saving
    save_dict = {
        f"{cat[0]}_{cat[1]}": dwpcs
        for cat, dwpcs in null_by_category.items()
    }

    np.savez_compressed(output_path, **save_dict)

    logger.info(
        f"Saved null distributions for {len(null_by_category)} categories "
        f"to {output_path}"
    )


def load_null_distributions(
    input_path: Path
) -> Dict[tuple, np.ndarray]:
    """
    Load null distributions from file.

    Parameters
    ----------
    input_path : Path
        Input file path (.npz format).

    Returns
    -------
    null_by_category : dict
        Null distributions by category.
    """
    data = np.load(input_path)

    # Convert string keys back to tuples
    null_by_category = {}
    for key, dwpcs in data.items():
        src_cat, tgt_cat = key.split('_')
        null_by_category[(src_cat, tgt_cat)] = dwpcs

    logger.info(
        f"Loaded null distributions for {len(null_by_category)} categories "
        f"from {input_path}"
    )

    return null_by_category


def build_null_distributions_for_samples(
    samples: List[Dict],
    metapath: str,
    perm_indices: List[int],
    damping_exponent: float = 0.5
) -> Dict[tuple, np.ndarray]:
    """
    Build null DWPC distributions for specific sample pairs.

    Uses the SAME node pairs across all permutations to maintain proper
    correspondence between observed and null distributions.

    Groups by EXACT (source_degree, target_degree) as per Himmelstein et al. 2023.

    Parameters
    ----------
    samples : list of dict
        Sample pairs with source_idx, target_idx, source_degree, target_degree.
    metapath : str
        Metapath abbreviation.
    perm_indices : list of int
        Permutation indices for null.
    damping_exponent : float
        DWPC damping exponent.

    Returns
    -------
    null_by_degree : dict
        Null DWPC distributions keyed by (source_degree, target_degree) tuple.
    """
    logger.info(
        f"Building null distributions for {metapath} using "
        f"permutations {perm_indices[0]}-{perm_indices[-1]} "
        f"with {len(samples)} specific sample pairs (exact degree grouping)"
    )

    # Organize samples by EXACT degree (not category)
    samples_by_degree = {}
    for sample in samples:
        degree_key = (sample['source_degree'], sample['target_degree'])
        if degree_key not in samples_by_degree:
            samples_by_degree[degree_key] = []
        samples_by_degree[degree_key].append(sample)

    # Build null for each exact degree combination
    null_by_degree = {}

    for degree_key, degree_samples in samples_by_degree.items():
        # Calculate null DWPCs for each permutation
        all_dwpcs = []

        for perm_idx in perm_indices:
            source = f"perm{perm_idx}"

            # Calculate DWPCs for the SAME node pairs
            dwpcs = dwpc_calculation.calculate_dwpc_for_samples(
                samples=degree_samples,
                metapath=metapath,
                source=source,
                damping_exponent=damping_exponent
            )
            all_dwpcs.extend(dwpcs)

        null_by_degree[degree_key] = np.array(all_dwpcs)

        logger.debug(
            f"{metapath} degrees {degree_key}: Built null with "
            f"{len(all_dwpcs)} DWPCs from {len(perm_indices)} permutations "
            f"({len(degree_samples)} pairs per perm)"
        )

    logger.info(
        f"Built null distributions for {len(null_by_degree)} exact degree combinations"
    )

    return null_by_degree
