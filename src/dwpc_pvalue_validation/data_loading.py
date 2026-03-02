"""
Data loading module for DWPC P-Value Validation

Loads Hetionet data using hetmatpy HetMat format.
Permutations are stored as HetMat directories (000.hetmat, 001.hetmat, etc.).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import hetmatpy.hetmat

from . import config
from .utils import setup_logging

logger = setup_logging(__name__)


class HetionetLoader:
    """
    Lazy loader for Hetionet and permuted networks using HetMat format.

    Caches loaded HetMat objects and matrices to avoid reloading.
    """

    def __init__(self):
        """Initialize the loader with empty cache."""
        self._hetmat_cache: Dict[str, hetmatpy.hetmat.HetMat] = {}
        self._degree_cache: Dict[Tuple[str, str, str, str], np.ndarray] = {}

    def load_hetmat(self, source: str = "true", perm_idx: Optional[int] = None) -> hetmatpy.hetmat.HetMat:
        """
        Load HetMat object.

        Parameters
        ----------
        source : str
            Source of the data: 'true' (Hetionet), 'perm0', or 'permX' where X is 1-20.
        perm_idx : int, optional
            Permutation index (alternative to source='permX').

        Returns
        -------
        hetmatpy.hetmat.HetMat
            Loaded HetMat object.
        """
        # Handle perm_idx parameter
        if perm_idx is not None:
            source = f"perm{perm_idx}"

        # Check cache
        if source in self._hetmat_cache:
            return self._hetmat_cache[source]

        # Determine path
        if source == "true":
            hetmat_dir = config.HETIONET_HETMAT_DIR
        elif source.startswith("perm"):
            perm_num = 0 if source == "perm0" else int(source[4:])
            hetmat_dir = config.get_permutation_hetmat_dir(perm_num)
        else:
            raise ValueError(
                f"Invalid source: {source}. Must be 'true', 'perm0', or 'permX'."
            )

        # Load HetMat
        logger.info(f"Loading HetMat from {source}: {hetmat_dir}")
        hetmat = hetmatpy.hetmat.HetMat(hetmat_dir)

        # Cache and return
        self._hetmat_cache[source] = hetmat
        return hetmat

    def load_edge_matrix(
        self,
        metaedge: str,
        source: str = "true",
        perm_idx: Optional[int] = None
    ) -> sp.spmatrix:
        """
        Load edge adjacency matrix for a metaedge.

        Parameters
        ----------
        metaedge : str
            Metaedge abbreviation (e.g., 'CbG' for Compound-binds-Gene).
        source : str
            Source of the matrix: 'true', 'perm0', or 'permX'.
        perm_idx : int, optional
            Permutation index.

        Returns
        -------
        scipy.sparse.spmatrix
            Edge adjacency matrix.
        """
        hetmat = self.load_hetmat(source, perm_idx)

        # Get adjacency matrix (returns tuple: source_nodes, target_nodes, matrix)
        _, _, matrix = hetmat.metaedge_to_adjacency_matrix(metaedge)

        logger.debug(
            f"{source}: {metaedge} shape={matrix.shape}, edges={matrix.nnz}"
        )
        return matrix

    def get_node_degrees(
        self,
        metaedge: str,
        source: str = "true",
        node_position: str = "source"
    ) -> np.ndarray:
        """
        Get node degrees for a metaedge.

        Parameters
        ----------
        metaedge : str
            Metaedge abbreviation (e.g., 'CbG').
        source : str
            Source of the matrix: 'true', 'perm0', or 'permX'.
        node_position : str
            'source' for source node degrees, 'target' for target node degrees.

        Returns
        -------
        np.ndarray
            Array of node degrees.
        """
        # Check cache
        cache_key = (metaedge, source, node_position)
        if cache_key in self._degree_cache:
            return self._degree_cache[cache_key]

        # Load matrix
        matrix = self.load_edge_matrix(metaedge, source)

        # Calculate degrees
        if node_position == "source":
            degrees = np.asarray(matrix.sum(axis=1)).flatten()
        elif node_position == "target":
            degrees = np.asarray(matrix.sum(axis=0)).flatten()
        else:
            raise ValueError(
                f"Invalid node_position: {node_position}. "
                "Must be 'source' or 'target'."
            )

        # Cache and return
        self._degree_cache[cache_key] = degrees
        logger.debug(
            f"{source} {metaedge} {node_position} degrees: "
            f"min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.2f}"
        )
        return degrees

    def get_connected_node_pairs(
        self,
        metaedge: str,
        source: str = "true"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all node pairs with at least one edge.

        Parameters
        ----------
        metaedge : str
            Metaedge abbreviation (e.g., 'CbG').
        source : str
            Source of the matrix.

        Returns
        -------
        source_nodes : np.ndarray
            Source node indices.
        target_nodes : np.ndarray
            Target node indices.
        """
        matrix = self.load_edge_matrix(metaedge, source)

        # Get nonzero entries
        source_nodes, target_nodes = matrix.nonzero()

        logger.info(
            f"{source} {metaedge}: {len(source_nodes)} connected pairs"
        )
        return source_nodes, target_nodes

    def clear_cache(self):
        """Clear all cached data."""
        self._hetmat_cache.clear()
        self._degree_cache.clear()
        logger.info("Cleared data cache")


def parse_metaedge(metaedge_abbrev: str) -> Tuple[str, str, str]:
    """
    Parse metaedge abbreviation into components.

    Parameters
    ----------
    metaedge_abbrev : str
        Metaedge abbreviation (e.g., 'CbG').

    Returns
    -------
    source_metanode : str
        Source metanode abbreviation (e.g., 'C' for Compound).
    edge_kind : str
        Edge kind abbreviation (e.g., 'b' for binds).
    target_metanode : str
        Target metanode abbreviation (e.g., 'G' for Gene).

    Examples
    --------
    >>> parse_metaedge('CbG')
    ('C', 'b', 'G')
    """
    if len(metaedge_abbrev) < 3:
        raise ValueError(f"Invalid metaedge: {metaedge_abbrev}")

    source_metanode = metaedge_abbrev[0]
    target_metanode = metaedge_abbrev[-1]
    edge_kind = metaedge_abbrev[1:-1]

    return source_metanode, edge_kind, target_metanode


def expand_metanode_abbrev(abbrev: str) -> str:
    """
    Expand metanode abbreviation to full name.

    Uses the metagraph to map abbreviations.
    """
    loader = get_loader()
    metagraph = loader.load_hetmat("true").metagraph

    # Find metanode with matching abbreviation
    for metanode in metagraph.get_metanodes():
        if metanode.identifier[0] == abbrev:
            return metanode.identifier

    raise ValueError(f"Unknown metanode abbreviation: {abbrev}")


def get_metaedges_for_metapath(metapath: str) -> List[str]:
    """
    Parse metapath into constituent metaedges.

    Parameters
    ----------
    metapath : str
        Metapath abbreviation (e.g., 'CbGpPW').

    Returns
    -------
    list of str
        List of metaedge abbreviations.

    Examples
    --------
    >>> get_metaedges_for_metapath('CbGpPW')
    ['CbG', 'GpPW']

    >>> get_metaedges_for_metapath('CbGpPWpG')
    ['CbG', 'GpPW', 'PWpG']
    """
    loader = get_loader()
    metagraph = loader.load_hetmat("true").metagraph

    # Use metagraph's built-in parsing
    metapath_obj = metagraph.metapath_from_abbrev(metapath)
    metaedges = [metaedge.get_abbrev() for metaedge in metapath_obj]

    return metaedges


def validate_data_availability(metapath: str) -> bool:
    """
    Check if all required data is available for a metapath.

    Parameters
    ----------
    metapath : str
        Metapath abbreviation.

    Returns
    -------
    bool
        True if all data is available, False otherwise.
    """
    try:
        # Check true Hetionet
        loader = get_loader()
        hetmat_true = loader.load_hetmat("true")

        # Check permutations
        for perm_idx in [config.PERMUTATION_OBSERVED] + list(
            range(config.PERMUTATION_START, config.PERMUTATION_END + 1)
        ):
            loader.load_hetmat(perm_idx=perm_idx)

        logger.info(f"All data available for metapath: {metapath}")
        return True

    except Exception as e:
        logger.error(f"Data availability check failed: {e}")
        return False


# Global loader instance
_global_loader: Optional[HetionetLoader] = None


def get_loader() -> HetionetLoader:
    """
    Get the global HetionetLoader instance.

    Returns
    -------
    HetionetLoader
        Global loader instance.
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = HetionetLoader()
    return _global_loader
