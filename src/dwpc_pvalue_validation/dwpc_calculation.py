"""
DWPC calculation module for DWPC P-Value Validation

Calculates degree-weighted path counts (DWPC) for metapaths using
edge-type-specific node degrees and per-node damping.

This matches het.io's Neo4j implementation where:
- Degrees are calculated per edge type (not total across all edges)
- Damping is applied once per node in the path (not per edge endpoint)
"""

from typing import List, Optional
import numpy as np
import scipy.sparse as sp

from . import config
from .data_loading import get_loader, get_metaedges_for_metapath
from .utils import setup_logging

logger = setup_logging(__name__)


def get_total_node_degrees(
    node_type: str,
    source: str = "true"
) -> np.ndarray:
    """
    Get total degree for all nodes of a given type across all edge types.

    In heterogeneous networks, each node type participates in multiple
    edge types. The total degree is the sum of degrees across all edge types.

    Parameters
    ----------
    node_type : str
        Node type name (e.g., 'Compound', 'Gene', 'Disease').
    source : str
        Data source: 'true', 'perm0', or 'permX'.

    Returns
    -------
    total_degrees : np.ndarray
        Total degrees for all nodes of this type.
    """
    loader = get_loader()
    hetmat = loader.load_hetmat(source)

    # Get metanode
    metanode = hetmat.metagraph.get_metanode(node_type)

    # Get all metaedges involving this node type
    # Track which edges to count and how
    metaedges_info = []

    for metaedge in hetmat.metagraph.get_edges():
        is_self_loop = metaedge.source == metanode and metaedge.target == metanode
        is_undirected = metaedge.direction == 'both'

        if is_self_loop and is_undirected:
            # Self-loop with undirected edge (e.g., GiG, CrC, GcG)
            # Matrix is symmetric, so row degrees = column degrees
            # Only count once to avoid double-counting
            metaedges_info.append((metaedge.get_abbrev(), 'source'))
        else:
            # Regular edge or directed self-loop
            # Count degrees for each position the node type appears in
            if metaedge.source == metanode:
                metaedges_info.append((metaedge.get_abbrev(), 'source'))
            if metaedge.target == metanode:
                metaedges_info.append((metaedge.get_abbrev(), 'target'))

    # Initialize total degrees
    n_nodes = len(hetmat.get_node_identifiers(metanode))
    total_degrees = np.zeros(n_nodes, dtype=float)

    # Sum degrees
    for metaedge_abbrev, position in metaedges_info:
        degrees = loader.get_node_degrees(metaedge_abbrev, source, position)
        total_degrees += degrees
        logger.debug(
            f"  {metaedge_abbrev} ({position}): "
            f"added to {metanode.identifier} degrees"
        )

    logger.debug(
        f"{source} {node_type} total degrees: "
        f"min={total_degrees.min()}, max={total_degrees.max()}, "
        f"mean={total_degrees.mean():.2f}"
    )

    return total_degrees


def get_node_types_for_metapath(metapath: str, hetmat) -> List[str]:
    """
    Get node types in order for a metapath.

    Parameters
    ----------
    metapath : str
        Metapath abbreviation (e.g., 'CbGpPW').
    hetmat : hetmatpy.hetmat.HetMat
        HetMat object for metagraph access.

    Returns
    -------
    node_types : list of str
        Node type names in path order (e.g., ['Compound', 'Gene', 'Pathway']).
    """
    metapath_obj = hetmat.metagraph.metapath_from_abbrev(metapath)

    node_types = []
    for i, metaedge in enumerate(metapath_obj):
        if i == 0:
            node_types.append(metaedge.source.identifier)
        node_types.append(metaedge.target.identifier)

    return node_types


def apply_degree_damping(
    adjacency_matrix: sp.spmatrix,
    source_degrees: np.ndarray,
    target_degrees: np.ndarray,
    damping_exponent: float
) -> sp.spmatrix:
    """
    Apply degree damping to an adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : scipy.sparse.spmatrix
        Adjacency matrix (rows=source, cols=target).
    source_degrees : np.ndarray
        Degrees of source nodes.
    target_degrees : np.ndarray
        Degrees of target nodes.
    damping_exponent : float
        Damping exponent w (typically 0.5).

    Returns
    -------
    damped_matrix : scipy.sparse.spmatrix
        Degree-damped adjacency matrix where each entry A[i,j] is multiplied
        by deg(i)^(-w) * deg(j)^(-w).
    """
    matrix = adjacency_matrix.tocsr()

    # Compute damping factors
    source_damping = np.power(
        source_degrees,
        -damping_exponent,
        where=source_degrees > 0,
        out=np.zeros_like(source_degrees, dtype=float)
    )
    target_damping = np.power(
        target_degrees,
        -damping_exponent,
        where=target_degrees > 0,
        out=np.zeros_like(target_degrees, dtype=float)
    )

    # Apply row damping (source nodes)
    row_damped = matrix.multiply(source_damping[:, np.newaxis])

    # Apply column damping (target nodes)
    damped_matrix = row_damped.multiply(target_damping[np.newaxis, :])

    return damped_matrix.tocsr()


def apply_degree_damping_target_only(
    adjacency_matrix: sp.spmatrix,
    target_degrees: np.ndarray,
    damping_exponent: float
) -> sp.spmatrix:
    """
    Apply degree damping to target nodes only.

    Used for non-initial edges in metapath where source nodes were already
    damped by the previous edge.

    Parameters
    ----------
    adjacency_matrix : scipy.sparse.spmatrix
        Adjacency matrix (rows=source, cols=target).
    target_degrees : np.ndarray
        Degrees of target nodes.
    damping_exponent : float
        Damping exponent w (typically 0.5).

    Returns
    -------
    damped_matrix : scipy.sparse.spmatrix
        Degree-damped adjacency matrix where each entry A[i,j] is multiplied
        by deg(j)^(-w).
    """
    matrix = adjacency_matrix.tocsr()

    # Compute damping factors
    target_damping = np.power(
        target_degrees,
        -damping_exponent,
        where=target_degrees > 0,
        out=np.zeros_like(target_degrees, dtype=float)
    )

    # Apply column damping (target nodes only)
    damped_matrix = matrix.multiply(target_damping[np.newaxis, :])

    return damped_matrix.tocsr()


def calculate_dwpc_pairs(
    metaedge: str,
    source_indices: List[int],
    target_indices: List[int],
    source: str = "true",
    damping_exponent: float = 0.5
) -> np.ndarray:
    """
    Calculate DWPC for source-target pairs connected by a single metaedge.

    Uses edge-type-specific degrees (het.io compatible).

    Parameters
    ----------
    metaedge : str
        Metaedge abbreviation (e.g., 'CbG').
    source_indices : list of int
        Source node indices.
    target_indices : list of int
        Target node indices.
    source : str
        Data source: 'true', 'perm0', or 'permX'.
    damping_exponent : float
        Damping exponent w.

    Returns
    -------
    dwpcs : np.ndarray
        DWPC values for each source-target pair.
    """
    loader = get_loader()
    hetmat = loader.load_hetmat(source)

    adjacency_matrix = loader.load_edge_matrix(metaedge, source)

    # Get metaedge object to find node types
    metaedge_obj = hetmat.metagraph.get_metaedge(metaedge)
    source_node_type = metaedge_obj.source.identifier
    target_node_type = metaedge_obj.target.identifier

    # Get edge-type-specific degrees (het.io compatible)
    source_degrees = loader.get_node_degrees(metaedge, source, 'source')
    target_degrees = loader.get_node_degrees(metaedge, source, 'target')

    # Apply degree damping
    damped_matrix = apply_degree_damping(
        adjacency_matrix,
        source_degrees,
        target_degrees,
        damping_exponent
    )

    # Extract DWPCs for requested pairs
    dwpcs = np.zeros(len(source_indices))
    for idx, (src, tgt) in enumerate(zip(source_indices, target_indices)):
        dwpcs[idx] = damped_matrix[src, tgt]

    logger.debug(
        f"{source} {metaedge}: Calculated {len(dwpcs)} DWPCs, "
        f"mean={np.mean(dwpcs):.6f}"
    )

    return dwpcs


def calculate_dwpc_metapath(
    metapath: str,
    source_indices: List[int],
    target_indices: List[int],
    source: str = "true",
    damping_exponent: float = 0.5
) -> np.ndarray:
    """
    Calculate DWPC for source-target pairs connected by a metapath.

    Uses edge-type-specific degrees and applies damping once per node
    in the path, matching het.io's implementation.

    For a path with nodes [n0, n1, ..., nk] and edges [e0, e1, ..., e(k-1)]:
    - DWPC = sum over paths of: product(deg_ei(ni)^-w for each edge-node pair)

    Implementation:
    - First edge: damp by source and target degrees
    - Subsequent edges: damp by target degree only (source already damped)

    Parameters
    ----------
    metapath : str
        Metapath abbreviation (e.g., 'CbGpPW').
    source_indices : list of int
        Source node indices (first node type in metapath).
    target_indices : list of int
        Target node indices (last node type in metapath).
    source : str
        Data source: 'true', 'perm0', or 'permX'.
    damping_exponent : float
        Damping exponent w.

    Returns
    -------
    dwpcs : np.ndarray
        DWPC values for each source-target pair.
    """
    loader = get_loader()
    hetmat = loader.load_hetmat(source)

    # Parse metapath into metaedges
    metaedges = get_metaedges_for_metapath(metapath)

    logger.info(
        f"Calculating DWPC for {metapath} ({len(metaedges)} edges) "
        f"on {len(source_indices)} pairs"
    )

    # Get node types in the path
    node_types = get_node_types_for_metapath(metapath, hetmat)

    logger.debug(f"Node types in path: {node_types}")

    # Process each edge matrix
    result_matrix = None

    for edge_idx, metaedge in enumerate(metaedges):
        # Load matrix
        adjacency_matrix = loader.load_edge_matrix(metaedge, source)

        # Get node types for this edge
        source_node_type = node_types[edge_idx]
        target_node_type = node_types[edge_idx + 1]

        # Get edge-type-specific degrees (het.io compatible)
        source_degrees = loader.get_node_degrees(metaedge, source, 'source')
        target_degrees = loader.get_node_degrees(metaedge, source, 'target')

        # Apply damping
        if edge_idx == 0:
            # First edge: damp both source and target
            damped_matrix = apply_degree_damping(
                adjacency_matrix,
                source_degrees,
                target_degrees,
                damping_exponent
            )
            logger.debug(
                f"Edge {edge_idx} ({metaedge}): damped source "
                f"({source_node_type}) and target ({target_node_type})"
            )
        else:
            # Subsequent edges: only damp target (source already damped)
            damped_matrix = apply_degree_damping_target_only(
                adjacency_matrix,
                target_degrees,
                damping_exponent
            )
            logger.debug(
                f"Edge {edge_idx} ({metaedge}): damped target "
                f"({target_node_type}) only"
            )

        # Multiply matrices
        if result_matrix is None:
            result_matrix = damped_matrix
        else:
            result_matrix = result_matrix @ damped_matrix

    # Extract DWPCs for requested pairs
    dwpcs = np.zeros(len(source_indices))
    for idx, (src, tgt) in enumerate(zip(source_indices, target_indices)):
        dwpcs[idx] = result_matrix[src, tgt]

    logger.info(
        f"{source} {metapath}: DWPC mean={np.mean(dwpcs):.6f}, "
        f"std={np.std(dwpcs):.6f}"
    )

    return dwpcs


def calculate_dwpc_for_samples(
    samples: List[dict],
    metapath: Optional[str] = None,
    metaedge: Optional[str] = None,
    source: str = "true",
    damping_exponent: float = 0.5
) -> np.ndarray:
    """
    Calculate DWPC for a list of sampled node pairs.

    Parameters
    ----------
    samples : list of dict
        Sampled pairs from sampling module, each with 'source_idx' and
        'target_idx'.
    metapath : str, optional
        Metapath abbreviation. Mutually exclusive with metaedge.
    metaedge : str, optional
        Metaedge abbreviation. Mutually exclusive with metapath.
    source : str
        Data source.
    damping_exponent : float
        Damping exponent.

    Returns
    -------
    dwpcs : np.ndarray
        DWPC values for each sample.
    """
    if (metapath is None) == (metaedge is None):
        raise ValueError("Must specify exactly one of metapath or metaedge")

    source_indices = [s['source_idx'] for s in samples]
    target_indices = [s['target_idx'] for s in samples]

    if metapath is not None:
        return calculate_dwpc_metapath(
            metapath, source_indices, target_indices, source, damping_exponent
        )
    else:
        return calculate_dwpc_pairs(
            metaedge, source_indices, target_indices, source, damping_exponent
        )
