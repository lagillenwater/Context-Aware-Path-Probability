"""
Theoretical corrections to analytical edge frequency formula.

This module implements corrections for known violations of the XSwap
analytical formula's assumptions:
1. Degree assortativity (assumes random edge placement)
2. Clustering (assumes edge independence)
3. Degree heterogeneity (assumes mean-field approximation)

All corrections use only features computable from the original graph.
No empirical frequencies needed for training.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict
import networkx as nx


def compute_degree_assortativity(edge_matrix: sp.spmatrix) -> float:
    """
    Compute degree assortativity coefficient.

    Measures correlation between degrees of connected nodes.
    Positive: high-degree nodes connect to high-degree nodes
    Negative: high-degree nodes connect to low-degree nodes
    Zero: random (XSwap assumption)

    Parameters
    ----------
    edge_matrix : sparse matrix
        Edge adjacency matrix

    Returns
    -------
    assortativity : float
        Pearson correlation coefficient of endpoint degrees (-1 to 1)
    """
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    src_nodes, tgt_nodes = edge_matrix.nonzero()

    src_degs_at_edges = source_degrees[src_nodes]
    tgt_degs_at_edges = target_degrees[tgt_nodes]

    if len(src_degs_at_edges) == 0:
        return 0.0

    correlation = np.corrcoef(src_degs_at_edges, tgt_degs_at_edges)[0, 1]

    if np.isnan(correlation):
        return 0.0

    return correlation


def compute_degree_heterogeneity(degrees: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics of degree distribution heterogeneity.

    Parameters
    ----------
    degrees : array
        Node degree sequence

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - gini: Gini coefficient (0=equal, 1=maximally unequal)
        - cv: Coefficient of variation (std/mean)
        - skewness: Distribution skewness
        - max_deg_ratio: Max degree / mean degree
    """
    degrees = degrees[degrees > 0]

    if len(degrees) == 0:
        return {'gini': 0.0, 'cv': 0.0, 'skewness': 0.0, 'max_deg_ratio': 1.0}

    sorted_degs = np.sort(degrees)
    n = len(sorted_degs)
    cumsum = np.cumsum(sorted_degs)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_degs))) / (n * np.sum(sorted_degs)) - (n + 1) / n

    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    cv = std_deg / mean_deg if mean_deg > 0 else 0.0

    from scipy import stats
    skewness = stats.skew(degrees)

    max_deg_ratio = np.max(degrees) / mean_deg if mean_deg > 0 else 1.0

    return {
        'gini': gini,
        'cv': cv,
        'skewness': skewness,
        'max_deg_ratio': max_deg_ratio
    }


def assortativity_correction(u: np.ndarray,
                              v: np.ndarray,
                              assortativity: float,
                              mean_u: float,
                              mean_v: float,
                              alpha: float = 0.1) -> np.ndarray:
    """
    Compute multiplicative correction for degree assortativity.

    Theory:
    - Positive assortativity: high × high and low × low more likely
    - Negative assortativity: high × low more likely, high × high less likely
    - Analytical formula assumes zero assortativity

    For negative assortativity (common in biological networks):
    - When both u and v are high: reduce predicted frequency
    - When u and v are dissimilar: increase predicted frequency

    Parameters
    ----------
    u : array
        Source degrees
    v : array
        Target degrees
    assortativity : float
        Degree assortativity coefficient (-1 to 1)
    mean_u, mean_v : float
        Mean degrees
    alpha : float
        Correction strength parameter (reduced to 0.1 for conservative corrections)

    Returns
    -------
    correction : array
        Multiplicative correction factors (typically 0.9 to 1.1)
    """
    if abs(assortativity) < 0.05:
        return np.ones_like(u, dtype=float)

    u_normalized = (u - mean_u) / mean_u if mean_u > 0 else np.zeros_like(u)
    v_normalized = (v - mean_v) / mean_v if mean_v > 0 else np.zeros_like(v)

    degree_product_normalized = u_normalized * v_normalized

    correction = 1.0 + alpha * assortativity * degree_product_normalized

    correction = np.clip(correction, 0.8, 1.2)

    return correction


def heterogeneity_correction(u: np.ndarray,
                             v: np.ndarray,
                             gini: float,
                             mean_u: float,
                             mean_v: float,
                             percentile_90_u: float,
                             percentile_90_v: float,
                             beta: float = 0.05) -> np.ndarray:
    """
    Compute multiplicative correction for degree heterogeneity.

    Theory:
    - Fat-tailed distributions have hub effects
    - Mean-field approximation underestimates high-degree pairs
    - Correction increases with Gini coefficient and extreme degrees

    Parameters
    ----------
    u, v : array
        Source and target degrees
    gini : float
        Gini coefficient of degree distribution
    mean_u, mean_v : float
        Mean degrees
    percentile_90_u, percentile_90_v : float
        90th percentile thresholds
    beta : float
        Correction strength parameter

    Returns
    -------
    correction : array
        Multiplicative correction factors
    """
    if gini < 0.3:
        return np.ones_like(u, dtype=float)

    high_degree_u = u > percentile_90_u
    high_degree_v = v > percentile_90_v

    correction = np.ones_like(u, dtype=float)

    high_degree_pairs = high_degree_u & high_degree_v
    if np.any(high_degree_pairs):
        u_ratio = (u[high_degree_pairs] / mean_u) if mean_u > 0 else 1.0
        v_ratio = (v[high_degree_pairs] / mean_v) if mean_v > 0 else 1.0

        correction[high_degree_pairs] = 1.0 + beta * (gini - 0.3) * np.sqrt(u_ratio * v_ratio)

    correction = np.clip(correction, 0.5, 2.0)

    return correction


def density_correction(u: np.ndarray,
                       v: np.ndarray,
                       density: float,
                       gamma: float = 0.05) -> np.ndarray:
    """
    Compute correction for network density effects.

    Theory:
    - Dense networks have more constraints on edge placement
    - Analytical formula assumes sparse network limit
    - Correction adjusts for finite-size effects

    Parameters
    ----------
    u, v : array
        Source and target degrees
    density : float
        Network density (m / (n_source * n_target))
    gamma : float
        Correction strength parameter

    Returns
    -------
    correction : array
        Multiplicative correction factors
    """
    if density < 0.01:
        return np.ones_like(u, dtype=float)

    correction = 1.0 - gamma * density * np.log1p(u * v)

    correction = np.clip(correction, 0.8, 1.0)

    return correction


def apply_all_corrections(P_analytical: np.ndarray,
                         u: np.ndarray,
                         v: np.ndarray,
                         graph_features: Dict,
                         degree_stratified: bool = False) -> np.ndarray:
    """
    Apply all theoretical corrections to analytical predictions.

    Parameters
    ----------
    P_analytical : array
        Analytical formula predictions
    u, v : array
        Source and target degrees
    graph_features : dict
        Dictionary containing:
        - assortativity: float
        - gini: float
        - density: float
        - mean_u, mean_v: float
        - percentile_90_u, percentile_90_v: float
    degree_stratified : bool
        If True, apply different corrections for different degree ranges

    Returns
    -------
    P_corrected : array
        Corrected predictions
    """
    if degree_stratified:
        return apply_degree_stratified_corrections(P_analytical, u, v, graph_features)

    corr_assort = assortativity_correction(
        u, v,
        graph_features['assortativity'],
        graph_features['mean_u'],
        graph_features['mean_v']
    )

    corr_heterog = heterogeneity_correction(
        u, v,
        graph_features['gini'],
        graph_features['mean_u'],
        graph_features['mean_v'],
        graph_features['percentile_90_u'],
        graph_features['percentile_90_v']
    )

    corr_density = density_correction(
        u, v,
        graph_features['density']
    )

    total_correction = corr_assort * corr_heterog * corr_density

    P_corrected = P_analytical * total_correction

    P_corrected = np.clip(P_corrected, 0.0, 1.0)

    return P_corrected


def apply_degree_stratified_corrections(P_analytical: np.ndarray,
                                       u: np.ndarray,
                                       v: np.ndarray,
                                       graph_features: Dict) -> np.ndarray:
    """
    Apply degree-range-specific corrections.

    Based on residual analysis, corrections help low/mid-degree pairs but
    hurt high-degree pairs. This function applies different correction
    strategies based on degree product.

    Strategy:
    - Low degree (0-33%): Full corrections (93.6% improved in AeG)
    - Mid degree (33-67%): Full corrections (69.6% improved in AeG)
    - High degree (67-100%): Skip assortativity correction (only 8.1% improved)

    The issue: For high-degree pairs under negative assortativity, the
    assortativity correction reduces predictions, but these pairs are
    already underestimated by the analytical formula. The heterogeneity
    correction tries to increase predictions for hubs, but the assortativity
    correction dominates and makes things worse.

    Parameters
    ----------
    P_analytical : array
        Analytical formula predictions
    u, v : array
        Source and target degrees
    graph_features : dict
        Graph features dictionary

    Returns
    -------
    P_corrected : array
        Corrected predictions with degree-stratified approach
    """
    degree_product = u * v

    threshold_33 = np.percentile(degree_product, 33)
    threshold_67 = np.percentile(degree_product, 67)

    low_mask = degree_product < threshold_33
    mid_mask = (degree_product >= threshold_33) & (degree_product < threshold_67)
    high_mask = degree_product >= threshold_67

    P_corrected = P_analytical.copy()

    corr_heterog = heterogeneity_correction(
        u, v,
        graph_features['gini'],
        graph_features['mean_u'],
        graph_features['mean_v'],
        graph_features['percentile_90_u'],
        graph_features['percentile_90_v']
    )

    corr_density = density_correction(
        u, v,
        graph_features['density']
    )

    if np.any(low_mask):
        corr_assort_low = assortativity_correction(
            u[low_mask], v[low_mask],
            graph_features['assortativity'],
            graph_features['mean_u'],
            graph_features['mean_v'],
            alpha=0.1
        )

        total_corr_low = corr_assort_low * corr_heterog[low_mask] * corr_density[low_mask]
        P_corrected[low_mask] = P_analytical[low_mask] * total_corr_low

    if np.any(mid_mask):
        corr_assort_mid = assortativity_correction(
            u[mid_mask], v[mid_mask],
            graph_features['assortativity'],
            graph_features['mean_u'],
            graph_features['mean_v'],
            alpha=0.1
        )

        total_corr_mid = corr_assort_mid * corr_heterog[mid_mask] * corr_density[mid_mask]
        P_corrected[mid_mask] = P_analytical[mid_mask] * total_corr_mid

    if np.any(high_mask):
        total_corr_high = corr_heterog[high_mask]
        P_corrected[high_mask] = P_analytical[high_mask] * total_corr_high

    P_corrected = np.clip(P_corrected, 0.0, 1.0)

    return P_corrected


def extract_graph_features(edge_matrix: sp.spmatrix) -> Dict:
    """
    Extract all graph features needed for corrections.

    Parameters
    ----------
    edge_matrix : sparse matrix
        Edge adjacency matrix

    Returns
    -------
    features : dict
        All features needed for corrections
    """
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    n_sources, n_targets = edge_matrix.shape
    m_edges = edge_matrix.nnz

    assortativity = compute_degree_assortativity(edge_matrix)

    source_heterog = compute_degree_heterogeneity(source_degrees)
    target_heterog = compute_degree_heterogeneity(target_degrees)

    gini = (source_heterog['gini'] + target_heterog['gini']) / 2

    density = m_edges / (n_sources * n_targets)

    mean_u = source_degrees[source_degrees > 0].mean() if np.any(source_degrees > 0) else 0.0
    mean_v = target_degrees[target_degrees > 0].mean() if np.any(target_degrees > 0) else 0.0

    percentile_90_u = np.percentile(source_degrees[source_degrees > 0], 90) if np.any(source_degrees > 0) else 0.0
    percentile_90_v = np.percentile(target_degrees[target_degrees > 0], 90) if np.any(target_degrees > 0) else 0.0

    features = {
        'assortativity': assortativity,
        'gini': gini,
        'density': density,
        'mean_u': mean_u,
        'mean_v': mean_v,
        'percentile_90_u': percentile_90_u,
        'percentile_90_v': percentile_90_v,
        'source_heterog': source_heterog,
        'target_heterog': target_heterog,
        'n_sources': n_sources,
        'n_targets': n_targets,
        'm_edges': m_edges
    }

    print("Graph features:")
    print(f"  Assortativity: {assortativity:.4f}")
    print(f"  Gini coefficient: {gini:.4f}")
    print(f"  Density: {density:.6f}")
    print(f"  Mean degrees: u={mean_u:.2f}, v={mean_v:.2f}")
    print(f"  90th percentile: u={percentile_90_u:.0f}, v={percentile_90_v:.0f}")

    return features
