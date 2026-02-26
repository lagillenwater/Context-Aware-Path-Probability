"""
Theory-Guided Feature Engineering for Edge Probability Prediction

Based on the XSwap Markov chain derivation from:
"The probability of edge existence due to node degree: a baseline for
network-based predictions"

This module provides feature engineering that relaxes the analytical formula's
assumptions to achieve better predictive accuracy with unbiased residuals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import scipy.sparse as sp


class TheoryGuidedFeatureEngineer:
    """
    Feature engineering based on XSwap theoretical derivation.

    Hierarchical feature sets:
    - Level 1: Direct analytical formula terms
    - Level 2: Non-linear transformations (relax linearity)
    - Level 3: Graph-specific statistics (edge-type adaptation)
    - Level 4: Polynomial corrections (bias reduction)
    - Level 5: Interaction terms (relax independence)
    """

    def __init__(self, edge_matrix: sp.spmatrix, edge_type: str):
        """
        Initialize feature engineer for a specific edge type.

        Parameters
        ----------
        edge_matrix : sparse matrix
            Adjacency matrix for the edge type
        edge_type : str
            Edge type identifier (e.g., 'CbG', 'AeG')
        """
        self.edge_type = edge_type
        self.edge_matrix = edge_matrix

        # Compute graph statistics (once)
        self.m = edge_matrix.nnz  # Total edges
        self.n_source = edge_matrix.shape[0]
        self.n_target = edge_matrix.shape[1]
        self.possible_edges = self.n_source * self.n_target
        self.density = self.m / self.possible_edges

        # Degree statistics
        self.source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
        self.target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

        self.mean_source_deg = self.source_degrees.mean()
        self.mean_target_deg = self.target_degrees.mean()
        self.std_source_deg = self.source_degrees.std()
        self.std_target_deg = self.target_degrees.std()

        # Total possible swaps (S in the paper)
        # Approximation: S ≈ m * (m-1) / 2 for simple graphs
        self.S_approx = self.m * (self.m - 1) / 2 if self.m > 1 else 1

    def compute_level1_analytical(self, u: np.ndarray, v: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Level 1: Direct analytical formula features.

        Based on equations from paper pages 2-3:
        - q_ij = (u × v) / S  [edge creation rate]
        - r_ij = (m - u - v + 1) / S  [edge removal rate]
        - P_L1 = q / (r + q)  [original formula]
        - P_L2 = q / sqrt(q² + r²)  [modified formula]

        Parameters
        ----------
        u : array
            Source node degrees
        v : array
            Target node degrees

        Returns
        -------
        features : dict
            Dictionary of feature arrays
        """
        # Core theoretical terms
        q = u * v  # Edge creation rate (numerator)
        r = self.m - u - v + 1  # Edge removal rate

        # Avoid division by zero
        r_safe = np.maximum(r, 1e-10)
        q_safe = np.maximum(q, 1e-10)

        features = {
            'u': u,
            'v': v,
            'degree_product': q,
            'removal_term': r,
            'q_normalized': q / self.S_approx,
            'r_normalized': r / self.S_approx,
            'P_L1_norm': q / (r_safe + q_safe),  # Original formula
            'P_L2_norm': q / np.sqrt(q**2 + r_safe**2),  # Modified formula
            'q_over_r': q / r_safe,  # Ratio of rates
        }

        return features

    def compute_level2_nonlinear(self, u: np.ndarray, v: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Level 2: Non-linear transformations to capture power-law effects.

        Relaxes assumption of linear relationship with degrees.
        Power-law networks have log-normal degree effects.
        """
        features = {
            'log_u': np.log1p(u),  # log(1 + u) for numerical stability
            'log_v': np.log1p(v),
            'log_product': np.log1p(u * v),
            'sqrt_product': np.sqrt(u * v),
            'geometric_mean': np.sqrt(u * v),
            'arithmetic_mean': (u + v) / 2,
            'degree_asymmetry': np.abs(u - v),
            'degree_ratio': u / (u + v + 1e-10),  # Proportion of source degree
            'harmonic_mean': 2 * u * v / (u + v + 1e-10),
        }

        return features

    def compute_level3_graph_stats(self, u: np.ndarray, v: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Level 3: Graph-specific statistics for edge-type adaptation.

        Each edge type has different density, degree distributions, and
        bipartite structure. These features allow the model to learn
        edge-type-specific corrections.
        """
        n = len(u)

        features = {
            # Global statistics (constant per edge type, but informative for multi-edge training)
            'm_total': np.full(n, self.m, dtype=np.float32),
            'density': np.full(n, self.density, dtype=np.float32),
            'log_density': np.full(n, np.log1p(self.density), dtype=np.float32),
            'n_source': np.full(n, self.n_source, dtype=np.float32),
            'n_target': np.full(n, self.n_target, dtype=np.float32),

            # Degree distribution statistics
            'mean_source_deg': np.full(n, self.mean_source_deg, dtype=np.float32),
            'mean_target_deg': np.full(n, self.mean_target_deg, dtype=np.float32),
            'std_source_deg': np.full(n, self.std_source_deg, dtype=np.float32),
            'std_target_deg': np.full(n, self.std_target_deg, dtype=np.float32),

            # Normalized degrees (z-scores)
            'u_zscore': (u - self.mean_source_deg) / (self.std_source_deg + 1e-10),
            'v_zscore': (v - self.mean_target_deg) / (self.std_target_deg + 1e-10),
        }

        return features

    def compute_level4_polynomial(self, u: np.ndarray, v: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Level 4: Polynomial corrections for systematic bias reduction.

        The analytical formula systematically underpredicts. Polynomial terms
        allow the model to learn corrections, especially at high frequencies.
        """
        q = u * v
        r = self.m - u - v + 1

        features = {
            # Quadratic terms
            'u_squared': u**2,
            'v_squared': v**2,
            'product_squared': q**2,
            'sum_squared': (u + v)**2,
            'removal_squared': r**2,

            # Cubic terms (high-degree saturation)
            'u_cubed': u**3,
            'v_cubed': v**3,

            # Mixed polynomial terms
            'u2_v': u**2 * v,
            'u_v2': u * v**2,
            'uv_sum': (u * v) * (u + v),
        }

        return features

    def compute_level5_interactions(self, u: np.ndarray, v: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Level 5: Cross-feature interactions to relax independence assumption.

        XSwap creates correlations between edges. These interaction terms
        allow the model to capture how graph structure modulates degree effects.
        """
        q = u * v
        r = self.m - u - v + 1

        features = {
            # Product with graph statistics
            'product_times_density': q * self.density,
            'product_div_graph_size': q / np.sqrt(self.n_source * self.n_target),
            'product_normalized_by_m': q / self.m,

            # Log-log interactions (power-law effects)
            'log_product_times_log_m': np.log1p(q) * np.log1p(self.m),
            'log_u_times_log_v': np.log1p(u) * np.log1p(v),

            # Degree sum interactions
            'degree_sum_times_removal': (u + v) * r,
            'degree_sum_div_m': (u + v) / self.m,

            # Analytical formula components normalized
            'analytical_normalized': np.sqrt(q**2 + r**2) / self.m,
            'q_over_sqrt_m': q / np.sqrt(self.m),
            'r_over_sqrt_m': r / np.sqrt(self.m),
        }

        return features

    def compute_all_features(self, u: np.ndarray, v: np.ndarray,
                            levels: Tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
        """
        Compute all requested feature levels.

        Parameters
        ----------
        u : array
            Source node degrees
        v : array
            Target node degrees
        levels : tuple
            Which feature levels to include (1-5)

        Returns
        -------
        features_df : DataFrame
            All features as columns
        """
        all_features = {}

        if 1 in levels:
            all_features.update(self.compute_level1_analytical(u, v))
        if 2 in levels:
            all_features.update(self.compute_level2_nonlinear(u, v))
        if 3 in levels:
            all_features.update(self.compute_level3_graph_stats(u, v))
        if 4 in levels:
            all_features.update(self.compute_level4_polynomial(u, v))
        if 5 in levels:
            all_features.update(self.compute_level5_interactions(u, v))

        return pd.DataFrame(all_features)

    def get_analytical_baseline(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute analytical formula predictions (L2-norm version).

        This serves as a baseline for comparison.
        """
        q = u * v
        r = self.m - u - v + 1
        return q / np.sqrt(q**2 + r**2)


def prepare_training_data(edge_matrix: sp.spmatrix,
                         edge_type: str,
                         empirical_frequencies: pd.DataFrame,
                         feature_levels: Tuple[int, ...] = (1, 2, 3, 4, 5),
                         sample_ratio: float = 0.1) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare training data with theory-guided features.

    Parameters
    ----------
    edge_matrix : sparse matrix
        Adjacency matrix for the edge type
    edge_type : str
        Edge type identifier
    empirical_frequencies : DataFrame
        Empirical frequencies by degree (from notebook 3)
        Columns: source_degree, target_degree, frequency
    feature_levels : tuple
        Which feature levels to include
    sample_ratio : float
        Sampling ratio for negative examples

    Returns
    -------
    X : DataFrame
        Feature matrix
    y : array
        Target empirical frequencies
    """
    # Initialize feature engineer
    fe = TheoryGuidedFeatureEngineer(edge_matrix, edge_type)

    # Extract degrees and frequencies
    u = empirical_frequencies['source_degree'].values
    v = empirical_frequencies['target_degree'].values
    y = empirical_frequencies['frequency'].values

    # Compute features
    X = fe.compute_all_features(u, v, levels=feature_levels)

    return X, y
