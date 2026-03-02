"""
Reduced Feature Sets for Theory-Guided Edge Probability Prediction

Defines three feature tiers based on feature importance analysis:
- Minimal (17 features): Universal core for easy edge types
- Standard (20 features): Add polynomial corrections for medium edge types
- Extended (25 features): Add graph stats for extreme edge types

Based on comprehensive feature importance analysis showing 65% of features
can be removed with minimal performance loss.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import scipy.sparse as sp
from theory_guided_features import TheoryGuidedFeatureEngineer


class ReducedFeatureSet:
    """
    Reduced feature sets based on feature importance analysis.
    """

    # Define feature sets based on importance rankings
    MINIMAL_FEATURES = {
        'Level 1 (Analytical)': [
            'u', 'v',
            'degree_product',
            'P_L2_norm',
            'q_normalized',
            'q_over_r',
        ],
        'Level 2 (Nonlinear)': [
            'log_u', 'log_v', 'log_product',
            'sqrt_product',
            'geometric_mean',
            'arithmetic_mean',
            'harmonic_mean',
        ],
    }

    STANDARD_ADDITIONS = {
        'Level 4 (Polynomial)': [
            'sum_squared',
            'v_squared',
            'u_v2',
        ],
    }

    EXTENDED_ADDITIONS = {
        'Level 3 (Graph Stats)': [
            'density',
            'u_zscore',
            'v_zscore',
        ],
        'Level 5 (Interactions)': [
            'log_product_times_log_m',
            'product_div_graph_size',
        ],
    }

    @staticmethod
    def get_feature_list(tier: str) -> List[str]:
        """
        Get list of features for a specific tier.

        Parameters
        ----------
        tier : str
            Feature tier: 'minimal', 'standard', or 'extended'

        Returns
        -------
        features : list
            List of feature names
        """
        features = []

        # Always include minimal features
        for level_features in ReducedFeatureSet.MINIMAL_FEATURES.values():
            features.extend(level_features)

        if tier in ['standard', 'extended']:
            # Add standard features
            for level_features in ReducedFeatureSet.STANDARD_ADDITIONS.values():
                features.extend(level_features)

        if tier == 'extended':
            # Add extended features
            for level_features in ReducedFeatureSet.EXTENDED_ADDITIONS.values():
                features.extend(level_features)

        return features

    @staticmethod
    def get_feature_count(tier: str) -> int:
        """Get number of features in tier."""
        return len(ReducedFeatureSet.get_feature_list(tier))

    @staticmethod
    def compute_reduced_features(fe: TheoryGuidedFeatureEngineer,
                                 u: np.ndarray,
                                 v: np.ndarray,
                                 tier: str = 'minimal') -> pd.DataFrame:
        """
        Compute only the features needed for specified tier.

        Parameters
        ----------
        fe : TheoryGuidedFeatureEngineer
            Feature engineer instance
        u : array
            Source degrees
        v : array
            Target degrees
        tier : str
            Feature tier: 'minimal', 'standard', or 'extended'

        Returns
        -------
        features_df : DataFrame
            Reduced feature matrix
        """
        # Determine which levels to compute
        if tier == 'minimal':
            levels = (1, 2)
        elif tier == 'standard':
            levels = (1, 2, 4)
        elif tier == 'extended':
            levels = (1, 2, 3, 4, 5)
        else:
            raise ValueError(f"Unknown tier: {tier}")

        # Compute all features for needed levels
        all_features = fe.compute_all_features(u, v, levels=levels)

        # Select only the features in this tier's list
        selected_features = ReducedFeatureSet.get_feature_list(tier)

        # Filter to features that exist
        available_features = [f for f in selected_features if f in all_features.columns]

        if len(available_features) < len(selected_features):
            missing = set(selected_features) - set(available_features)
            print(f"Warning: {len(missing)} features not available: {missing}")

        return all_features[available_features]


def classify_edge_type_difficulty(edge_matrix: sp.spmatrix,
                                  analytical_correlation: float = None,
                                  empirical_df: pd.DataFrame = None) -> str:
    """
    Classify edge type as easy, medium, or hard based on characteristics.

    Criteria:
    - Easy: analytical r > 0.985, degree range < 500
    - Hard: analytical r < 0.975 OR max degree > 5000
    - Medium: everything else

    Parameters
    ----------
    edge_matrix : sparse matrix
        Edge adjacency matrix
    analytical_correlation : float, optional
        Correlation of analytical formula with empirical
    empirical_df : DataFrame, optional
        Empirical frequencies for computing characteristics

    Returns
    -------
    difficulty : str
        'easy', 'medium', or 'hard'
    """
    # Compute degree statistics
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    max_source_deg = source_degrees.max()
    max_target_deg = target_degrees.max()
    max_degree = max(max_source_deg, max_target_deg)

    # Compute density
    density = edge_matrix.nnz / (edge_matrix.shape[0] * edge_matrix.shape[1])

    # Classification logic
    if analytical_correlation is not None:
        if analytical_correlation > 0.985 and max_degree < 500:
            return 'easy'
        elif analytical_correlation < 0.975 or max_degree > 5000:
            return 'hard'
        else:
            return 'medium'
    else:
        # Heuristic based on degree range only
        if max_degree < 500:
            return 'easy'
        elif max_degree > 5000 or density > 0.25:
            return 'hard'
        else:
            return 'medium'


def recommend_feature_tier(edge_type: str,
                           edge_matrix: sp.spmatrix,
                           analytical_correlation: float = None) -> Tuple[str, str]:
    """
    Recommend appropriate feature tier for an edge type.

    Parameters
    ----------
    edge_type : str
        Edge type identifier
    edge_matrix : sparse matrix
        Edge adjacency matrix
    analytical_correlation : float, optional
        Analytical formula correlation

    Returns
    -------
    tier : str
        Recommended tier ('minimal', 'standard', 'extended')
    reasoning : str
        Explanation of recommendation
    """
    difficulty = classify_edge_type_difficulty(edge_matrix, analytical_correlation)

    # Compute statistics for reasoning
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()
    max_degree = max(source_degrees.max(), target_degrees.max())
    density = edge_matrix.nnz / (edge_matrix.shape[0] * edge_matrix.shape[1])

    # Map difficulty to tier
    tier_map = {
        'easy': 'minimal',
        'medium': 'standard',
        'hard': 'extended'
    }
    tier = tier_map[difficulty]

    # Generate reasoning
    if analytical_correlation is not None:
        reasoning = f"""
Edge type: {edge_type}
Difficulty: {difficulty}
Recommended tier: {tier} ({ReducedFeatureSet.get_feature_count(tier)} features)

Characteristics:
  - Max degree: {max_degree}
  - Density: {density:.4f}
  - Analytical r: {analytical_correlation:.4f}

Reasoning:
"""
    else:
        reasoning = f"""
Edge type: {edge_type}
Difficulty: {difficulty} (heuristic)
Recommended tier: {tier} ({ReducedFeatureSet.get_feature_count(tier)} features)

Characteristics:
  - Max degree: {max_degree}
  - Density: {density:.4f}
  - Analytical r: unknown

Reasoning:
"""

    if difficulty == 'easy':
        reasoning += f"  - Well-behaved degree distribution (max={max_degree})\n"
        reasoning += f"  - Analytical formula performs well (r={analytical_correlation:.3f})\n" if analytical_correlation else ""
        reasoning += "  - Minimal feature set (17 features) should achieve r > 0.97\n"
    elif difficulty == 'hard':
        if max_degree > 5000:
            reasoning += f"  - Extreme degree distribution (max={max_degree})\n"
        if analytical_correlation and analytical_correlation < 0.975:
            reasoning += f"  - Analytical formula underperforms (r={analytical_correlation:.3f})\n"
        if density > 0.25:
            reasoning += f"  - High density ({density:.2%}) creates complex patterns\n"
        reasoning += "  - Extended feature set (25 features) needed for edge-type adaptation\n"
    else:  # medium
        reasoning += "  - Moderate complexity, standard polynomial corrections needed\n"
        reasoning += "  - Standard feature set (20 features) balances performance and simplicity\n"

    return tier, reasoning


def print_feature_set_summary():
    """
    Print summary of all feature tiers.
    """
    print("="*80)
    print("REDUCED FEATURE SETS SUMMARY")
    print("="*80)

    for tier in ['minimal', 'standard', 'extended']:
        features = ReducedFeatureSet.get_feature_list(tier)
        print(f"\n{tier.upper()} ({len(features)} features):")
        print("-"*60)

        # Group by level
        level_features = {}
        for f in features:
            # Determine level based on feature name patterns
            if any(x in f for x in ['degree_product', 'removal_term', 'P_L', 'q_', 'r_', 'u', 'v']):
                if f not in ['u_zscore', 'v_zscore', 'u_v2', 'u2_v', 'u_cubed', 'v_cubed', 'u_squared', 'v_squared']:
                    level = 'Level 1: Analytical'
            if any(x in f for x in ['log_', 'sqrt_', 'geometric', 'arithmetic', 'harmonic', 'asymmetry', 'ratio']):
                level = 'Level 2: Nonlinear'
            elif any(x in f for x in ['m_total', 'density', 'zscore', 'n_source', 'n_target', 'mean_', 'std_']):
                level = 'Level 3: Graph Stats'
            elif any(x in f for x in ['squared', 'cubed', 'u2_v', 'u_v2', 'uv_sum']):
                level = 'Level 4: Polynomial'
            elif any(x in f for x in ['times_', 'div_', 'normalized_by']):
                level = 'Level 5: Interactions'

            if level not in level_features:
                level_features[level] = []
            level_features[level].append(f)

        for level, feats in sorted(level_features.items()):
            print(f"\n  {level}: ({len(feats)} features)")
            for f in feats:
                print(f"    - {f}")

    print("\n" + "="*80)
    print("FEATURE REDUCTION SUMMARY")
    print("="*80)
    print(f"Original: 49 features (all 5 levels)")
    print(f"Minimal:  {ReducedFeatureSet.get_feature_count('minimal')} features (73% reduction)")
    print(f"Standard: {ReducedFeatureSet.get_feature_count('standard')} features (67% reduction)")
    print(f"Extended: {ReducedFeatureSet.get_feature_count('extended')} features (57% reduction)")
    print("="*80)
