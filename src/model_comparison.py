"""
Model definitions for edge probability prediction comparison.
Contains 4 different models: Simple NN, Random Forest, Logistic Regression, and Polynomial Logistic Regression.
"""

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, Optional


class SimpleNN(nn.Module):
    """High-performance Neural Network with engineered features for edge probability prediction."""

    def __init__(self, input_dim: int = 2, hidden_dims: Tuple[int, int, int] = (128, 64, 32), dropout_rate: float = 0.3, use_class_weights: bool = False):
        super(SimpleNN, self).__init__()
        self.use_class_weights = use_class_weights

        self.network = nn.Sequential(
            # First hidden layer - increased capacity
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Second hidden layer
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Third hidden layer for better representation
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout in final layer

            # Output layer
            nn.Linear(hidden_dims[2], 1)
        )

        # Initialize weights with better initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights with improved initialization."""
        if isinstance(module, nn.Linear):
            # Use He initialization for ReLU networks
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        output = self.network(x).squeeze()
        # Apply sigmoid if not using class weights (BCEWithLogitsLoss handles sigmoid internally)
        if not self.use_class_weights:
            output = torch.sigmoid(output)
        return output


class ModelCollection:
    """Collection of different models for edge probability prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.input_dim = 2  # Track input dimension for model descriptions
        self.rf_params = {}  # Track RF parameters for descriptions

    def create_models(self, use_class_weights: bool = False, input_dim: int = 2,
                     edge_file_path: str = None) -> Dict[str, Any]:
        """Create all models for comparison with intelligent parameter adaptation.

        Parameters:
        -----------
        use_class_weights : bool
            Whether to configure neural network for class imbalance handling
        input_dim : int
            Number of input features for the neural network (always 2: source_degree, target_degree)
        edge_file_path : str
            Optional path to edge file for intelligent parameter adaptation

        Returns:
        --------
        Dict[str, Any]
            Dictionary of model name to model instance
        """
        # Store input dimension for model descriptions
        self.input_dim = 2  # Always use 2 features

        # Calculate adaptive Random Forest parameters if edge file provided
        if edge_file_path is not None:
            rf_params = _calculate_adaptive_rf_parameters(edge_file_path, self.random_state)
            self.rf_params = rf_params
        else:
            # Default parameters for large, dense datasets
            rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            self.rf_params = rf_params

        # 1. Simple Neural Network - high performance architecture
        torch.manual_seed(self.random_state)
        simple_nn = SimpleNN(input_dim=2, hidden_dims=(128, 64, 32), dropout_rate=0.3, use_class_weights=use_class_weights)

        # 2. Random Forest Classifier with adaptive parameters
        random_forest = RandomForestClassifier(**rf_params)

        # 3. Logistic Regression
        logistic_regression = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )

        # 4. Polynomial Logistic Regression (degree 2)
        polynomial_logistic_regression = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('logistic', LogisticRegression(random_state=self.random_state, max_iter=1000))
        ])

        self.models = {
            'Simple NN': simple_nn,
            'Random Forest': random_forest,
            'Logistic Regression': logistic_regression,
            'Polynomial Logistic Regression': polynomial_logistic_regression
        }

        return self.models

    def get_model_info(self) -> Dict[str, str]:
        """Get description of each model."""
        # Build RF description from actual parameters
        if self.rf_params:
            rf_desc = (f"Random Forest Classifier with {self.rf_params.get('n_estimators', 100)} trees, "
                      f"max_depth={self.rf_params.get('max_depth', 10)}, "
                      f"min_samples_split={self.rf_params.get('min_samples_split', 2)}, "
                      f"min_samples_leaf={self.rf_params.get('min_samples_leaf', 1)}")
        else:
            rf_desc = 'Random Forest Classifier with 100 trees, max_depth=10'

        return {
            'Simple NN': 'High-performance Neural Network with 3 hidden layers (128, 64, 32 neurons), BatchNorm, Weighted Cross-Entropy, 2 features (source_degree, target_degree)',
            'Random Forest': rf_desc,
            'Logistic Regression': 'Standard logistic regression with L2 regularization',
            'Polynomial Logistic Regression': 'Polynomial features (degree=2) + Logistic regression'
        }


def filter_zero_degree_nodes(edge_matrix: sp.csr_matrix) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Filter out nodes with zero degree from the bipartite graph.

    Zero-degree nodes provide no signal for degree-based prediction and only add noise.
    Filtering them improves model performance, especially for sparse datasets.

    Parameters:
    -----------
    edge_matrix : sp.csr_matrix
        The bipartite graph adjacency matrix

    Returns:
    --------
    Tuple[sp.csr_matrix, np.ndarray, np.ndarray]
        - filtered_matrix: Matrix with only non-zero-degree nodes
        - source_idx_mapping: Original indices of retained source nodes
        - target_idx_mapping: Original indices of retained target nodes
    """
    # Calculate degrees
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    # Find non-zero degree nodes
    non_zero_sources = source_degrees > 0
    non_zero_targets = target_degrees > 0

    # Count removals
    n_sources_removed = np.sum(~non_zero_sources)
    n_targets_removed = np.sum(~non_zero_targets)

    # Filter matrix
    filtered_matrix = edge_matrix[non_zero_sources, :][:, non_zero_targets]

    # Create index mappings (for potential future use)
    source_idx_mapping = np.where(non_zero_sources)[0]
    target_idx_mapping = np.where(non_zero_targets)[0]

    # Print filtering statistics
    if n_sources_removed > 0 or n_targets_removed > 0:
        orig_density = edge_matrix.nnz / (edge_matrix.shape[0] * edge_matrix.shape[1])
        new_density = filtered_matrix.nnz / (filtered_matrix.shape[0] * filtered_matrix.shape[1])

        print(f"\n  Zero-degree node filtering:")
        print(f"    Removed {n_sources_removed} sources and {n_targets_removed} targets with degree=0")
        print(f"    Retained: {filtered_matrix.shape[0]} sources × {filtered_matrix.shape[1]} targets")
        print(f"    Density: {orig_density:.6f} → {new_density:.6f} ({new_density/orig_density:.1f}x increase)")

    return filtered_matrix, source_idx_mapping, target_idx_mapping


def _calculate_adaptive_rf_parameters(edge_file_path: str, random_state: int = 42) -> Dict[str, Any]:
    """
    Calculate adaptive Random Forest parameters based on dataset characteristics.

    Strategy:
    - Small, sparse, low-degree datasets: Simpler RF (fewer trees, shallower depth, more regularization)
    - Large, dense, high-degree datasets: Complex RF (more trees, deeper depth, less regularization)

    Parameters:
    -----------
    edge_file_path : str
        Path to the sparse edge matrix file
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    Dict[str, Any]
        Dictionary of Random Forest parameters
    """
    # Load and filter dataset
    edge_matrix_original = sp.load_npz(edge_file_path)
    edge_matrix, _, _ = filter_zero_degree_nodes(edge_matrix_original)

    n_sources, n_targets = edge_matrix.shape
    n_edges = edge_matrix.nnz
    edge_density = n_edges / (n_sources * n_targets)

    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()
    mean_source_degree = source_degrees.mean()
    mean_target_degree = target_degrees.mean()
    avg_mean_degree = (mean_source_degree + mean_target_degree) / 2

    # Calculate dataset complexity scores
    size_score = min(1.0, np.log10(max(n_edges, 1)) / 5.0)  # 0-1, where 1 = 100K+ edges
    sparsity_score = min(1.0, edge_density / 0.1)  # 0-1, where 1 = 10%+ density
    degree_score = min(1.0, avg_mean_degree / 10.0)  # 0-1, where 1 = mean degree 10+

    # Combined complexity score
    complexity_score = 0.4 * size_score + 0.4 * sparsity_score + 0.2 * degree_score

    # Special handling for tiny datasets (< 1000 edges): apply penalty to prevent overfitting
    if n_edges < 1000:
        penalty_factor = min(0.5, n_edges / 2000)  # 0-0.5 penalty for datasets < 1000 edges
        complexity_score = complexity_score * penalty_factor
        print(f"  Tiny dataset detected ({n_edges} edges) - applying penalty factor {penalty_factor:.3f}")

    # Adapt parameters based on complexity
    # Low complexity (small, sparse, low-degree) → simpler model with strong regularization
    # High complexity (large, dense, high-degree) → complex model

    # Number of estimators: 20-50 (reduced max from 100 to 50)
    n_estimators = int(20 + 30 * complexity_score)

    # Max depth: 3-5 (reduced max from 10 to 5)
    max_depth = int(3 + 2 * complexity_score)

    # Min samples split: 50-10 (increased min from 2 to 10)
    min_samples_split = int(50 - 40 * complexity_score)

    # Min samples leaf: 30-5 (increased from 20-1 to 30-5)
    min_samples_leaf = int(30 - 25 * complexity_score)

    # Estimate training samples (assuming 80% train split and adaptive sampling)
    # This is a rough estimate - actual value depends on sampling ratio
    total_possible = n_sources * n_targets
    estimated_neg_samples = int((total_possible - n_edges) * 0.15)  # Conservative estimate
    estimated_training = int((n_edges + estimated_neg_samples) * 0.8)

    # Add training size-aware max_depth cap to prevent overfitting
    # Rule: max_depth should not create more leaves than we have samples per tree
    # Each tree uses bootstrap sample (typically ~63% of data)
    samples_per_tree = int(estimated_training * 0.63)
    max_leaves_per_tree = 2 ** max_depth

    if samples_per_tree < max_leaves_per_tree * 10:  # Want at least 10 samples per leaf
        size_aware_max_depth = max(3, int(np.log2(samples_per_tree / 10)))
        if size_aware_max_depth < max_depth:
            print(f"  Training size cap: Reducing max_depth from {max_depth} to {size_aware_max_depth}")
            print(f"    (estimated {samples_per_tree} samples/tree, need 10+ samples/leaf)")
            max_depth = size_aware_max_depth

    # Add bootstrap aggregation control for small datasets
    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'random_state': random_state,
        'n_jobs': -1
    }

    # Add max_samples for small datasets to reduce overfitting
    if estimated_training < 5000:
        params['max_samples'] = 0.8
        print(f"  Small dataset: Setting max_samples=0.8 for additional regularization")

    print(f"\nAdaptive Random Forest parameters:")
    print(f"  Dataset size: {n_edges} edges, ~{estimated_training} training samples")
    print(f"  Complexity score: {complexity_score:.3f}")
    print(f"    (size: {size_score:.3f}, sparsity: {sparsity_score:.3f}, degree: {degree_score:.3f})")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth} (→ max {2**max_depth} leaves/tree)")
    print(f"  min_samples_split: {min_samples_split}")
    print(f"  min_samples_leaf: {min_samples_leaf}")
    if 'max_samples' in params:
        print(f"  max_samples: {params['max_samples']}")

    return params


def load_edge_data(edge_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load edge data from sparse matrix file.

    Parameters:
    -----------
    edge_file_path : str
        Path to the sparse edge matrix file

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Source degrees and target degrees for all possible edges
    """
    # Load sparse matrix
    edge_matrix = sp.load_npz(edge_file_path)

    # Calculate degrees
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()  # Row sums
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()  # Column sums

    return source_degrees, target_degrees


def prepare_edge_features_and_labels(edge_file_path: str, sample_ratio: float = 0.1,
                                   adaptive_sampling: bool = False, enhanced_features: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features (source_degree, target_degree) and labels (edge_exists) from edge matrix.
    Intelligently adapts sampling ratio based on dataset characteristics.

    Parameters:
    -----------
    edge_file_path : str
        Path to the sparse edge matrix file
    sample_ratio : float
        Base ratio of negative samples to include (adapted based on dataset)
    adaptive_sampling : bool
        Whether to use adaptive sampling based on dataset characteristics
    enhanced_features : bool
        Unused parameter, kept for compatibility

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Features array (N, 2) with source_degree and target_degree, and labels array (N,)
    """
    # Load sparse matrix
    edge_matrix_original = sp.load_npz(edge_file_path)

    print(f"Original edge matrix:")
    print(f"  Shape: {edge_matrix_original.shape}")
    print(f"  Edges: {edge_matrix_original.nnz:,}")

    # ALWAYS filter zero-degree nodes
    edge_matrix, source_mapping, target_mapping = filter_zero_degree_nodes(edge_matrix_original)

    # Work with filtered matrix from here on
    n_sources, n_targets = edge_matrix.shape

    # Calculate degrees (from filtered matrix)
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    # Get positive edges (existing edges)
    positive_edges = list(zip(*edge_matrix.nonzero()))
    n_positive = len(positive_edges)

    # Calculate dataset characteristics for adaptive sampling
    total_possible = n_sources * n_targets
    edge_density = n_positive / total_possible
    mean_source_degree = source_degrees.mean()
    mean_target_degree = target_degrees.mean()

    print(f"\nFiltered edge matrix statistics:")
    print(f"  Shape: {edge_matrix.shape}")
    print(f"  Existing edges: {n_positive:,}")
    print(f"  Edge density: {edge_density:.6f} ({edge_density*100:.3f}%)")
    print(f"  Mean degrees: source={mean_source_degree:.2f}, target={mean_target_degree:.2f}")

    # Intelligent adaptive sampling based on dataset characteristics
    if adaptive_sampling:
        adapted_ratio = _calculate_adaptive_sample_ratio(
            n_positive, edge_density, mean_source_degree, mean_target_degree, sample_ratio
        )
    else:
        adapted_ratio = sample_ratio

    print(f"  Sampling ratio: {adapted_ratio:.3f} (1 positive : {1/adapted_ratio:.1f} negatives)")

    # Sample negative edges (non-existing edges)
    all_possible_edges = set((i, j) for i in range(n_sources) for j in range(n_targets))
    negative_edges = list(all_possible_edges - set(positive_edges))

    # Sample negative edges to balance the dataset
    n_negative_sample = min(int(n_positive / adapted_ratio), len(negative_edges))
    np.random.seed(42)  # For reproducibility
    negative_sample = np.random.choice(len(negative_edges), n_negative_sample, replace=False)
    negative_edges_sampled = [negative_edges[i] for i in negative_sample]

    # Combine positive and negative examples
    all_edges = positive_edges + negative_edges_sampled
    all_labels = [1.0] * n_positive + [0.0] * len(negative_edges_sampled)

    # Create features array: only source_degree and target_degree
    features = np.array([[source_degrees[i], target_degrees[j]]
                         for i, j in all_edges])
    labels = np.array(all_labels)

    print(f"Dataset prepared:")
    print(f"  Positive examples: {n_positive:,}")
    print(f"  Negative examples: {len(negative_edges_sampled):,}")
    print(f"  Total examples: {len(features):,}")
    print(f"  Positive ratio: {n_positive / len(features):.1%}")
    print(f"  Feature shape: {features.shape}")

    return features, labels


def _calculate_adaptive_sample_ratio(n_positive: int, edge_density: float,
                                     mean_source_degree: float, mean_target_degree: float,
                                     base_ratio: float) -> float:
    """
    Calculate adaptive sampling ratio based on dataset characteristics.

    Strategy:
    - Small, sparse, low-degree datasets (like CtD): More balanced sampling (0.1-0.2)
    - Large, dense, high-degree datasets (like AeG): Standard sparse sampling (0.01)

    Parameters:
    -----------
    n_positive : int
        Number of positive edges
    edge_density : float
        Proportion of actual edges to possible edges
    mean_source_degree : float
        Mean degree of source nodes
    mean_target_degree : float
        Mean degree of target nodes
    base_ratio : float
        Base sampling ratio to start from

    Returns:
    --------
    float
        Adapted sampling ratio
    """
    # Calculate dataset size score (0 = very small, 1 = very large)
    # Using log scale: < 1K edges = small, > 100K edges = large
    size_score = min(1.0, np.log10(max(n_positive, 1)) / 5.0)  # 10^5 = 100K edges

    # Calculate sparsity score (0 = very sparse, 1 = dense)
    # < 0.001 = very sparse, > 0.1 = dense
    sparsity_score = min(1.0, edge_density / 0.1)

    # Calculate degree score (0 = low degree, 1 = high degree)
    # Mean degree < 1 = low, > 10 = high
    avg_mean_degree = (mean_source_degree + mean_target_degree) / 2
    degree_score = min(1.0, avg_mean_degree / 10.0)

    # Combine scores (weighted average)
    combined_score = 0.4 * size_score + 0.4 * sparsity_score + 0.2 * degree_score

    # Adapt ratio based on combined score
    # High score (large, dense, high-degree) → use base_ratio (e.g., 0.01)
    # Low score (small, sparse, low-degree) → use more balanced ratio

    # Special handling for tiny datasets (< 1000 edges): use even more balanced sampling
    if n_positive < 1000:
        # For tiny datasets, we need more negative examples to learn from
        # Use up to 0.3 ratio (1:3.3 pos:neg) for datasets < 500 edges
        # Scale to 0.15 for datasets approaching 1000 edges
        max_ratio = 0.3 - 0.15 * (n_positive / 1000)  # 0.3 at 0 edges, 0.15 at 1000 edges
        print(f"  Tiny dataset ({n_positive} edges): Increasing max_ratio to {max_ratio:.3f} for better learning")
    else:
        max_ratio = 0.15  # Standard for small/sparse datasets

    min_ratio = base_ratio  # Original ratio for large/dense datasets

    adapted_ratio = max_ratio - (max_ratio - min_ratio) * combined_score

    print(f"  Adaptive sampling analysis:")
    print(f"    Size score: {size_score:.3f}, Sparsity score: {sparsity_score:.3f}, Degree score: {degree_score:.3f}")
    print(f"    Combined score: {combined_score:.3f} → Adapted ratio: {adapted_ratio:.3f}")
    print(f"    This gives ~{int(n_positive / adapted_ratio)} total samples ({n_positive} pos + {int(n_positive / adapted_ratio - n_positive)} neg)")

    return adapted_ratio


def _stratified_negative_sampling(n_sources: int, n_targets: int, positive_edges: list,
                                source_degrees: np.ndarray, target_degrees: np.ndarray,
                                n_positive: int, sample_ratio: float) -> list:
    """
    Stratified negative sampling to ensure degree distribution coverage.
    """
    from collections import defaultdict

    # Create degree bins
    source_bins = np.percentile(source_degrees[source_degrees > 0], [25, 50, 75]) if np.sum(source_degrees > 0) > 0 else [1, 2, 3]
    target_bins = np.percentile(target_degrees[target_degrees > 0], [25, 50, 75]) if np.sum(target_degrees > 0) > 0 else [1, 2, 3]

    def get_bin(value, bins):
        return np.digitize(value, bins)

    # Group positive edges by degree bins
    positive_by_bin = defaultdict(list)
    for i, j in positive_edges:
        src_bin = get_bin(source_degrees[i], source_bins)
        tgt_bin = get_bin(target_degrees[j], target_bins)
        positive_by_bin[(src_bin, tgt_bin)].append((i, j))

    # Sample negatives proportionally from each bin
    negative_edges_sampled = []
    total_negative_needed = int(n_positive / sample_ratio)

    # Generate potential negative edges by bin
    for src_bin in range(len(source_bins) + 1):
        for tgt_bin in range(len(target_bins) + 1):
            # Find nodes in this bin
            src_nodes = [i for i in range(n_sources)
                        if get_bin(source_degrees[i], source_bins) == src_bin]
            tgt_nodes = [j for j in range(n_targets)
                        if get_bin(target_degrees[j], target_bins) == tgt_bin]

            if not src_nodes or not tgt_nodes:
                continue

            # Calculate proportion for this bin
            bin_positive_count = len(positive_by_bin[(src_bin, tgt_bin)])
            if bin_positive_count == 0:
                continue

            bin_proportion = bin_positive_count / n_positive
            bin_negative_needed = int(total_negative_needed * bin_proportion)

            # Generate potential negative edges in this bin
            potential_negatives = []
            for src in src_nodes[:min(100, len(src_nodes))]:  # Limit for efficiency
                for tgt in tgt_nodes[:min(100, len(tgt_nodes))]:
                    if (src, tgt) not in positive_edges:
                        potential_negatives.append((src, tgt))

            # Sample from potential negatives
            if potential_negatives and bin_negative_needed > 0:
                sample_size = min(bin_negative_needed, len(potential_negatives))
                sampled = np.random.choice(len(potential_negatives), sample_size, replace=False)
                negative_edges_sampled.extend([potential_negatives[i] for i in sampled])

    # If we didn't get enough samples, fill with random sampling
    if len(negative_edges_sampled) < total_negative_needed:
        all_possible = set((i, j) for i in range(n_sources) for j in range(n_targets))
        remaining_negatives = list(all_possible - set(positive_edges) - set(negative_edges_sampled))
        additional_needed = total_negative_needed - len(negative_edges_sampled)

        if remaining_negatives:
            additional_sample = np.random.choice(
                len(remaining_negatives),
                min(additional_needed, len(remaining_negatives)),
                replace=False
            )
            negative_edges_sampled.extend([remaining_negatives[i] for i in additional_sample])

    print(f"  Stratified sampling: {len(negative_edges_sampled):,} negative samples from {len(source_bins)+1}x{len(target_bins)+1} degree bins")
    return negative_edges_sampled


def _create_enhanced_features(all_edges: list, source_degrees: np.ndarray, target_degrees: np.ndarray,
                            n_sources: int, n_targets: int, edge_density: float) -> np.ndarray:
    """
    Create enhanced feature set with domain-specific engineering.
    """
    features_list = []

    # Calculate global statistics for normalization
    max_source_degree = source_degrees.max() if len(source_degrees) > 0 else 1
    max_target_degree = target_degrees.max() if len(target_degrees) > 0 else 1
    mean_source_degree = source_degrees.mean()
    mean_target_degree = target_degrees.mean()

    for i, j in all_edges:
        src_deg = source_degrees[i]
        tgt_deg = target_degrees[j]

        # Basic features
        features = [
            src_deg,  # source_degree
            tgt_deg,  # target_degree
        ]

        # Normalized degrees
        features.extend([
            src_deg / max_source_degree,  # source_degree_normalized
            tgt_deg / max_target_degree,  # target_degree_normalized
        ])

        # Relative degrees (compared to mean)
        features.extend([
            src_deg / (mean_source_degree + 1e-8),  # source_degree_relative
            tgt_deg / (mean_target_degree + 1e-8),  # target_degree_relative
        ])

        # Degree interactions
        features.extend([
            src_deg * tgt_deg,  # degree_product
            src_deg + tgt_deg,  # degree_sum
            abs(src_deg - tgt_deg),  # degree_difference
            min(src_deg, tgt_deg),  # degree_min
            max(src_deg, tgt_deg),  # degree_max
        ])

        # Logarithmic features (handle zeros)
        features.extend([
            np.log1p(src_deg),  # log_source_degree
            np.log1p(tgt_deg),  # log_target_degree
            np.log1p(src_deg * tgt_deg),  # log_degree_product
        ])

        # Degree ratios (handle division by zero)
        if tgt_deg > 0:
            degree_ratio = src_deg / tgt_deg
        else:
            degree_ratio = src_deg  # or some default value
        features.append(degree_ratio)  # degree_ratio

        # Centrality-inspired features
        features.extend([
            src_deg / n_sources,  # source_centrality
            tgt_deg / n_targets,  # target_centrality
        ])

        # Edge density context
        features.append(edge_density)  # edge_density_context

        features_list.append(features)

    return np.array(features_list)


def _get_enhanced_feature_names() -> list:
    """
    Get names of enhanced features for documentation.
    """
    return [
        'source_degree',
        'target_degree',
        'source_degree_normalized',
        'target_degree_normalized',
        'source_degree_relative',
        'target_degree_relative',
        'degree_product',
        'degree_sum',
        'degree_difference',
        'degree_min',
        'degree_max',
        'log_source_degree',
        'log_target_degree',
        'log_degree_product',
        'degree_ratio',
        'source_centrality',
        'target_centrality',
        'edge_density_context'
    ]


def create_degree_grid(source_degrees: np.ndarray, target_degrees: np.ndarray,
                      n_bins: int = 50, enhanced_features: bool = False,
                      edge_matrix: Optional[sp.csr_matrix] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid of degree combinations for prediction visualization.

    Parameters:
    -----------
    source_degrees : np.ndarray
        Array of source degrees
    target_degrees : np.ndarray
        Array of target degrees
    n_bins : int
        Number of bins for each degree dimension
    enhanced_features : bool
        Unused parameter, kept for compatibility
    edge_matrix : Optional[sp.csr_matrix]
        Unused parameter, kept for compatibility

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Source degree bins, target degree bins, and grid features for prediction
    """
    # Create degree bins
    source_min, source_max = source_degrees.min(), source_degrees.max()
    target_min, target_max = target_degrees.min(), target_degrees.max()

    source_bins = np.linspace(source_min, source_max, n_bins)
    target_bins = np.linspace(target_min, target_max, n_bins)

    # Create meshgrid
    source_grid, target_grid = np.meshgrid(source_bins, target_bins)

    # Flatten for prediction
    grid_features = np.column_stack([source_grid.ravel(), target_grid.ravel()])

    return source_bins, target_bins, grid_features


def _create_grid_enhanced_features(basic_features: np.ndarray, edge_matrix: sp.csr_matrix) -> np.ndarray:
    """
    Create enhanced features for visualization grid points.

    Parameters:
    -----------
    basic_features : np.ndarray
        Array of shape (N, 2) with source and target degrees
    edge_matrix : sp.csr_matrix
        Sparse edge matrix for calculating context features

    Returns:
    --------
    np.ndarray
        Enhanced features array (N, 18)
    """
    n_sources, n_targets = edge_matrix.shape
    total_edges = edge_matrix.nnz
    edge_density = total_edges / (n_sources * n_targets)

    # Calculate global statistics for normalization
    source_degrees_all = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees_all = np.array(edge_matrix.sum(axis=0)).flatten()

    max_source_degree = source_degrees_all.max() if len(source_degrees_all) > 0 else 1
    max_target_degree = target_degrees_all.max() if len(target_degrees_all) > 0 else 1
    mean_source_degree = source_degrees_all.mean()
    mean_target_degree = target_degrees_all.mean()

    features_list = []

    for src_deg, tgt_deg in basic_features:
        # Create the exact same 18 enhanced features as in the training data
        features = [
            src_deg,  # source_degree
            tgt_deg,  # target_degree
        ]

        # Normalized degrees
        features.extend([
            src_deg / max_source_degree,  # source_degree_normalized
            tgt_deg / max_target_degree,  # target_degree_normalized
        ])

        # Relative degrees (compared to mean)
        features.extend([
            src_deg / (mean_source_degree + 1e-8),  # source_degree_relative
            tgt_deg / (mean_target_degree + 1e-8),  # target_degree_relative
        ])

        # Degree interactions
        features.extend([
            src_deg * tgt_deg,  # degree_product
            src_deg + tgt_deg,  # degree_sum
            abs(src_deg - tgt_deg),  # degree_difference
            min(src_deg, tgt_deg),  # degree_min
            max(src_deg, tgt_deg),  # degree_max
        ])

        # Logarithmic features (handle zeros)
        features.extend([
            np.log1p(src_deg),  # log_source_degree
            np.log1p(tgt_deg),  # log_target_degree
            np.log1p(src_deg * tgt_deg),  # log_degree_product
        ])

        # Degree ratios (handle division by zero)
        if tgt_deg > 0:
            degree_ratio = src_deg / tgt_deg
        else:
            degree_ratio = src_deg  # or some default value
        features.append(degree_ratio)  # degree_ratio

        # Centrality-inspired features
        features.extend([
            src_deg / n_sources,  # source_centrality
            tgt_deg / n_targets,  # target_centrality
        ])

        # Edge density context
        features.append(edge_density)  # edge_density_context

        features_list.append(features)

    return np.array(features_list)