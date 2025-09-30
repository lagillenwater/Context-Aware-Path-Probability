"""
Model definitions for edge probability prediction comparison.
Contains 4 different models: Simple NN, Random Forest, Logistic Regression, and Polynomial Logistic Regression.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple


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

    def create_models(self, use_class_weights: bool = False, input_dim: int = 2) -> Dict[str, Any]:
        """Create all models for comparison.

        Parameters:
        -----------
        use_class_weights : bool
            Whether to configure neural network for class imbalance handling
        input_dim : int
            Number of input features for the neural network
        """
        # Store input dimension for model descriptions
        self.input_dim = input_dim

        # 1. Simple Neural Network - high performance architecture
        torch.manual_seed(self.random_state)
        simple_nn = SimpleNN(input_dim=input_dim, hidden_dims=(128, 64, 32), dropout_rate=0.3, use_class_weights=use_class_weights)

        # 2. Random Forest Classifier
        random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )

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
        if self.input_dim == 2:
            nn_description = f'High-performance Neural Network with 3 hidden layers (128, 64, 32 neurons), BatchNorm, Focal Loss, {self.input_dim} features (source_degree, target_degree)'
        else:
            nn_description = f'High-performance Neural Network with 3 hidden layers (128, 64, 32 neurons), BatchNorm, Focal Loss, {self.input_dim} enhanced features (degrees, interactions, transformations)'

        return {
            'Simple NN': nn_description,
            'Random Forest': 'Random Forest Classifier with 100 trees, max_depth=10',
            'Logistic Regression': 'Standard logistic regression with L2 regularization',
            'Polynomial Logistic Regression': 'Polynomial features (degree=2) + Logistic regression'
        }


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
    import scipy.sparse as sp

    # Load sparse matrix
    edge_matrix = sp.load_npz(edge_file_path)

    # Calculate degrees
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()  # Row sums
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()  # Column sums

    return source_degrees, target_degrees


def prepare_edge_features_and_labels(edge_file_path: str, sample_ratio: float = 0.1,
                                   adaptive_sampling: bool = True, enhanced_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels from edge matrix with adaptive sampling and enhanced features.

    Parameters:
    -----------
    edge_file_path : str
        Path to the sparse edge matrix file
    sample_ratio : float
        Base ratio of negative samples to include (adapted based on edge density)
    adaptive_sampling : bool
        Whether to use adaptive sampling strategy based on edge density
    enhanced_features : bool
        Whether to include domain-specific engineered features

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Features array (N, feature_dim) and labels array (N,)
    """
    import scipy.sparse as sp

    # Load sparse matrix
    edge_matrix = sp.load_npz(edge_file_path)
    n_sources, n_targets = edge_matrix.shape

    # Calculate degrees
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    # Calculate edge density for adaptive sampling
    total_possible = n_sources * n_targets
    n_positive = edge_matrix.nnz
    edge_density = n_positive / total_possible

    print(f"Edge matrix statistics:")
    print(f"  Shape: {edge_matrix.shape}")
    print(f"  Existing edges: {n_positive:,}")
    print(f"  Edge density: {edge_density:.6f}")

    # Adaptive sampling based on edge density
    if adaptive_sampling:
        if edge_density < 0.0001:  # Very sparse (< 0.01%)
            adapted_ratio = 0.05  # More balanced sampling
            stratified = True
            print(f"  Very sparse edge type - using adapted ratio: {adapted_ratio}")
        elif edge_density < 0.001:  # Sparse (< 0.1%)
            adapted_ratio = 0.02  # Slightly more balanced
            stratified = True
            print(f"  Sparse edge type - using adapted ratio: {adapted_ratio}")
        else:  # Dense enough for standard sampling
            adapted_ratio = sample_ratio
            stratified = False
            print(f"  Dense edge type - using standard ratio: {adapted_ratio}")
    else:
        adapted_ratio = sample_ratio
        stratified = False

    # Get positive edges (existing edges)
    positive_edges = list(zip(*edge_matrix.nonzero()))

    # Enhanced negative sampling
    if stratified and adaptive_sampling:
        # Stratified sampling by degree bins to ensure coverage
        negative_edges_sampled = _stratified_negative_sampling(
            n_sources, n_targets, positive_edges, source_degrees, target_degrees,
            n_positive, adapted_ratio
        )
    else:
        # Standard random sampling
        all_possible_edges = set((i, j) for i in range(n_sources) for j in range(n_targets))
        negative_edges = list(all_possible_edges - set(positive_edges))

        n_negative_sample = min(int(n_positive / adapted_ratio), len(negative_edges))
        np.random.seed(42)  # For reproducibility
        negative_sample = np.random.choice(len(negative_edges), n_negative_sample, replace=False)
        negative_edges_sampled = [negative_edges[i] for i in negative_sample]

    # Combine positive and negative examples
    all_edges = positive_edges + negative_edges_sampled
    all_labels = [1.0] * n_positive + [0.0] * len(negative_edges_sampled)

    # Enhanced feature engineering
    if enhanced_features:
        features = _create_enhanced_features(all_edges, source_degrees, target_degrees,
                                           n_sources, n_targets, edge_density)
        feature_names = _get_enhanced_feature_names()
    else:
        # Basic features: source_degree, target_degree
        features = np.array([[source_degrees[i], target_degrees[j]]
                             for i, j in all_edges])
        feature_names = ['source_degree', 'target_degree']

    labels = np.array(all_labels)

    print(f"Dataset prepared:")
    print(f"  Positive examples: {n_positive:,}")
    print(f"  Negative examples: {len(negative_edges_sampled):,}")
    print(f"  Total examples: {len(features):,}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Features: {feature_names}")

    return features, labels


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
                      n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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