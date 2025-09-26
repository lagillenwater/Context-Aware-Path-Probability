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
    """Simple Neural Network with 2 hidden layers for continuous edge probability prediction."""

    def __init__(self, input_dim: int = 2, hidden_dims: Tuple[int, int] = (128, 64), dropout_rate: float = 0.3, use_class_weights: bool = False):
        super(SimpleNN, self).__init__()
        self.use_class_weights = use_class_weights

        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Second hidden layer
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Output layer (logits for BCEWithLogitsLoss if using class weights, else sigmoid)
            nn.Linear(hidden_dims[1], 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
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

    def create_models(self, use_class_weights: bool = False) -> Dict[str, Any]:
        """Create all models for comparison.

        Parameters:
        -----------
        use_class_weights : bool
            Whether to configure neural network for class imbalance handling
        """

        # 1. Simple Neural Network
        torch.manual_seed(self.random_state)
        simple_nn = SimpleNN(input_dim=2, hidden_dims=(128, 64), dropout_rate=0.3, use_class_weights=use_class_weights)

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
        return {
            'Simple NN': 'Neural Network with 2 hidden layers (128, 64 neurons), ReLU activation, dropout=0.3',
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


def prepare_edge_features_and_labels(edge_file_path: str, sample_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features (source_degree, target_degree) and labels (edge_exists) from edge matrix.

    Parameters:
    -----------
    edge_file_path : str
        Path to the sparse edge matrix file
    sample_ratio : float
        Ratio of negative samples to include (to balance dataset)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Features array (N, 2) and labels array (N,)
    """
    import scipy.sparse as sp

    # Load sparse matrix
    edge_matrix = sp.load_npz(edge_file_path)
    n_sources, n_targets = edge_matrix.shape

    # Calculate degrees
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    # Get positive edges (existing edges)
    positive_edges = list(zip(*edge_matrix.nonzero()))
    n_positive = len(positive_edges)

    # Sample negative edges (non-existing edges)
    all_possible_edges = set((i, j) for i in range(n_sources) for j in range(n_targets))
    negative_edges = list(all_possible_edges - set(positive_edges))

    # Sample negative edges to balance the dataset
    n_negative_sample = min(int(n_positive / sample_ratio), len(negative_edges))
    np.random.seed(42)  # For reproducibility
    negative_sample = np.random.choice(len(negative_edges), n_negative_sample, replace=False)
    negative_edges_sampled = [negative_edges[i] for i in negative_sample]

    # Combine positive and negative examples
    all_edges = positive_edges + negative_edges_sampled
    all_labels = [1.0] * n_positive + [0.0] * len(negative_edges_sampled)

    # Create features array
    features = np.array([[source_degrees[i], target_degrees[j]] for i, j in all_edges])
    labels = np.array(all_labels)

    print(f"Dataset prepared:")
    print(f"  Positive examples: {n_positive}")
    print(f"  Negative examples: {len(negative_edges_sampled)}")
    print(f"  Total examples: {len(features)}")
    print(f"  Feature shape: {features.shape}")

    return features, labels


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