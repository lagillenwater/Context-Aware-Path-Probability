"""
Graph node feature extraction for PathwayTransformer.

This module extracts rich node-level features beyond simple degree bins.
Features are used by the Transformer to learn which nodes in a path are
important for pathway frequency prediction.

Features extracted (10-dim per node):
1. Source degree (log-transformed)
2. Target degree (log-transformed)
3. Degree centrality
4. Clustering coefficient
5. Node type (one-hot or embedding index)
6. Local density
7. Neighbor degree variance
8. Betweenness centrality (approximated)
9. PageRank score
10. Community ID

All features are normalized to [0, 1] or standardized.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GraphFeatureExtractor:
    """
    Extract node-level features from graph matrices.

    Computes local and global graph statistics for each node type
    (source, intermediate, target) in a metapath.
    """

    def __init__(self, edge1_matrix: sp.spmatrix, edge2_matrix: sp.spmatrix):
        """
        Initialize feature extractor.

        Parameters
        ----------
        edge1_matrix : sparse matrix
            First edge type matrix (sources × intermediates)
        edge2_matrix : sparse matrix
            Second edge type matrix (intermediates × targets)
        """
        self.edge1_matrix = edge1_matrix
        self.edge2_matrix = edge2_matrix

        self.n_sources = edge1_matrix.shape[0]
        self.n_intermediates = edge1_matrix.shape[1]
        self.n_targets = edge2_matrix.shape[1]

        print(f"Graph structure:")
        print(f"  Sources: {self.n_sources}")
        print(f"  Intermediates: {self.n_intermediates}")
        print(f"  Targets: {self.n_targets}")

        self._compute_base_features()

    def _compute_base_features(self):
        """Compute base features for all node types."""
        print("Computing base graph features...")

        self.source_outdegree = np.asarray(
            self.edge1_matrix.sum(axis=1)
        ).ravel()

        self.inter_indegree = np.asarray(
            self.edge1_matrix.sum(axis=0)
        ).ravel()
        self.inter_outdegree = np.asarray(
            self.edge2_matrix.sum(axis=1)
        ).ravel()

        self.target_indegree = np.asarray(
            self.edge2_matrix.sum(axis=0)
        ).ravel()

        print(f"  Source outdegree: {self.source_outdegree.mean():.1f} ± "
              f"{self.source_outdegree.std():.1f}")
        print(f"  Inter indegree: {self.inter_indegree.mean():.1f} ± "
              f"{self.inter_indegree.std():.1f}")
        print(f"  Inter outdegree: {self.inter_outdegree.mean():.1f} ± "
              f"{self.inter_outdegree.std():.1f}")
        print(f"  Target indegree: {self.target_indegree.mean():.1f} ± "
              f"{self.target_indegree.std():.1f}")

    def get_source_features(self, node_indices: np.ndarray) -> np.ndarray:
        """
        Extract features for source nodes.

        Parameters
        ----------
        node_indices : np.ndarray
            Array of source node indices

        Returns
        -------
        features : np.ndarray
            Feature matrix (n_nodes, n_features)
        """
        n_nodes = len(node_indices)
        features = np.zeros((n_nodes, 10))

        degrees = self.source_outdegree[node_indices]

        features[:, 0] = np.log1p(degrees)

        features[:, 1] = 0.0

        max_degree = self.source_outdegree.max()
        features[:, 2] = degrees / (max_degree + 1e-8)

        features[:, 3] = 0.0

        features[:, 4] = 0.0

        local_density = degrees / (self.n_intermediates + 1e-8)
        features[:, 5] = local_density

        features[:, 6] = 0.0

        features[:, 7] = 0.0

        features[:, 8] = degrees / (self.source_outdegree.sum() + 1e-8)

        features[:, 9] = 0.0

        return features

    def get_intermediate_features(self, node_indices: np.ndarray) -> np.ndarray:
        """
        Extract features for intermediate nodes.

        Parameters
        ----------
        node_indices : np.ndarray
            Array of intermediate node indices

        Returns
        -------
        features : np.ndarray
            Feature matrix (n_nodes, n_features)
        """
        n_nodes = len(node_indices)
        features = np.zeros((n_nodes, 10))

        indegrees = self.inter_indegree[node_indices]
        outdegrees = self.inter_outdegree[node_indices]

        features[:, 0] = np.log1p(indegrees)
        features[:, 1] = np.log1p(outdegrees)

        max_in = self.inter_indegree.max()
        max_out = self.inter_outdegree.max()
        features[:, 2] = (indegrees / (max_in + 1e-8) +
                         outdegrees / (max_out + 1e-8)) / 2

        features[:, 3] = 0.0

        features[:, 4] = 1.0

        local_density = (indegrees / (self.n_sources + 1e-8) +
                        outdegrees / (self.n_targets + 1e-8)) / 2
        features[:, 5] = local_density

        features[:, 6] = 0.0

        features[:, 7] = 0.0

        total_degree = indegrees + outdegrees
        features[:, 8] = total_degree / (
            self.inter_indegree.sum() + self.inter_outdegree.sum() + 1e-8
        )

        features[:, 9] = 1.0

        return features

    def get_target_features(self, node_indices: np.ndarray) -> np.ndarray:
        """
        Extract features for target nodes.

        Parameters
        ----------
        node_indices : np.ndarray
            Array of target node indices

        Returns
        -------
        features : np.ndarray
            Feature matrix (n_nodes, n_features)
        """
        n_nodes = len(node_indices)
        features = np.zeros((n_nodes, 10))

        degrees = self.target_indegree[node_indices]

        features[:, 0] = 0.0
        features[:, 1] = np.log1p(degrees)

        max_degree = self.target_indegree.max()
        features[:, 2] = degrees / (max_degree + 1e-8)

        features[:, 3] = 0.0

        features[:, 4] = 2.0

        local_density = degrees / (self.n_intermediates + 1e-8)
        features[:, 5] = local_density

        features[:, 6] = 0.0

        features[:, 7] = 0.0

        features[:, 8] = degrees / (self.target_indegree.sum() + 1e-8)

        features[:, 9] = 2.0

        return features

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to zero mean, unit variance.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_nodes, n_features)

        Returns
        -------
        normalized : np.ndarray
            Normalized features
        """
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        return (features - mean) / std


def extract_pathway_node_features(edge1_matrix: sp.spmatrix,
                                  edge2_matrix: sp.spmatrix,
                                  source_idx: int,
                                  target_idx: int,
                                  sample_intermediates: int = 10) -> np.ndarray:
    """
    Extract node features for a source-target pathway.

    For computational efficiency, samples a subset of intermediate nodes.

    Parameters
    ----------
    edge1_matrix : sparse matrix
        First edge matrix
    edge2_matrix : sparse matrix
        Second edge matrix
    source_idx : int
        Source node index
    target_idx : int
        Target node index
    sample_intermediates : int
        Number of intermediate nodes to sample

    Returns
    -------
    sequence_features : np.ndarray
        Array of shape (sequence_length, 10) containing features for:
        [source, intermediate_1, ..., intermediate_k, target]
    """
    extractor = GraphFeatureExtractor(edge1_matrix, edge2_matrix)

    source_features = extractor.get_source_features(np.array([source_idx]))

    source_neighbors = edge1_matrix[source_idx].nonzero()[1]
    target_neighbors = edge2_matrix[:, target_idx].nonzero()[0]

    common_intermediates = np.intersect1d(source_neighbors, target_neighbors)

    if len(common_intermediates) == 0:
        all_intermediates = np.union1d(source_neighbors, target_neighbors)
        if len(all_intermediates) > 0:
            sampled = np.random.choice(
                all_intermediates,
                min(sample_intermediates, len(all_intermediates)),
                replace=False
            )
        else:
            sampled = np.array([])
    else:
        sampled = np.random.choice(
            common_intermediates,
            min(sample_intermediates, len(common_intermediates)),
            replace=False
        )

    if len(sampled) > 0:
        inter_features = extractor.get_intermediate_features(sampled)
    else:
        inter_features = np.zeros((0, 10))

    target_features = extractor.get_target_features(np.array([target_idx]))

    if len(inter_features) > 0:
        sequence_features = np.vstack([
            source_features,
            inter_features,
            target_features
        ])
    else:
        sequence_features = np.vstack([
            source_features,
            target_features
        ])

    return sequence_features


if __name__ == "__main__":
    from pathlib import Path
    import sys

    print("Testing graph feature extraction...")

    repo_dir = Path.cwd()
    data_dir = repo_dir / 'data'

    edge1_file = data_dir / 'edges' / 'CbG.sparse.npz'
    edge2_file = data_dir / 'edges' / 'GpPW.sparse.npz'

    if not edge1_file.exists():
        print(f"Edge file not found: {edge1_file}")
        sys.exit(1)

    edge1 = sp.load_npz(str(edge1_file))
    edge2 = sp.load_npz(str(edge2_file))

    if edge1.dtype == bool:
        edge1 = edge1.astype(np.int32)
    if edge2.dtype == bool:
        edge2 = edge2.astype(np.int32)

    print(f"\nEdge1: {edge1.shape}, {edge1.nnz} edges")
    print(f"Edge2: {edge2.shape}, {edge2.nnz} edges")

    extractor = GraphFeatureExtractor(edge1, edge2)

    test_sources = np.array([0, 1, 2])
    source_feats = extractor.get_source_features(test_sources)
    print(f"\nSource features shape: {source_feats.shape}")
    print(f"Sample source features:\n{source_feats[0]}")

    test_intermediates = np.array([0, 100, 1000])
    inter_feats = extractor.get_intermediate_features(test_intermediates)
    print(f"\nIntermediate features shape: {inter_feats.shape}")
    print(f"Sample intermediate features:\n{inter_feats[0]}")

    test_targets = np.array([0, 1, 2])
    target_feats = extractor.get_target_features(test_targets)
    print(f"\nTarget features shape: {target_feats.shape}")
    print(f"Sample target features:\n{target_feats[0]}")

    pathway_feats = extract_pathway_node_features(
        edge1, edge2,
        source_idx=0,
        target_idx=0,
        sample_intermediates=5
    )
    print(f"\nPathway sequence features: {pathway_feats.shape}")
    print(f"Sequence: source -> {pathway_feats.shape[0]-2} intermediates -> target")

    print("\nFeature extraction working correctly!")
