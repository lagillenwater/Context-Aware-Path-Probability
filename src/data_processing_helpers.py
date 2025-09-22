"""
Data loading and feature extraction helpers for edge prediction.
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, List

def load_permutation_data(perm_dir: Path, edge_type: str) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    edge_file = perm_dir / 'edges' / f'{edge_type}.sparse.npz'
    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")
    edge_matrix = sp.load_npz(edge_file).astype(bool).tocsr()
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()
    return edge_matrix, source_degrees, target_degrees

def get_available_permutations(permutations_dir: Path) -> List[str]:
    perm_dirs = []
    for item in permutations_dir.iterdir():
        if item.is_dir() and item.name.endswith('.hetmat'):
            perm_dirs.append(item.name)
    return sorted(perm_dirs)

def extract_improved_edge_features_and_labels(edge_matrix: sp.csr_matrix, source_degrees: np.ndarray, target_degrees: np.ndarray, negative_ratio: float = 0.5, use_normalized_features: bool = True, use_regression: bool = True):
    pos_edges = list(zip(*edge_matrix.nonzero()))
    n_pos = len(pos_edges)
    n_neg = int(n_pos * negative_ratio)
    neg_edges = []
    n_source, n_target = edge_matrix.shape
    source_probs = (source_degrees + 1) / (source_degrees + 1).sum()
    target_probs = (target_degrees + 1) / (target_degrees + 1).sum()
    attempts = 0
    max_attempts = n_neg * 20
    while len(neg_edges) < n_neg and attempts < max_attempts:
        source = np.random.choice(n_source, p=source_probs)
        target = np.random.choice(n_target, p=target_probs)
        if edge_matrix[source, target] == 0:
            neg_edges.append((source, target))
        attempts += 1
    while len(neg_edges) < n_neg:
        source = np.random.randint(0, n_source)
        target = np.random.randint(0, n_target)
        if edge_matrix[source, target] == 0:
            neg_edges.append((source, target))
    all_edges = pos_edges + neg_edges
    n_total = len(all_edges)
    n_features = 2
    features = np.zeros((n_total, n_features))
    targets = np.zeros(n_total)
    for i, (source, target) in enumerate(all_edges):
        source_deg = source_degrees[source]
        target_deg = target_degrees[target]
        features[i, 0] = source_deg
        features[i, 1] = target_deg
        if use_regression:
            targets[i] = 1.0 if i < n_pos else 0.0
        else:
            targets[i] = 1.0 if i < n_pos else 0.0
    return features, targets
