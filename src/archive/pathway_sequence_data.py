"""
Convert degree-binned pathway data to sequential format for Transformer.

This module prepares training data for PathwayTransformer by:
1. Sampling actual node pairs from each degree bin
2. Extracting node sequences (source -> intermediates -> target)
3. Computing node-level features for each position
4. Creating variable-length sequences with padding

Output format for each sample:
    - node_features: (max_seq_len, feature_dim) padded sequence
    - attention_mask: (max_seq_len,) binary mask (1=real, 0=padding)
    - pathway_count: scalar target
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset

from src.graph_features import GraphFeatureExtractor
from src.intermediate_signatures import assign_to_bins


class PathwaySequenceDataset(Dataset):
    """
    PyTorch Dataset for pathway sequences.

    Each sample is a pathway with:
    - Variable-length node sequence
    - Node features at each position
    - Target pathway count
    """

    def __init__(self,
                 sequences: List[np.ndarray],
                 targets: np.ndarray,
                 max_seq_len: int = 20):
        """
        Initialize dataset.

        Parameters
        ----------
        sequences : list of np.ndarray
            List of node feature sequences, each (seq_len, feature_dim)
        targets : np.ndarray
            Target pathway counts (n_samples,)
        max_seq_len : int
            Maximum sequence length for padding
        """
        self.sequences = sequences
        self.targets = targets
        self.max_seq_len = max_seq_len
        self.feature_dim = sequences[0].shape[1] if len(sequences) > 0 else 10

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get one sample."""
        seq = self.sequences[idx]
        target = self.targets[idx]

        seq_len = len(seq)

        if seq_len > self.max_seq_len:
            seq = seq[:self.max_seq_len]
            seq_len = self.max_seq_len

        padded_seq = np.zeros((self.max_seq_len, self.feature_dim))
        padded_seq[:seq_len] = seq

        mask = np.zeros(self.max_seq_len)
        mask[:seq_len] = 1

        return {
            'node_features': torch.FloatTensor(padded_seq),
            'attention_mask': torch.FloatTensor(mask),
            'pathway_count': torch.FloatTensor([target])
        }


def sample_node_pairs_from_bins(edge1_matrix: sp.spmatrix,
                                edge2_matrix: sp.spmatrix,
                                degree_bins_df: pd.DataFrame,
                                n_samples_per_bin: int = 10,
                                random_seed: int = 42) -> Tuple[List, List]:
    """
    Sample actual node pairs from each degree bin combination.

    Parameters
    ----------
    edge1_matrix : sparse matrix
        First edge matrix
    edge2_matrix : sparse matrix
        Second edge matrix
    degree_bins_df : pd.DataFrame
        Degree-binned data from notebook 18a with columns:
        source_bin, target_bin, pathway_count_mean, ...
    n_samples_per_bin : int
        Number of node pairs to sample from each bin
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    source_indices : list
        List of source node indices
    target_indices : list
        List of target node indices
    """
    np.random.seed(random_seed)

    print(f"Sampling {n_samples_per_bin} node pairs per degree bin...")

    source_degrees = np.asarray(edge1_matrix.sum(axis=1)).ravel()
    target_degrees = np.asarray(edge2_matrix.sum(axis=0)).ravel()

    from src.intermediate_signatures import create_degree_bins
    source_bins = create_degree_bins(source_degrees, n_bins=10)
    target_bins = create_degree_bins(target_degrees, n_bins=10)

    source_bin_assignments = assign_to_bins(source_degrees, source_bins)
    target_bin_assignments = assign_to_bins(target_degrees, target_bins)

    source_indices = []
    target_indices = []

    pathway_matrix = edge1_matrix @ edge2_matrix
    pathway_coo = pathway_matrix.tocoo()

    pathway_dict = {}
    for i, j, v in zip(pathway_coo.row, pathway_coo.col, pathway_coo.data):
        src_bin = source_bin_assignments[i]
        tgt_bin = target_bin_assignments[j]
        key = (src_bin, tgt_bin)
        if key not in pathway_dict:
            pathway_dict[key] = []
        pathway_dict[key].append((i, j))

    for _, row in degree_bins_df.iterrows():
        src_bin = int(row['source_bin'])
        tgt_bin = int(row['target_bin'])

        pairs = pathway_dict.get((src_bin, tgt_bin), [])

        if len(pairs) == 0:
            continue

        if len(pairs) <= n_samples_per_bin:
            sampled_pairs = pairs
        else:
            sampled_indices = np.random.choice(
                len(pairs), n_samples_per_bin, replace=False
            )
            sampled_pairs = [pairs[i] for i in sampled_indices]

        for src_idx, tgt_idx in sampled_pairs:
            source_indices.append(src_idx)
            target_indices.append(tgt_idx)

    print(f"  Sampled {len(source_indices)} total node pairs")

    return source_indices, target_indices


def create_pathway_sequences(edge1_matrix: sp.spmatrix,
                             edge2_matrix: sp.spmatrix,
                             source_indices: List[int],
                             target_indices: List[int],
                             n_intermediate_samples: int = 5) -> List[np.ndarray]:
    """
    Create node feature sequences for each pathway.

    Parameters
    ----------
    edge1_matrix : sparse matrix
        First edge matrix
    edge2_matrix : sparse matrix
        Second edge matrix
    source_indices : list
        Source node indices
    target_indices : list
        Target node indices
    n_intermediate_samples : int
        Number of intermediate nodes to sample per pathway

    Returns
    -------
    sequences : list of np.ndarray
        List of node feature sequences, each (seq_len, feature_dim)
    """
    print(f"Creating node feature sequences...")

    extractor = GraphFeatureExtractor(edge1_matrix, edge2_matrix)

    sequences = []

    for i, (src_idx, tgt_idx) in enumerate(zip(source_indices, target_indices)):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(source_indices)} pathways")

        src_features = extractor.get_source_features(np.array([src_idx]))

        src_neighbors = edge1_matrix[src_idx].nonzero()[1]
        tgt_neighbors = edge2_matrix[:, tgt_idx].nonzero()[0]

        common_inter = np.intersect1d(src_neighbors, tgt_neighbors)

        if len(common_inter) > 0:
            n_sample = min(n_intermediate_samples, len(common_inter))
            sampled_inter = np.random.choice(common_inter, n_sample, replace=False)
        elif len(src_neighbors) > 0 and len(tgt_neighbors) > 0:
            all_inter = np.union1d(src_neighbors, tgt_neighbors)
            n_sample = min(n_intermediate_samples, len(all_inter))
            sampled_inter = np.random.choice(all_inter, n_sample, replace=False)
        else:
            sampled_inter = np.array([])

        if len(sampled_inter) > 0:
            inter_features = extractor.get_intermediate_features(sampled_inter)
        else:
            inter_features = np.zeros((0, 10))

        tgt_features = extractor.get_target_features(np.array([tgt_idx]))

        if len(inter_features) > 0:
            sequence = np.vstack([src_features, inter_features, tgt_features])
        else:
            sequence = np.vstack([src_features, tgt_features])

        sequences.append(sequence)

    print(f"  Created {len(sequences)} sequences")

    seq_lengths = [len(seq) for seq in sequences]
    print(f"  Sequence length range: {min(seq_lengths)} - {max(seq_lengths)}")
    print(f"  Mean sequence length: {np.mean(seq_lengths):.1f}")

    return sequences


def prepare_transformer_data(degree_bins_df: pd.DataFrame,
                            edge1_matrix: sp.spmatrix,
                            edge2_matrix: sp.spmatrix,
                            n_samples_per_bin: int = 10,
                            n_intermediate_samples: int = 5,
                            random_seed: int = 42) -> Tuple[PathwaySequenceDataset,
                                                            np.ndarray]:
    """
    Prepare complete dataset for PathwayTransformer training.

    Parameters
    ----------
    degree_bins_df : pd.DataFrame
        Degree-binned data from notebook 18a
    edge1_matrix : sparse matrix
        First edge matrix
    edge2_matrix : sparse matrix
        Second edge matrix
    n_samples_per_bin : int
        Samples per degree bin
    n_intermediate_samples : int
        Intermediate nodes per pathway
    random_seed : int
        Random seed

    Returns
    -------
    dataset : PathwaySequenceDataset
        PyTorch dataset with sequences
    targets : np.ndarray
        Target pathway counts
    """
    print("=" * 80)
    print("PREPARING SEQUENTIAL DATA FOR PATHWAYTRANSFORMER")
    print("=" * 80)

    source_indices, target_indices = sample_node_pairs_from_bins(
        edge1_matrix, edge2_matrix, degree_bins_df,
        n_samples_per_bin, random_seed
    )

    sequences = create_pathway_sequences(
        edge1_matrix, edge2_matrix,
        source_indices, target_indices,
        n_intermediate_samples
    )

    pathway_matrix = edge1_matrix @ edge2_matrix

    targets = []
    for src_idx, tgt_idx in zip(source_indices, target_indices):
        count = pathway_matrix[src_idx, tgt_idx]
        targets.append(count)

    targets = np.array(targets, dtype=np.float32)

    print(f"\nTarget statistics:")
    print(f"  Range: {targets.min()} - {targets.max()}")
    print(f"  Mean: {targets.mean():.2f}")
    print(f"  Std: {targets.std():.2f}")

    max_seq_len = max(len(seq) for seq in sequences)
    print(f"\nMax sequence length: {max_seq_len}")

    dataset = PathwaySequenceDataset(sequences, targets, max_seq_len=max_seq_len+2)

    print(f"\nDataset created:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Feature dim: {dataset.feature_dim}")
    print(f"  Max sequence length: {dataset.max_seq_len}")

    return dataset, targets


if __name__ == "__main__":
    import sys

    print("Testing pathway sequence data preparation...")

    repo_dir = Path.cwd()
    if repo_dir.name == 'src':
        repo_dir = repo_dir.parent
    sys.path.insert(0, str(repo_dir))

    data_dir = repo_dir / 'data'

    training_data_file = repo_dir / 'results' / 'pathway_nn' / \
                        'training_data' / 'CbGpPW_training_data.csv'

    if not training_data_file.exists():
        print(f"Training data not found: {training_data_file}")
        print("Run prepare_and_test_pathway_data.py first")
        import sys
        sys.exit(1)

    df = pd.read_csv(training_data_file)
    print(f"Loaded degree bins: {len(df)} bins")

    edge1_file = data_dir / 'edges' / 'CbG.sparse.npz'
    edge2_file = data_dir / 'edges' / 'GpPW.sparse.npz'

    edge1 = sp.load_npz(str(edge1_file))
    edge2 = sp.load_npz(str(edge2_file))

    if edge1.dtype == bool:
        edge1 = edge1.astype(np.int32)
    if edge2.dtype == bool:
        edge2 = edge2.astype(np.int32)

    dataset, targets = prepare_transformer_data(
        df, edge1, edge2,
        n_samples_per_bin=3,
        n_intermediate_samples=3,
        random_seed=42
    )

    print(f"\n\nTest sample:")
    sample = dataset[0]
    for key, value in sample.items():
        print(f"  {key}: {value.shape}")

    print("\nSequence data preparation working correctly!")
