"""
Compute edge frequencies from original Hetionet graph.

This module computes edge frequencies grouped by (source_degree, target_degree)
pairs from the original Hetionet graph (not permutations). These frequencies
serve as training targets for neural network models.

Frequency definition:
    frequency(u, v) = (number of edges between nodes with degrees u, v) /
                      (number of possible node pairs with degrees u, v)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, Dict
from collections import defaultdict


def compute_original_graph_frequencies(edge_type: str,
                                       data_dir: Path) -> pd.DataFrame:
    """
    Compute edge frequencies by degree pair from original Hetionet graph.

    Parameters
    ----------
    edge_type : str
        Edge type identifier (e.g., 'CbG')
    data_dir : Path
        Data directory containing edges/ subdirectory

    Returns
    -------
    frequencies_df : DataFrame
        DataFrame with columns:
        - source_degree: int
        - target_degree: int
        - frequency: float (0-1)
        - edge_count: int (number of edges with this degree pair)
        - possible_pairs: int (number of possible node pairs)
    """
    edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'

    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    print(f"Loading original graph: {edge_file}")

    edge_matrix = sp.load_npz(str(edge_file))
    n_sources, n_targets = edge_matrix.shape
    n_edges = edge_matrix.nnz

    print(f"Graph shape: {n_sources} sources x {n_targets} targets")
    print(f"Total edges: {n_edges}")

    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    print(f"Source degree range: {source_degrees.min()}-{source_degrees.max()}")
    print(f"Target degree range: {target_degrees.min()}-{target_degrees.max()}")

    edge_counts = defaultdict(int)
    src_nodes, tgt_nodes = edge_matrix.nonzero()

    for s, t in zip(src_nodes, tgt_nodes):
        s_deg = int(source_degrees[s])
        t_deg = int(target_degrees[t])
        edge_counts[(s_deg, t_deg)] += 1

    src_degree_counts = np.bincount(source_degrees.astype(int))
    tgt_degree_counts = np.bincount(target_degrees.astype(int))

    possible_pairs = {}
    for src_deg in range(len(src_degree_counts)):
        if src_degree_counts[src_deg] == 0:
            continue
        for tgt_deg in range(len(tgt_degree_counts)):
            if tgt_degree_counts[tgt_deg] == 0:
                continue
            possible_pairs[(src_deg, tgt_deg)] = int(
                src_degree_counts[src_deg] * tgt_degree_counts[tgt_deg]
            )

    frequencies_data = []
    for (src_deg, tgt_deg), edge_count in edge_counts.items():
        possible = possible_pairs.get((src_deg, tgt_deg), 0)
        if possible > 0:
            frequency = edge_count / possible
            frequencies_data.append({
                'source_degree': src_deg,
                'target_degree': tgt_deg,
                'frequency': frequency,
                'edge_count': edge_count,
                'possible_pairs': possible
            })

    frequencies_df = pd.DataFrame(frequencies_data)
    frequencies_df = frequencies_df.sort_values(['source_degree', 'target_degree'])

    print(f"Unique degree pairs: {len(frequencies_df)}")
    print(f"Frequency range: {frequencies_df['frequency'].min():.6f} - {frequencies_df['frequency'].max():.6f}")
    print(f"Mean frequency: {frequencies_df['frequency'].mean():.6f}")

    return frequencies_df


def analyze_degree_pair_coverage(original_df: pd.DataFrame,
                                 empirical_df: pd.DataFrame) -> Dict:
    """
    Analyze overlap between original graph and 200-permutation degree pairs.

    Parameters
    ----------
    original_df : DataFrame
        Frequencies from original graph
    empirical_df : DataFrame
        Frequencies from 200 permutations

    Returns
    -------
    coverage_stats : dict
        Statistics about degree pair overlap and coverage
    """
    original_pairs = set(zip(original_df['source_degree'],
                            original_df['target_degree']))
    empirical_pairs = set(zip(empirical_df['source_degree'],
                             empirical_df['target_degree']))

    overlap_pairs = original_pairs & empirical_pairs
    unseen_pairs = empirical_pairs - original_pairs
    original_only_pairs = original_pairs - empirical_pairs

    stats = {
        'n_original': len(original_pairs),
        'n_empirical': len(empirical_pairs),
        'n_overlap': len(overlap_pairs),
        'n_unseen': len(unseen_pairs),
        'n_original_only': len(original_only_pairs),
        'overlap_pct': 100.0 * len(overlap_pairs) / len(empirical_pairs) if len(empirical_pairs) > 0 else 0,
        'unseen_pct': 100.0 * len(unseen_pairs) / len(empirical_pairs) if len(empirical_pairs) > 0 else 0,
    }

    print("\nDegree Pair Coverage Analysis:")
    print(f"  Original graph degree pairs: {stats['n_original']}")
    print(f"  200-permutation degree pairs: {stats['n_empirical']}")
    print(f"  Overlap (in both): {stats['n_overlap']} ({stats['overlap_pct']:.1f}%)")
    print(f"  Unseen (only in 200-perm): {stats['n_unseen']} ({stats['unseen_pct']:.1f}%)")
    print(f"  Original only: {stats['n_original_only']}")

    return stats


def get_unseen_degree_pairs(original_df: pd.DataFrame,
                            empirical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract degree pairs that exist in empirical data but not in original graph.

    Parameters
    ----------
    original_df : DataFrame
        Frequencies from original graph
    empirical_df : DataFrame
        Frequencies from 200 permutations

    Returns
    -------
    unseen_df : DataFrame
        Subset of empirical_df containing only unseen degree pairs
    """
    original_pairs = set(zip(original_df['source_degree'],
                            original_df['target_degree']))

    unseen_mask = ~empirical_df.apply(
        lambda row: (row['source_degree'], row['target_degree']) in original_pairs,
        axis=1
    )

    unseen_df = empirical_df[unseen_mask].copy()

    print(f"\nUnseen degree pairs: {len(unseen_df)}")

    return unseen_df
