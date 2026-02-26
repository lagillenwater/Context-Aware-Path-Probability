"""
Diagnostic script to show actual path counts for specific degree combinations.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import pandas as pd


def load_edge_matrix(edge_type, perm_id=None, data_dir=None, perm_dir=None):
    """Load edge matrix."""
    if perm_id is None:
        edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    else:
        edge_file = perm_dir / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

    return sp.load_npz(str(edge_file))


def compute_metapath_counts(edge_matrices):
    """Compute path counts."""
    # Convert to float to ensure we get actual counts, not boolean
    result = edge_matrices[0].astype(float)
    for matrix in edge_matrices[1:]:
        result = result @ matrix.astype(float)
    return result


def main():
    # Setup paths
    repo_dir = Path(__file__).parent.parent
    data_dir = repo_dir / 'data'
    perm_dir = repo_dir / 'data' / 'permutations'

    print("Loading CbGpPWpG metapath data...")

    # Define metapath
    metapath_edge_types = ['CbG', 'GpPW', 'GpPW']
    transpose_flags = [False, False, True]

    # Load unpermuted
    hetionet_matrices = []
    for edge_type, needs_transpose in zip(metapath_edge_types, transpose_flags):
        matrix = load_edge_matrix(edge_type, perm_id=None,
                                  data_dir=data_dir, perm_dir=perm_dir)
        if needs_transpose:
            matrix = matrix.T
        hetionet_matrices.append(matrix)

    hetionet_paths = compute_metapath_counts(hetionet_matrices)

    # Get degrees
    source_degrees = np.array(hetionet_matrices[0].sum(axis=1)).flatten()
    target_degrees = np.array(hetionet_matrices[2].sum(axis=0)).flatten()

    print(f"Source (Compound) degrees: min={source_degrees.min()}, max={source_degrees.max()}")
    print(f"Target (Gene) degrees: min={target_degrees.min()}, max={target_degrees.max()}")

    # Load permutations
    print("\nLoading permutations 0-4...")
    perm_paths_list = []
    for perm_id in range(5):
        perm_matrices = []
        for edge_type, needs_transpose in zip(metapath_edge_types, transpose_flags):
            matrix = load_edge_matrix(edge_type, perm_id=perm_id,
                                     data_dir=data_dir, perm_dir=perm_dir)
            if needs_transpose:
                matrix = matrix.T
            perm_matrices.append(matrix)

        perm_paths = compute_metapath_counts(perm_matrices)
        perm_paths_list.append(perm_paths)

    # Select quantile examples
    src_quantiles = [0, 0.25, 0.5, 0.75, 0.95]
    tgt_quantiles = [0, 0.25, 0.5, 0.75, 0.95]

    src_degrees_to_check = [int(np.quantile(source_degrees[source_degrees > 0], q))
                            for q in src_quantiles]
    tgt_degrees_to_check = [int(np.quantile(target_degrees[target_degrees > 0], q))
                            for q in tgt_quantiles]

    print(f"\nSource degree quantiles: {src_degrees_to_check}")
    print(f"Target degree quantiles: {tgt_degrees_to_check}")

    # Build diagnostic table
    results = []

    for src_deg in src_degrees_to_check:
        for tgt_deg in tgt_degrees_to_check:
            # Find nodes with these degrees
            src_nodes = np.where(source_degrees == src_deg)[0]
            tgt_nodes = np.where(target_degrees == tgt_deg)[0]

            if len(src_nodes) == 0 or len(tgt_nodes) == 0:
                continue

            # Take first node of each
            src_idx = src_nodes[0]
            tgt_idx = tgt_nodes[0]

            # Get path counts
            hetionet_count = hetionet_paths[src_idx, tgt_idx]

            perm_counts = []
            for perm_paths in perm_paths_list:
                perm_counts.append(perm_paths[src_idx, tgt_idx])

            results.append({
                'source_degree': src_deg,
                'target_degree': tgt_deg,
                'source_idx': src_idx,
                'target_idx': tgt_idx,
                'hetionet_count': hetionet_count,
                'perm_0': perm_counts[0],
                'perm_1': perm_counts[1],
                'perm_2': perm_counts[2],
                'perm_3': perm_counts[3],
                'perm_4': perm_counts[4],
                'perm_sum': sum(perm_counts),
                'perm_mean': np.mean(perm_counts)
            })

    df = pd.DataFrame(results)

    print("\n" + "="*100)
    print("PATH COUNTS BY DEGREE COMBINATION")
    print("="*100)
    print(df.to_string(index=False))

    # Now show what happens when aggregating by degree bin
    print("\n" + "="*100)
    print("AGGREGATED BY DEGREE BIN")
    print("="*100)

    # Compute totals for each degree combination across ALL pairs with those degrees
    agg_results = []

    for src_deg in src_degrees_to_check[:3]:  # Just a few examples
        for tgt_deg in tgt_degrees_to_check[:3]:
            src_nodes = np.where(source_degrees == src_deg)[0]
            tgt_nodes = np.where(target_degrees == tgt_deg)[0]

            if len(src_nodes) == 0 or len(tgt_nodes) == 0:
                continue

            # Sum across all pairs with these degrees
            total_perm_sum = 0
            total_perm_mean = 0
            n_pairs = 0

            for src_idx in src_nodes:
                for tgt_idx in tgt_nodes:
                    perm_counts = [perm_paths[src_idx, tgt_idx] for perm_paths in perm_paths_list]
                    total_perm_sum += sum(perm_counts)
                    total_perm_mean += np.mean(perm_counts)
                    n_pairs += 1

            agg_results.append({
                'source_degree': src_deg,
                'target_degree': tgt_deg,
                'n_pairs': n_pairs,
                'total_perm_sum': total_perm_sum,
                'total_perm_mean': total_perm_mean,
                'mean_per_pair_sum': total_perm_sum / n_pairs if n_pairs > 0 else 0,
                'mean_per_pair_mean': total_perm_mean / n_pairs if n_pairs > 0 else 0
            })

    agg_df = pd.DataFrame(agg_results)
    print(agg_df.to_string(index=False))


if __name__ == '__main__':
    main()
