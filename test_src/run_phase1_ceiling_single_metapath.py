"""
Run Phase 1 Ceiling Analysis for a single 2-hop metapath.

This script runs oracle upper bound, binning resolution, and feature
sufficiency tests for one metapath.

Usage:
    python run_phase1_ceiling_single_metapath.py --metapath CbGaD

For HPC array job, use:
    sbatch scripts/phase1_ceiling_analysis_array.sh
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import sys
import argparse

from src.ceiling_analysis import run_ceiling_analysis


def load_edge_matrix_from_hetmat(hetmat_dir: str, edge_abbrev: str) -> sp.spmatrix:
    """
    Load edge adjacency matrix from hetmat directory.

    Parameters:
    - hetmat_dir: Path to .hetmat directory (e.g., 'data' or 'data/permutations/000.hetmat')
    - edge_abbrev: Edge abbreviation (e.g., 'CbG')

    Returns:
    - edge_matrix: Sparse adjacency matrix
    """
    edge_path = os.path.join(hetmat_dir, 'edges', f'{edge_abbrev}.sparse.npz')

    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge file not found: {edge_path}")

    return sp.load_npz(edge_path)


def load_pathway_counts_from_original(data_dir: str,
                                        edge1_abbrev: str,
                                        edge2_abbrev: str,
                                        n_bins: int = 10) -> pd.DataFrame:
    """
    Load pathway counts from original Hetionet, aggregated by degree bins.

    Parameters:
    - data_dir: Path to original Hetionet data
    - edge1_abbrev: First edge (e.g., 'CbG')
    - edge2_abbrev: Second edge (e.g., 'GaD')
    - n_bins: Number of degree quantile bins

    Returns:
    - df: DataFrame with bin-level statistics
    """
    print(f"  Loading pathway counts from original Hetionet...")
    edge1 = load_edge_matrix_from_hetmat(data_dir, edge1_abbrev)
    edge2 = load_edge_matrix_from_hetmat(data_dir, edge2_abbrev)

    pathway_matrix = edge1.dot(edge2)

    if sp.issparse(pathway_matrix):
        pathway_matrix = pathway_matrix.toarray()

    source_degrees = np.array(edge1.sum(axis=1)).flatten()
    target_degrees = np.array(edge2.sum(axis=0)).flatten()

    # Define bins from original degrees
    source_bins, source_bin_edges = pd.qcut(source_degrees, q=n_bins,
                                             labels=False, retbins=True,
                                             duplicates='drop')
    target_bins, target_bin_edges = pd.qcut(target_degrees, q=n_bins,
                                             labels=False, retbins=True,
                                             duplicates='drop')

    print(f"  Defined {len(np.unique(source_bins))} source bins, " +
          f"{len(np.unique(target_bins))} target bins")

    # Aggregate pathway counts by bin
    bin_data = {}
    for i in range(pathway_matrix.shape[0]):
        for j in range(pathway_matrix.shape[1]):
            bin_key = (source_bins[i], target_bins[j])
            if bin_key not in bin_data:
                bin_data[bin_key] = []
            bin_data[bin_key].append(pathway_matrix[i, j])

    rows = []
    for (src_bin, tgt_bin), counts in bin_data.items():
        rows.append({
            'source_bin': src_bin,
            'target_bin': tgt_bin,
            'mean_count': np.mean(counts),
            'std_count': np.std(counts),
            'n_pairs': len(counts)
        })

    df = pd.DataFrame(rows)
    print(f"  Original Hetionet: {len(df)} bins with pathway data")

    return df, source_bin_edges, target_bin_edges


def load_pathway_counts_from_permutations(perm_dir: str,
                                            edge1_abbrev: str,
                                            edge2_abbrev: str,
                                            perm_ids: range,
                                            source_bin_edges: np.ndarray,
                                            target_bin_edges: np.ndarray) -> list:
    """
    Load pathway counts aggregated by degree bins from permutations.

    Uses bin edges defined from original Hetionet for consistent binning.

    Parameters:
    - perm_dir: Base permutation directory (e.g., 'data/permutations')
    - edge1_abbrev: First edge (e.g., 'CbG')
    - edge2_abbrev: Second edge (e.g., 'GaD')
    - perm_ids: Which permutations to load (e.g., range(0, 20))
    - source_bin_edges: Bin edges for source degrees (from original Hetionet)
    - target_bin_edges: Bin edges for target degrees (from original Hetionet)

    Returns:
    - pathway_data: List of DataFrames with bin-level statistics
                     Columns: [source_bin, target_bin, mean_count, std_count, n_pairs]
    """

    pathway_data = []

    for perm_id in perm_ids:
        hetmat_path = os.path.join(perm_dir, f'{perm_id:03d}.hetmat')

        if not os.path.exists(hetmat_path):
            print(f"Warning: Permutation {perm_id:03d} not found, skipping")
            continue

        try:
            edge1 = load_edge_matrix_from_hetmat(hetmat_path, edge1_abbrev)
            edge2 = load_edge_matrix_from_hetmat(hetmat_path, edge2_abbrev)
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping permutation {perm_id:03d}")
            continue

        pathway_matrix = edge1.dot(edge2)

        if sp.issparse(pathway_matrix):
            pathway_matrix = pathway_matrix.toarray()

        source_degrees = np.array(edge1.sum(axis=1)).flatten()
        target_degrees = np.array(edge2.sum(axis=0)).flatten()

        source_bins = pd.cut(source_degrees, bins=source_bin_edges,
                             labels=False, include_lowest=True)
        target_bins = pd.cut(target_degrees, bins=target_bin_edges,
                             labels=False, include_lowest=True)

        bin_data = {}
        for i in range(pathway_matrix.shape[0]):
            for j in range(pathway_matrix.shape[1]):
                bin_key = (source_bins[i], target_bins[j])
                if bin_key not in bin_data:
                    bin_data[bin_key] = []
                bin_data[bin_key].append(pathway_matrix[i, j])

        rows = []
        for (src_bin, tgt_bin), counts in bin_data.items():
            rows.append({
                'source_bin': src_bin,
                'target_bin': tgt_bin,
                'mean_count': np.mean(counts),
                'std_count': np.std(counts),
                'n_pairs': len(counts),
                'permutation': perm_id
            })

        perm_df = pd.DataFrame(rows)
        pathway_data.append(perm_df)

        if perm_id % 5 == 0 or perm_id == max(perm_ids):
            print(f"  Loaded perm {perm_id:03d}: {len(perm_df)} bins")

    return pathway_data


def main():
    """Run Phase 1 ceiling analysis for single metapath."""
    parser = argparse.ArgumentParser(
        description='Run ceiling analysis for single 2-hop metapath'
    )
    parser.add_argument('--metapath', type=str, required=True,
                        help='Metapath abbreviation (e.g., CbGaD)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--perm_dir', type=str, default='data/permutations',
                        help='Permutation directory')
    parser.add_argument('--output_dir', type=str,
                        default='results/ceiling_analysis',
                        help='Output directory')
    parser.add_argument('--test_start', type=int, default=0,
                        help='Test permutations start (inclusive)')
    parser.add_argument('--test_end', type=int, default=20,
                        help='Test permutations end (exclusive)')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of degree bins')

    args = parser.parse_args()

    print("=" * 70)
    print(f"PHASE 1: CEILING ANALYSIS FOR METAPATH {args.metapath}")
    print("=" * 70)

    if len(args.metapath) < 5:
        print(f"ERROR: Invalid metapath '{args.metapath}'")
        print("Expected format: XrYsZ (e.g., CbGaD)")
        sys.exit(1)

    if len(args.metapath) > 2 and args.metapath[2] == '>':
        edge1_abbrev = args.metapath[:4]
        edge2_node_type = args.metapath[3]
        edge2_abbrev = edge2_node_type + args.metapath[4:]
    else:
        edge1_abbrev = args.metapath[:3]
        edge2_node_type = args.metapath[2]
        edge2_abbrev = edge2_node_type + args.metapath[3:]

    print(f"\nMetapath: {args.metapath}")
    print(f"  Edge 1: {edge1_abbrev}")
    print(f"  Edge 2: {edge2_abbrev}")

    print(f"\nLoading original Hetionet edges from {args.data_dir}...")
    try:
        edge1_matrix = load_edge_matrix_from_hetmat(args.data_dir, edge1_abbrev)
        edge2_matrix = load_edge_matrix_from_hetmat(args.data_dir, edge2_abbrev)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  {edge1_abbrev}: {edge1_matrix.shape}, {edge1_matrix.nnz} edges")
    print(f"  {edge2_abbrev}: {edge2_matrix.shape}, {edge2_matrix.nnz} edges")

    print(f"\nLoading pathway counts...")
    print(f"  Training: Original Hetionet")
    train_df, source_bin_edges, target_bin_edges = load_pathway_counts_from_original(
        args.data_dir, edge1_abbrev, edge2_abbrev, n_bins=args.n_bins
    )

    print(f"  Testing: perms {args.test_start:03d}-{args.test_end-1:03d}")
    test_permutations = load_pathway_counts_from_permutations(
        args.perm_dir, edge1_abbrev, edge2_abbrev,
        range(args.test_start, args.test_end),
        source_bin_edges, target_bin_edges
    )

    if len(test_permutations) == 0:
        print("\nERROR: No permutation data found!")
        sys.exit(1)

    n_test = sum(len(df) for df in test_permutations)

    print(f"\n  Training bins: {len(train_df)}")
    print(f"  Total test pathways: {n_test}")

    if n_test == 0:
        print("\nERROR: No pathways found for this metapath!")
        sys.exit(1)

    output_subdir = os.path.join(args.output_dir, args.metapath)
    os.makedirs(output_subdir, exist_ok=True)

    print(f"\nRunning ceiling analysis...")
    print(f"  Output: {output_subdir}")

    results = run_ceiling_analysis(
        edge1_matrix=edge1_matrix,
        edge2_matrix=edge2_matrix,
        train_permutations=[train_df],
        test_permutations=test_permutations,
        output_dir=output_subdir,
        n_bins=args.n_bins
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Metapath: {args.metapath}")
    print(f"Oracle (exact): r = {results['oracle_exact']['r']:.4f}")
    print(f"Oracle (binned): r = {results['oracle_binned']['r']:.4f}")
    print(f"Gap: {results['gap_exact_binned']:.4f}")

    print("\nFeature tests: Skipped (requires trained model)")
    print("Gap analysis will be performed in Phase 2 with actual models")

    gap_to_oracle = results['gap_exact_binned']

    if np.isnan(gap_to_oracle) or np.isnan(results['oracle_binned']['r']):
        print(f"\nGap (exact - binned): NaN")
        conclusion = "Constant pathway counts (no variance to predict)"
    else:
        print(f"\nGap (exact - binned): {gap_to_oracle:.4f}")
        if gap_to_oracle < 0.01:
            conclusion = "Near-optimal (within 1% of ceiling)"
        elif gap_to_oracle < 0.03:
            conclusion = "Close to ceiling (1-3% gap)"
        else:
            conclusion = "Significant headroom (>3% gap)"

    print(f"Conclusion: {conclusion}")

    summary_file = os.path.join(output_subdir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Metapath: {args.metapath}\n")
        if np.isnan(results['oracle_exact']['r']):
            f.write(f"Oracle (exact): r = NaN (constant values)\n")
        else:
            f.write(f"Oracle (exact): r = {results['oracle_exact']['r']:.4f}\n")

        if np.isnan(results['oracle_binned']['r']):
            f.write(f"Oracle (binned): r = NaN (constant values)\n")
        else:
            f.write(f"Oracle (binned): r = {results['oracle_binned']['r']:.4f}\n")

        f.write(f"Gap (exact - binned): {gap_to_oracle:.4f}\n")
        f.write(f"Conclusion: {conclusion}\n")

    print(f"\nSummary saved to: {summary_file}")
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
