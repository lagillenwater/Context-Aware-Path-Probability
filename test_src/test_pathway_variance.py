"""
Verify pathway count variance within single permutation.

Tests hypothesis about why oracle r=NaN: Are pathway counts constant
within degree pairs, or do they vary?

Usage:
    python test_pathway_variance.py [--metapath METAPATH]

Default tests AdGpMF metapath.
"""

import scipy.sparse as sp
import numpy as np
import argparse
import os


def test_pathway_variance(metapath='AdGpMF', perm_id=0):
    """
    Test if pathway counts vary within degree pairs.

    Parameters:
    - metapath: Metapath abbreviation (e.g., 'AdGpMF')
    - perm_id: Which permutation to test (default 0)

    Returns:
    - stats: Dict with variance statistics
    """
    edge1_abbrev = metapath[:3]
    edge2_abbrev = metapath[2:]

    if len(metapath) > 2 and metapath[2] == '>':
        edge1_abbrev = metapath[:4]
        edge2_node_type = metapath[3]
        edge2_abbrev = edge2_node_type + metapath[4:]

    perm_dir = f'data/permutations/{perm_id:03d}.hetmat'

    print(f"Testing metapath: {metapath}")
    print(f"  Edge 1: {edge1_abbrev}")
    print(f"  Edge 2: {edge2_abbrev}")
    print(f"  Permutation: {perm_id:03d}")
    print()

    edge1_path = os.path.join(perm_dir, 'edges', f'{edge1_abbrev}.sparse.npz')
    edge2_path = os.path.join(perm_dir, 'edges', f'{edge2_abbrev}.sparse.npz')

    if not os.path.exists(edge1_path):
        print(f"ERROR: {edge1_path} not found")
        return None

    if not os.path.exists(edge2_path):
        print(f"ERROR: {edge2_path} not found")
        return None

    edge1 = sp.load_npz(edge1_path)
    edge2 = sp.load_npz(edge2_path)

    print(f"Edge matrices loaded:")
    print(f"  {edge1_abbrev}: {edge1.shape}, {edge1.nnz} edges")
    print(f"  {edge2_abbrev}: {edge2.shape}, {edge2.nnz} edges")
    print()

    pathway_matrix = edge1.dot(edge2)

    if sp.issparse(pathway_matrix):
        pathway_matrix = pathway_matrix.toarray()

    source_degrees = np.array(edge1.sum(axis=1)).flatten()
    target_degrees = np.array(edge2.sum(axis=0)).flatten()

    counts = pathway_matrix[pathway_matrix > 0]

    print("=" * 70)
    print("PATHWAY COUNT DISTRIBUTION")
    print("=" * 70)
    print(f"Total pathways: {len(counts):,}")
    print(f"Count = 1: {(counts==1).sum():,} ({100*(counts==1).sum()/len(counts):.1f}%)")
    print(f"Count > 1: {(counts>1).sum():,} ({100*(counts>1).sum()/len(counts):.1f}%)")
    print(f"Count > 5: {(counts>5).sum():,} ({100*(counts>5).sum()/len(counts):.1f}%)")
    print(f"Mean: {counts.mean():.2f}, Std: {counts.std():.2f}")
    print(f"Min: {counts.min()}, Max: {counts.max()}")
    print()

    degree_counts = {}
    for i in range(pathway_matrix.shape[0]):
        for j in range(pathway_matrix.shape[1]):
            count = pathway_matrix[i, j]
            if count > 0:
                key = (source_degrees[i], target_degrees[j])
                if key not in degree_counts:
                    degree_counts[key] = []
                degree_counts[key].append(count)

    multi_obs = {k: v for k, v in degree_counts.items() if len(v) > 1}
    varied = {k: v for k, v in multi_obs.items() if np.std(v) > 0}

    print("=" * 70)
    print("DEGREE PAIR VARIANCE ANALYSIS")
    print("=" * 70)
    print(f"Unique degree pairs: {len(degree_counts):,}")
    print(f"Degree pairs with 2+ pathways: {len(multi_obs):,} ({100*len(multi_obs)/len(degree_counts):.1f}%)")
    print(f"Degree pairs with variance: {len(varied):,} ({100*len(varied)/len(degree_counts):.1f}%)")
    print()

    if len(varied) > 0:
        print("Sample degree pairs with variance:")
        for i, (key, vals) in enumerate(list(varied.items())[:5]):
            print(f"  {key}: n={len(vals)}, counts={vals[:10]}, mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")
        print()

    print("=" * 70)
    print("ORACLE PREDICTION")
    print("=" * 70)

    if len(varied) == 0:
        print("RESULT: All degree pairs have constant counts")
        print("Expected oracle r: NaN (no variance to predict)")
        print()
        print("INTERPRETATION: Metapath is too sparse.")
        print("Each degree combination produces same count (usually 1).")
        print("Oracle cannot learn degree-conditioned distribution.")
    elif len(varied) < len(degree_counts) * 0.05:
        print(f"RESULT: Only {100*len(varied)/len(degree_counts):.1f}% of degree pairs have variance")
        print("Expected oracle r: Close to NaN or very low")
        print()
        print("INTERPRETATION: Metapath is mostly sparse.")
        print("Most degree combinations have constant counts.")
        print("Oracle has minimal signal to learn from.")
    else:
        print(f"RESULT: {100*len(varied)/len(degree_counts):.1f}% of degree pairs have variance")
        print(f"Expected oracle r: > 0.5 (potentially good prediction)")
        print()
        print("INTERPRETATION: Metapath has sufficient complexity.")
        print("Degree-conditioned distributions vary meaningfully.")
        print("Oracle should be able to predict count distributions.")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if len(varied) > len(degree_counts) * 0.1:
        print("FIX WILL HELP: Removing node indices should reveal variance.")
        print("After fix, expect oracle r > 0.5 for this metapath.")
    else:
        print("FIX WON'T HELP: Metapath inherently has minimal variance.")
        print("NaN is expected and correct for this metapath.")
        print("Focus modeling efforts on denser metapaths.")

    print()

    stats = {
        'metapath': metapath,
        'total_pathways': len(counts),
        'pct_count_1': 100 * (counts == 1).sum() / len(counts),
        'mean_count': counts.mean(),
        'std_count': counts.std(),
        'unique_degree_pairs': len(degree_counts),
        'degree_pairs_with_variance': len(varied),
        'pct_varied': 100 * len(varied) / len(degree_counts) if len(degree_counts) > 0 else 0
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Test pathway count variance within single permutation'
    )
    parser.add_argument('--metapath', type=str, default='AdGpMF',
                        help='Metapath to test (default: AdGpMF)')
    parser.add_argument('--perm_id', type=int, default=0,
                        help='Permutation ID (default: 0)')

    args = parser.parse_args()

    stats = test_pathway_variance(args.metapath, args.perm_id)

    if stats:
        print(f"\nStats saved: {stats}")


if __name__ == "__main__":
    main()
