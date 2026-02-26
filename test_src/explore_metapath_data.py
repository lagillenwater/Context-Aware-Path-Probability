"""
Data exploration for Phase 2 feature ablation study.

Checks data availability and statistics for 5 selected metapaths:
1. CbGpPW - Benchmark (r=0.94)
2. GiGiG - Dense symmetric
3. CtDaG - Sparse heterogeneous
4. AdGpBP - Medium density
5. CrCbG - Compound resemblance

Verification checklist:
- [ ] All edge files exist
- [ ] Pathway counts computed correctly
- [ ] Sufficient variance for evaluation
- [ ] Train/test data format verified
"""

import numpy as np
import scipy.sparse as sp
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def load_edge_matrix(data_dir, edge_abbrev):
    """Load edge adjacency matrix."""
    edge_path = os.path.join(data_dir, 'edges', f'{edge_abbrev}.sparse.npz')
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge file not found: {edge_path}")
    return sp.load_npz(edge_path)


def compute_pathway_stats(edge1, edge2, name):
    """Compute pathway count statistics."""
    # Convert to int32 for pathway counting
    if edge1.dtype == bool:
        edge1 = edge1.astype(np.int32)
    if edge2.dtype == bool:
        edge2 = edge2.astype(np.int32)

    pathway_matrix = edge1.dot(edge2)

    if sp.issparse(pathway_matrix):
        pathway_matrix = pathway_matrix.toarray()

    # Get non-zero pathways
    counts = pathway_matrix[pathway_matrix > 0]

    stats = {
        'metapath': name,
        'edge1_shape': edge1.shape,
        'edge2_shape': edge2.shape,
        'edge1_nnz': edge1.nnz,
        'edge2_nnz': edge2.nnz,
        'total_pairs': pathway_matrix.shape[0] * pathway_matrix.shape[1],
        'pathway_pairs': len(counts),
        'density': len(counts) / (pathway_matrix.shape[0] * pathway_matrix.shape[1]),
        'count_mean': counts.mean() if len(counts) > 0 else 0,
        'count_std': counts.std() if len(counts) > 0 else 0,
        'count_min': counts.min() if len(counts) > 0 else 0,
        'count_max': counts.max() if len(counts) > 0 else 0,
        'count_median': np.median(counts) if len(counts) > 0 else 0,
        'pct_count_1': 100 * (counts == 1).sum() / len(counts) if len(counts) > 0 else 0,
        'pct_count_gt_5': 100 * (counts > 5).sum() / len(counts) if len(counts) > 0 else 0
    }

    return stats


def explore_metapaths():
    """Explore all 5 metapaths."""
    metapaths = [
        ('CbGpPW', 'CbG', 'GpPW', 'Compound-binds-Gene-participates-Pathway'),
        ('GiGiG', 'GiG', 'GiG', 'Gene-interacts-Gene-interacts-Gene'),
        ('CtDaG', 'CtD', 'DaG', 'Compound-treats-Disease-associates-Gene'),
        ('AdGpBP', 'AdG', 'GpBP', 'Anatomy-downregulates-Gene-participates-BiologicalProcess'),
        ('CrCbG', 'CrC', 'CbG', 'Compound-resembles-Compound-binds-Gene')
    ]

    print("=" * 80)
    print("PHASE 2: DATA EXPLORATION FOR FEATURE ABLATION STUDY")
    print("=" * 80)
    print()

    all_stats = []

    for name, edge1_abbrev, edge2_abbrev, description in metapaths:
        print(f"Metapath: {name}")
        print(f"  Description: {description}")
        print(f"  Edge 1: {edge1_abbrev}")
        print(f"  Edge 2: {edge2_abbrev}")

        try:
            # Load original Hetionet edges
            edge1 = load_edge_matrix('data', edge1_abbrev)
            edge2 = load_edge_matrix('data', edge2_abbrev)

            # Compute pathway statistics
            stats = compute_pathway_stats(edge1, edge2, name)
            all_stats.append(stats)

            print(f"  Edge 1 shape: {stats['edge1_shape']}, edges: {stats['edge1_nnz']:,}")
            print(f"  Edge 2 shape: {stats['edge2_shape']}, edges: {stats['edge2_nnz']:,}")
            print(f"  Total possible pairs: {stats['total_pairs']:,}")
            print(f"  Pathway pairs: {stats['pathway_pairs']:,} ({stats['density']:.6f} density)")
            print(f"  Pathway counts: mean={stats['count_mean']:.2f}, std={stats['count_std']:.2f}")
            print(f"                  min={stats['count_min']}, max={stats['count_max']}, median={stats['count_median']:.1f}")
            print(f"  Count=1: {stats['pct_count_1']:.1f}%, Count>5: {stats['pct_count_gt_5']:.1f}%")
            print()

        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print()
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            continue

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if len(all_stats) == 0:
        print("ERROR: No metapaths could be loaded!")
        return

    print(f"Successfully loaded: {len(all_stats)}/5 metapaths")
    print()

    # Sort by density
    all_stats_sorted = sorted(all_stats, key=lambda x: x['density'], reverse=True)

    print("Metapaths by density:")
    for stats in all_stats_sorted:
        print(f"  {stats['metapath']:10s}: {stats['pathway_pairs']:,} pathways "
              f"({stats['density']:.6f} density, "
              f"{stats['pct_count_1']:.0f}% count=1)")

    print()
    print("VERIFICATION CHECKLIST:")
    print(f"  [{'X' if len(all_stats) == 5 else ' '}] All 5 metapaths loaded successfully")

    has_variance = all([s['count_std'] > 0 for s in all_stats])
    print(f"  [{'X' if has_variance else ' '}] All metapaths have pathway count variance")

    has_density = all([s['density'] > 0 for s in all_stats])
    print(f"  [{'X' if has_density else ' '}] All metapaths have non-zero pathways")

    has_multi_paths = any([s['pct_count_1'] < 99 for s in all_stats])
    print(f"  [{'X' if has_multi_paths else ' '}] At least one metapath has multi-pathways (count>1)")

    print()

    if len(all_stats) < 5:
        print("WARNING: Some metapaths could not be loaded. Check edge file names.")

    if not has_variance:
        print("WARNING: Some metapaths have no variance (all counts same). "
              "Oracle r will be NaN.")

    if not has_multi_paths:
        print("WARNING: All metapaths are extremely sparse (100% count=1). "
              "Limited signal for ML models.")

    return all_stats


if __name__ == "__main__":
    explore_metapaths()
