"""
Comprehensive metapath analysis: Test 5 diverse metapaths at each path length (3, 4, 5).

This script systematically evaluates endpoint-only prediction across:
- 15 metapaths (5 per length)
- Different biological contexts (therapeutic, similarity, interaction)
- Negative control (shuffled labels)

Goal: Assess generalization and identify path characteristics that affect performance.

Usage:
    python test_src/test_multipath_comprehensive.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
from scipy.sparse import csr_matrix
import time

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

from test_src.validate_mean_variance_prediction import (
    load_permuted_edge_matrices
)


METAPATHS = {
    'length3': [
        {
            'name': 'CbGpPW',
            'description': 'Compound-Gene-Pathway',
            'edges': ['CbG', 'GpPW'],
            'context': 'Gene function'
        },
        {
            'name': 'CtDaG',
            'description': 'Compound-Disease-Gene',
            'edges': ['CtD', 'DaG'],
            'context': 'Therapeutic'
        },
        {
            'name': 'CrCbG',
            'description': 'Compound-Compound-Gene',
            'edges': ['CrC', 'CbG'],
            'context': 'Compound similarity'
        },
        {
            'name': 'GiGaD',
            'description': 'Gene-Gene-Disease',
            'edges': ['GiG', 'DaG'],
            'context': 'Gene network'
        },
        {
            'name': 'DaGiG',
            'description': 'Disease-Gene-Gene',
            'edges': ['DaG', 'GiG'],
            'context': 'Disease mechanism'
        }
    ],
    'length4': [
        {
            'name': 'CbGiGpPW',
            'description': 'Compound-Gene-Gene-Pathway',
            'edges': ['CbG', 'GiG', 'GpPW'],
            'context': 'Extended function'
        },
        {
            'name': 'CtDaGiG',
            'description': 'Compound-Disease-Gene-Gene',
            'edges': ['CtD', 'DaG', 'GiG'],
            'context': 'Therapeutic network'
        },
        {
            'name': 'CrCbGaD',
            'description': 'Compound-Compound-Gene-Disease',
            'edges': ['CrC', 'CbG', 'DaG'],
            'context': 'Similar drug effects'
        },
        {
            'name': 'DaGiGaD',
            'description': 'Disease-Gene-Gene-Disease',
            'edges': ['DaG', 'GiG', 'DaG'],
            'context': 'Comorbidity'
        },
        {
            'name': 'CbGpBPpG',
            'description': 'Compound-Gene-BioProcess-Gene',
            'edges': ['CbG', 'GpBP', 'GpBP'],
            'context': 'Biological process'
        }
    ],
    'length5': [
        {
            'name': 'CbGiGiGpPW',
            'description': 'Compound-Gene-Gene-Gene-Pathway',
            'edges': ['CbG', 'GiG', 'GiG', 'GpPW'],
            'context': 'Deep function'
        },
        {
            'name': 'CtDaGiGpPW',
            'description': 'Compound-Disease-Gene-Gene-Pathway',
            'edges': ['CtD', 'DaG', 'GiG', 'GpPW'],
            'context': 'Therapeutic pathway'
        },
        {
            'name': 'CrCbGiGaD',
            'description': 'Compound-Compound-Gene-Gene-Disease',
            'edges': ['CrC', 'CbG', 'GiG', 'DaG'],
            'context': 'Similar drug disease'
        },
        {
            'name': 'DaGiGiGaD',
            'description': 'Disease-Gene-Gene-Gene-Disease',
            'edges': ['DaG', 'GiG', 'GiG', 'DaG'],
            'context': 'Deep comorbidity'
        },
        {
            'name': 'CbGiGpBPpG',
            'description': 'Compound-Gene-Gene-BioProcess-Gene',
            'edges': ['CbG', 'GiG', 'GpBP', 'GpBP'],
            'context': 'Process pathway'
        }
    ]
}


def load_and_multiply_edges(edge_types, perm, data_dir):
    """
    Load edge matrices and compute pathway matrix via multiplication.

    Args:
        edge_types: List of edge type strings
        perm: Permutation number
        data_dir: Data directory path

    Returns:
        Pathway matrix (sparse)
    """
    # Load first edge
    edge_file = data_dir / 'permutations' / f'{perm:03d}.hetmat' / 'edges' / f'{edge_types[0]}.sparse.npz'
    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    import scipy.sparse as sp
    result = sp.load_npz(str(edge_file))
    if result.dtype == bool:
        result = result.astype(np.int32)

    # Multiply through subsequent edges
    for edge_type in edge_types[1:]:
        edge_file = data_dir / 'permutations' / f'{perm:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
        if not edge_file.exists():
            raise FileNotFoundError(f"Edge file not found: {edge_file}")

        edge = sp.load_npz(str(edge_file))
        if edge.dtype == bool:
            edge = edge.astype(np.int32)

        result = result @ edge

    return result


def compute_pathway_counts_from_matrix(pairs, pathway_matrix):
    """Extract counts for specific pairs from precomputed pathway matrix."""
    counts = np.zeros(len(pairs))
    for i, (src, tgt) in enumerate(pairs):
        counts[i] = pathway_matrix[src, tgt]
    return counts


def analyze_single_metapath(metapath_info, length, n_samples=10000, random_state=42):
    """
    Analyze a single metapath with endpoint-only prediction.

    Returns:
        Dictionary with results
    """
    data_dir = repo_dir / 'data'

    print(f"\n{'='*70}")
    print(f"Analyzing: {metapath_info['name']} ({metapath_info['description']})")
    print(f"Context: {metapath_info['context']}")
    print(f"Edges: {' @ '.join(metapath_info['edges'])}")
    print(f"{'='*70}")

    try:
        # Sample pairs from perm 0
        print("  Loading edges for sampling...")
        pathway_perm0 = load_and_multiply_edges(metapath_info['edges'], 0, data_dir)

        n_sources = pathway_perm0.shape[0]
        n_targets = pathway_perm0.shape[1]

        print(f"  Matrix shape: {n_sources} x {n_targets}")

        # Sample pairs
        np.random.seed(random_state)
        pathway_coo = pathway_perm0.tocoo()
        pathway_pairs = np.column_stack([pathway_coo.row, pathway_coo.col])

        print(f"  Pairs with pathways in perm 0: {len(pathway_pairs)}")

        if len(pathway_pairs) == 0:
            print("  ERROR: No pathways found, skipping...")
            return None

        n_with_pathways = min(n_samples // 2, len(pathway_pairs))
        idx = np.random.choice(len(pathway_pairs), n_with_pathways, replace=False)
        sampled_with_pathways = pathway_pairs[idx]

        n_random = n_samples - n_with_pathways
        random_sources = np.random.randint(0, n_sources, n_random)
        random_targets = np.random.randint(0, n_targets, n_random)
        random_pairs = np.column_stack([random_sources, random_targets])

        pairs = np.vstack([sampled_with_pathways, random_pairs])
        np.random.shuffle(pairs)

        print(f"  Sampled {len(pairs)} pairs")

        # Extract endpoint degrees from first and last edge
        import scipy.sparse as sp
        first_edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f"{metapath_info['edges'][0]}.sparse.npz"
        last_edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f"{metapath_info['edges'][-1]}.sparse.npz"

        first_edge = sp.load_npz(str(first_edge_file))
        last_edge = sp.load_npz(str(last_edge_file))

        source_degrees = np.asarray(first_edge.sum(axis=1)).ravel()
        target_degrees = np.asarray(last_edge.sum(axis=0)).ravel()

        deg_src = source_degrees[pairs[:, 0]]
        deg_tgt = target_degrees[pairs[:, 1]]

        X = np.column_stack([
            deg_src,
            deg_tgt,
            deg_src * deg_tgt,
            deg_src ** 2,
            deg_tgt ** 2
        ])

        # Compute training counts
        train_perms = list(range(5))
        test_perms = list(range(15, 20))

        print(f"  Computing counts for training perms {train_perms}...")
        counts_train = []
        for perm in train_perms:
            pathway = load_and_multiply_edges(metapath_info['edges'], perm, data_dir)
            counts = compute_pathway_counts_from_matrix(pairs, pathway)
            counts_train.append(counts)

        counts_train = np.column_stack(counts_train)
        mean_train = counts_train.mean(axis=1)

        print(f"    Mean count: {mean_train.mean():.3f}, Pairs with pathways: {(mean_train>0).mean()*100:.1f}%")

        # Train model
        print("  Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, mean_train)

        # Train negative control (shuffled labels)
        print("  Training negative control (shuffled labels)...")
        mean_train_shuffled = mean_train.copy()
        np.random.seed(random_state + 1)
        np.random.shuffle(mean_train_shuffled)

        model_control = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model_control.fit(X, mean_train_shuffled)

        # Compute test counts
        print(f"  Computing counts for test perms {test_perms}...")
        counts_test = []
        for perm in test_perms:
            pathway = load_and_multiply_edges(metapath_info['edges'], perm, data_dir)
            counts = compute_pathway_counts_from_matrix(pairs, pathway)
            counts_test.append(counts)

        counts_test = np.column_stack(counts_test)

        # Evaluate
        predictions = model.predict(X)
        predictions_control = model_control.predict(X)

        results = []
        for test_idx, perm in enumerate(test_perms):
            actual = counts_test[:, test_idx]

            # Main model
            r = np.corrcoef(actual, predictions)[0, 1]
            mae = np.abs(actual - predictions).mean()
            qq = stats.probplot(actual - predictions)[1][2]

            # Control model
            r_control = np.corrcoef(actual, predictions_control)[0, 1]
            mae_control = np.abs(actual - predictions_control).mean()

            results.append({
                'metapath': metapath_info['name'],
                'length': length,
                'context': metapath_info['context'],
                'perm': perm,
                'r': r,
                'mae': mae,
                'qq': qq,
                'r_control': r_control,
                'mae_control': mae_control,
                'mean_count': mean_train.mean(),
                'pairs_with_pathways': (mean_train > 0).mean()
            })

        results_df = pd.DataFrame(results)

        print(f"\n  Results: r={results_df['r'].mean():.3f}±{results_df['r'].std():.3f}, " +
              f"Q-Q={results_df['qq'].mean():.3f}±{results_df['qq'].std():.3f}")
        print(f"  Control: r={results_df['r_control'].mean():.3f}±{results_df['r_control'].std():.3f}")

        return results_df

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comprehensive_analysis():
    """Run analysis on all metapaths."""

    output_dir = repo_dir / 'results' / 'multipath_comprehensive'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("COMPREHENSIVE METAPATH ANALYSIS")
    print("="*70)
    print("\nTesting 15 metapaths (5 per length: 3, 4, 5)")
    print("Including negative control (shuffled labels)")
    print()

    all_results = []

    for length_key, metapaths in METAPATHS.items():
        length = int(length_key.replace('length', ''))
        print(f"\n{'#'*70}")
        print(f"# PATH LENGTH {length}")
        print(f"{'#'*70}")

        for metapath_info in metapaths:
            start_time = time.time()

            results_df = analyze_single_metapath(metapath_info, length)

            if results_df is not None:
                all_results.append(results_df)

            elapsed = time.time() - start_time
            print(f"  Time: {elapsed:.1f}s")

    # Combine results
    if len(all_results) == 0:
        print("\nERROR: No successful analyses")
        return

    combined_df = pd.concat(all_results, ignore_index=True)

    # Save results
    results_file = output_dir / 'comprehensive_results.csv'
    combined_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    summary = combined_df.groupby(['length', 'metapath']).agg({
        'r': ['mean', 'std'],
        'qq': ['mean', 'std'],
        'mae': 'mean',
        'r_control': 'mean',
        'mean_count': 'first',
        'pairs_with_pathways': 'first'
    }).round(3)

    print(summary)

    # Summary by length
    print("\n" + "-"*70)
    print("BY PATH LENGTH")
    print("-"*70)

    length_summary = combined_df.groupby('length').agg({
        'r': ['mean', 'std', 'min', 'max'],
        'qq': ['mean', 'std', 'min', 'max'],
        'r_control': ['mean', 'std']
    }).round(3)

    print(length_summary)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_comprehensive_visualizations(combined_df, output_dir)

    return combined_df


def generate_comprehensive_visualizations(df, output_dir):
    """Generate comprehensive comparison visualizations."""

    # Aggregate by metapath
    metapath_summary = df.groupby(['length', 'metapath', 'context']).agg({
        'r': 'mean',
        'qq': 'mean',
        'r_control': 'mean',
        'mae': 'mean',
        'mean_count': 'first',
        'pairs_with_pathways': 'first'
    }).reset_index()

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: r by length and metapath
    ax1 = fig.add_subplot(gs[0, 0])
    for length in [3, 4, 5]:
        data = metapath_summary[metapath_summary['length'] == length]
        ax1.scatter([length]*len(data), data['r'], s=100, alpha=0.6, label=f'Length {length}')

    ax1.axhline(0, color='red', linestyle='--', lw=1, alpha=0.5, label='Negative control')
    ax1.set_xlabel('Path Length')
    ax1.set_ylabel('Correlation (r)')
    ax1.set_title('Performance by Path Length')
    ax1.set_xticks([3, 4, 5])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Q-Q by length
    ax2 = fig.add_subplot(gs[0, 1])
    for length in [3, 4, 5]:
        data = metapath_summary[metapath_summary['length'] == length]
        ax2.scatter([length]*len(data), data['qq'], s=100, alpha=0.6, label=f'Length {length}')

    ax2.set_xlabel('Path Length')
    ax2.set_ylabel('Q-Q Correlation')
    ax2.set_title('Calibration by Path Length')
    ax2.set_xticks([3, 4, 5])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: r vs control
    ax3 = fig.add_subplot(gs[0, 2])
    colors = {3: 'blue', 4: 'green', 5: 'orange'}
    for length in [3, 4, 5]:
        data = metapath_summary[metapath_summary['length'] == length]
        ax3.scatter(data['r_control'], data['r'], s=100, alpha=0.6,
                   color=colors[length], label=f'Length {length}')

    ax3.plot([-0.2, 0.2], [-0.2, 0.2], 'r--', lw=2, alpha=0.5)
    ax3.set_xlabel('Control r (shuffled labels)')
    ax3.set_ylabel('Model r (true labels)')
    ax3.set_title('Model vs Negative Control')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mean counts by length
    ax4 = fig.add_subplot(gs[1, 0])
    lengths = []
    mean_counts = []
    for length in [3, 4, 5]:
        data = metapath_summary[metapath_summary['length'] == length]
        for count in data['mean_count']:
            lengths.append(length)
            mean_counts.append(count)

    ax4.scatter(lengths, mean_counts, s=100, alpha=0.6)
    ax4.set_xlabel('Path Length')
    ax4.set_ylabel('Mean Count')
    ax4.set_yscale('log')
    ax4.set_title('Count Magnitude by Length')
    ax4.set_xticks([3, 4, 5])
    ax4.grid(True, alpha=0.3)

    # Plot 5: Pathway prevalence
    ax5 = fig.add_subplot(gs[1, 1])
    for length in [3, 4, 5]:
        data = metapath_summary[metapath_summary['length'] == length]
        ax5.scatter([length]*len(data), data['pairs_with_pathways']*100,
                   s=100, alpha=0.6, label=f'Length {length}')

    ax5.set_xlabel('Path Length')
    ax5.set_ylabel('Pairs with Pathways (%)')
    ax5.set_title('Pathway Prevalence')
    ax5.set_xticks([3, 4, 5])
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: r vs Q-Q tradeoff
    ax6 = fig.add_subplot(gs[1, 2])
    for length in [3, 4, 5]:
        data = metapath_summary[metapath_summary['length'] == length]
        ax6.scatter(data['r'], data['qq'], s=100, alpha=0.6,
                   color=colors[length], label=f'Length {length}')

    ax6.set_xlabel('Correlation (r)')
    ax6.set_ylabel('Q-Q Correlation')
    ax6.set_title('Performance vs Calibration Tradeoff')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7-9: Individual metapath performance by length
    for idx, length in enumerate([3, 4, 5]):
        ax = fig.add_subplot(gs[2, idx])
        data = metapath_summary[metapath_summary['length'] == length].sort_values('r', ascending=True)

        y_pos = np.arange(len(data))
        ax.barh(y_pos, data['r'], alpha=0.7, label='Model')
        ax.barh(y_pos, data['r_control'], alpha=0.4, label='Control')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(data['metapath'], fontsize=8)
        ax.set_xlabel('Correlation (r)')
        ax.set_title(f'Length-{length} Metapaths')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'comprehensive_comparison.png'}")
    plt.close()


if __name__ == '__main__':
    results = run_comprehensive_analysis()
