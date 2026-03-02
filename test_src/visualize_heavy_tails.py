"""
Visualize heavy-tail phenomenon across path lengths 2-8.

Creates three visualizations:
1. Coefficient of Variation (CV) trajectory
2. Log-scale count distributions
3. Outlier prevalence (% beyond 3 SD)

Shows why Q-Q calibration craters at length 5 and recovers at 6-7.

Usage:
    python test_src/visualize_heavy_tails.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor
import scipy.sparse as sp

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))


def load_and_multiply_edges(edge_types, perm, data_dir):
    """Load edge matrices and compute pathway matrix."""
    perm_dir = data_dir / 'permutations' / f'{perm:03d}.hetmat' / 'edges'

    edge_file = perm_dir / f'{edge_types[0]}.sparse.npz'
    result = sp.load_npz(str(edge_file))
    if result.dtype == bool:
        result = result.astype(np.int32)

    for edge_type in edge_types[1:]:
        edge_file = perm_dir / f'{edge_type}.sparse.npz'
        edge = sp.load_npz(str(edge_file))
        if edge.dtype == bool:
            edge = edge.astype(np.int32)
        result = result @ edge

    return result


def compute_pathway_counts(pairs, pathway_matrix):
    """Extract counts for pairs from pathway matrix."""
    counts = np.zeros(len(pairs))
    for i, (src, tgt) in enumerate(pairs):
        counts[i] = pathway_matrix[src, tgt]
    return counts


def analyze_length_detailed(edge_types, length, n_samples=10000, random_state=42):
    """
    Analyze a single length with detailed outputs.

    Returns:
        Dictionary with predictions, actuals, residuals, and statistics
    """
    data_dir = repo_dir / 'data'

    print(f"  Processing length {length}...")

    # Compute pathway matrix for perm 0
    pathway_perm0 = load_and_multiply_edges(edge_types, 0, data_dir)

    n_sources = pathway_perm0.shape[0]
    n_targets = pathway_perm0.shape[1]

    # Sample pairs
    np.random.seed(random_state)
    pathway_coo = pathway_perm0.tocoo()
    pathway_pairs = np.column_stack([pathway_coo.row, pathway_coo.col])

    if len(pathway_pairs) == 0:
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

    # Extract endpoint degrees
    first_edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_types[0]}.sparse.npz'
    last_edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_types[-1]}.sparse.npz'

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

    counts_train = []
    for perm in train_perms:
        pathway = load_and_multiply_edges(edge_types, perm, data_dir)
        counts = compute_pathway_counts(pairs, pathway)
        counts_train.append(counts)

    counts_train = np.column_stack(counts_train)
    mean_train = counts_train.mean(axis=1)
    std_train = counts_train.std(axis=1)

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, mean_train)

    # Compute test counts
    counts_test = []
    for perm in test_perms:
        pathway = load_and_multiply_edges(edge_types, perm, data_dir)
        counts = compute_pathway_counts(pairs, pathway)
        counts_test.append(counts)

    counts_test = np.column_stack(counts_test)

    # Predictions
    predictions = model.predict(X)

    # Collect all actuals and residuals
    all_actuals = counts_test.flatten()
    all_predictions = np.repeat(predictions, len(test_perms))
    all_residuals = all_actuals - all_predictions

    # Statistics
    cv = std_train.mean() / (mean_train.mean() + 1e-10)  # Avoid division by zero
    outlier_threshold = 3 * std_train.mean()
    outlier_pct = (np.abs(all_residuals) > outlier_threshold).mean() * 100

    return {
        'length': length,
        'predictions': predictions,
        'actuals_test': counts_test,
        'residuals': all_residuals,
        'mean_count': mean_train.mean(),
        'std_count': std_train.mean(),
        'cv': cv,
        'outlier_pct': outlier_pct,
        'all_actuals': all_actuals,
        'all_predictions': all_predictions
    }


def generate_heavy_tail_visualizations():
    """Generate three-panel visualization of heavy-tail phenomenon."""

    print("="*70)
    print("HEAVY-TAIL VISUALIZATION ANALYSIS")
    print("="*70)

    # Define metapaths
    metapaths = {
        2: ['CbG'],
        3: ['CbG', 'GpPW'],
        4: ['CbG', 'GiG', 'GpPW'],
        5: ['CbG', 'GiG', 'GiG', 'GpPW'],
        6: ['CbG', 'GiG', 'GiG', 'GiG', 'GpPW'],
        7: ['CbG', 'GiG', 'GiG', 'GiG', 'GiG', 'GpPW'],
        8: ['CbG', 'GiG', 'GiG', 'GiG', 'GiG', 'GiG', 'GpPW']
    }

    # Analyze each length
    results = {}
    for length in [2, 3, 4, 5, 6, 7, 8]:
        result = analyze_length_detailed(metapaths[length], length)
        if result is not None:
            results[length] = result

    # Extract statistics
    lengths = sorted(results.keys())
    cvs = [results[l]['cv'] for l in lengths]
    outlier_pcts = [results[l]['outlier_pct'] for l in lengths]
    mean_counts = [results[l]['mean_count'] for l in lengths]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Coefficient of Variation
    ax = axes[0]
    ax.plot(lengths, cvs, marker='o', markersize=10, linewidth=2, color='purple')
    ax.set_xlabel('Path Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (std/mean)', fontsize=12, fontweight='bold')
    ax.set_title('Variance Grows Faster Than Mean at Length 5', fontsize=13, fontweight='bold')
    ax.set_xticks(lengths)
    ax.grid(True, alpha=0.3)

    # Annotate peak
    peak_idx = cvs.index(max(cvs))
    peak_length = lengths[peak_idx]
    ax.annotate(f'Peak CV: {max(cvs):.2f}\n(Length {peak_length})',
                xy=(peak_length, max(cvs)),
                xytext=(peak_length + 0.5, max(cvs) + max(cvs)*0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Panel 2: Log-Scale Count Distributions
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lengths)))

    for i, length in enumerate(lengths):
        actuals = results[length]['all_actuals']
        actuals_nonzero = actuals[actuals > 0]

        if len(actuals_nonzero) > 0:
            log_counts = np.log10(actuals_nonzero)
            ax.hist(log_counts, bins=50, alpha=0.6, label=f'Length {length}',
                   color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('log10(Count)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Count Distributions Show Widest Spread at Length 5', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Outlier Prevalence
    ax = axes[2]
    ax.plot(lengths, outlier_pcts, marker='s', markersize=10, linewidth=2, color='red')
    ax.set_xlabel('Path Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Pairs Beyond ±3 SD', fontsize=12, fontweight='bold')
    ax.set_title('Heavy-Tail Severity Peaks at Length 4-5', fontsize=13, fontweight='bold')
    ax.set_xticks(lengths)
    ax.grid(True, alpha=0.3)

    # Annotate peak
    peak_idx = outlier_pcts.index(max(outlier_pcts))
    peak_length = lengths[peak_idx]
    ax.annotate(f'Peak outliers: {max(outlier_pcts):.1f}%\n(Length {peak_length})',
                xy=(peak_length, max(outlier_pcts)),
                xytext=(peak_length - 1, max(outlier_pcts) + 2),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()

    # Save
    output_dir = repo_dir / 'results' / 'length_degradation'
    output_file = output_dir / 'heavy_tail_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    summary_df = pd.DataFrame({
        'Length': lengths,
        'CV': cvs,
        'Outlier %': outlier_pcts,
        'Mean Count': mean_counts
    })
    print(summary_df.to_string(index=False))

    return results


if __name__ == '__main__':
    results = generate_heavy_tail_visualizations()
