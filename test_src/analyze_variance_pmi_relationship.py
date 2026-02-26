"""
Analyze the relationship between variance across permutations and PMI.

This script tests whether the apparent contradiction between:
1. High variance at high-degree nodes (from path count heatmaps)
2. Low PMI at high-degree nodes (from compositionality analysis)

is real or explained by scale effects / measurement differences.

Usage:
    python test_src/analyze_variance_pmi_relationship.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from pathlib import Path
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Setup paths
repo_dir = Path(__file__).parent.parent
data_dir = repo_dir / 'data'
perm_dir = data_dir / 'permutations'
results_dir = repo_dir / 'results' / 'variance_pmi_analysis'
results_dir.mkdir(parents=True, exist_ok=True)


def load_edge_matrix(edge_type, perm_id=None):
    """Load edge matrix for given edge type and permutation."""
    if perm_id is None:
        edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    else:
        edge_file = perm_dir / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    return sp.load_npz(str(edge_file))


def compute_metapath_counts(edge1, edge2):
    """Compute path counts for 2-edge metapath."""
    return (edge1.astype(float) @ edge2.astype(float)).toarray()


def compute_compositional_probability_fast(edge1_probs, edge2_probs):
    """
    Compute compositional probability using simple multiplicative approximation.

    For PMI analysis, we use the log-space simplification:
    log(P_observed / P_compositional) = log(P_observed) - log(P_compositional)

    P_compositional = E[path_count] / max_path_count
                    ≈ sum_over_j(P(i->j) * P(j->k))

    This is much faster than Option A and appropriate for PMI calculation.
    """
    # Simple matrix multiplication: compositional path probability
    # This gives expected path count under independence assumption
    compositional = edge1_probs @ edge2_probs

    return compositional


def compute_edge_probabilities(edge_matrices):
    """Compute empirical edge probabilities from list of permutation matrices."""
    # Stack all permutations
    n_perms = len(edge_matrices)
    edge_sum = sum(m.toarray() for m in edge_matrices)

    # Probability = frequency of edge across permutations
    return edge_sum / n_perms


def stratify_by_degree(values, source_degrees, target_degrees,
                       min_src=1, max_src=20, min_tgt=1, max_tgt=40):
    """
    Stratify values by (source_degree, target_degree) bins.

    Returns heatmap array and count of samples per bin.
    """
    n_src_bins = max_src - min_src + 1
    n_tgt_bins = max_tgt - min_tgt + 1

    sum_values = np.zeros((n_src_bins, n_tgt_bins))
    count = np.zeros((n_src_bins, n_tgt_bins))

    for i in range(len(source_degrees)):
        src_deg = int(source_degrees[i])
        if src_deg < min_src or src_deg > max_src:
            continue

        for j in range(len(target_degrees)):
            tgt_deg = int(target_degrees[j])
            if tgt_deg < min_tgt or tgt_deg > max_tgt:
                continue

            src_idx = src_deg - min_src
            tgt_idx = tgt_deg - min_tgt

            sum_values[src_idx, tgt_idx] += values[i, j]
            count[src_idx, tgt_idx] += 1

    # Compute mean per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_values = np.where(count > 0, sum_values / count, np.nan)

    return mean_values, count


def main():
    """Main analysis function."""

    print("="*70)
    print("VARIANCE-PMI RELATIONSHIP ANALYSIS")
    print("="*70)
    print("\nMetapath: CbGpPW (Compound -> Gene -> Pathway)")
    print("Analyzing first 20 permutations")

    # Define metapath
    edge1_type = 'CbG'  # Compound-binds-Gene
    edge2_type = 'GpPW'  # Gene-participates-Pathway
    n_perms = 20

    # Load permutations
    print(f"\nLoading {n_perms} permutations...")
    edge1_perms = []
    edge2_perms = []
    path_counts_perms = []

    for perm_id in range(n_perms):
        edge1 = load_edge_matrix(edge1_type, perm_id)
        edge2 = load_edge_matrix(edge2_type, perm_id)
        path_counts = compute_metapath_counts(edge1, edge2)

        edge1_perms.append(edge1)
        edge2_perms.append(edge2)
        path_counts_perms.append(path_counts)

        if (perm_id + 1) % 5 == 0:
            print(f"  Loaded {perm_id + 1}/{n_perms} permutations")

    # Compute degrees from first permutation
    # Source: Compound degree (CbG rows = compounds)
    source_degrees = np.array(edge1_perms[0].sum(axis=1)).flatten()
    # Target: Pathway degree (GpPW columns = pathways)
    target_degrees = np.array(edge2_perms[0].sum(axis=0)).flatten()

    print(f"\nDegree statistics:")
    print(f"  Source (Compound): min={source_degrees.min()}, max={source_degrees.max()}, mean={source_degrees.mean():.1f}")
    print(f"  Target (Pathway): min={target_degrees.min()}, max={target_degrees.max()}, mean={target_degrees.mean():.1f}")

    # Stack path counts into 3D array (n_perms, n_compounds, n_genes)
    path_counts_3d = np.stack(path_counts_perms, axis=0)

    # Compute statistics across permutations
    print("\nComputing statistics across permutations...")
    mean_counts = np.mean(path_counts_3d, axis=0)
    var_counts = np.var(path_counts_3d, axis=0)
    std_counts = np.std(path_counts_3d, axis=0)

    # Coefficient of variation (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv_counts = np.where(mean_counts > 0, std_counts / mean_counts, 0)

    # Compute compositional probability
    print("\nComputing compositional probabilities...")
    edge1_probs = compute_edge_probabilities(edge1_perms)
    edge2_probs = compute_edge_probabilities(edge2_perms)

    # Compositional expected count under independence
    compositional_expected_count = edge1_probs @ edge2_probs

    # Normalize both by total sum to make them comparable
    total_observed = mean_counts.sum()
    total_compositional = compositional_expected_count.sum()

    observed_normalized = mean_counts / total_observed
    compositional_normalized = compositional_expected_count / total_compositional

    # Compute PMI = log2(P_observed / P_compositional)
    print("\nComputing PMI...")
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2(observed_normalized / (compositional_normalized + 1e-10))
        pmi = np.where(np.isfinite(pmi), pmi, 0)  # Replace inf/nan with 0

    # Stratify metrics by degree
    print("\nStratifying by degree bins...")
    # Use actual degree ranges from data
    degree_range = dict(min_src=1, max_src=min(20, int(source_degrees.max())),
                       min_tgt=1, max_tgt=min(40, int(target_degrees.max())))

    mean_by_degree, count_by_degree = stratify_by_degree(
        mean_counts, source_degrees, target_degrees, **degree_range)

    var_by_degree, _ = stratify_by_degree(
        var_counts, source_degrees, target_degrees, **degree_range)

    cv_by_degree, _ = stratify_by_degree(
        cv_counts, source_degrees, target_degrees, **degree_range)

    pmi_by_degree, _ = stratify_by_degree(
        pmi, source_degrees, target_degrees, **degree_range)

    # Flatten for correlation analysis (exclude bins with no data)
    valid_mask = ~np.isnan(mean_by_degree) & (count_by_degree > 0)

    mean_flat = mean_by_degree[valid_mask]
    var_flat = var_by_degree[valid_mask]
    cv_flat = cv_by_degree[valid_mask]
    pmi_flat = pmi_by_degree[valid_mask]

    # Diagnostic: print PMI statistics
    print(f"\nPMI statistics:")
    print(f"  Min: {pmi_flat.min():.3f}")
    print(f"  Max: {pmi_flat.max():.3f}")
    print(f"  Mean: {pmi_flat.mean():.3f}")
    print(f"  Median: {np.median(pmi_flat):.3f}")
    print(f"  Std: {pmi_flat.std():.3f}")

    # Compute correlations
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    corr_var_pmi = pearsonr(var_flat, pmi_flat)
    corr_cv_pmi = pearsonr(cv_flat, pmi_flat)
    corr_mean_pmi = pearsonr(mean_flat, pmi_flat)
    corr_mean_var = pearsonr(mean_flat, var_flat)

    print(f"\nPearson correlations:")
    print(f"  Variance vs PMI:  r={corr_var_pmi[0]:6.3f}, p={corr_var_pmi[1]:.2e}")
    print(f"  CV vs PMI:        r={corr_cv_pmi[0]:6.3f}, p={corr_cv_pmi[1]:.2e}")
    print(f"  Mean vs PMI:      r={corr_mean_pmi[0]:6.3f}, p={corr_mean_pmi[1]:.2e}")
    print(f"  Mean vs Variance: r={corr_mean_var[0]:6.3f}, p={corr_mean_var[1]:.2e}")

    # Create visualizations
    print("\nCreating visualizations...")

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # Define tick positions based on actual degree ranges
    max_src = degree_range['max_src']
    max_tgt = degree_range['max_tgt']
    tick_pos_x = range(0, max_tgt, max(1, max_tgt // 8))
    tick_lab_x = range(1, max_tgt + 1, max(1, max_tgt // 8))
    tick_pos_y = range(0, max_src, max(1, max_src // 4))
    tick_lab_y = range(1, max_src + 1, max(1, max_src // 4))

    # Row 1: Heatmaps
    # Panel 1: Mean count heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mean_by_degree, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title('Mean Path Count', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Target Pathway Degree')
    ax1.set_ylabel('Source Compound Degree')
    ax1.set_xticks(tick_pos_x)
    ax1.set_xticklabels(tick_lab_x)
    ax1.set_yticks(tick_pos_y)
    ax1.set_yticklabels(tick_lab_y)
    plt.colorbar(im1, ax=ax1, label='Mean Count')

    # Panel 2: Variance heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(var_by_degree, aspect='auto', origin='lower', cmap='plasma')
    ax2.set_title('Variance of Path Count', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Target Pathway Degree')
    ax2.set_ylabel('Source Compound Degree')
    ax2.set_xticks(tick_pos_x)
    ax2.set_xticklabels(tick_lab_x)
    ax2.set_yticks(tick_pos_y)
    ax2.set_yticklabels(tick_lab_y)
    plt.colorbar(im2, ax=ax2, label='Variance')

    # Panel 3: CV heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(cv_by_degree, aspect='auto', origin='lower', cmap='magma', vmax=1.0)
    ax3.set_title('Coefficient of Variation', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Target Pathway Degree')
    ax3.set_ylabel('Source Compound Degree')
    ax3.set_xticks(tick_pos_x)
    ax3.set_xticklabels(tick_lab_x)
    ax3.set_yticks(tick_pos_y)
    ax3.set_yticklabels(tick_lab_y)
    plt.colorbar(im3, ax=ax3, label='CV (std/mean)')

    # Row 2: PMI and correlation
    # Panel 4: PMI heatmap
    ax4 = fig.add_subplot(gs[1, 0])
    # Use automatic scaling based on actual data range
    pmi_vmin = np.nanpercentile(pmi_by_degree, 1)
    pmi_vmax = np.nanpercentile(pmi_by_degree, 99)
    im4 = ax4.imshow(pmi_by_degree, aspect='auto', origin='lower', cmap='coolwarm',
                     vmin=pmi_vmin, vmax=pmi_vmax)
    ax4.set_title('PMI (Compositional Fit)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Target Pathway Degree')
    ax4.set_ylabel('Source Compound Degree')
    ax4.set_xticks(tick_pos_x)
    ax4.set_xticklabels(tick_lab_x)
    ax4.set_yticks(tick_pos_y)
    ax4.set_yticklabels(tick_lab_y)
    plt.colorbar(im4, ax=ax4, label='PMI (lower=better fit)')

    # Panel 5: Variance vs PMI scatter
    ax5 = fig.add_subplot(gs[1, 1])
    scatter1 = ax5.scatter(var_flat, pmi_flat, c=mean_flat, cmap='viridis', alpha=0.5, s=20)
    ax5.set_xlabel('Variance')
    ax5.set_ylabel('PMI')
    ax5.set_title(f'Variance vs PMI\nr={corr_var_pmi[0]:.3f}, p={corr_var_pmi[1]:.2e}',
                  fontsize=12, fontweight='bold')
    ax5.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.colorbar(scatter1, ax=ax5, label='Mean Count')

    # Panel 6: CV vs PMI scatter
    ax6 = fig.add_subplot(gs[1, 2])
    scatter2 = ax6.scatter(cv_flat, pmi_flat, c=mean_flat, cmap='viridis', alpha=0.5, s=20)
    ax6.set_xlabel('Coefficient of Variation')
    ax6.set_ylabel('PMI')
    ax6.set_title(f'CV vs PMI\nr={corr_cv_pmi[0]:.3f}, p={corr_cv_pmi[1]:.2e}',
                  fontsize=12, fontweight='bold')
    ax6.set_xlim(0, 1)
    ax6.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.colorbar(scatter2, ax=ax6, label='Mean Count')

    # Row 3: Additional analysis
    # Panel 7: Mean vs Variance
    ax7 = fig.add_subplot(gs[2, 0])
    scatter3 = ax7.scatter(mean_flat, var_flat, c=pmi_flat, cmap='coolwarm', alpha=0.5, s=20,
                          vmin=pmi_vmin, vmax=pmi_vmax)
    ax7.set_xlabel('Mean Count')
    ax7.set_ylabel('Variance')
    ax7.set_title(f'Mean vs Variance\nr={corr_mean_var[0]:.3f}, p={corr_mean_var[1]:.2e}',
                  fontsize=12, fontweight='bold')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    plt.colorbar(scatter3, ax=ax7, label='PMI')

    # Panel 8: Mean vs PMI
    ax8 = fig.add_subplot(gs[2, 1])
    scatter4 = ax8.scatter(mean_flat, pmi_flat, c=cv_flat, cmap='magma', alpha=0.5, s=20, vmin=0, vmax=1)
    ax8.set_xlabel('Mean Count')
    ax8.set_ylabel('PMI')
    ax8.set_title(f'Mean vs PMI\nr={corr_mean_pmi[0]:.3f}, p={corr_mean_pmi[1]:.2e}',
                  fontsize=12, fontweight='bold')
    ax8.set_xscale('log')
    ax8.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.colorbar(scatter4, ax=ax8, label='CV')

    # Panel 9: Distribution comparison
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.hist(cv_flat, bins=50, alpha=0.5, label='CV', density=True, color='blue')
    ax9_twin = ax9.twinx()
    ax9_twin.hist(pmi_flat, bins=50, alpha=0.5, label='PMI', density=True, color='red')
    ax9.set_xlabel('Value')
    ax9.set_ylabel('Density (CV)', color='blue')
    ax9_twin.set_ylabel('Density (PMI)', color='red')
    ax9.set_title('Distribution of CV and PMI', fontsize=12, fontweight='bold')
    ax9.legend(loc='upper left')
    ax9_twin.legend(loc='upper right')

    plt.suptitle('Variance-PMI Relationship Analysis: CbGpPW Metapath',
                 fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    output_file = results_dir / 'variance_pmi_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {output_file}")

    # Save correlation table
    print("\nSaving correlation table...")
    corr_df = pd.DataFrame({
        'Comparison': ['Variance vs PMI', 'CV vs PMI', 'Mean vs PMI', 'Mean vs Variance'],
        'Pearson_r': [corr_var_pmi[0], corr_cv_pmi[0], corr_mean_pmi[0], corr_mean_var[0]],
        'p_value': [corr_var_pmi[1], corr_cv_pmi[1], corr_mean_pmi[1], corr_mean_var[1]]
    })

    corr_file = results_dir / 'variance_pmi_correlations.csv'
    corr_df.to_csv(corr_file, index=False)
    print(f"Saved correlations: {corr_file}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if abs(corr_var_pmi[0]) > 0.5:
        print("\nSTRONG correlation between variance and PMI detected!")
        if corr_var_pmi[0] < 0:
            print("  → High variance = Low PMI (better compositional fit)")
            print("  → This suggests the contradiction is REAL - needs investigation")
        else:
            print("  → High variance = High PMI (worse compositional fit)")
            print("  → This makes sense - unstable systems are less compositional")
    else:
        print("\nWEAK correlation between variance and PMI detected.")
        print("  → These metrics measure different properties")

    if abs(corr_cv_pmi[0]) < abs(corr_var_pmi[0]):
        print("\nCV-PMI correlation is weaker than Variance-PMI correlation.")
        print("  → This supports SCALE EFFECT hypothesis:")
        print("  → High-degree nodes have high absolute variance but low relative variance")
        print("  → PMI depends on relative deviation, not absolute")

    if corr_mean_var[0] > 0.8:
        print("\nSTRONG positive correlation between mean and variance detected.")
        print("  → Variance increases with mean count (heteroscedasticity)")
        print("  → This explains why high-count (high-degree) pairs have high variance")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
