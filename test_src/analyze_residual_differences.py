"""
Analyze residual differences between analytical and corrected predictions.

This script examines how theoretical corrections change residual patterns
and identifies which degree pairs are improved vs made worse.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path.cwd() / 'src'))

from theory_guided_features import TheoryGuidedFeatureEngineer
from theoretical_corrections import (
    extract_graph_features,
    apply_all_corrections
)


def analyze_residual_changes(edge_type: str,
                             data_dir: Path,
                             results_dir: Path):
    """
    Analyze how corrections change residuals.

    Parameters
    ----------
    edge_type : str
        Edge type identifier
    data_dir : Path
        Data directory
    results_dir : Path
        Results directory
    """
    print(f"\n{'='*80}")
    print(f"Residual Analysis for {edge_type}")
    print(f"{'='*80}\n")

    edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    edge_matrix = sp.load_npz(str(edge_file))

    graph_features = extract_graph_features(edge_matrix)

    fe = TheoryGuidedFeatureEngineer(edge_matrix, edge_type)

    empirical_file = results_dir / 'empirical_edge_frequencies' / f'edge_frequency_by_degree_{edge_type}.csv'
    empirical_df = pd.read_csv(empirical_file)

    u = empirical_df['source_degree'].values
    v = empirical_df['target_degree'].values
    y_empirical = empirical_df['frequency'].values

    analytical_pred = fe.get_analytical_baseline(u, v)
    corrected_pred = apply_all_corrections(analytical_pred, u, v, graph_features)

    analytical_residual = analytical_pred - y_empirical
    corrected_residual = corrected_pred - y_empirical

    residual_change = corrected_residual - analytical_residual

    improved = np.abs(corrected_residual) < np.abs(analytical_residual)
    worsened = np.abs(corrected_residual) > np.abs(analytical_residual)

    print("Overall Statistics:")
    print(f"  Points improved: {improved.sum()} ({100*improved.mean():.1f}%)")
    print(f"  Points worsened: {worsened.sum()} ({100*worsened.mean():.1f}%)")
    print(f"  Mean |residual| change: {np.mean(np.abs(corrected_residual) - np.abs(analytical_residual)):.6f}")

    print("\nResidual Statistics:")
    print(f"  Analytical - Mean: {np.mean(analytical_residual):.6f}, Std: {np.std(analytical_residual):.6f}")
    print(f"  Corrected  - Mean: {np.mean(corrected_residual):.6f}, Std: {np.std(corrected_residual):.6f}")

    degree_product = u * v

    low_deg = degree_product < np.percentile(degree_product, 33)
    mid_deg = (degree_product >= np.percentile(degree_product, 33)) & (degree_product < np.percentile(degree_product, 67))
    high_deg = degree_product >= np.percentile(degree_product, 67)

    print("\nBy Degree Product Range:")
    for name, mask in [('Low (0-33%)', low_deg), ('Mid (33-67%)', mid_deg), ('High (67-100%)', high_deg)]:
        if mask.sum() == 0:
            continue

        improved_pct = 100 * improved[mask].mean()
        mean_analytical = np.mean(analytical_residual[mask])
        mean_corrected = np.mean(corrected_residual[mask])
        mean_change = np.mean(residual_change[mask])

        print(f"\n  {name}:")
        print(f"    N points: {mask.sum()}")
        print(f"    Improved: {improved_pct:.1f}%")
        print(f"    Mean residual: {mean_analytical:.6f} → {mean_corrected:.6f} (Δ={mean_change:+.6f})")

    high_freq = y_empirical > np.percentile(y_empirical, 75)
    low_freq = y_empirical < np.percentile(y_empirical, 25)

    print("\nBy Empirical Frequency:")
    for name, mask in [('Low freq (0-25%)', low_freq), ('High freq (75-100%)', high_freq)]:
        if mask.sum() == 0:
            continue

        improved_pct = 100 * improved[mask].mean()
        mean_analytical = np.mean(analytical_residual[mask])
        mean_corrected = np.mean(corrected_residual[mask])

        print(f"\n  {name}:")
        print(f"    N points: {mask.sum()}")
        print(f"    Improved: {improved_pct:.1f}%")
        print(f"    Mean residual: {mean_analytical:.6f} → {mean_corrected:.6f}")

    high_u = u > np.percentile(u, 90)
    high_v = v > np.percentile(v, 90)
    hub_pairs = high_u & high_v

    if hub_pairs.sum() > 0:
        print(f"\nHub-Hub Pairs (u,v both > 90th percentile):")
        print(f"  N points: {hub_pairs.sum()}")
        print(f"  Improved: {100*improved[hub_pairs].mean():.1f}%")
        print(f"  Mean residual: {np.mean(analytical_residual[hub_pairs]):.6f} → {np.mean(corrected_residual[hub_pairs]):.6f}")
        print(f"  Assortativity effect: {graph_features['assortativity']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].scatter(y_empirical, analytical_residual, alpha=0.3, s=5, c='blue', label='Analytical')
    axes[0, 0].scatter(y_empirical, corrected_residual, alpha=0.3, s=5, c='red', label='Corrected')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Empirical Frequency')
    axes[0, 0].set_ylabel('Residual (Predicted - Empirical)')
    axes[0, 0].set_title('Residuals vs Empirical Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(degree_product, residual_change, alpha=0.3, s=5, c='purple')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Degree Product (u × v)')
    axes[0, 1].set_ylabel('Residual Change (Corrected - Analytical)')
    axes[0, 1].set_title('How Corrections Changed Residuals')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    bins = 50
    axes[1, 0].hist(analytical_residual, bins=bins, alpha=0.5, label='Analytical', color='blue', density=True)
    axes[1, 0].hist(corrected_residual, bins=bins, alpha=0.5, label='Corrected', color='red', density=True)
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    improved_color = np.where(improved, 'green', 'red')
    axes[1, 1].scatter(u, v, c=improved_color, alpha=0.5, s=5)
    axes[1, 1].set_xlabel('Source Degree (u)')
    axes[1, 1].set_ylabel('Target Degree (v)')
    axes[1, 1].set_title('Improved (green) vs Worsened (red) by Degree')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label=f'Improved ({100*improved.mean():.1f}%)'),
        Patch(facecolor='red', alpha=0.5, label=f'Worsened ({100*worsened.mean():.1f}%)')
    ]
    axes[1, 1].legend(handles=legend_elements)

    plt.tight_layout()

    output_dir = results_dir / 'theoretical_corrections' / edge_type
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResidual analysis plot saved to: {output_dir / 'residual_analysis.png'}")

    analysis_df = pd.DataFrame({
        'source_degree': u,
        'target_degree': v,
        'degree_product': degree_product,
        'empirical_frequency': y_empirical,
        'analytical_pred': analytical_pred,
        'corrected_pred': corrected_pred,
        'analytical_residual': analytical_residual,
        'corrected_residual': corrected_residual,
        'residual_change': residual_change,
        'improved': improved
    })

    analysis_df.to_csv(output_dir / 'residual_analysis.csv', index=False)
    print(f"Detailed residual data saved to: {output_dir / 'residual_analysis.csv'}")


def main():
    repo_dir = Path.cwd()
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    edge_types = ['CbG', 'AeG']

    for edge_type in edge_types:
        try:
            analyze_residual_changes(edge_type, data_dir, results_dir)
        except Exception as e:
            print(f"\nERROR analyzing {edge_type}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
