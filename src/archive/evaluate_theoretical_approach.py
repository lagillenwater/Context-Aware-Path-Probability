"""
Evaluate theoretical corrections to analytical formula.

This script tests theoretical corrections that use ONLY features from
the original graph (no empirical frequencies in training).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from typing import Dict

from theory_guided_features import TheoryGuidedFeatureEngineer
from theoretical_corrections import (
    extract_graph_features,
    apply_all_corrections
)
from evaluate_theory_guided_models import analyze_residuals


def evaluate_single_edge_type(edge_type: str,
                              data_dir: Path,
                              results_dir: Path) -> Dict:
    """
    Evaluate theoretical corrections on one edge type.

    Parameters
    ----------
    edge_type : str
        Edge type identifier (e.g., 'CbG', 'AeG')
    data_dir : Path
        Data directory
    results_dir : Path
        Results directory

    Returns
    -------
    results : dict
        Evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating edge type: {edge_type}")
    print(f"{'='*80}\n")

    edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    edge_matrix = sp.load_npz(str(edge_file))

    print(f"Graph: {edge_matrix.shape[0]} sources x {edge_matrix.shape[1]} targets")
    print(f"Edges: {edge_matrix.nnz}")

    graph_features = extract_graph_features(edge_matrix)

    fe = TheoryGuidedFeatureEngineer(edge_matrix, edge_type)

    empirical_file = results_dir / 'empirical_edge_frequencies' / f'edge_frequency_by_degree_{edge_type}.csv'
    empirical_df = pd.read_csv(empirical_file)

    print(f"\nEmpirical data: {len(empirical_df)} degree pairs")

    u = empirical_df['source_degree'].values
    v = empirical_df['target_degree'].values
    y_empirical = empirical_df['frequency'].values

    analytical_pred = fe.get_analytical_baseline(u, v)

    analytical_r = pearsonr(y_empirical, analytical_pred)[0]
    analytical_bias = np.mean(analytical_pred - y_empirical)
    analytical_rmse = np.sqrt(np.mean((analytical_pred - y_empirical)**2))

    print(f"\n{'-'*60}")
    print("BASELINE: Analytical Formula")
    print(f"{'-'*60}")
    print(f"Correlation r: {analytical_r:.4f}")
    print(f"Bias: {analytical_bias:+.4f}")
    print(f"RMSE: {analytical_rmse:.4f}")

    corrected_pred = apply_all_corrections(
        analytical_pred, u, v, graph_features
    )

    corrected_r = pearsonr(y_empirical, corrected_pred)[0]
    corrected_bias = np.mean(corrected_pred - y_empirical)
    corrected_rmse = np.sqrt(np.mean((corrected_pred - y_empirical)**2))

    print(f"\n{'-'*60}")
    print("WITH CORRECTIONS: Theoretical Corrections")
    print(f"{'-'*60}")
    print(f"Correlation r: {corrected_r:.4f}")
    print(f"Bias: {corrected_bias:+.4f}")
    print(f"RMSE: {corrected_rmse:.4f}")

    improvement_r = corrected_r - analytical_r
    improvement_bias = abs(analytical_bias) - abs(corrected_bias)
    improvement_rmse = analytical_rmse - corrected_rmse

    print(f"\n{'-'*60}")
    print("IMPROVEMENTS")
    print(f"{'-'*60}")
    print(f"Delta r: {improvement_r:+.4f}")
    print(f"Bias reduction: {improvement_bias:+.4f}")
    print(f"RMSE reduction: {improvement_rmse:+.4f}")

    edge_results_dir = results_dir / 'theoretical_corrections' / edge_type
    edge_results_dir.mkdir(parents=True, exist_ok=True)

    analytical_metrics_dict, analytical_fig = analyze_residuals(
        y_empirical, analytical_pred,
        title=f"Analytical Formula - {edge_type}"
    )
    analytical_fig.savefig(edge_results_dir / 'analytical_residuals.png',
                          dpi=300, bbox_inches='tight')
    plt.close(analytical_fig)

    corrected_metrics_dict, corrected_fig = analyze_residuals(
        y_empirical, corrected_pred,
        title=f"With Theoretical Corrections - {edge_type}"
    )
    corrected_fig.savefig(edge_results_dir / 'corrected_residuals.png',
                         dpi=300, bbox_inches='tight')
    plt.close(corrected_fig)

    comparison_df = pd.DataFrame({
        'Analytical': {
            'r': analytical_r,
            'bias': analytical_bias,
            'rmse': analytical_rmse
        },
        'Corrected': {
            'r': corrected_r,
            'bias': corrected_bias,
            'rmse': corrected_rmse
        },
        'Improvement': {
            'r': improvement_r,
            'bias': improvement_bias,
            'rmse': improvement_rmse
        }
    }).T

    comparison_df.to_csv(edge_results_dir / 'comparison_metrics.csv')

    with open(edge_results_dir / 'graph_features.txt', 'w') as f:
        f.write("Graph Structure Features\n")
        f.write("="*60 + "\n\n")
        for key, value in graph_features.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"\nResults saved to: {edge_results_dir}")

    return {
        'edge_type': edge_type,
        'analytical_r': analytical_r,
        'corrected_r': corrected_r,
        'improvement_r': improvement_r,
        'analytical_bias': analytical_bias,
        'corrected_bias': corrected_bias,
        'improvement_bias': improvement_bias,
        'graph_features': graph_features
    }


def main():
    """
    Main evaluation script.
    """
    repo_dir = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    edge_types = ['CbG', 'AeG']

    print("Testing theoretical corrections")
    print("="*80)
    print("Approach: Use only original graph features for corrections")
    print("No empirical frequencies used in training")
    print("="*80)

    all_results = []

    for edge_type in edge_types:
        try:
            result = evaluate_single_edge_type(
                edge_type=edge_type,
                data_dir=data_dir,
                results_dir=results_dir
            )
            all_results.append(result)

        except FileNotFoundError as e:
            print(f"\nWARNING: Skipping {edge_type} - file not found: {e}")
            continue

        except Exception as e:
            print(f"\nERROR evaluating {edge_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        print("\n" + "="*80)
        print("SUMMARY ACROSS EDGE TYPES")
        print("="*80 + "\n")

        summary_df = pd.DataFrame([
            {
                'Edge Type': r['edge_type'],
                'Analytical r': r['analytical_r'],
                'Corrected r': r['corrected_r'],
                'Delta r': r['improvement_r'],
                'Analytical Bias': r['analytical_bias'],
                'Corrected Bias': r['corrected_bias'],
                'Bias Reduction': r['improvement_bias'],
                'Assortativity': r['graph_features']['assortativity'],
                'Gini': r['graph_features']['gini']
            }
            for r in all_results
        ])

        print(summary_df.to_string(index=False))

        summary_dir = results_dir / 'theoretical_corrections'
        summary_df.to_csv(summary_dir / 'summary.csv', index=False)

        print(f"\nSummary saved to: {summary_dir / 'summary.csv'}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
