#!/usr/bin/env python3
"""
Fix for the analytical correlation calculation in notebook 5.

This script provides the correct method to calculate real Pearson correlations
for the analytical formula within each degree combination, rather than the
fake "correlation = 1.0 - relative_error" that was being used.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_analytical_prediction(source_degree, target_degree, total_edges):
    """
    Calculate analytical prediction using the correct formula.
    P(u,v) = (u*v) / sqrt((u*v)^2 + (m-u-v+1)^2)
    """
    if source_degree <= 0 or target_degree <= 0:
        return 0.0

    uv_product = source_degree * target_degree
    denominator_term = total_edges - source_degree - target_degree + 1

    if denominator_term <= 0:
        denominator_term = 1  # Fallback for edge cases

    prediction = uv_product / np.sqrt(uv_product**2 + denominator_term**2)
    return float(prediction)

def calculate_real_analytical_correlation(edge_type, data_dir, results_dir):
    """
    Calculate real Pearson correlation between analytical predictions and empirical values
    within each degree combination.

    Returns:
        dict: Results with correlation metrics by degree combination
    """

    # Load individual predictions data
    pred_file = results_dir / f'{edge_type}_results' / f'{edge_type}_all_model_predictions.csv'
    if not pred_file.exists():
        print(f"Predictions file not found: {pred_file}")
        return {}

    predictions_df = pd.read_csv(pred_file)

    # Get total edges for analytical formula
    edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
    if not edge_file.exists():
        print(f"Edge file not found: {edge_file}")
        return {}

    import scipy.sparse as sp
    edge_matrix = sp.load_npz(str(edge_file))
    total_edges = edge_matrix.nnz

    # Calculate analytical predictions for each row
    predictions_df['analytical_prediction'] = predictions_df.apply(
        lambda row: calculate_analytical_prediction(
            row['source_degree'],
            row['target_degree'],
            total_edges
        ), axis=1
    )

    # Create degree combinations (binned)
    def create_degree_bin(degree):
        if degree <= 1:
            return "1"
        elif degree <= 5:
            return "2-5"
        elif degree <= 10:
            return "6-10"
        elif degree <= 20:
            return "11-20"
        elif degree <= 50:
            return "21-50"
        else:
            return "50+"

    predictions_df['source_bin'] = predictions_df['source_degree'].apply(create_degree_bin)
    predictions_df['target_bin'] = predictions_df['target_degree'].apply(create_degree_bin)
    predictions_df['degree_combination'] = predictions_df['source_bin'] + "_" + predictions_df['target_bin']

    # Calculate real correlations within each degree combination
    results = {}

    for degree_combo in predictions_df['degree_combination'].unique():
        combo_data = predictions_df[predictions_df['degree_combination'] == degree_combo]

        if len(combo_data) < 3:  # Need at least 3 points for meaningful correlation
            continue

        # Calculate Pearson correlation
        analytical_pred = combo_data['analytical_prediction'].values
        empirical_values = combo_data['edge_exists'].values

        if len(np.unique(analytical_pred)) > 1 and len(np.unique(empirical_values)) > 1:
            try:
                correlation, p_value = pearsonr(analytical_pred, empirical_values)

                results[degree_combo] = {
                    'edge_type': edge_type,
                    'degree_combination': degree_combo,
                    'n_samples': len(combo_data),
                    'mean_source_degree': combo_data['source_degree'].mean(),
                    'mean_target_degree': combo_data['target_degree'].mean(),
                    'mean_analytical_pred': analytical_pred.mean(),
                    'mean_empirical': empirical_values.mean(),
                    'real_correlation': correlation,
                    'correlation_p_value': p_value,
                    'mae': np.abs(analytical_pred - empirical_values).mean(),
                    'rmse': np.sqrt(((analytical_pred - empirical_values)**2).mean()),
                    'bias': (analytical_pred - empirical_values).mean()
                }
            except Exception as e:
                print(f"Error calculating correlation for {degree_combo}: {e}")
                continue
        else:
            # No variance in predictions or empirical values
            results[degree_combo] = {
                'edge_type': edge_type,
                'degree_combination': degree_combo,
                'n_samples': len(combo_data),
                'mean_source_degree': combo_data['source_degree'].mean(),
                'mean_target_degree': combo_data['target_degree'].mean(),
                'mean_analytical_pred': analytical_pred.mean(),
                'mean_empirical': empirical_values.mean(),
                'real_correlation': np.nan,
                'correlation_p_value': np.nan,
                'mae': np.abs(analytical_pred - empirical_values).mean(),
                'rmse': np.sqrt(((analytical_pred - empirical_values)**2).mean()),
                'bias': (analytical_pred - empirical_values).mean()
            }

    return results

def test_correlation_fix():
    """
    Test the correlation fix on a sample edge type.
    """
    repo_dir = Path.cwd()
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results' / 'model_comparison'

    # Test on CrC edge type (known to exist)
    edge_type = 'CrC'
    print(f"Testing analytical correlation fix for {edge_type}...")

    results = calculate_real_analytical_correlation(edge_type, data_dir, results_dir)

    if results:
        print(f"\n✓ Successfully calculated real correlations for {len(results)} degree combinations")
        print("\nSample results:")
        for combo, metrics in list(results.items())[:3]:
            print(f"  {combo}: r={metrics['real_correlation']:.4f}, p={metrics['correlation_p_value']:.3e}, n={metrics['n_samples']}")
    else:
        print("✗ No correlation results generated")

    return results

if __name__ == "__main__":
    test_correlation_fix()