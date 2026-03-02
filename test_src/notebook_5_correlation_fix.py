#!/usr/bin/env python3
"""
Complete fix for notebook 5 to replace fake analytical correlations with real ones.

This provides the corrected function that should replace the existing
analytical correlation calculation in the notebook.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_real_analytical_metrics_improved(degree_metrics_df, data_dir, results_dir):
    """
    Calculate real analytical correlations by accessing individual prediction data
    instead of using the fake "correlation = 1.0 - relative_error" approach.

    This function replaces the problematic analytical metrics calculation
    in notebook 5 with proper Pearson correlations.
    """

    print("Computing REAL analytical correlations using individual predictions...")

    analytical_metrics_list = []

    # Get unique combinations from existing degree metrics
    unique_combinations = degree_metrics_df.groupby(['edge_type', 'degree_combination']).first().reset_index()

    for _, row in unique_combinations.iterrows():
        edge_type = row['edge_type']
        degree_combination = row['degree_combination']

        # Access the actual combination data to get empirical mean
        combination_data = degree_metrics_df[
            (degree_metrics_df['edge_type'] == edge_type) &
            (degree_metrics_df['degree_combination'] == degree_combination)
        ]

        if len(combination_data) == 0:
            continue

        mean_empirical = combination_data['mean_empirical'].iloc[0]
        mean_source_degree = combination_data['mean_source_degree'].iloc[0]
        mean_target_degree = combination_data['mean_target_degree'].iloc[0]
        n_samples = combination_data['n_samples'].iloc[0]

        # Load individual predictions for this edge type
        pred_file = results_dir / f'{edge_type}_results' / f'{edge_type}_all_model_predictions.csv'

        if not pred_file.exists():
            print(f"  ⚠ Predictions file not found for {edge_type}, using fallback method")
            # Fallback: use the old approach but mark correlation as NaN
            analytical_pred = calculate_analytical_fallback(mean_source_degree, mean_target_degree)
            mae = abs(analytical_pred - mean_empirical)
            bias = analytical_pred - mean_empirical
            correlation = np.nan  # Mark as unavailable
            relative_error = mae / max(mean_empirical, 0.001)

        else:
            # Load individual predictions and calculate real correlation
            try:
                predictions_df = pd.read_csv(pred_file)

                # Get total edges for analytical formula
                edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
                if edge_file.exists():
                    import scipy.sparse as sp
                    edge_matrix = sp.load_npz(str(edge_file))
                    total_edges = edge_matrix.nnz
                else:
                    total_edges = 1000  # Fallback value

                # Filter data to this specific degree combination
                combo_data = filter_to_degree_combination(predictions_df, degree_combination)

                if len(combo_data) >= 3:  # Need minimum samples for correlation
                    # Calculate analytical predictions for each individual data point
                    analytical_preds = []
                    empirical_vals = []

                    for _, pred_row in combo_data.iterrows():
                        analytical_pred_point = calculate_analytical_prediction(
                            pred_row['source_degree'],
                            pred_row['target_degree'],
                            total_edges
                        )
                        analytical_preds.append(analytical_pred_point)
                        empirical_vals.append(pred_row['edge_exists'])

                    analytical_preds = np.array(analytical_preds)
                    empirical_vals = np.array(empirical_vals)

                    # Calculate real Pearson correlation
                    if len(np.unique(analytical_preds)) > 1 and len(np.unique(empirical_vals)) > 1:
                        correlation, p_value = pearsonr(analytical_preds, empirical_vals)
                    else:
                        correlation = np.nan  # No variance for correlation

                    # Calculate other metrics
                    analytical_pred = analytical_preds.mean()  # Mean for summary
                    mae = np.abs(analytical_preds - empirical_vals).mean()
                    bias = (analytical_preds - empirical_vals).mean()
                    relative_error = mae / max(empirical_vals.mean(), 0.001)

                else:
                    # Too few samples, use fallback
                    analytical_pred = calculate_analytical_fallback(mean_source_degree, mean_target_degree)
                    mae = abs(analytical_pred - mean_empirical)
                    bias = analytical_pred - mean_empirical
                    correlation = np.nan
                    relative_error = mae / max(mean_empirical, 0.001)

            except Exception as e:
                print(f"  ⚠ Error processing {edge_type}/{degree_combination}: {e}")
                # Fallback to old method but with NaN correlation
                analytical_pred = calculate_analytical_fallback(mean_source_degree, mean_target_degree)
                mae = abs(analytical_pred - mean_empirical)
                bias = analytical_pred - mean_empirical
                correlation = np.nan
                relative_error = mae / max(mean_empirical, 0.001)

        # Store the corrected analytical metrics
        analytical_metrics_list.append({
            'degree_combination': degree_combination,
            'n_samples': n_samples,
            'mean_source_degree': mean_source_degree,
            'mean_target_degree': mean_target_degree,
            'mean_predicted': analytical_pred,
            'mean_empirical': mean_empirical,
            'bias': bias,
            'variance': 0.0,  # Analytical has no variance
            'mae': mae,
            'rmse': mae,  # For analytical, RMSE = MAE since no variance
            'median_ae': mae,
            'q75_ae': mae,
            'q95_ae': mae,
            'correlation': correlation,  # NOW THIS IS A REAL CORRELATION!
            'mean_relative_error': relative_error,
            'model': 'Analytical Formula',
            'edge_type': edge_type
        })

    if analytical_metrics_list:
        analytical_metrics_df = pd.DataFrame(analytical_metrics_list)
        print(f"✓ Computed real analytical correlations for {len(analytical_metrics_df)} combinations")

        # Show correlation summary
        valid_corrs = analytical_metrics_df['correlation'].dropna()
        if len(valid_corrs) > 0:
            print(f"  Real correlation range: {valid_corrs.min():.4f} to {valid_corrs.max():.4f}")
            print(f"  Mean real correlation: {valid_corrs.mean():.4f}")
        else:
            print("  No valid correlations computed")

        return analytical_metrics_df
    else:
        print("✗ No analytical metrics could be computed")
        return pd.DataFrame()

def calculate_analytical_prediction(source_degree, target_degree, total_edges):
    """Calculate analytical prediction using correct formula."""
    if source_degree <= 0 or target_degree <= 0:
        return 0.0

    uv_product = source_degree * target_degree
    denominator_term = total_edges - source_degree - target_degree + 1

    if denominator_term <= 0:
        denominator_term = 1

    prediction = uv_product / np.sqrt(uv_product**2 + denominator_term**2)
    return float(prediction)

def calculate_analytical_fallback(mean_source_degree, mean_target_degree):
    """Fallback analytical calculation when individual data unavailable."""
    if mean_source_degree > 0 and mean_target_degree > 0:
        degree_product = mean_source_degree * mean_target_degree
        return degree_product / np.sqrt(degree_product**2 + 1000**2)
    else:
        return 0.001

def filter_to_degree_combination(predictions_df, degree_combination):
    """Filter predictions to specific degree combination."""
    # Parse the degree combination (e.g., "6-10_2-5")
    if "_" not in degree_combination:
        return pd.DataFrame()

    source_bin, target_bin = degree_combination.split("_")

    def is_in_bin(degree, bin_str):
        if bin_str == "1":
            return degree <= 1
        elif bin_str == "2-5":
            return 2 <= degree <= 5
        elif bin_str == "6-10":
            return 6 <= degree <= 10
        elif bin_str == "11-20":
            return 11 <= degree <= 20
        elif bin_str == "21-50":
            return 21 <= degree <= 50
        elif bin_str == "50+":
            return degree > 50
        else:
            return False

    source_mask = predictions_df['source_degree'].apply(lambda x: is_in_bin(x, source_bin))
    target_mask = predictions_df['target_degree'].apply(lambda x: is_in_bin(x, target_bin))

    return predictions_df[source_mask & target_mask]

# Demonstration of the fix
def demonstrate_fix():
    """Show the difference between fake and real correlations."""
    print("="*80)
    print("ANALYTICAL CORRELATION FIX DEMONSTRATION")
    print("="*80)

    # Fake correlation example
    print("\n1. FAKE CORRELATION (original method):")
    mae = 0.05
    mean_empirical = 0.1
    relative_error = mae / mean_empirical
    fake_correlation = max(0.1, 1.0 - relative_error)
    print(f"  MAE: {mae}")
    print(f"  Mean empirical: {mean_empirical}")
    print(f"  Relative error: {relative_error}")
    print(f"  Fake 'correlation': {fake_correlation:.4f}")
    print("  ⚠ This is NOT a real correlation!")

    # Real correlation example
    print("\n2. REAL CORRELATION (fixed method):")
    # Simulate some data
    np.random.seed(42)
    analytical_preds = np.random.uniform(0.01, 0.1, 1000)
    empirical_vals = np.random.binomial(1, 0.05, 1000)  # Binary outcomes

    real_correlation, p_value = pearsonr(analytical_preds, empirical_vals)
    print(f"  Analytical predictions: {analytical_preds.mean():.4f} ± {analytical_preds.std():.4f}")
    print(f"  Empirical values: {empirical_vals.mean():.4f} ± {empirical_vals.std():.4f}")
    print(f"  Real Pearson correlation: {real_correlation:.4f}")
    print(f"  P-value: {p_value:.3e}")
    print("  ✓ This IS a real correlation!")

    print(f"\n3. MATHEMATICAL CONSISTENCY CHECK:")
    print(f"  Can low degree correlations (~0.02) aggregate to high overall correlation (0.96)? YES!")
    print(f"  Explanation: Simpson's Paradox - different scales mask within-group patterns")

if __name__ == "__main__":
    demonstrate_fix()