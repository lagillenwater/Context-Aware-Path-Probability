"""
Generate analytical vs empirical comparison CSV files for all edge types.

This script loads existing empirical frequency data and compares it against
the analytical formula without re-running the full model testing pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.sparse as sp
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def analytical_prior(u, v, m):
    """
    Analytical formula for edge probability.

    P = uv / sqrt(uv^2 + (m - u - v + 1)^2)
    """
    numerator = u * v
    denominator = np.sqrt(u * v**2 + (m - u - v + 1)**2)
    return numerator / denominator


def generate_comparison_for_edge_type(edge_type):
    """Generate analytical vs empirical comparison for a single edge type."""

    # Paths
    empirical_path = Path('results/empirical_edge_frequencies') / f'edge_frequency_by_degree_{edge_type}.csv'
    edge_matrix_path = Path('data/edges') / f'{edge_type}.sparse.npz'
    output_dir = Path('results/model_comparison') / f'{edge_type}_results'
    output_path = output_dir / 'analytical_vs_empirical_comparison.csv'

    # Check if empirical data exists
    if not empirical_path.exists():
        print(f"Skipping {edge_type}: empirical data not found at {empirical_path}")
        return None

    # Check if edge matrix exists
    if not edge_matrix_path.exists():
        print(f"Skipping {edge_type}: edge matrix not found at {edge_matrix_path}")
        return None

    # Load empirical data
    df = pd.read_csv(empirical_path)

    # Get degrees and empirical frequencies
    source_degrees = df['source_degree'].values
    target_degrees = df['target_degree'].values

    # Handle both 'frequency' and 'empirical_frequency' column names
    if 'empirical_frequency' in df.columns:
        empirical_frequencies = df['empirical_frequency'].values
    elif 'frequency' in df.columns:
        empirical_frequencies = df['frequency'].values
    else:
        print(f"Skipping {edge_type}: no frequency column found")
        return None

    # Load edge matrix to get total edges (m)
    edge_matrix = sp.load_npz(edge_matrix_path)
    m = edge_matrix.nnz  # Number of non-zero elements (total edges)

    # Compute analytical probabilities
    analytical_probs = np.array([
        analytical_prior(u, v, m)
        for u, v in zip(source_degrees, target_degrees)
    ])

    # Compute metrics
    mask = ~(np.isnan(empirical_frequencies) | np.isnan(analytical_probs))
    emp_clean = empirical_frequencies[mask]
    ana_clean = analytical_probs[mask]

    if len(emp_clean) < 2:
        print(f"Skipping {edge_type}: insufficient valid data points")
        return None

    mae = mean_absolute_error(emp_clean, ana_clean)
    rmse = np.sqrt(mean_squared_error(emp_clean, ana_clean))
    r2 = r2_score(emp_clean, ana_clean)

    try:
        pearson_corr, pearson_p = pearsonr(emp_clean, ana_clean)
    except:
        pearson_corr, pearson_p = np.nan, np.nan

    try:
        spearman_corr, spearman_p = spearmanr(emp_clean, ana_clean)
    except:
        spearman_corr, spearman_p = np.nan, np.nan

    # Create results DataFrame matching the format from model_evaluation.py
    results_df = pd.DataFrame([{
        'Model': 'Current Analytical',
        'Comparison': 'vs_empirical',
        'edge_type': edge_type,
        'total_edges_m': m,
        'n_samples': len(emp_clean),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p,
        'mean_analytical': np.mean(ana_clean),
        'mean_empirical': np.mean(emp_clean),
        'std_analytical': np.std(ana_clean),
        'std_empirical': np.std(emp_clean),
        'min_analytical': np.min(ana_clean),
        'max_analytical': np.max(ana_clean),
        'min_empirical': np.min(emp_clean),
        'max_empirical': np.max(emp_clean)
    }])

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results_df.to_csv(output_path, index=False)

    print(f"✓ {edge_type}: r={pearson_corr:.4f}, saved to {output_path}")

    return results_df


def main():
    """Generate comparison files for all edge types."""

    # Get all edge types from empirical frequency directory
    empirical_dir = Path('results/empirical_edge_frequencies')

    if not empirical_dir.exists():
        print(f"Error: {empirical_dir} does not exist")
        return

    # Find all empirical frequency files
    edge_type_files = list(empirical_dir.glob('edge_frequency_by_degree_*.csv'))
    edge_types = [f.stem.replace('edge_frequency_by_degree_', '') for f in edge_type_files]

    print(f"Found {len(edge_types)} edge types to process\n")

    # Process each edge type
    results = []
    for edge_type in sorted(edge_types):
        result = generate_comparison_for_edge_type(edge_type)
        if result is not None:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Successfully processed {len(results)} / {len(edge_types)} edge types")
    print(f"{'='*60}")

    if results:
        all_results = pd.concat(results, ignore_index=True)
        print("\nSummary statistics:")
        print(all_results[['edge_type', 'pearson_r', 'mae', 'rmse']].to_string(index=False))


if __name__ == '__main__':
    main()
