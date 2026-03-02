"""
Example: Feature Importance Analysis for Theory-Guided Models

This script demonstrates how to determine which features are most important
for edge probability prediction.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from theory_guided_features import TheoryGuidedFeatureEngineer
from theory_guided_model import train_theory_guided_model
from feature_importance_analysis import comprehensive_feature_analysis


def main():
    """
    Run feature importance analysis for CbG edge type.
    """
    # Setup paths
    repo_dir = Path.cwd().parent if (Path.cwd().parent / 'data').exists() else Path.cwd()
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    edge_type = 'CbG'
    print(f"Analyzing feature importance for {edge_type}")
    print("="*80 + "\n")

    # Load data
    edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
    edge_matrix = sp.load_npz(str(edge_file))

    empirical_file = results_dir / 'empirical_edge_frequencies' / f'edge_frequency_by_degree_{edge_type}.csv'
    empirical_df = pd.read_csv(empirical_file)

    print(f"Loaded {len(empirical_df)} degree combinations\n")

    # Initialize feature engineer
    fe = TheoryGuidedFeatureEngineer(edge_matrix, edge_type)

    # Extract degrees and frequencies
    u = empirical_df['source_degree'].values
    v = empirical_df['target_degree'].values
    y = empirical_df['frequency'].values

    # Compute all features
    X = fe.compute_all_features(u, v, levels=(1, 2, 3, 4, 5))
    print(f"Generated {X.shape[1]} features across 5 levels\n")

    # Show feature breakdown
    print("Feature Breakdown:")
    feature_counts = {
        'Level 1 (Analytical)': len([f for f in X.columns if any(x in f for x in ['degree_product', 'removal_term', 'P_L', 'q_'])]),
        'Level 2 (Nonlinear)': len([f for f in X.columns if any(x in f for x in ['log_', 'sqrt_', 'geometric'])]),
        'Level 3 (Graph Stats)': len([f for f in X.columns if any(x in f for x in ['m_total', 'density', 'zscore'])]),
        'Level 4 (Polynomial)': len([f for f in X.columns if 'squared' in f or 'cubed' in f]),
        'Level 5 (Interactions)': len([f for f in X.columns if 'times_' in f or 'div_' in f]),
    }

    for level, count in feature_counts.items():
        print(f"  {level}: {count} features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Train model
    print("\nTraining model...")
    results = train_theory_guided_model(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        model_type='full',
        hidden_dims=(64, 32, 16),
        n_epochs=100,  # Fewer epochs for demo
        device='cpu'
    )

    model = results['model']
    print(f"\nFinal validation correlation: {results['final_val_corr']:.4f}\n")

    # Define feature groups
    feature_groups = {
        'Level 1: Analytical': [f for f in X.columns if any(x in f for x in ['degree_product', 'removal_term', 'P_L', 'q_over_r'])],
        'Level 2: Nonlinear': [f for f in X.columns if any(x in f for x in ['log_', 'sqrt_', 'geometric', 'asymmetry'])],
        'Level 3: Graph Stats': [f for f in X.columns if any(x in f for x in ['m_total', 'density', 'zscore'])],
        'Level 4: Polynomial': [f for f in X.columns if any(x in f for x in ['squared', 'cubed'])],
        'Level 5: Interactions': [f for f in X.columns if any(x in f for x in ['times_', 'div_', 'normalized_by'])],
    }

    # Run comprehensive analysis
    importance_dir = results_dir / 'feature_importance_example' / edge_type
    importance_dir.mkdir(parents=True, exist_ok=True)

    importance_results = comprehensive_feature_analysis(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_groups=feature_groups,
        device='cpu',
        save_dir=str(importance_dir)
    )

    print(f"\n\nResults saved to: {importance_dir}")
    print("\nGenerated files:")
    print("  - permutation_importance.csv/png")
    print("  - gradient_importance.csv/png")
    print("  - weight_importance.csv/png")
    print("  - ablation_importance.csv")
    print("  - top5_comparison.csv")


if __name__ == '__main__':
    main()
