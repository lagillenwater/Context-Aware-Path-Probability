"""
Comprehensive evaluation of theory-guided models vs analytical formula.

This script:
1. Trains theory-guided NN models with hierarchical features
2. Compares to analytical formula baseline
3. Analyzes residuals for systematic bias
4. Validates across multiple edge types
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from typing import Dict, Tuple

from theory_guided_features import TheoryGuidedFeatureEngineer, prepare_training_data
from theory_guided_model import train_theory_guided_model
from feature_importance_analysis import comprehensive_feature_analysis


def analyze_residuals(empirical: np.ndarray,
                      predicted: np.ndarray,
                      title: str = "Residual Analysis") -> Tuple[Dict, plt.Figure]:
    """
    Comprehensive residual analysis.

    Parameters
    ----------
    empirical : array
        True empirical frequencies
    predicted : array
        Model predictions
    title : str
        Plot title

    Returns
    -------
    metrics : dict
        Residual statistics
    """
    residuals = predicted - empirical

    # Calculate metrics
    metrics = {
        'mae': np.mean(np.abs(residuals)),
        'rmse': np.sqrt(np.mean(residuals**2)),
        'mean_bias': np.mean(residuals),
        'median_bias': np.median(residuals),
        'std_residuals': np.std(residuals),
        'pearson_r': pearsonr(empirical, predicted)[0],
        'spearman_r': spearmanr(empirical, predicted)[0],
    }

    # Residual plots (4-panel layout)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel [0,0]: Predicted vs Empirical with density coloring
    hb = axes[0, 0].hexbin(empirical, predicted, gridsize=50,
                          cmap='viridis', mincnt=1, alpha=0.8)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect match')
    plt.colorbar(hb, ax=axes[0, 0], label='Point density')
    axes[0, 0].set_xlabel('Empirical Frequency', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Frequency', fontsize=11)
    axes[0, 0].set_title(f'{title}\nr = {metrics["pearson_r"]:.4f}',
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel [0,1]: Residuals vs Empirical (check for bias)
    axes[0, 1].scatter(empirical, residuals, alpha=0.3, s=10, c='steelblue')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero bias')
    axes[0, 1].axhline(y=metrics['mean_bias'], color='orange',
                      linestyle='-', linewidth=2,
                      label=f'Mean bias = {metrics["mean_bias"]:.4f}')
    axes[0, 1].set_xlabel('Empirical Frequency', fontsize=11)
    axes[0, 1].set_ylabel('Residuals (Predicted - Empirical)', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel [1,0]: Residual histogram
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
    axes[1, 0].axvline(x=metrics['mean_bias'], color='orange',
                      linestyle='-', linewidth=2, label='Mean')
    axes[1, 0].set_xlabel('Residuals', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title(f'Residual Distribution\nMean = {metrics["mean_bias"]:.4f}',
                        fontsize=12, fontweight='bold')
    axes[1, 0].legend()

    # Panel [1,1]: Q-Q plot for normality check
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)',
                        fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].get_lines()[0].set_markersize(4)
    axes[1, 1].get_lines()[0].set_alpha(0.6)

    plt.tight_layout()

    return metrics, fig


def evaluate_edge_type(edge_type: str,
                      data_dir: Path,
                      results_dir: Path,
                      feature_levels: tuple = (1, 2, 3, 4, 5),
                      model_type: str = 'full',
                      device: str = 'cpu') -> Dict:
    """
    Complete evaluation pipeline for one edge type.

    Parameters
    ----------
    edge_type : str
        Edge type identifier (e.g., 'CbG')
    data_dir : Path
        Data directory
    results_dir : Path
        Results output directory
    feature_levels : tuple
        Which feature levels to use
    model_type : str
        'full' or 'residual'
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    results : dict
        Complete evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating edge type: {edge_type}")
    print(f"{'='*80}\n")

    # Load data
    edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
    edge_matrix = sp.load_npz(str(edge_file))

    empirical_file = results_dir / 'empirical_edge_frequencies' / f'edge_frequency_by_degree_{edge_type}.csv'
    empirical_df = pd.read_csv(empirical_file)

    print(f"Loaded {len(empirical_df)} degree combinations")
    print(f"Edge matrix: {edge_matrix.shape}, {edge_matrix.nnz} edges")

    # Initialize feature engineer
    fe = TheoryGuidedFeatureEngineer(edge_matrix, edge_type)

    # Extract degrees and frequencies
    u = empirical_df['source_degree'].values
    v = empirical_df['target_degree'].values
    y = empirical_df['frequency'].values

    # Compute analytical baseline
    analytical_pred = fe.get_analytical_baseline(u, v)

    # Compute theory-guided features
    X = fe.compute_all_features(u, v, levels=feature_levels)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Feature levels used: {feature_levels}")

    # Train/test split
    X_train, X_test, y_train, y_test, analytical_train, analytical_test = train_test_split(
        X, y, analytical_pred, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train model
    print(f"\nTraining {model_type} model...")
    training_results = train_theory_guided_model(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        analytical_baseline=analytical_pred if model_type == 'residual' else None,
        model_type=model_type,
        hidden_dims=(64, 32, 16),
        learning_rate=0.001,
        n_epochs=200,
        batch_size=256,
        patience=20,
        device=device
    )

    model = training_results['model']

    # Generate predictions
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled.values).to(device)

        if model_type == 'full':
            nn_pred_test = model(X_test_tensor).cpu().numpy()
        else:
            analytical_test_tensor = torch.FloatTensor(analytical_test).to(device)
            nn_pred_test = model(X_test_tensor, analytical_test_tensor).cpu().numpy()

    # Analyze residuals: Analytical formula
    print("\n" + "="*60)
    print("ANALYTICAL FORMULA RESIDUALS")
    print("="*60)
    analytical_metrics, analytical_fig = analyze_residuals(
        y_test, analytical_test,
        title=f"Analytical Formula - {edge_type}"
    )

    # Analyze residuals: NN model
    print("\n" + "="*60)
    print("NEURAL NETWORK RESIDUALS")
    print("="*60)
    nn_metrics, nn_fig = analyze_residuals(
        y_test, nn_pred_test,
        title=f"Theory-Guided NN - {edge_type}"
    )

    # Comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    comparison = pd.DataFrame({
        'Analytical Formula': analytical_metrics,
        'Theory-Guided NN': nn_metrics
    }).T
    print(comparison.to_string())

    # Improvement metrics
    improvement = {
        'pearson_r_improvement': nn_metrics['pearson_r'] - analytical_metrics['pearson_r'],
        'bias_reduction': abs(analytical_metrics['mean_bias']) - abs(nn_metrics['mean_bias']),
        'rmse_reduction': analytical_metrics['rmse'] - nn_metrics['rmse'],
    }

    print("\nImprovements:")
    for key, value in improvement.items():
        print(f"  {key}: {value:+.4f}")

    # Feature importance analysis
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Define feature groups for ablation
    feature_groups = {
        'Level 1: Analytical': [f for f in X.columns if any(x in f for x in ['degree_product', 'removal_term', 'P_L', 'q_over_r', 'q_normalized', 'r_normalized'])],
        'Level 2: Nonlinear': [f for f in X.columns if any(x in f for x in ['log_', 'sqrt_', 'geometric', 'arithmetic', 'asymmetry', 'ratio', 'harmonic'])],
        'Level 3: Graph Stats': [f for f in X.columns if any(x in f for x in ['m_total', 'density', 'n_source', 'n_target', 'mean_', 'std_', 'zscore'])],
        'Level 4: Polynomial': [f for f in X.columns if any(x in f for x in ['squared', 'cubed', 'u2_v', 'u_v2', 'uv_sum'])],
        'Level 5: Interactions': [f for f in X.columns if any(x in f for x in ['times_', 'div_', 'normalized_by'])],
    }

    # Save results
    edge_results_dir = results_dir / 'theory_guided_evaluation' / edge_type
    edge_results_dir.mkdir(parents=True, exist_ok=True)

    # Run comprehensive feature importance
    importance_results = comprehensive_feature_analysis(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_groups=feature_groups,
        device=device,
        save_dir=str(edge_results_dir)
    )

    # Save figures
    analytical_fig.savefig(edge_results_dir / 'analytical_residuals.png', dpi=300, bbox_inches='tight')
    nn_fig.savefig(edge_results_dir / 'nn_residuals.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    # Save metrics
    comparison.to_csv(edge_results_dir / 'comparison_metrics.csv')

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'feature_levels': feature_levels,
        'scaler': scaler,
        'feature_columns': list(X.columns),
        'metrics': nn_metrics,
    }, edge_results_dir / 'trained_model.pt')

    print(f"\nResults saved to: {edge_results_dir}")

    return {
        'edge_type': edge_type,
        'analytical_metrics': analytical_metrics,
        'nn_metrics': nn_metrics,
        'improvement': improvement,
        'model': model,
        'scaler': scaler,
        'feature_columns': list(X.columns),
        'feature_importance': importance_results,
    }


def main():
    """
    Main evaluation script.
    """
    # Paths
    repo_dir = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    # Edge types to evaluate
    edge_types = ['CbG', 'AeG']  # Start with these two

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Evaluate each edge type
    all_results = []

    for edge_type in edge_types:
        try:
            results = evaluate_edge_type(
                edge_type=edge_type,
                data_dir=data_dir,
                results_dir=results_dir,
                feature_levels=(1, 2, 3, 4, 5),  # All features
                model_type='full',
                device=device
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nERROR evaluating {edge_type}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-edge-type summary
    if all_results:
        print("\n" + "="*80)
        print("CROSS-EDGE-TYPE SUMMARY")
        print("="*80 + "\n")

        summary_data = []
        for res in all_results:
            summary_data.append({
                'Edge Type': res['edge_type'],
                'Analytical r': res['analytical_metrics']['pearson_r'],
                'NN r': res['nn_metrics']['pearson_r'],
                'r Improvement': res['improvement']['pearson_r_improvement'],
                'Analytical Bias': res['analytical_metrics']['mean_bias'],
                'NN Bias': res['nn_metrics']['mean_bias'],
                'Bias Reduction': res['improvement']['bias_reduction'],
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Save cross-edge summary
        summary_dir = results_dir / 'theory_guided_evaluation'
        summary_df.to_csv(summary_dir / 'cross_edge_summary.csv', index=False)


if __name__ == '__main__':
    main()
