"""
Evaluate physics-informed neural network approach.

This script trains and evaluates a physics-informed NN that uses ONLY
theoretical constraints from XSwap, no empirical frequencies.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.stats import pearsonr
from typing import Dict

from physics_informed_nn import (
    train_physics_informed_nn,
    PhysicsInformedNN
)
from theory_guided_features import TheoryGuidedFeatureEngineer
from theoretical_corrections import (
    extract_graph_features,
    apply_all_corrections
)
from evaluate_theory_guided_models import analyze_residuals


def evaluate_pinn_single_edge_type(edge_type: str,
                                   data_dir: Path,
                                   results_dir: Path,
                                   n_epochs: int = 1000,
                                   learning_rate: float = 0.001) -> Dict:
    """
    Evaluate physics-informed NN on one edge type.

    Parameters
    ----------
    edge_type : str
        Edge type identifier
    data_dir : Path
        Data directory
    results_dir : Path
        Results directory
    n_epochs : int
        Training epochs
    learning_rate : float
        Learning rate

    Returns
    -------
    results : dict
        Evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Physics-Informed NN Evaluation: {edge_type}")
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

    degree_pairs = np.column_stack([u, v])

    print(f"\n{'-'*60}")
    print("Training Physics-Informed Neural Network")
    print(f"{'-'*60}")
    print(f"Training epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("Loss components: conservation, marginal, detailed balance, monotonicity")
    print(f"{'-'*60}\n")

    model = train_physics_informed_nn(
        edge_matrix=edge_matrix,
        degree_pairs=degree_pairs,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        hidden_dims=[64, 32, 16],
        loss_weights={
            'conservation': 1.0,
            'marginal': 1.0,
            'detailed_balance': 0.5,
            'monotonicity': 0.5
        },
        verbose=True
    )

    print(f"\n{'-'*60}")
    print("Generating Predictions")
    print(f"{'-'*60}")

    u_tensor = torch.tensor(u, dtype=torch.float32)
    v_tensor = torch.tensor(v, dtype=torch.float32)

    u_normalized = (u_tensor - u_tensor.mean()) / (u_tensor.std() + 1e-8)
    v_normalized = (v_tensor - v_tensor.mean()) / (v_tensor.std() + 1e-8)

    with torch.no_grad():
        pinn_pred = model(u_normalized, v_normalized).numpy()

    analytical_pred = fe.get_analytical_baseline(u, v)

    corrected_pred = apply_all_corrections(analytical_pred, u, v, graph_features)

    pinn_r = pearsonr(y_empirical, pinn_pred)[0]
    pinn_bias = np.mean(pinn_pred - y_empirical)
    pinn_rmse = np.sqrt(np.mean((pinn_pred - y_empirical)**2))

    analytical_r = pearsonr(y_empirical, analytical_pred)[0]
    analytical_bias = np.mean(analytical_pred - y_empirical)
    analytical_rmse = np.sqrt(np.mean((analytical_pred - y_empirical)**2))

    corrected_r = pearsonr(y_empirical, corrected_pred)[0]
    corrected_bias = np.mean(corrected_pred - y_empirical)
    corrected_rmse = np.sqrt(np.mean((corrected_pred - y_empirical)**2))

    print(f"\n{'-'*60}")
    print("RESULTS COMPARISON")
    print(f"{'-'*60}")
    print(f"\n{'Method':<25} {'Correlation r':<15} {'Bias':<15} {'RMSE':<15}")
    print(f"{'-'*60}")
    print(f"{'Analytical':<25} {analytical_r:<15.4f} {analytical_bias:<+15.4f} {analytical_rmse:<15.4f}")
    print(f"{'Corrected':<25} {corrected_r:<15.4f} {corrected_bias:<+15.4f} {corrected_rmse:<15.4f}")
    print(f"{'Physics-Informed NN':<25} {pinn_r:<15.4f} {pinn_bias:<+15.4f} {pinn_rmse:<15.4f}")
    print(f"{'-'*60}")
    print(f"\nImprovement over analytical:")
    print(f"  Corrected:  Δr = {corrected_r - analytical_r:+.4f}")
    print(f"  PINN:       Δr = {pinn_r - analytical_r:+.4f}")
    print(f"\nImprovement over corrected:")
    print(f"  PINN:       Δr = {pinn_r - corrected_r:+.4f}")

    output_dir = results_dir / 'physics_informed_nn' / edge_type
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / 'model.pt')
    print(f"\nModel saved to: {output_dir / 'model.pt'}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].scatter(y_empirical, analytical_pred, alpha=0.3, s=5, c='blue', label='Analytical')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 0].set_xlabel('Empirical Frequency')
    axes[0, 0].set_ylabel('Predicted Frequency')
    axes[0, 0].set_title(f'Analytical (r = {analytical_r:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(y_empirical, corrected_pred, alpha=0.3, s=5, c='red', label='Corrected')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 1].set_xlabel('Empirical Frequency')
    axes[0, 1].set_ylabel('Predicted Frequency')
    axes[0, 1].set_title(f'Corrected (r = {corrected_r:.4f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(y_empirical, pinn_pred, alpha=0.3, s=5, c='green', label='PINN')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1, 0].set_xlabel('Empirical Frequency')
    axes[1, 0].set_ylabel('Predicted Frequency')
    axes[1, 0].set_title(f'Physics-Informed NN (r = {pinn_r:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    methods = ['Analytical', 'Corrected', 'PINN']
    correlations = [analytical_r, corrected_r, pinn_r]
    colors = ['blue', 'red', 'green']

    axes[1, 1].bar(methods, correlations, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Correlation (r)')
    axes[1, 1].set_title('Method Comparison')
    axes[1, 1].set_ylim([min(correlations) - 0.01, 1.0])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for i, (method, corr) in enumerate(zip(methods, correlations)):
        axes[1, 1].text(i, corr + 0.002, f'{corr:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to: {output_dir / 'comparison.png'}")

    comparison_df = pd.DataFrame({
        'Method': ['Analytical', 'Corrected', 'PINN'],
        'Correlation_r': [analytical_r, corrected_r, pinn_r],
        'Bias': [analytical_bias, corrected_bias, pinn_bias],
        'RMSE': [analytical_rmse, corrected_rmse, pinn_rmse],
        'Improvement_over_analytical': [
            0.0,
            corrected_r - analytical_r,
            pinn_r - analytical_r
        ]
    })

    comparison_df.to_csv(output_dir / 'comparison_metrics.csv', index=False)
    print(f"Metrics saved to: {output_dir / 'comparison_metrics.csv'}")

    predictions_df = pd.DataFrame({
        'source_degree': u,
        'target_degree': v,
        'empirical': y_empirical,
        'analytical': analytical_pred,
        'corrected': corrected_pred,
        'pinn': pinn_pred
    })

    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    print(f"Predictions saved to: {output_dir / 'predictions.csv'}")

    return {
        'edge_type': edge_type,
        'analytical_r': analytical_r,
        'corrected_r': corrected_r,
        'pinn_r': pinn_r,
        'analytical_bias': analytical_bias,
        'corrected_bias': corrected_bias,
        'pinn_bias': pinn_bias,
        'improvement_corrected': corrected_r - analytical_r,
        'improvement_pinn': pinn_r - analytical_r,
        'pinn_vs_corrected': pinn_r - corrected_r
    }


def main():
    """
    Main evaluation script.
    """
    repo_dir = Path.cwd()
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    edge_types = ['CbG', 'AeG']

    print("="*80)
    print("PHYSICS-INFORMED NEURAL NETWORK EVALUATION")
    print("="*80)
    print("\nApproach: Train NN using ONLY XSwap theoretical constraints")
    print("No empirical frequencies used in training")
    print("\nLoss components:")
    print("  1. Conservation: Total predicted edges = actual edge count")
    print("  2. Marginal: Marginal distributions match observed degrees")
    print("  3. Detailed Balance: XSwap equilibrium constraints")
    print("  4. Monotonicity: Higher degree product implies higher probability")
    print("="*80)

    all_results = []

    for edge_type in edge_types:
        try:
            result = evaluate_pinn_single_edge_type(
                edge_type=edge_type,
                data_dir=data_dir,
                results_dir=results_dir,
                n_epochs=1000,
                learning_rate=0.001
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

        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))

        summary_dir = results_dir / 'physics_informed_nn'
        summary_df.to_csv(summary_dir / 'summary.csv', index=False)

        print(f"\nSummary saved to: {summary_dir / 'summary.csv'}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
