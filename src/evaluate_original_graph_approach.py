"""
Evaluate neural network models trained on original graph frequencies.

This script implements the correct training approach:
1. Compute frequencies from original Hetionet graph by (source_degree, target_degree)
2. Split degree pairs into train/test
3. Train models on original graph frequencies
4. Validate against 200-permutation empirical frequencies
5. Measure interpolation quality for unseen degree pairs
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from original_graph_frequencies import (
    compute_original_graph_frequencies,
    analyze_degree_pair_coverage,
    get_unseen_degree_pairs
)
from theory_guided_features import TheoryGuidedFeatureEngineer
from theory_guided_model import train_theory_guided_model
from reduced_feature_sets import ReducedFeatureSet
from model_comparison import SimpleNN
from evaluate_theory_guided_models import analyze_residuals


def train_simplenn_on_frequencies(X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  device: str = 'cpu') -> Dict:
    """
    Train SimpleNN on continuous frequency targets.

    Parameters
    ----------
    X_train : array (N, 2)
        Training degree pairs
    y_train : array (N,)
        Training frequencies
    X_test : array (N, 2)
        Test degree pairs
    y_test : array (N,)
        Test frequencies
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    results : dict
        Model, predictions, and metrics
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)

    model = SimpleNN(input_dim=2, hidden_dims=(128, 64, 32), dropout_rate=0.3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    training_time = time.time() - start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).cpu().numpy()
        test_pred = model(X_test_t).cpu().numpy()

    test_corr = pearsonr(y_test, test_pred)[0]
    train_corr = pearsonr(y_train, train_pred)[0]

    return {
        'model': model,
        'scaler': scaler,
        'test_predictions': test_pred,
        'train_predictions': train_pred,
        'test_correlation': test_corr,
        'train_correlation': train_corr,
        'training_time': training_time,
        'bias': np.mean(test_pred - y_test),
        'rmse': np.sqrt(np.mean((test_pred - y_test)**2))
    }


def evaluate_on_empirical_frequencies(model: nn.Module,
                                     scaler: StandardScaler,
                                     empirical_df: pd.DataFrame,
                                     device: str = 'cpu') -> Tuple[np.ndarray, Dict]:
    """
    Evaluate model predictions on 200-permutation empirical frequencies.

    Parameters
    ----------
    model : nn.Module
        Trained model
    scaler : StandardScaler
        Feature scaler fitted on training data
    empirical_df : DataFrame
        200-permutation empirical frequencies
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    predictions : array
        Model predictions for all empirical degree pairs
    metrics : dict
        Validation metrics
    """
    X_empirical = empirical_df[['source_degree', 'target_degree']].values
    y_empirical = empirical_df['frequency'].values

    X_empirical_scaled = scaler.transform(X_empirical)
    X_empirical_t = torch.FloatTensor(X_empirical_scaled).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(X_empirical_t).cpu().numpy()

    correlation = pearsonr(y_empirical, predictions)[0]
    bias = np.mean(predictions - y_empirical)
    rmse = np.sqrt(np.mean((predictions - y_empirical)**2))

    metrics = {
        'pearson_r': correlation,
        'mean_bias': bias,
        'rmse': rmse,
        'n_samples': len(predictions)
    }

    return predictions, metrics


def evaluate_single_edge_type(edge_type: str,
                              data_dir: Path,
                              results_dir: Path,
                              feature_tiers: List[str] = ['minimal'],
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
        Results directory
    feature_tiers : list
        Which feature tiers to evaluate
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

    original_df = compute_original_graph_frequencies(edge_type, data_dir)

    empirical_file = results_dir / 'empirical_edge_frequencies' / f'edge_frequency_by_degree_{edge_type}.csv'
    empirical_df = pd.read_csv(empirical_file)

    print(f"\n200-permutation empirical data: {len(empirical_df)} degree pairs")

    coverage_stats = analyze_degree_pair_coverage(original_df, empirical_df)
    unseen_df = get_unseen_degree_pairs(original_df, empirical_df)

    train_df, test_df = train_test_split(original_df, test_size=0.2, random_state=42)

    print(f"\nTrain/test split:")
    print(f"  Train: {len(train_df)} degree pairs")
    print(f"  Test: {len(test_df)} degree pairs")

    X_train = train_df[['source_degree', 'target_degree']].values
    y_train = train_df['frequency'].values
    X_test = test_df[['source_degree', 'target_degree']].values
    y_test = test_df['frequency'].values

    edge_file = data_dir / 'edges' / f'{edge_type}.sparse.npz'
    edge_matrix = sp.load_npz(str(edge_file))
    fe = TheoryGuidedFeatureEngineer(edge_matrix, edge_type)

    u_train = train_df['source_degree'].values
    v_train = train_df['target_degree'].values
    u_test = test_df['source_degree'].values
    v_test = test_df['target_degree'].values

    tier_results = {}

    edge_results_dir = results_dir / 'original_graph_training' / edge_type
    edge_results_dir.mkdir(parents=True, exist_ok=True)

    original_df.to_csv(edge_results_dir / 'original_frequencies.csv', index=False)

    with open(edge_results_dir / 'degree_pair_coverage.txt', 'w') as f:
        f.write("Degree Pair Coverage Analysis\n")
        f.write("="*60 + "\n\n")
        for key, value in coverage_stats.items():
            f.write(f"{key}: {value}\n")

    print(f"\n{'-'*60}")
    print("BASELINE: Analytical Formula")
    print(f"{'-'*60}")

    analytical_train = fe.get_analytical_baseline(u_train, v_train)
    analytical_test = fe.get_analytical_baseline(u_test, v_test)

    u_empirical = empirical_df['source_degree'].values
    v_empirical = empirical_df['target_degree'].values
    analytical_empirical = fe.get_analytical_baseline(u_empirical, v_empirical)

    analytical_val_metrics = {
        'pearson_r': pearsonr(empirical_df['frequency'].values, analytical_empirical)[0],
        'mean_bias': np.mean(analytical_empirical - empirical_df['frequency'].values),
        'rmse': np.sqrt(np.mean((analytical_empirical - empirical_df['frequency'].values)**2)),
        'n_samples': len(analytical_empirical)
    }

    print(f"Test r (on original graph): {pearsonr(y_test, analytical_test)[0]:.4f}")
    print(f"Validation r (on 200-perm): {analytical_val_metrics['pearson_r']:.4f}")
    print(f"Bias: {analytical_val_metrics['mean_bias']:+.4f}")
    print(f"RMSE: {analytical_val_metrics['rmse']:.4f}")

    tier_results['analytical'] = {
        'test_r': pearsonr(y_test, analytical_test)[0],
        'validation_r': analytical_val_metrics['pearson_r'],
        'mean_bias': analytical_val_metrics['mean_bias'],
        'rmse': analytical_val_metrics['rmse'],
        'n_features': 0,
        'training_time': 0
    }

    analytical_metrics_dict, analytical_fig = analyze_residuals(
        empirical_df['frequency'].values, analytical_empirical,
        title=f"Analytical Formula - {edge_type}"
    )
    analytical_fig.savefig(edge_results_dir / 'analytical_residuals.png',
                          dpi=300, bbox_inches='tight')
    plt.close(analytical_fig)

    print(f"\n{'-'*60}")
    print("BASELINE: SimpleNN (2 features)")
    print(f"{'-'*60}")

    simplenn_results = train_simplenn_on_frequencies(
        X_train, y_train, X_test, y_test, device=device
    )

    print(f"Training time: {simplenn_results['training_time']:.2f}s")
    print(f"Train r: {simplenn_results['train_correlation']:.4f}")
    print(f"Test r (on original graph): {simplenn_results['test_correlation']:.4f}")

    simplenn_empirical_pred, simplenn_val_metrics = evaluate_on_empirical_frequencies(
        simplenn_results['model'], simplenn_results['scaler'], empirical_df, device
    )

    print(f"Validation r (on 200-perm): {simplenn_val_metrics['pearson_r']:.4f}")
    print(f"Bias: {simplenn_val_metrics['mean_bias']:+.4f}")
    print(f"RMSE: {simplenn_val_metrics['rmse']:.4f}")

    if len(unseen_df) > 0:
        X_unseen = unseen_df[['source_degree', 'target_degree']].values
        y_unseen = unseen_df['frequency'].values
        X_unseen_scaled = simplenn_results['scaler'].transform(X_unseen)
        X_unseen_t = torch.FloatTensor(X_unseen_scaled).to(device)

        simplenn_results['model'].eval()
        with torch.no_grad():
            unseen_pred = simplenn_results['model'](X_unseen_t).cpu().numpy()

        unseen_r = pearsonr(y_unseen, unseen_pred)[0]
        print(f"Interpolation r (unseen degree pairs): {unseen_r:.4f}")
    else:
        unseen_r = None

    tier_results['simplenn'] = {
        'test_r': simplenn_results['test_correlation'],
        'validation_r': simplenn_val_metrics['pearson_r'],
        'interpolation_r': unseen_r,
        'mean_bias': simplenn_val_metrics['mean_bias'],
        'rmse': simplenn_val_metrics['rmse'],
        'n_features': 2,
        'training_time': simplenn_results['training_time']
    }

    simplenn_metrics_dict, simplenn_fig = analyze_residuals(
        empirical_df['frequency'].values, simplenn_empirical_pred,
        title=f"SimpleNN (2 features) - {edge_type}"
    )
    simplenn_fig.savefig(edge_results_dir / 'simplenn_residuals.png',
                        dpi=300, bbox_inches='tight')
    plt.close(simplenn_fig)

    for tier in feature_tiers:
        print(f"\n{'-'*60}")
        print(f"TIER: {tier.upper()} ({ReducedFeatureSet.get_feature_count(tier)} features)")
        print(f"{'-'*60}")

        X_train_features = ReducedFeatureSet.compute_reduced_features(fe, u_train, v_train, tier=tier)
        X_test_features = ReducedFeatureSet.compute_reduced_features(fe, u_test, v_test, tier=tier)

        print(f"Feature matrix shape: {X_train_features.shape}")

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_features),
            columns=X_train_features.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_features),
            columns=X_test_features.columns
        )

        print("Training model...")
        start_time = time.time()

        training_results = train_theory_guided_model(
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            model_type='full',
            hidden_dims=(64, 32, 16),
            learning_rate=0.001,
            n_epochs=200,
            batch_size=256,
            patience=20,
            device=device
        )

        training_time = time.time() - start_time

        model = training_results['model']
        model.eval()

        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled.values).to(device)
            test_pred = model(X_test_tensor).cpu().numpy()

        test_corr = pearsonr(y_test, test_pred)[0]

        print(f"Training time: {training_time:.2f}s")
        print(f"Test r (on original graph): {test_corr:.4f}")

        u_empirical = empirical_df['source_degree'].values
        v_empirical = empirical_df['target_degree'].values
        X_empirical_features = ReducedFeatureSet.compute_reduced_features(
            fe, u_empirical, v_empirical, tier=tier
        )
        X_empirical_scaled = pd.DataFrame(
            scaler.transform(X_empirical_features),
            columns=X_empirical_features.columns
        )

        with torch.no_grad():
            X_empirical_tensor = torch.FloatTensor(X_empirical_scaled.values).to(device)
            empirical_pred = model(X_empirical_tensor).cpu().numpy()

        val_r = pearsonr(empirical_df['frequency'].values, empirical_pred)[0]
        val_bias = np.mean(empirical_pred - empirical_df['frequency'].values)
        val_rmse = np.sqrt(np.mean((empirical_pred - empirical_df['frequency'].values)**2))

        print(f"Validation r (on 200-perm): {val_r:.4f}")
        print(f"Bias: {val_bias:+.4f}")
        print(f"RMSE: {val_rmse:.4f}")

        if len(unseen_df) > 0:
            u_unseen = unseen_df['source_degree'].values
            v_unseen = unseen_df['target_degree'].values
            y_unseen = unseen_df['frequency'].values

            X_unseen_features = ReducedFeatureSet.compute_reduced_features(
                fe, u_unseen, v_unseen, tier=tier
            )
            X_unseen_scaled = pd.DataFrame(
                scaler.transform(X_unseen_features),
                columns=X_unseen_features.columns
            )

            with torch.no_grad():
                X_unseen_tensor = torch.FloatTensor(X_unseen_scaled.values).to(device)
                unseen_pred = model(X_unseen_tensor).cpu().numpy()

            unseen_r = pearsonr(y_unseen, unseen_pred)[0]
            print(f"Interpolation r (unseen degree pairs): {unseen_r:.4f}")
        else:
            unseen_r = None

        tier_results[tier] = {
            'test_r': test_corr,
            'validation_r': val_r,
            'interpolation_r': unseen_r,
            'mean_bias': val_bias,
            'rmse': val_rmse,
            'n_features': X_train_features.shape[1],
            'training_time': training_time
        }

        tier_metrics_dict, tier_fig = analyze_residuals(
            empirical_df['frequency'].values, empirical_pred,
            title=f"{tier.capitalize()} Features ({ReducedFeatureSet.get_feature_count(tier)}) - {edge_type}"
        )
        tier_fig.savefig(edge_results_dir / f'{tier}_residuals.png',
                        dpi=300, bbox_inches='tight')
        plt.close(tier_fig)

    comparison_df = pd.DataFrame(tier_results).T
    comparison_df.to_csv(edge_results_dir / 'comparison_metrics.csv')

    print(f"\nResults saved to: {edge_results_dir}")

    return {
        'edge_type': edge_type,
        'tier_results': tier_results,
        'coverage_stats': coverage_stats
    }


def main():
    """
    Main evaluation script.
    """
    repo_dir = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    edge_types = ['CbG']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    all_results = []

    for edge_type in edge_types:
        try:
            result = evaluate_single_edge_type(
                edge_type=edge_type,
                data_dir=data_dir,
                results_dir=results_dir,
                feature_tiers=['minimal', 'standard', 'extended'],
                device=device
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

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
