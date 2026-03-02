"""
Evaluate Feature Reduction Across Edge Types

Tests whether reduced feature sets (17, 20, 25 features) maintain performance
compared to full feature set (49 features) across diverse edge types.

Validates hypothesis:
- Easy edge types: minimal features sufficient
- Medium edge types: standard features needed
- Hard edge types: extended features required
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import time
from typing import Dict, List
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from theory_guided_features import TheoryGuidedFeatureEngineer
from theory_guided_model import train_theory_guided_model
from reduced_feature_sets import (
    ReducedFeatureSet,
    classify_edge_type_difficulty,
    recommend_feature_tier
)
from model_comparison import SimpleNN, prepare_edge_features_and_labels
from evaluate_theory_guided_models import analyze_residuals


def train_simplenn_baseline(u_train, v_train, y_train,
                           u_test, v_test, y_test,
                           device='cpu'):
    """
    Train SimpleNN baseline using only (u, v) features.

    Parameters
    ----------
    u_train, v_train : array
        Training degrees
    y_train : array
        Training targets
    u_test, v_test : array
        Test degrees
    y_test : array
        Test targets
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    results : dict
        Model, predictions, metrics, training_time
    """
    # Prepare 2-feature input
    X_train = np.column_stack([u_train, v_train])
    X_test = np.column_stack([u_test, v_test])

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)

    # Initialize model
    model = SimpleNN(input_dim=2, hidden_dims=(128, 64, 32), dropout_rate=0.3).to(device)

    # Training configuration
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    # Train with early stopping
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

        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    training_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Generate predictions
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).cpu().numpy()
        test_pred = model(X_test_t).cpu().numpy()

    # Calculate metrics
    test_corr = pearsonr(y_test, test_pred)[0]
    train_corr = pearsonr(y_train, train_pred)[0]

    return {
        'model': model,
        'test_predictions': test_pred,
        'train_predictions': train_pred,
        'test_correlation': test_corr,
        'train_correlation': train_corr,
        'training_time': training_time,
        'bias': np.mean(test_pred - y_test),
        'rmse': np.sqrt(np.mean((test_pred - y_test)**2))
    }


def load_single_permutation_data(edge_type: str,
                                 data_dir: Path,
                                 permutation_id: str = '001',
                                 sample_ratio: float = 0.01):
    """
    Load binary training data from single permutation.

    Parameters
    ----------
    edge_type : str
        Edge type identifier (e.g., 'CbG')
    data_dir : Path
        Data directory
    permutation_id : str
        Permutation ID (default '001')
    sample_ratio : float
        Ratio of negative edges to sample

    Returns
    -------
    X : array (N, 2)
        Degree pairs [source_degree, target_degree]
    y : array (N,)
        Binary labels (0=no edge, 1=edge)
    edge_matrix : sparse matrix
        The edge matrix (for feature engineering)
    """
    edge_file = data_dir / 'permutations' / f'{permutation_id}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    print(f"Loading single permutation from: {edge_file}")

    edge_matrix = sp.load_npz(str(edge_file))

    X, y = prepare_edge_features_and_labels(
        str(edge_file),
        sample_ratio=sample_ratio,
        adaptive_sampling=True,
        enhanced_features=False
    )

    print(f"Loaded {len(y)} samples from single permutation ({int(y.sum())} positive, {len(y) - int(y.sum())} negative)")

    return X, y, edge_matrix


def validate_against_empirical_frequencies(predictions: np.ndarray,
                                          X_data: np.ndarray,
                                          empirical_df: pd.DataFrame) -> Dict:
    """
    Group predictions by degree pair and compare to empirical frequencies.

    This function:
    1. Groups individual predictions by (source_degree, target_degree)
    2. Averages predictions within each degree pair
    3. Compares averaged predictions to empirical frequencies (gold standard)

    Parameters
    ----------
    predictions : array (N,)
        Individual edge probability predictions
    X_data : array (N, 2)
        Degree pairs [source_degree, target_degree]
    empirical_df : DataFrame
        Empirical frequencies with columns: source_degree, target_degree, frequency

    Returns
    -------
    validation_metrics : dict
        Correlation, bias, RMSE comparing grouped predictions to empirical frequencies
    grouped_pred : array
        Averaged predictions for each degree pair
    grouped_emp : array
        Empirical frequencies for each degree pair
    """
    empirical_lookup = {}
    for _, row in empirical_df.iterrows():
        key = (int(row['source_degree']), int(row['target_degree']))
        empirical_lookup[key] = row['frequency']

    from collections import defaultdict
    prediction_groups = defaultdict(list)

    for i in range(len(X_data)):
        u = int(X_data[i, 0])
        v = int(X_data[i, 1])
        key = (u, v)
        prediction_groups[key].append(predictions[i])

    matched_pred = []
    matched_emp = []

    for key, pred_list in prediction_groups.items():
        if key in empirical_lookup:
            avg_pred = np.mean(pred_list)
            matched_pred.append(avg_pred)
            matched_emp.append(empirical_lookup[key])

    matched_pred = np.array(matched_pred)
    matched_emp = np.array(matched_emp)

    if len(matched_pred) == 0:
        print("WARNING: No matches between predictions and empirical frequencies")
        return {
            'pearson_r': 0.0,
            'mean_bias': 0.0,
            'rmse': 0.0,
            'n_matched': 0
        }, matched_pred, matched_emp

    correlation = pearsonr(matched_emp, matched_pred)[0]
    bias = np.mean(matched_pred - matched_emp)
    rmse = np.sqrt(np.mean((matched_pred - matched_emp)**2))

    return {
        'pearson_r': correlation,
        'mean_bias': bias,
        'rmse': rmse,
        'n_matched': len(matched_pred)
    }, matched_pred, matched_emp


def evaluate_single_edge_type(edge_type: str,
                               data_dir: Path,
                               results_dir: Path,
                               feature_tiers: List[str] = ['minimal', 'standard', 'extended'],
                               device: str = 'cpu') -> Dict:
    """
    Evaluate all feature tiers on one edge type.

    Parameters
    ----------
    edge_type : str
        Edge type identifier (e.g., 'CbG')
    data_dir : Path
        Data directory
    results_dir : Path
        Results directory
    feature_tiers : list
        Which tiers to test
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    results : dict
        Comprehensive evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating edge type: {edge_type}")
    print(f"{'='*80}\n")

    # Load single permutation data (binary labels for training)
    X_full, y_binary, edge_matrix = load_single_permutation_data(
        edge_type=edge_type,
        data_dir=data_dir,
        permutation_id='001',
        sample_ratio=0.01
    )

    # Load empirical frequencies (gold standard for validation)
    empirical_file = results_dir / 'empirical_edge_frequencies' / f'edge_frequency_by_degree_{edge_type}.csv'
    empirical_df = pd.read_csv(empirical_file)

    print(f"Empirical frequencies: {len(empirical_df)} degree combinations")
    print(f"Edge matrix: {edge_matrix.shape}, {edge_matrix.nnz} edges")

    # Initialize feature engineer
    fe = TheoryGuidedFeatureEngineer(edge_matrix, edge_type)

    # Extract degrees from training data
    u_full = X_full[:, 0]
    v_full = X_full[:, 1]

    # Compute analytical baseline for empirical degree pairs (for difficulty classification)
    u_emp = empirical_df['source_degree'].values
    v_emp = empirical_df['target_degree'].values
    y_emp = empirical_df['frequency'].values
    analytical_emp = fe.get_analytical_baseline(u_emp, v_emp)
    analytical_corr = pearsonr(y_emp, analytical_emp)[0]

    print(f"Analytical correlation (on empirical frequencies): {analytical_corr:.4f}")

    # Classify difficulty
    difficulty = classify_edge_type_difficulty(edge_matrix, analytical_corr)
    recommended_tier, reasoning = recommend_feature_tier(edge_type, edge_matrix, analytical_corr)

    print(f"\nEdge type difficulty: {difficulty}")
    print(f"Recommended tier: {recommended_tier}")
    print(reasoning)

    # Train/test split on binary data
    indices = np.arange(len(X_full))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    u_train, u_test = u_full[train_idx], u_full[test_idx]
    v_train, v_test = v_full[train_idx], v_full[test_idx]
    y_train, y_test = y_binary[train_idx], y_binary[test_idx]
    X_train, X_test = X_full[train_idx], X_full[test_idx]

    # Store results for each tier
    tier_results = {}

    # Baseline: Analytical formula only
    print(f"\n{'-'*60}")
    print("BASELINE: Analytical Formula")
    print(f"{'-'*60}")

    # Compute analytical predictions for test set
    analytical_test_pred = fe.get_analytical_baseline(u_test, v_test)

    # Validate against empirical frequencies
    analytical_val_metrics, analytical_grouped_pred, analytical_grouped_emp = validate_against_empirical_frequencies(
        analytical_test_pred, X_test, empirical_df
    )

    analytical_metrics = {
        'pearson_r': analytical_val_metrics['pearson_r'],
        'mean_bias': analytical_val_metrics['mean_bias'],
        'rmse': analytical_val_metrics['rmse'],
        'n_features': 0,
        'training_time': 0,
        'train_test_gap': 0,
        'n_matched': analytical_val_metrics['n_matched']
    }

    print(f"Test r (vs empirical): {analytical_metrics['pearson_r']:.4f}")
    print(f"Bias: {analytical_metrics['mean_bias']:+.4f}")
    print(f"RMSE: {analytical_metrics['rmse']:.4f}")
    print(f"Matched degree pairs: {analytical_metrics['n_matched']}")

    tier_results['analytical'] = analytical_metrics

    # Create edge-specific results directory
    edge_results_dir = results_dir / 'feature_reduction_evaluation' / edge_type
    edge_results_dir.mkdir(parents=True, exist_ok=True)

    # Generate analytical residual plot (grouped predictions vs empirical)
    analytical_metrics_dict, analytical_fig = analyze_residuals(
        analytical_grouped_emp, analytical_grouped_pred,
        title=f"Analytical Formula - {edge_type}"
    )
    analytical_fig.savefig(edge_results_dir / 'analytical_residuals.png',
                          dpi=300, bbox_inches='tight')
    plt.close(analytical_fig)

    # Baseline: SimpleNN (2 features)
    print(f"\n{'-'*60}")
    print("BASELINE: SimpleNN (2 features: u, v)")
    print(f"{'-'*60}")

    # Train on binary labels
    simplenn_results = train_simplenn_baseline(
        u_train, v_train, y_train,
        u_test, v_test, y_test,
        device=device
    )

    print(f"Training time: {simplenn_results['training_time']:.2f}s")
    print(f"Train r (on binary labels): {simplenn_results['train_correlation']:.4f}")
    print(f"Test r (on binary labels): {simplenn_results['test_correlation']:.4f}")

    # Validate against empirical frequencies
    simplenn_val_metrics, simplenn_grouped_pred, simplenn_grouped_emp = validate_against_empirical_frequencies(
        simplenn_results['test_predictions'], X_test, empirical_df
    )

    print(f"Test r (vs empirical): {simplenn_val_metrics['pearson_r']:.4f}")
    print(f"Bias (vs empirical): {simplenn_val_metrics['mean_bias']:+.4f}")
    print(f"RMSE (vs empirical): {simplenn_val_metrics['rmse']:.4f}")
    print(f"Matched degree pairs: {simplenn_val_metrics['n_matched']}")

    # Generate SimpleNN residual plot (grouped predictions vs empirical)
    simplenn_metrics_dict, simplenn_fig = analyze_residuals(
        simplenn_grouped_emp, simplenn_grouped_pred,
        title=f"SimpleNN (2 features) - {edge_type}"
    )
    simplenn_fig.savefig(edge_results_dir / 'simplenn_residuals.png',
                        dpi=300, bbox_inches='tight')
    plt.close(simplenn_fig)

    # Store SimpleNN results (using empirical validation metrics)
    tier_results['simplenn'] = {
        'pearson_r': simplenn_val_metrics['pearson_r'],
        'mean_bias': simplenn_val_metrics['mean_bias'],
        'rmse': simplenn_val_metrics['rmse'],
        'n_features': 2,
        'training_time': simplenn_results['training_time'],
        'train_test_gap': 0,
        'train_corr': simplenn_results['train_correlation'],
        'n_matched': simplenn_val_metrics['n_matched']
    }

    # Compute improvements over analytical
    tier_results['simplenn']['r_improvement'] = (
        tier_results['simplenn']['pearson_r'] - analytical_metrics['pearson_r']
    )
    tier_results['simplenn']['bias_reduction'] = (
        abs(analytical_metrics['mean_bias']) - abs(tier_results['simplenn']['mean_bias'])
    )
    tier_results['simplenn']['rmse_reduction'] = (
        analytical_metrics['rmse'] - tier_results['simplenn']['rmse']
    )

    # Test each feature tier
    for tier in feature_tiers:
        print(f"\n{'-'*60}")
        print(f"TIER: {tier.upper()} ({ReducedFeatureSet.get_feature_count(tier)} features)")
        print(f"{'-'*60}")

        # Compute features for this tier
        start_time = time.time()

        X_train_features = ReducedFeatureSet.compute_reduced_features(fe, u_train, v_train, tier=tier)
        X_test_features = ReducedFeatureSet.compute_reduced_features(fe, u_test, v_test, tier=tier)

        print(f"Feature computation time: {time.time() - start_time:.2f}s")
        print(f"Feature matrix shape: {X_train_features.shape}")

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_features),
            columns=X_train_features.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_features),
            columns=X_test_features.columns
        )

        # Train model on binary labels
        print("Training model on binary labels...")
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

        # Generate predictions
        model = training_results['model']
        model.eval()

        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train_scaled.values).to(device)
            X_test_tensor = torch.FloatTensor(X_test_scaled.values).to(device)

            train_pred = model(X_train_tensor).cpu().numpy()
            test_pred = model(X_test_tensor).cpu().numpy()

        # Calculate metrics on binary labels
        train_corr_binary = pearsonr(y_train, train_pred)[0]
        test_corr_binary = pearsonr(y_test, test_pred)[0]

        print(f"Training time: {training_time:.2f}s")
        print(f"Train r (on binary labels): {train_corr_binary:.4f}")
        print(f"Test r (on binary labels): {test_corr_binary:.4f}")

        # Validate against empirical frequencies
        tier_val_metrics, tier_grouped_pred, tier_grouped_emp = validate_against_empirical_frequencies(
            test_pred, X_test, empirical_df
        )

        print(f"Test r (vs empirical): {tier_val_metrics['pearson_r']:.4f}")
        print(f"Bias (vs empirical): {tier_val_metrics['mean_bias']:+.4f}")
        print(f"RMSE (vs empirical): {tier_val_metrics['rmse']:.4f}")
        print(f"Matched degree pairs: {tier_val_metrics['n_matched']}")

        tier_metrics = {
            'pearson_r': tier_val_metrics['pearson_r'],
            'mean_bias': tier_val_metrics['mean_bias'],
            'rmse': tier_val_metrics['rmse'],
            'n_features': X_train_features.shape[1],
            'training_time': training_time,
            'train_test_gap': 0,
            'train_corr': train_corr_binary,
            'n_matched': tier_val_metrics['n_matched']
        }

        # Generate residual plot for minimal tier (grouped predictions vs empirical)
        if tier == 'minimal':
            tier_metrics_dict, tier_fig = analyze_residuals(
                tier_grouped_emp, tier_grouped_pred,
                title=f"Minimal Features (13) - {edge_type}"
            )
            tier_fig.savefig(edge_results_dir / 'minimal_residuals.png',
                            dpi=300, bbox_inches='tight')
            plt.close(tier_fig)

        tier_results[tier] = tier_metrics

    # Compute improvements over analytical
    for tier in feature_tiers:
        tier_results[tier]['r_improvement'] = (
            tier_results[tier]['pearson_r'] - analytical_metrics['pearson_r']
        )
        tier_results[tier]['bias_reduction'] = (
            abs(analytical_metrics['mean_bias']) - abs(tier_results[tier]['mean_bias'])
        )
        tier_results[tier]['rmse_reduction'] = (
            analytical_metrics['rmse'] - tier_results[tier]['rmse']
        )

    return {
        'edge_type': edge_type,
        'difficulty': difficulty,
        'recommended_tier': recommended_tier,
        'analytical_corr': analytical_corr,
        'tier_results': tier_results
    }


def compare_across_edge_types(all_results: List[Dict],
                              save_path: Path = None) -> pd.DataFrame:
    """
    Generate comparison table across edge types.

    Parameters
    ----------
    all_results : list
        Results from evaluate_single_edge_type for multiple edge types
    save_path : Path, optional
        Path to save comparison table

    Returns
    -------
    comparison_df : DataFrame
        Comparison table
    """
    comparison_data = []

    for result in all_results:
        edge_type = result['edge_type']
        difficulty = result['difficulty']
        recommended = result['recommended_tier']

        # Get metrics for each tier
        for tier in ['analytical', 'simplenn', 'minimal', 'standard', 'extended']:
            if tier in result['tier_results']:
                metrics = result['tier_results'][tier]

                row = {
                    'edge_type': edge_type,
                    'difficulty': difficulty,
                    'recommended_tier': recommended,
                    'tier': tier,
                    'n_features': metrics['n_features'],
                    'pearson_r': metrics['pearson_r'],
                    'mean_bias': metrics['mean_bias'],
                    'rmse': metrics['rmse'],
                    'training_time': metrics['training_time'],
                    'train_test_gap': metrics.get('train_test_gap', 0),
                }

                # Add improvements (if not analytical)
                if tier != 'analytical':
                    row['r_improvement'] = metrics['r_improvement']
                    row['bias_reduction'] = metrics['bias_reduction']
                    row['rmse_reduction'] = metrics['rmse_reduction']

                comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"\nComparison table saved to: {save_path}")

    return comparison_df


def print_summary(comparison_df: pd.DataFrame):
    """
    Print summary of feature reduction evaluation.

    Parameters
    ----------
    comparison_df : DataFrame
        Comparison table from compare_across_edge_types
    """
    print("\n" + "="*80)
    print("FEATURE REDUCTION EVALUATION SUMMARY")
    print("="*80)

    # Summary by difficulty
    print("\n1. Performance by Edge Type Difficulty:")
    print("-"*60)

    for difficulty in ['easy', 'medium', 'hard']:
        subset = comparison_df[comparison_df['difficulty'] == difficulty]
        if len(subset) == 0:
            continue

        print(f"\n{difficulty.upper()} edge types:")

        # Best tier for this difficulty
        edge_types = subset['edge_type'].unique()
        print(f"  Edge types: {', '.join(edge_types)}")

        for tier in ['minimal', 'standard', 'extended']:
            tier_subset = subset[subset['tier'] == tier]
            if len(tier_subset) == 0:
                continue

            avg_r = tier_subset['pearson_r'].mean()
            avg_improvement = tier_subset['r_improvement'].mean()
            avg_time = tier_subset['training_time'].mean()

            print(f"  {tier.capitalize()}: r={avg_r:.4f} (Δr={avg_improvement:+.4f}), "
                  f"time={avg_time:.1f}s")

    # Recommended tier validation
    print("\n2. Recommended Tier Validation:")
    print("-"*60)

    for idx, row in comparison_df[comparison_df['tier'] == comparison_df['recommended_tier']].iterrows():
        edge_type = row['edge_type']
        recommended = row['recommended_tier']
        r = row['pearson_r']
        improvement = row['r_improvement']

        print(f"{edge_type}: {recommended} (r={r:.4f}, Δr={improvement:+.4f})")

    # Feature reduction trade-offs
    print("\n3. Feature Reduction Trade-offs:")
    print("-"*60)

    ml_tiers = comparison_df[comparison_df['tier'] != 'analytical']

    minimal = ml_tiers[ml_tiers['tier'] == 'minimal']
    extended = ml_tiers[ml_tiers['tier'] == 'extended']

    print(f"Minimal (17 features):")
    print(f"  Avg r: {minimal['pearson_r'].mean():.4f}")
    print(f"  Avg training time: {minimal['training_time'].mean():.1f}s")
    print(f"  Avg overfitting gap: {minimal['train_test_gap'].mean():+.4f}")

    print(f"\nExtended (25 features):")
    print(f"  Avg r: {extended['pearson_r'].mean():.4f}")
    print(f"  Avg training time: {extended['training_time'].mean():.1f}s")
    print(f"  Avg overfitting gap: {extended['train_test_gap'].mean():+.4f}")

    print(f"\nPerformance difference: {extended['pearson_r'].mean() - minimal['pearson_r'].mean():+.4f}")
    print(f"Time difference: {extended['training_time'].mean() - minimal['training_time'].mean():+.1f}s")

    # Best universal tier
    print("\n4. Best Universal Feature Tier:")
    print("-"*60)

    for tier in ['minimal', 'standard', 'extended']:
        tier_subset = ml_tiers[ml_tiers['tier'] == tier]
        if len(tier_subset) == 0:
            continue

        min_r = tier_subset['pearson_r'].min()
        avg_r = tier_subset['pearson_r'].mean()
        max_r = tier_subset['pearson_r'].max()

        print(f"{tier.capitalize()}: min={min_r:.4f}, avg={avg_r:.4f}, max={max_r:.4f}")

    print("\n" + "="*80)


def main():
    """
    Main evaluation script.
    """
    # Paths
    repo_dir = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    # Edge types: diverse set covering easy, medium, hard
    edge_types = [
        'CbG',   # Compound-binds-Gene (easy: small degrees, high analytical r)
        'CtD',   # Compound-treats-Disease (medium)
        'GpPW',  # Gene-participates-Pathway (medium)
        'AeG',   # Anatomy-expresses-Gene (hard: extreme degrees)
        'DdG'    # Disease-downregulates-Gene (hard: sparse, variable)
    ]

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Evaluate each edge type
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

    if not all_results:
        print("\nNo edge types successfully evaluated!")
        return

    # Generate comparison table
    save_dir = results_dir / 'feature_reduction_evaluation'
    save_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = compare_across_edge_types(
        all_results,
        save_path=save_dir / 'feature_tier_comparison.csv'
    )

    # Print summary
    print_summary(comparison_df)

    # Save detailed results
    import json
    with open(save_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()
