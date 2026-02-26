"""
Feature Importance Analysis for Theory-Guided Neural Networks

Implements multiple methods to determine which features are most important:
1. Permutation importance (model-agnostic)
2. Gradient-based importance (NN-specific)
3. Weight magnitude analysis (first layer)
4. Ablation analysis (remove feature groups)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def permutation_importance(model: nn.Module,
                          X: pd.DataFrame,
                          y: np.ndarray,
                          n_repeats: int = 10,
                          device: str = 'cpu') -> pd.DataFrame:
    """
    Compute permutation importance for each feature.

    Measures how much prediction accuracy drops when a feature is randomly
    shuffled (breaking its relationship with target).

    Parameters
    ----------
    model : nn.Module
        Trained model
    X : DataFrame
        Feature matrix
    y : array
        Target values
    n_repeats : int
        Number of permutation repeats
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    importance_df : DataFrame
        Feature importance scores with mean and std
    """
    model.eval()

    # Baseline performance
    X_tensor = torch.FloatTensor(X.values).to(device)
    y_tensor = torch.FloatTensor(y).to(device)

    with torch.no_grad():
        baseline_pred = model(X_tensor).cpu().numpy()

    baseline_mse = mean_squared_error(y, baseline_pred)
    baseline_corr = pearsonr(y.flatten(), baseline_pred.flatten())[0]

    print(f"Baseline - MSE: {baseline_mse:.6f}, Correlation: {baseline_corr:.4f}")

    # Permutation importance for each feature
    importances = []

    for feature_idx, feature_name in enumerate(X.columns):
        feature_importance = []

        for _ in range(n_repeats):
            # Create permuted data
            X_permuted = X.copy()
            X_permuted.iloc[:, feature_idx] = np.random.permutation(X_permuted.iloc[:, feature_idx])

            X_perm_tensor = torch.FloatTensor(X_permuted.values).to(device)

            with torch.no_grad():
                perm_pred = model(X_perm_tensor).cpu().numpy()

            perm_mse = mean_squared_error(y, perm_pred)
            perm_corr = pearsonr(y.flatten(), perm_pred.flatten())[0]

            # Importance = drop in performance
            importance_mse = perm_mse - baseline_mse
            importance_corr = baseline_corr - perm_corr

            feature_importance.append({
                'mse_increase': importance_mse,
                'corr_decrease': importance_corr
            })

        # Aggregate across repeats
        importances.append({
            'feature': feature_name,
            'importance_mse_mean': np.mean([x['mse_increase'] for x in feature_importance]),
            'importance_mse_std': np.std([x['mse_increase'] for x in feature_importance]),
            'importance_corr_mean': np.mean([x['corr_decrease'] for x in feature_importance]),
            'importance_corr_std': np.std([x['corr_decrease'] for x in feature_importance]),
        })

        print(f"  {feature_name}: corr_drop={importances[-1]['importance_corr_mean']:.4f} ± {importances[-1]['importance_corr_std']:.4f}")

    importance_df = pd.DataFrame(importances)
    importance_df = importance_df.sort_values('importance_corr_mean', ascending=False)

    return importance_df


def gradient_based_importance(model: nn.Module,
                              X: pd.DataFrame,
                              y: np.ndarray,
                              device: str = 'cpu') -> pd.DataFrame:
    """
    Compute gradient-based feature importance.

    Measures average gradient magnitude with respect to each input feature.
    High gradient = output is sensitive to changes in that feature.

    Parameters
    ----------
    model : nn.Module
        Trained model
    X : DataFrame
        Feature matrix
    y : array
        Target values
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    importance_df : DataFrame
        Gradient-based importance scores
    """
    model.eval()

    X_tensor = torch.FloatTensor(X.values).to(device)
    X_tensor.requires_grad = True

    # Forward pass
    output = model(X_tensor)

    # Compute gradients for each sample
    gradients = []

    for i in range(len(output)):
        # Gradient of output[i] with respect to input features
        if X_tensor.grad is not None:
            X_tensor.grad.zero_()

        output[i].backward(retain_graph=True)
        grad = X_tensor.grad[i].cpu().numpy()
        gradients.append(np.abs(grad))

    # Average gradient magnitude per feature
    avg_gradient = np.mean(gradients, axis=0)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'gradient_importance': avg_gradient
    })
    importance_df = importance_df.sort_values('gradient_importance', ascending=False)

    return importance_df


def weight_magnitude_importance(model: nn.Module,
                                feature_names: List[str]) -> pd.DataFrame:
    """
    Compute importance based on first layer weight magnitudes.

    For linear models or first layer of NN, weight magnitude indicates
    how much each feature contributes to predictions.

    Parameters
    ----------
    model : nn.Module
        Trained model
    feature_names : list
        List of feature names

    Returns
    -------
    importance_df : DataFrame
        Weight-based importance scores
    """
    # Get first layer weights
    first_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            first_layer = module
            break

    if first_layer is None:
        raise ValueError("No linear layer found in model")

    # Get weight matrix [out_features, in_features]
    weights = first_layer.weight.data.cpu().numpy()

    # Average absolute weight per input feature (across output neurons)
    avg_weight_magnitude = np.mean(np.abs(weights), axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'weight_importance': avg_weight_magnitude
    })
    importance_df = importance_df.sort_values('weight_importance', ascending=False)

    return importance_df


def ablation_importance(model: nn.Module,
                       X: pd.DataFrame,
                       y: np.ndarray,
                       feature_groups: Dict[str, List[str]],
                       device: str = 'cpu') -> pd.DataFrame:
    """
    Compute importance by feature group ablation.

    Tests performance when entire feature groups (levels) are removed.

    Parameters
    ----------
    model : nn.Module
        Trained model
    X : DataFrame
        Feature matrix
    y : array
        Target values
    feature_groups : dict
        Dictionary mapping group names to lists of feature names
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    importance_df : DataFrame
        Group-level importance scores
    """
    model.eval()

    # Baseline performance
    X_tensor = torch.FloatTensor(X.values).to(device)

    with torch.no_grad():
        baseline_pred = model(X_tensor).cpu().numpy()

    baseline_corr = pearsonr(y.flatten(), baseline_pred.flatten())[0]

    print(f"Baseline correlation: {baseline_corr:.4f}\n")

    # Test ablation of each group
    importances = []

    for group_name, feature_list in feature_groups.items():
        # Set features in this group to zero (ablation)
        X_ablated = X.copy()
        for feature in feature_list:
            if feature in X_ablated.columns:
                X_ablated[feature] = 0

        X_abl_tensor = torch.FloatTensor(X_ablated.values).to(device)

        with torch.no_grad():
            abl_pred = model(X_abl_tensor).cpu().numpy()

        abl_corr = pearsonr(y.flatten(), abl_pred.flatten())[0]
        corr_drop = baseline_corr - abl_corr

        importances.append({
            'group': group_name,
            'baseline_corr': baseline_corr,
            'ablated_corr': abl_corr,
            'correlation_drop': corr_drop,
            'n_features': len([f for f in feature_list if f in X.columns])
        })

        print(f"{group_name}: {abl_corr:.4f} (drop: {corr_drop:.4f})")

    importance_df = pd.DataFrame(importances)
    importance_df = importance_df.sort_values('correlation_drop', ascending=False)

    return importance_df


def visualize_feature_importance(importance_df: pd.DataFrame,
                                 top_n: int = 20,
                                 method: str = 'permutation',
                                 save_path: str = None):
    """
    Visualize feature importance.

    Parameters
    ----------
    importance_df : DataFrame
        Feature importance results
    top_n : int
        Number of top features to show
    method : str
        Method name for title
    save_path : str, optional
        Path to save figure
    """
    # Select appropriate importance column
    if 'importance_corr_mean' in importance_df.columns:
        importance_col = 'importance_corr_mean'
        error_col = 'importance_corr_std'
        xlabel = 'Correlation Drop (Importance)'
    elif 'gradient_importance' in importance_df.columns:
        importance_col = 'gradient_importance'
        error_col = None
        xlabel = 'Gradient Magnitude (Importance)'
    elif 'weight_importance' in importance_df.columns:
        importance_col = 'weight_importance'
        error_col = None
        xlabel = 'Weight Magnitude (Importance)'
    else:
        raise ValueError("Unknown importance type")

    # Get top N features
    top_features = importance_df.head(top_n)

    # Create barplot
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(top_features))
    importance_values = top_features[importance_col].values

    if error_col is not None:
        errors = top_features[error_col].values
        ax.barh(y_pos, importance_values, xerr=errors, alpha=0.7,
                color='steelblue', ecolor='gray', capsize=3)
    else:
        ax.barh(y_pos, importance_values, alpha=0.7, color='steelblue')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(f'Top {top_n} Features by {method.title()} Importance',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def comprehensive_feature_analysis(model: nn.Module,
                                   X_train: pd.DataFrame,
                                   y_train: np.ndarray,
                                   X_test: pd.DataFrame,
                                   y_test: np.ndarray,
                                   feature_groups: Dict[str, List[str]] = None,
                                   device: str = 'cpu',
                                   save_dir: str = None) -> Dict:
    """
    Run all feature importance analyses.

    Parameters
    ----------
    model : nn.Module
        Trained model
    X_train : DataFrame
        Training features
    y_train : array
        Training targets
    X_test : DataFrame
        Test features
    y_test : array
        Test targets
    feature_groups : dict, optional
        Feature group definitions for ablation
    device : str
        'cpu' or 'cuda'
    save_dir : str, optional
        Directory to save results

    Returns
    -------
    results : dict
        All importance analysis results
    """
    print("="*80)
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    results = {}

    # 1. Permutation importance
    print("\n1. Permutation Importance (Test Set)")
    print("-" * 60)
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, device=device)
    results['permutation'] = perm_importance

    if save_dir:
        perm_importance.to_csv(f"{save_dir}/permutation_importance.csv", index=False)
        fig = visualize_feature_importance(perm_importance, top_n=20, method='permutation',
                                          save_path=f"{save_dir}/permutation_importance.png")
        plt.close(fig)

    # 2. Gradient-based importance
    print("\n2. Gradient-Based Importance")
    print("-" * 60)
    grad_importance = gradient_based_importance(model, X_test, y_test, device=device)
    results['gradient'] = grad_importance

    print("\nTop 10 features by gradient magnitude:")
    print(grad_importance.head(10).to_string(index=False))

    if save_dir:
        grad_importance.to_csv(f"{save_dir}/gradient_importance.csv", index=False)
        fig = visualize_feature_importance(grad_importance, top_n=20, method='gradient',
                                          save_path=f"{save_dir}/gradient_importance.png")
        plt.close(fig)

    # 3. Weight magnitude importance
    print("\n3. Weight Magnitude Importance (First Layer)")
    print("-" * 60)
    weight_importance = weight_magnitude_importance(model, list(X_train.columns))
    results['weight'] = weight_importance

    print("\nTop 10 features by weight magnitude:")
    print(weight_importance.head(10).to_string(index=False))

    if save_dir:
        weight_importance.to_csv(f"{save_dir}/weight_importance.csv", index=False)
        fig = visualize_feature_importance(weight_importance, top_n=20, method='weight',
                                          save_path=f"{save_dir}/weight_importance.png")
        plt.close(fig)

    # 4. Ablation analysis (if groups provided)
    if feature_groups:
        print("\n4. Feature Group Ablation Analysis")
        print("-" * 60)
        ablation_results = ablation_importance(model, X_test, y_test, feature_groups, device=device)
        results['ablation'] = ablation_results

        if save_dir:
            ablation_results.to_csv(f"{save_dir}/ablation_importance.csv", index=False)

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: Top 5 Features by Each Method")
    print("="*80)

    comparison_data = []
    for rank in range(5):
        row = {'Rank': rank + 1}
        if rank < len(perm_importance):
            row['Permutation'] = perm_importance.iloc[rank]['feature']
        if rank < len(grad_importance):
            row['Gradient'] = grad_importance.iloc[rank]['feature']
        if rank < len(weight_importance):
            row['Weight'] = weight_importance.iloc[rank]['feature']
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    if save_dir:
        comparison_df.to_csv(f"{save_dir}/top5_comparison.csv", index=False)

    return results
