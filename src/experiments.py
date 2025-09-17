"""
Experiment Utilities for Edge Prediction Analysis

This module provides functions for running experiments, calculating stability metrics,
and analyzing model performance across different sample sizes and configurations.

MIGRATED FROM: notebooks/3_learn_null_edge.ipynb  
This code was previously defined inline in the notebook and has been moved here
for better modularity, reusability, and maintainability.

Functions:
    - calculate_prediction_stability: Calculate coefficient of variation across CV folds
    - run_experiment: Run complete experiments with different sample sizes
    - analyze_experiment_results: Comprehensive analysis of experiment results

Usage:
    from src.experiments import run_experiment
    
    results_df = run_experiment(
        create_dataset_func=your_dataset_function,
        sample_sizes=[100, 500, 1000],
        n_runs=5,
        cv_folds=3
    )
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_prediction_stability(y_true: np.ndarray, 
                                 predictions: List[np.ndarray], 
                                 metric: str = 'auc') -> float:
    """
    Calculate prediction stability using coefficient of variation.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    predictions : list of np.ndarray
        List of prediction arrays from different CV folds
    metric : str
        Metric to calculate stability for ('auc', 'accuracy', 'precision', 'recall', 'f1')
    
    Returns:
    --------
    cv_stability : float
        Coefficient of variation (lower = more stable)
    """
    scores = []
    
    for pred in predictions:
        if metric == 'auc':
            if len(np.unique(y_true)) > 1:  # Check if both classes present
                score = roc_auc_score(y_true, pred)
            else:
                score = 0.5  # Default for single class
        elif metric == 'accuracy':
            score = accuracy_score(y_true, pred > 0.5)
        elif metric == 'precision':
            score = precision_score(y_true, pred > 0.5, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, pred > 0.5, zero_division=0)
        elif metric == 'f1':
            score = f1_score(y_true, pred > 0.5, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    scores = np.array(scores)
    if np.mean(scores) == 0:
        return np.inf
    
    return np.std(scores) / np.mean(scores)


def run_experiment(create_dataset_func,
                  sample_sizes: List[int],
                  n_runs: int = 5,
                  cv_folds: int = 3,
                  models: Optional[Dict[str, Any]] = None,
                  random_state: int = 42,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Run comprehensive experiment across different sample sizes and models.
    
    Parameters:
    -----------
    create_dataset_func : callable
        Function that creates dataset, should return (X, y, report)
    sample_sizes : list of int
        List of sample sizes to test
    n_runs : int
        Number of independent runs per sample size
    cv_folds : int
        Number of cross-validation folds
    models : dict, optional
        Dictionary of models to test. If None, uses default RF and LR
    random_state : int
        Base random seed
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    results_df : pd.DataFrame
        Comprehensive results dataframe
    """
    if models is None:
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state),
            'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000)
        }
    
    results = []
    
    for sample_size in sample_sizes:
        if verbose:
            print(f"\n--- Testing sample size: {sample_size} ---")
        
        size_results = {
            'sample_size': sample_size,
            'runs': [],
            'cv_scores': {model_name: [] for model_name in models.keys()},
            'stability_metrics': {model_name: [] for model_name in models.keys()}
        }
        
        # Run multiple independent experiments
        for run in range(n_runs):
            run_seed = random_state + run * 1000 + sample_size
            
            try:
                # Create dataset for this run
                X, y, sampling_report = create_dataset_func(
                    n_positive=sample_size, 
                    n_negative=sample_size,
                    random_state=run_seed
                )
                
                if verbose and run == 0:
                    print(f"  Dataset shape: {X.shape}, Positive ratio: {np.mean(y):.3f}")
                
                # Test each model
                for model_name, model in models.items():
                    # Cross-validation
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=run_seed)
                    fold_scores = []
                    fold_predictions = []
                    all_y_true = []
                    
                    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        # Train model
                        model_copy = type(model)(**model.get_params())
                        model_copy.fit(X_train, y_train)
                        
                        # Predict
                        if hasattr(model_copy, "predict_proba"):
                            y_pred_proba = model_copy.predict_proba(X_val)[:, 1]
                        else:
                            y_pred_proba = model_copy.decision_function(X_val)
                        
                        # Calculate metrics
                        if len(np.unique(y_val)) > 1:
                            auc_score = roc_auc_score(y_val, y_pred_proba)
                        else:
                            auc_score = 0.5
                        
                        fold_scores.append(auc_score)
                        fold_predictions.append(y_pred_proba)
                        all_y_true.extend(y_val)
                    
                    # Calculate stability
                    all_y_true = np.array(all_y_true)
                    stability = calculate_prediction_stability(all_y_true, fold_predictions, 'auc')
                    
                    # Store results
                    mean_auc = np.mean(fold_scores)
                    std_auc = np.std(fold_scores)
                    
                    size_results['cv_scores'][model_name].append(mean_auc)
                    size_results['stability_metrics'][model_name].append(stability)
                    
                    if verbose and run == 0:
                        print(f"    {model_name}: AUC={mean_auc:.3f}±{std_auc:.3f}, Stability={stability:.3f}")
                
                # Store run info
                size_results['runs'].append({
                    'run': run,
                    'dataset_size': len(X),
                    'positive_ratio': np.mean(y),
                    'feature_correlation': np.corrcoef(X[:, 0], X[:, 1])[0, 1] if X.shape[1] >= 2 else 0,
                    'sampling_report': sampling_report
                })
                
            except Exception as e:
                if verbose:
                    print(f"    Error in run {run}: {str(e)}")
                continue
        
        # Aggregate results for this sample size
        for model_name in models.keys():
            cv_scores = size_results['cv_scores'][model_name]
            stability_scores = size_results['stability_metrics'][model_name]
            
            if len(cv_scores) > 0:
                result_row = {
                    'sample_size': sample_size,
                    'model': model_name,
                    'n_successful_runs': len(cv_scores),
                    'mean_auc': np.mean(cv_scores),
                    'std_auc': np.std(cv_scores),
                    'mean_stability': np.mean(stability_scores),
                    'std_stability': np.std(stability_scores),
                    'cv_of_auc': np.std(cv_scores) / np.mean(cv_scores) if np.mean(cv_scores) > 0 else np.inf,
                    'best_auc': np.max(cv_scores),
                    'worst_auc': np.min(cv_scores),
                    'median_stability': np.median(stability_scores)
                }
                
                # Add sampling quality metrics from first successful run
                if size_results['runs']:
                    first_report = size_results['runs'][0]['sampling_report']
                    if 'sampling_quality' in first_report:
                        result_row['degree_balance_quality'] = first_report['sampling_quality']['degree_balance_quality']
                    if 'dataset_stats' in first_report:
                        result_row['feature_correlation'] = first_report['dataset_stats']['feature_correlation']
                
                results.append(result_row)
        
        if verbose:
            print(f"  Completed {len(size_results['runs'])} successful runs")
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n=== Experiment Complete ===")
        print(f"Total experiments: {len(results_df)}")
        print(f"Sample sizes tested: {sorted(sample_sizes)}")
        print(f"Models tested: {list(models.keys())}")
    
    return results_df


def analyze_experiment_results(results_df: pd.DataFrame, 
                             save_path: Optional[str] = None,
                             verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze experiment results and generate summary statistics.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_experiment
    save_path : str, optional
        Path to save detailed results
    verbose : bool
        Whether to print analysis
        
    Returns:
    --------
    analysis : dict
        Comprehensive analysis results
    """
    analysis = {}
    
    # Overall statistics
    analysis['overall'] = {
        'total_experiments': len(results_df),
        'sample_sizes': sorted(results_df['sample_size'].unique()),
        'models': sorted(results_df['model'].unique()),
        'successful_runs_pct': results_df['n_successful_runs'].mean() / results_df['n_successful_runs'].max() * 100
    }
    
    # Best performing configurations
    best_auc_idx = results_df['mean_auc'].idxmax()
    best_stability_idx = results_df['mean_stability'].idxmin()  # Lower is better for stability
    
    analysis['best_performance'] = {
        'best_auc': {
            'value': results_df.loc[best_auc_idx, 'mean_auc'],
            'model': results_df.loc[best_auc_idx, 'model'],
            'sample_size': results_df.loc[best_auc_idx, 'sample_size'],
            'stability': results_df.loc[best_auc_idx, 'mean_stability']
        },
        'best_stability': {
            'value': results_df.loc[best_stability_idx, 'mean_stability'],
            'model': results_df.loc[best_stability_idx, 'model'],
            'sample_size': results_df.loc[best_stability_idx, 'sample_size'],
            'auc': results_df.loc[best_stability_idx, 'mean_auc']
        }
    }
    
    # Model comparison
    model_comparison = results_df.groupby('model').agg({
        'mean_auc': ['mean', 'std', 'max'],
        'mean_stability': ['mean', 'std', 'min'],
        'cv_of_auc': 'mean'
    }).round(4)
    
    analysis['model_comparison'] = model_comparison
    
    # Sample size effects
    size_effects = results_df.groupby('sample_size').agg({
        'mean_auc': ['mean', 'std'],
        'mean_stability': ['mean', 'std'],
        'cv_of_auc': 'mean'
    }).round(4)
    
    analysis['sample_size_effects'] = size_effects
    
    # Correlation analysis
    numeric_cols = ['sample_size', 'mean_auc', 'mean_stability', 'cv_of_auc']
    if 'degree_balance_quality' in results_df.columns:
        numeric_cols.append('degree_balance_quality')
    if 'feature_correlation' in results_df.columns:
        numeric_cols.append('feature_correlation')
    
    correlation_matrix = results_df[numeric_cols].corr().round(3)
    analysis['correlations'] = correlation_matrix
    
    if verbose:
        print("=== Experiment Analysis ===")
        print(f"Total experiments: {analysis['overall']['total_experiments']}")
        print(f"Sample sizes: {analysis['overall']['sample_sizes']}")
        print(f"Models: {analysis['overall']['models']}")
        
        print(f"\nBest AUC: {analysis['best_performance']['best_auc']['value']:.3f}")
        print(f"  Model: {analysis['best_performance']['best_auc']['model']}")
        print(f"  Sample size: {analysis['best_performance']['best_auc']['sample_size']}")
        
        print(f"\nBest Stability: {analysis['best_performance']['best_stability']['value']:.3f}")
        print(f"  Model: {analysis['best_performance']['best_stability']['model']}")
        print(f"  Sample size: {analysis['best_performance']['best_stability']['sample_size']}")
        
        print(f"\nModel Comparison (Mean AUC):")
        for model in analysis['overall']['models']:
            model_data = results_df[results_df['model'] == model]
            print(f"  {model}: {model_data['mean_auc'].mean():.3f}±{model_data['mean_auc'].std():.3f}")
    
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
    
    return analysis