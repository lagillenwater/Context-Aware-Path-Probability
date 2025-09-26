"""
Model evaluation functions for edge probability prediction comparison.
Includes AUC, ROC curve, and other classification/regression metrics.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
    classification_report, confusion_matrix
)
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from pathlib import Path


class ModelEvaluator:
    """Comprehensive model evaluation for edge probability prediction."""

    def __init__(self):
        self.evaluation_results = {}
        self.empirical_frequencies = None

    def evaluate_binary_classification(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                     threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate binary classification performance.

        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        threshold : float
            Classification threshold

        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        # Binary predictions
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = {}

        # AUC and ROC
        if len(np.unique(y_true)) > 1:  # Only if both classes are present
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics['roc_curve'] = (fpr, tpr)
        else:
            metrics['auc'] = np.nan
            metrics['roc_curve'] = (None, None)

        # Precision-Recall
        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_curve'] = (precision, recall)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        else:
            metrics['pr_curve'] = (None, None)
            metrics['average_precision'] = np.nan

        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp

        # Derived metrics
        if tp + fp > 0:
            metrics['precision'] = tp / (tp + fp)
        else:
            metrics['precision'] = 0.0

        if tp + fn > 0:
            metrics['recall'] = tp / (tp + fn)
            metrics['sensitivity'] = metrics['recall']  # Same as recall
        else:
            metrics['recall'] = 0.0
            metrics['sensitivity'] = 0.0

        if tn + fp > 0:
            metrics['specificity'] = tn / (tn + fp)
        else:
            metrics['specificity'] = 0.0

        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0

        return metrics

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression performance.

        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values

        Returns:
        --------
        Dict[str, float]
            Dictionary of regression metrics
        """
        metrics = {}

        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

        # Additional metrics
        metrics['mean_prediction'] = np.mean(y_pred)
        metrics['std_prediction'] = np.std(y_pred)
        metrics['mean_true'] = np.mean(y_true)
        metrics['std_true'] = np.std(y_true)

        # Correlation
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            metrics['correlation'] = np.nan

        return metrics

    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               model_name: str = "Model") -> Dict[str, Any]:
        """
        Perform comprehensive evaluation combining classification and regression metrics.

        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        model_name : str
            Name of the model being evaluated

        Returns:
        --------
        Dict[str, Any]
            Comprehensive evaluation results
        """
        results = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'n_positive': np.sum(y_true),
            'n_negative': len(y_true) - np.sum(y_true),
            'positive_ratio': np.mean(y_true)
        }

        # Classification evaluation
        classification_metrics = self.evaluate_binary_classification(y_true, y_pred_proba)
        results['classification'] = classification_metrics

        # Regression evaluation (treating as continuous prediction)
        regression_metrics = self.evaluate_regression(y_true, y_pred_proba)
        results['regression'] = regression_metrics

        return results

    def evaluate_all_models(self, models_results: Dict[str, Dict[str, Any]],
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models.

        Parameters:
        -----------
        models_results : Dict[str, Dict[str, Any]]
            Results from model training
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Evaluation results for all models
        """
        from model_training import predict_with_model

        evaluation_results = {}

        print("="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        print(f"Test samples: {len(X_test)}")
        print(f"Positive samples: {np.sum(y_test)} ({np.mean(y_test):.3f})")
        print()

        for model_name, model_result in models_results.items():
            if model_name == 'data_splits':
                continue

            print(f"Evaluating {model_name}...")

            model = model_result['model']
            scaler = model_result['training_result'].get('scaler')

            # Make predictions
            y_pred_proba = predict_with_model(model, X_test, model_name, scaler)

            # Comprehensive evaluation
            eval_result = self.comprehensive_evaluation(y_test, y_pred_proba, model_name)

            evaluation_results[model_name] = eval_result

            # Print key metrics
            print(f"  AUC: {eval_result['classification']['auc']:.4f}")
            print(f"  Accuracy: {eval_result['classification']['accuracy']:.4f}")
            print(f"  F1 Score: {eval_result['classification']['f1_score']:.4f}")
            print(f"  RMSE: {eval_result['regression']['rmse']:.4f}")
            print(f"  Correlation: {eval_result['regression']['correlation']:.4f}")
            print()

        self.evaluation_results = evaluation_results
        return evaluation_results

    def load_empirical_frequencies(self, empirical_freq_file: str) -> pd.DataFrame:
        """
        Load empirical edge frequencies from CSV file.

        Parameters:
        -----------
        empirical_freq_file : str
            Path to the empirical frequency CSV file

        Returns:
        --------
        pd.DataFrame
            DataFrame with empirical frequencies
        """
        try:
            empirical_df = pd.read_csv(empirical_freq_file)

            # Standardize column names
            if 'frequency' in empirical_df.columns:
                empirical_df = empirical_df.rename(columns={'frequency': 'empirical_frequency'})

            self.empirical_frequencies = empirical_df

            print(f"Loaded empirical frequencies from: {empirical_freq_file}")
            print(f"  Records: {len(empirical_df)}")
            print(f"  Source degree range: {empirical_df['source_degree'].min()} - {empirical_df['source_degree'].max()}")
            print(f"  Target degree range: {empirical_df['target_degree'].min()} - {empirical_df['target_degree'].max()}")
            print(f"  Frequency range: {empirical_df['empirical_frequency'].min():.6f} - {empirical_df['empirical_frequency'].max():.6f}")

            return empirical_df

        except Exception as e:
            print(f"Error loading empirical frequencies: {e}")
            raise

    def compare_test_predictions_with_empirical(self, evaluation_results: Dict[str, Dict[str, Any]],
                                              models_results: Dict[str, Dict[str, Any]],
                                              X_test: np.ndarray, empirical_freq_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare test set predictions with empirical frequencies for the same degree combinations.

        Parameters:
        -----------
        evaluation_results : Dict[str, Dict[str, Any]]
            Results from model evaluation containing predictions
        models_results : Dict[str, Dict[str, Any]]
            Results from model training
        X_test : np.ndarray
            Test features (degree combinations)
        empirical_freq_file : str
            Path to the empirical frequency CSV file

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Comparison results for all models
        """
        from model_training import predict_with_model

        # Load empirical frequencies
        empirical_df = self.load_empirical_frequencies(empirical_freq_file)

        # Create a lookup dictionary for empirical frequencies
        empirical_lookup = {}
        for _, row in empirical_df.iterrows():
            key = (int(row['source_degree']), int(row['target_degree']))
            empirical_lookup[key] = row['empirical_frequency']

        comparison_results = {}

        print("\n" + "="*60)
        print("COMPARING TEST PREDICTIONS WITH EMPIRICAL FREQUENCIES")
        print("="*60)

        for model_name in evaluation_results.keys():
            print(f"\nComparing {model_name} test predictions with empirical frequencies...")

            # Get model predictions for test set
            model = models_results[model_name]['model']
            scaler = models_results[model_name]['training_result'].get('scaler')
            test_predictions = predict_with_model(model, X_test, model_name, scaler)

            # Match test predictions with empirical frequencies
            matched_predictions = []
            matched_empirical = []
            unmatched_count = 0

            for i, (source_deg, target_deg) in enumerate(X_test):
                key = (int(source_deg), int(target_deg))
                if key in empirical_lookup:
                    matched_predictions.append(test_predictions[i])
                    matched_empirical.append(empirical_lookup[key])
                else:
                    unmatched_count += 1

            if len(matched_predictions) == 0:
                print(f"  Warning: No matching degree combinations found!")
                continue

            matched_predictions = np.array(matched_predictions)
            matched_empirical = np.array(matched_empirical)

            # Calculate comparison metrics
            mae = mean_absolute_error(matched_empirical, matched_predictions)
            rmse = np.sqrt(mean_squared_error(matched_empirical, matched_predictions))

            # Correlation
            if len(matched_empirical) > 1 and np.std(matched_empirical) > 0 and np.std(matched_predictions) > 0:
                correlation = np.corrcoef(matched_empirical, matched_predictions)[0, 1]
            else:
                correlation = np.nan

            # R² score
            r2 = r2_score(matched_empirical, matched_predictions)

            comparison_results[model_name] = {
                'mae_vs_empirical': mae,
                'rmse_vs_empirical': rmse,
                'r2_vs_empirical': r2,
                'correlation_vs_empirical': correlation,
                'n_matched': len(matched_predictions),
                'n_unmatched': unmatched_count,
                'match_ratio': len(matched_predictions) / len(X_test),
                'mean_prediction': np.mean(matched_predictions),
                'mean_empirical': np.mean(matched_empirical),
                'std_prediction': np.std(matched_predictions),
                'std_empirical': np.std(matched_empirical),
                'matched_predictions': matched_predictions,
                'matched_empirical': matched_empirical
            }

            print(f"  Matched degree combinations: {len(matched_predictions)}/{len(X_test)} ({len(matched_predictions)/len(X_test)*100:.1f}%)")
            print(f"  MAE vs Empirical: {mae:.6f}")
            print(f"  RMSE vs Empirical: {rmse:.6f}")
            print(f"  R² vs Empirical: {r2:.6f}")
            print(f"  Correlation vs Empirical: {correlation:.6f}")

        return comparison_results

    def compare_with_empirical_frequencies(self, models_results: Dict[str, Dict[str, Any]],
                                         empirical_freq_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare model predictions with empirical frequencies.

        Parameters:
        -----------
        models_results : Dict[str, Dict[str, Any]]
            Results from model training
        empirical_freq_file : str
            Path to the empirical frequency CSV file

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Empirical comparison results for all models
        """
        from model_training import predict_with_model

        # Load empirical frequencies
        empirical_df = self.load_empirical_frequencies(empirical_freq_file)

        # Prepare features from empirical data
        empirical_features = empirical_df[['source_degree', 'target_degree']].values.astype(np.float32)
        empirical_values = empirical_df['empirical_frequency'].values

        empirical_comparison = {}

        print("\n" + "="*60)
        print("COMPARING WITH EMPIRICAL FREQUENCIES")
        print("="*60)

        for model_name, model_result in models_results.items():
            if model_name == 'data_splits':
                continue

            print(f"\nComparing {model_name} with empirical frequencies...")

            model = model_result['model']
            scaler = model_result['training_result'].get('scaler')

            # Make predictions on empirical degree combinations
            predictions = predict_with_model(model, empirical_features, model_name, scaler)

            # Ensure predictions are probabilities
            predictions = np.clip(predictions, 0.0, 1.0)

            # Calculate comparison metrics
            mae = mean_absolute_error(empirical_values, predictions)
            rmse = np.sqrt(mean_squared_error(empirical_values, predictions))
            r2 = r2_score(empirical_values, predictions)

            # Correlation
            if len(empirical_values) > 1 and np.std(empirical_values) > 0 and np.std(predictions) > 0:
                correlation = np.corrcoef(empirical_values, predictions)[0, 1]
            else:
                correlation = np.nan

            # Calculate percentile-based metrics
            prediction_percentiles = np.percentile(predictions, [25, 50, 75, 90, 95])
            empirical_percentiles = np.percentile(empirical_values, [25, 50, 75, 90, 95])

            empirical_comparison[model_name] = {
                'mae_vs_empirical': mae,
                'rmse_vs_empirical': rmse,
                'r2_vs_empirical': r2,
                'correlation_vs_empirical': correlation,
                'mean_prediction': np.mean(predictions),
                'mean_empirical': np.mean(empirical_values),
                'std_prediction': np.std(predictions),
                'std_empirical': np.std(empirical_values),
                'prediction_percentiles': prediction_percentiles,
                'empirical_percentiles': empirical_percentiles,
                'predictions': predictions,
                'empirical_values': empirical_values,
                'degree_combinations': empirical_features
            }

            print(f"  MAE vs Empirical: {mae:.6f}")
            print(f"  RMSE vs Empirical: {rmse:.6f}")
            print(f"  R² vs Empirical: {r2:.6f}")
            print(f"  Correlation vs Empirical: {correlation:.6f}")

        return empirical_comparison

    def create_test_empirical_comparison_dataframe(self, test_empirical_comparison: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame comparing test predictions against empirical frequencies.

        Parameters:
        -----------
        test_empirical_comparison : Dict[str, Dict[str, Any]]
            Test empirical comparison results

        Returns:
        --------
        pd.DataFrame
            Comparison DataFrame
        """
        comparison_data = []

        for model_name, results in test_empirical_comparison.items():
            row = {
                'Model': model_name,
                'MAE vs Empirical': results['mae_vs_empirical'],
                'RMSE vs Empirical': results['rmse_vs_empirical'],
                'R² vs Empirical': results['r2_vs_empirical'],
                'Correlation vs Empirical': results['correlation_vs_empirical'],
                'Matched Samples': results['n_matched'],
                'Match Ratio': results['match_ratio'],
                'Mean Prediction': results['mean_prediction'],
                'Mean Empirical': results['mean_empirical']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by correlation with empirical (descending)
        df = df.sort_values('Correlation vs Empirical', ascending=False).reset_index(drop=True)

        return df

    def print_test_empirical_comparison_summary(self, test_empirical_comparison: Dict[str, Dict[str, Any]]):
        """Print summary of test predictions vs empirical frequency comparison."""

        print("\n" + "="*80)
        print("TEST PREDICTIONS VS EMPIRICAL FREQUENCIES SUMMARY")
        print("="*80)

        # Find best model for each metric
        best_mae_model = min(test_empirical_comparison.keys(),
                           key=lambda x: test_empirical_comparison[x]['mae_vs_empirical'])
        best_correlation_model = max(test_empirical_comparison.keys(),
                                   key=lambda x: test_empirical_comparison[x]['correlation_vs_empirical']
                                   if not np.isnan(test_empirical_comparison[x]['correlation_vs_empirical']) else -1)

        print(f"Best MAE vs Empirical: {best_mae_model} "
              f"(MAE = {test_empirical_comparison[best_mae_model]['mae_vs_empirical']:.6f})")
        print(f"Best Correlation vs Empirical: {best_correlation_model} "
              f"(Correlation = {test_empirical_comparison[best_correlation_model]['correlation_vs_empirical']:.6f})")

        print(f"\nMatching Statistics:")
        print(f"{'Model':<25} {'Matched':<10} {'Total':<10} {'Match %':<10} {'Mean Pred':<12} {'Mean Emp':<12}")
        print("-" * 85)

        for model_name, results in test_empirical_comparison.items():
            match_pct = results['match_ratio'] * 100
            print(f"{model_name:<25} {results['n_matched']:<10} "
                  f"{results['n_matched'] + results['n_unmatched']:<10} {match_pct:<10.1f} "
                  f"{results['mean_prediction']:<12.6f} {results['mean_empirical']:<12.6f}")

        print("\n" + "="*80)

    def create_empirical_comparison_dataframe(self, empirical_comparison: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame comparing models against empirical frequencies.

        Parameters:
        -----------
        empirical_comparison : Dict[str, Dict[str, Any]]
            Empirical comparison results

        Returns:
        --------
        pd.DataFrame
            Comparison DataFrame
        """
        comparison_data = []

        for model_name, results in empirical_comparison.items():
            row = {
                'Model': model_name,
                'MAE vs Empirical': results['mae_vs_empirical'],
                'RMSE vs Empirical': results['rmse_vs_empirical'],
                'R² vs Empirical': results['r2_vs_empirical'],
                'Correlation vs Empirical': results['correlation_vs_empirical'],
                'Mean Prediction': results['mean_prediction'],
                'Mean Empirical': results['mean_empirical'],
                'Std Prediction': results['std_prediction'],
                'Std Empirical': results['std_empirical']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by correlation with empirical (descending)
        df = df.sort_values('Correlation vs Empirical', ascending=False).reset_index(drop=True)

        return df

    def print_empirical_comparison_summary(self, empirical_comparison: Dict[str, Dict[str, Any]]):
        """Print summary of empirical frequency comparison."""

        print("\n" + "="*80)
        print("EMPIRICAL FREQUENCY COMPARISON SUMMARY")
        print("="*80)

        # Find best model for each metric
        best_mae_model = min(empirical_comparison.keys(),
                           key=lambda x: empirical_comparison[x]['mae_vs_empirical'])
        best_correlation_model = max(empirical_comparison.keys(),
                                   key=lambda x: empirical_comparison[x]['correlation_vs_empirical']
                                   if not np.isnan(empirical_comparison[x]['correlation_vs_empirical']) else -1)

        print(f"Best MAE vs Empirical: {best_mae_model} "
              f"(MAE = {empirical_comparison[best_mae_model]['mae_vs_empirical']:.6f})")
        print(f"Best Correlation vs Empirical: {best_correlation_model} "
              f"(Correlation = {empirical_comparison[best_correlation_model]['correlation_vs_empirical']:.6f})")

        print(f"\nDistribution Comparison:")
        print(f"{'Model':<25} {'Mean Pred':<12} {'Mean Emp':<12} {'Std Pred':<12} {'Std Emp':<12}")
        print("-" * 75)

        for model_name, results in empirical_comparison.items():
            print(f"{model_name:<25} {results['mean_prediction']:<12.6f} "
                  f"{results['mean_empirical']:<12.6f} {results['std_prediction']:<12.6f} "
                  f"{results['std_empirical']:<12.6f}")

        print("\n" + "="*80)

    def create_comparison_dataframe(self, evaluation_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame comparing all models.

        Parameters:
        -----------
        evaluation_results : Dict[str, Dict[str, Any]]
            Evaluation results from all models

        Returns:
        --------
        pd.DataFrame
            Comparison DataFrame
        """
        comparison_data = []

        for model_name, results in evaluation_results.items():
            row = {
                'Model': model_name,
                'AUC': results['classification']['auc'],
                'Accuracy': results['classification']['accuracy'],
                'Precision': results['classification']['precision'],
                'Recall': results['classification']['recall'],
                'F1 Score': results['classification']['f1_score'],
                'Average Precision': results['classification']['average_precision'],
                'RMSE': results['regression']['rmse'],
                'MAE': results['regression']['mae'],
                'R²': results['regression']['r2'],
                'Correlation': results['regression']['correlation']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by AUC (descending)
        df = df.sort_values('AUC', ascending=False).reset_index(drop=True)

        return df

    def print_detailed_results(self, evaluation_results: Dict[str, Dict[str, Any]]):
        """Print detailed evaluation results for all models."""

        print("\n" + "="*80)
        print("DETAILED EVALUATION RESULTS")
        print("="*80)

        for model_name, results in evaluation_results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 50)

            # Dataset info
            print(f"Test Samples: {results['n_samples']}")
            print(f"Positive Samples: {results['n_positive']} ({results['positive_ratio']:.1%})")

            # Classification metrics
            cls_metrics = results['classification']
            print(f"\nClassification Metrics:")
            print(f"  AUC-ROC: {cls_metrics['auc']:.4f}")
            print(f"  Accuracy: {cls_metrics['accuracy']:.4f}")
            print(f"  Precision: {cls_metrics['precision']:.4f}")
            print(f"  Recall/Sensitivity: {cls_metrics['recall']:.4f}")
            print(f"  Specificity: {cls_metrics['specificity']:.4f}")
            print(f"  F1 Score: {cls_metrics['f1_score']:.4f}")
            print(f"  Average Precision: {cls_metrics['average_precision']:.4f}")

            # Confusion matrix
            print(f"\nConfusion Matrix:")
            print(f"  True Positives: {cls_metrics['true_positives']}")
            print(f"  False Positives: {cls_metrics['false_positives']}")
            print(f"  True Negatives: {cls_metrics['true_negatives']}")
            print(f"  False Negatives: {cls_metrics['false_negatives']}")

            # Regression metrics
            reg_metrics = results['regression']
            print(f"\nRegression Metrics:")
            print(f"  RMSE: {reg_metrics['rmse']:.4f}")
            print(f"  MAE: {reg_metrics['mae']:.4f}")
            print(f"  R²: {reg_metrics['r2']:.4f}")
            print(f"  Correlation: {reg_metrics['correlation']:.4f}")

        print("\n" + "="*80)

    def get_total_edges_from_file(self, edge_file_path: str) -> int:
        """
        Load edge matrix and return total number of edges.

        Parameters:
        -----------
        edge_file_path : str
            Path to the sparse edge matrix file

        Returns:
        --------
        int
            Total number of edges in the matrix
        """
        import scipy.sparse as sp

        # Load sparse matrix
        edge_matrix = sp.load_npz(edge_file_path)
        total_edges = edge_matrix.nnz  # Number of non-zero elements (edges)

        print(f"Loaded edge matrix from {edge_file_path}")
        print(f"  Shape: {edge_matrix.shape}")
        print(f"  Total edges: {total_edges}")

        return total_edges

    def analytical_approximation(self, source_degrees: np.ndarray, target_degrees: np.ndarray, total_edges_m: int) -> np.ndarray:
        """
        Calculate analytical approximation for edge probabilities.

        Formula: P_{i,j} = (u_i * v_j) / sqrt((u_i * v_j)^2 + (m - u_i - v_j + 1)^2)

        Parameters:
        -----------
        source_degrees : np.ndarray
            Source node degrees
        target_degrees : np.ndarray
            Target node degrees
        total_edges_m : int
            Total number of edges in the network

        Returns:
        --------
        np.ndarray
            Analytical approximation probabilities
        """
        u_i = source_degrees.astype(np.float64)
        v_j = target_degrees.astype(np.float64)
        m = float(total_edges_m)

        # Calculate the analytical approximation
        numerator = u_i * v_j
        denominator_term = (m - u_i - v_j + 1.0)
        denominator = np.sqrt(numerator**2 + denominator_term**2)

        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-10, denominator)
        probabilities = numerator / denominator

        # Ensure probabilities are in [0, 1] range
        probabilities = np.clip(probabilities, 0.0, 1.0)

        return probabilities

    def validate_analytical_approximation_vs_empirical(self, source_degrees: np.ndarray, target_degrees: np.ndarray,
                                                     empirical_frequencies: np.ndarray, total_edges_m: int) -> Dict[str, Any]:
        """
        Validate analytical approximation against empirical frequencies.

        Parameters:
        -----------
        source_degrees : np.ndarray
            Source degrees from empirical data
        target_degrees : np.ndarray
            Target degrees from empirical data
        empirical_frequencies : np.ndarray
            Empirical frequencies
        total_edges_m : int
            Total number of edges

        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        # Calculate analytical approximation
        analytical_probs = self.analytical_approximation(source_degrees, target_degrees, total_edges_m)

        # Calculate comparison metrics
        mae = mean_absolute_error(empirical_frequencies, analytical_probs)
        rmse = np.sqrt(mean_squared_error(empirical_frequencies, analytical_probs))
        r2 = r2_score(empirical_frequencies, analytical_probs)

        # Correlation
        if len(empirical_frequencies) > 1 and np.std(empirical_frequencies) > 0 and np.std(analytical_probs) > 0:
            correlation = np.corrcoef(empirical_frequencies, analytical_probs)[0, 1]
        else:
            correlation = np.nan

        validation_results = {
            'total_edges_m': total_edges_m,
            'mae_vs_empirical': mae,
            'rmse_vs_empirical': rmse,
            'r2_vs_empirical': r2,
            'correlation_vs_empirical': correlation,
            'mean_analytical': np.mean(analytical_probs),
            'mean_empirical': np.mean(empirical_frequencies),
            'std_analytical': np.std(analytical_probs),
            'std_empirical': np.std(empirical_frequencies),
            'analytical_probabilities': analytical_probs,
            'empirical_frequencies': empirical_frequencies,
            'source_degrees': source_degrees,
            'target_degrees': target_degrees
        }

        print(f"\nAnalytical Approximation Validation vs Empirical:")
        print(f"  Total edges (m): {total_edges_m}")
        print(f"  MAE vs Empirical: {mae:.6f}")
        print(f"  RMSE vs Empirical: {rmse:.6f}")
        print(f"  R² vs Empirical: {r2:.6f}")
        print(f"  Correlation vs Empirical: {correlation:.6f}")
        print(f"  Mean Analytical: {np.mean(analytical_probs):.6f}")
        print(f"  Mean Empirical: {np.mean(empirical_frequencies):.6f}")

        return validation_results

    def compare_models_vs_analytical_approximation(self, evaluation_results: Dict[str, Dict[str, Any]],
                                                  models_results: Dict[str, Dict[str, Any]],
                                                  X_test: np.ndarray, edge_file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare all model predictions with analytical approximation.

        Parameters:
        -----------
        evaluation_results : Dict[str, Dict[str, Any]]
            Model evaluation results
        models_results : Dict[str, Dict[str, Any]]
            Model training results
        X_test : np.ndarray
            Test features (source_degree, target_degree)
        edge_file_path : str
            Path to edge file to get total edges

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Comparison results for all models
        """
        from model_training import predict_with_model

        # Get total edges
        total_edges_m = self.get_total_edges_from_file(edge_file_path)

        # Calculate analytical approximation for test set
        source_degrees_test = X_test[:, 0]
        target_degrees_test = X_test[:, 1]
        analytical_probs = self.analytical_approximation(source_degrees_test, target_degrees_test, total_edges_m)

        analytical_comparison = {}

        print("\n" + "="*80)
        print("COMPARING MODEL PREDICTIONS WITH ANALYTICAL APPROXIMATION")
        print("="*80)
        print(f"Total edges (m): {total_edges_m}")
        print(f"Test samples: {len(X_test)}")
        print()

        for model_name in evaluation_results.keys():
            print(f"Comparing {model_name} with analytical approximation...")

            # Get model predictions
            model = models_results[model_name]['model']
            scaler = models_results[model_name]['training_result'].get('scaler')
            model_predictions = predict_with_model(model, X_test, model_name, scaler)

            # Calculate comparison metrics
            mae = mean_absolute_error(analytical_probs, model_predictions)
            rmse = np.sqrt(mean_squared_error(analytical_probs, model_predictions))
            r2 = r2_score(analytical_probs, model_predictions)

            # Correlation
            if len(analytical_probs) > 1 and np.std(analytical_probs) > 0 and np.std(model_predictions) > 0:
                correlation = np.corrcoef(analytical_probs, model_predictions)[0, 1]
            else:
                correlation = np.nan

            analytical_comparison[model_name] = {
                'mae_vs_analytical': mae,
                'rmse_vs_analytical': rmse,
                'r2_vs_analytical': r2,
                'correlation_vs_analytical': correlation,
                'mean_model_prediction': np.mean(model_predictions),
                'mean_analytical': np.mean(analytical_probs),
                'std_model_prediction': np.std(model_predictions),
                'std_analytical': np.std(analytical_probs),
                'model_predictions': model_predictions,
                'analytical_probabilities': analytical_probs
            }

            print(f"  MAE vs Analytical: {mae:.6f}")
            print(f"  RMSE vs Analytical: {rmse:.6f}")
            print(f"  R² vs Analytical: {r2:.6f}")
            print(f"  Correlation vs Analytical: {correlation:.6f}")
            print()

        return analytical_comparison

    def print_analytical_comparison_summary(self, analytical_comparison: Dict[str, Dict[str, Any]]):
        """Print summary of analytical approximation comparison."""

        print("="*80)
        print("MODEL PREDICTIONS VS ANALYTICAL APPROXIMATION SUMMARY")
        print("="*80)

        # Find best models
        best_mae_model = min(analytical_comparison.keys(),
                           key=lambda x: analytical_comparison[x]['mae_vs_analytical'])
        best_correlation_model = max(analytical_comparison.keys(),
                                   key=lambda x: analytical_comparison[x]['correlation_vs_analytical']
                                   if not np.isnan(analytical_comparison[x]['correlation_vs_analytical']) else -1)

        print(f"Best MAE vs Analytical: {best_mae_model} "
              f"(MAE = {analytical_comparison[best_mae_model]['mae_vs_analytical']:.6f})")
        print(f"Best Correlation vs Analytical: {best_correlation_model} "
              f"(Correlation = {analytical_comparison[best_correlation_model]['correlation_vs_analytical']:.6f})")

        print(f"\nDistribution Comparison:")
        print(f"{'Model':<30} {'Mean Model':<15} {'Mean Analytical':<15} {'Std Model':<15} {'Std Analytical':<15}")
        print("-" * 95)

        for model_name, results in analytical_comparison.items():
            print(f"{model_name:<30} {results['mean_model_prediction']:<15.6f} "
                  f"{results['mean_analytical']:<15.6f} {results['std_model_prediction']:<15.6f} "
                  f"{results['std_analytical']:<15.6f}")

        print("\n" + "="*80)

    def create_analytical_comparison_dataframe(self, analytical_comparison: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame comparing models vs analytical approximation.

        Parameters:
        -----------
        analytical_comparison : Dict[str, Dict[str, Any]]
            Analytical comparison results

        Returns:
        --------
        pd.DataFrame
            Comparison DataFrame
        """
        comparison_data = []

        for model_name, results in analytical_comparison.items():
            row = {
                'Model': model_name,
                'MAE vs Analytical': results['mae_vs_analytical'],
                'RMSE vs Analytical': results['rmse_vs_analytical'],
                'R² vs Analytical': results['r2_vs_analytical'],
                'Correlation vs Analytical': results['correlation_vs_analytical'],
                'Mean Model Prediction': results['mean_model_prediction'],
                'Mean Analytical': results['mean_analytical'],
                'Std Model Prediction': results['std_model_prediction'],
                'Std Analytical': results['std_analytical']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by correlation with analytical (descending)
        df = df.sort_values('Correlation vs Analytical', ascending=False).reset_index(drop=True)

        return df


def get_best_models(evaluation_results: Dict[str, Dict[str, Any]],
                   metrics: List[str] = None) -> Dict[str, str]:
    """
    Identify the best model for each metric.

    Parameters:
    -----------
    evaluation_results : Dict[str, Dict[str, Any]]
        Evaluation results from all models
    metrics : List[str], optional
        List of metrics to consider. If None, uses default set.

    Returns:
    --------
    Dict[str, str]
        Dictionary mapping metric names to best model names
    """
    if metrics is None:
        metrics = ['auc', 'accuracy', 'f1_score', 'average_precision', 'rmse', 'correlation']

    best_models = {}

    for metric in metrics:
        best_value = None
        best_model = None

        for model_name, results in evaluation_results.items():
            if metric in ['auc', 'accuracy', 'f1_score', 'average_precision', 'correlation']:
                # Higher is better
                if metric == 'correlation':
                    current_value = results['regression'][metric]
                else:
                    current_value = results['classification'][metric]

                if not np.isnan(current_value) and (best_value is None or current_value > best_value):
                    best_value = current_value
                    best_model = model_name

            elif metric in ['rmse', 'mae']:
                # Lower is better
                current_value = results['regression'][metric]
                if not np.isnan(current_value) and (best_value is None or current_value < best_value):
                    best_value = current_value
                    best_model = model_name

        best_models[metric] = best_model

    return best_models


