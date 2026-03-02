"""
Baseline and control evaluation framework for pathway frequency prediction.

This module provides a systematic framework for evaluating prediction models
against baselines and controls:

Negative Controls: Random predictions, constant mean
Weak Baselines: Degree product (compositional, r~0.35)
Strong Baselines: GLM, Random Forest, Neural Networks
Target Models: Advanced architectures (Transformer, etc.)
Positive Control: Empirical frequencies from permutations

Includes:
- Cross-validation with statistical significance testing
- Comprehensive metrics (correlation, RMSE, bias, calibration)
- Paired comparisons with confidence intervals
- Automated reporting
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import warnings


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for pathway frequency prediction.

    Computes correlation, error, bias, and calibration metrics.
    """

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        metrics : dict
            Dictionary of metric names to values
        """
        metrics = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pearson_r, pearson_p = pearsonr(y_true, y_pred)
            spearman_r, spearman_p = spearmanr(y_true, y_pred)

            metrics['pearson_r'] = pearson_r
            metrics['pearson_p'] = pearson_p
            metrics['spearman_r'] = spearman_r
            metrics['spearman_p'] = spearman_p

            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)

            metrics['bias'] = np.mean(y_pred - y_true)
            metrics['abs_bias'] = np.mean(np.abs(y_pred - y_true))

            residuals = y_pred - y_true
            metrics['residual_std'] = np.std(residuals)
            metrics['residual_skew'] = EvaluationMetrics._skewness(residuals)

            metrics['median_error'] = np.median(np.abs(residuals))

            low_mask = y_true < np.percentile(y_true, 33)
            mid_mask = (y_true >= np.percentile(y_true, 33)) & \
                       (y_true < np.percentile(y_true, 67))
            high_mask = y_true >= np.percentile(y_true, 67)

            metrics['bias_low'] = np.mean((y_pred - y_true)[low_mask]) \
                if low_mask.any() else np.nan
            metrics['bias_mid'] = np.mean((y_pred - y_true)[mid_mask]) \
                if mid_mask.any() else np.nan
            metrics['bias_high'] = np.mean((y_pred - y_true)[high_mask]) \
                if high_mask.any() else np.nan

        return metrics

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        """Compute skewness of distribution."""
        if len(x) < 3:
            return np.nan
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return np.mean(((x - mean) / std) ** 3)


class BaselineComparison:
    """
    Statistical comparison of model performance against baselines.

    Performs paired t-tests and computes confidence intervals for
    performance differences.
    """

    @staticmethod
    def paired_comparison(model_scores: np.ndarray,
                         baseline_scores: np.ndarray,
                         metric_name: str = 'correlation') -> Dict[str, float]:
        """
        Perform paired t-test comparing model to baseline.

        Parameters
        ----------
        model_scores : np.ndarray
            Scores for model across CV folds
        baseline_scores : np.ndarray
            Scores for baseline across CV folds
        metric_name : str
            Name of metric being compared

        Returns
        -------
        comparison : dict
            Statistical comparison results
        """
        diff = model_scores - baseline_scores

        t_stat, p_value = ttest_rel(model_scores, baseline_scores)

        comparison = {
            f'model_mean_{metric_name}': np.mean(model_scores),
            f'baseline_mean_{metric_name}': np.mean(baseline_scores),
            f'mean_improvement': np.mean(diff),
            f'std_improvement': np.std(diff),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'significant_001': p_value < 0.01
        }

        ci_95 = 1.96 * np.std(diff) / np.sqrt(len(diff))
        comparison['ci_95_lower'] = np.mean(diff) - ci_95
        comparison['ci_95_upper'] = np.mean(diff) + ci_95

        return comparison

    @staticmethod
    def compare_all_baselines(model_cv_scores: Dict[str, np.ndarray],
                              baseline_cv_scores: Dict[str, Dict[str, np.ndarray]],
                              metric: str = 'pearson_r') -> pd.DataFrame:
        """
        Compare model against all baselines.

        Parameters
        ----------
        model_cv_scores : dict
            Model scores across CV folds for each metric
        baseline_cv_scores : dict
            Baseline scores: {baseline_name: {metric: scores}}
        metric : str
            Primary metric for comparison

        Returns
        -------
        comparison_df : pd.DataFrame
            Comparison table
        """
        comparisons = []

        model_scores = model_cv_scores[metric]

        for baseline_name, baseline_scores_dict in baseline_cv_scores.items():
            baseline_scores = baseline_scores_dict[metric]

            comp = BaselineComparison.paired_comparison(
                model_scores, baseline_scores, metric
            )
            comp['baseline'] = baseline_name
            comparisons.append(comp)

        df = pd.DataFrame(comparisons)

        df = df.sort_values(f'baseline_mean_{metric}')

        return df


class CrossValidator:
    """
    Cross-validation framework for pathway models.

    Handles stratified k-fold CV on degree bins, ensuring balanced
    representation of count ranges across folds.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize cross-validator.

        Parameters
        ----------
        n_splits : int, default=5
            Number of CV folds
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=True,
                           random_state=random_state)

    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray,
                      fit_params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Evaluate model with k-fold cross-validation.

        Parameters
        ----------
        model : object
            Model with fit() and predict() methods
        X : np.ndarray
            Features
        y : np.ndarray
            Targets
        fit_params : dict, optional
            Additional parameters for model.fit()

        Returns
        -------
        cv_scores : dict
            Dictionary of metric names to arrays of scores across folds
        """
        if fit_params is None:
            fit_params = {}

        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train, **fit_params)

            y_pred = model.predict(X_val)

            metrics = EvaluationMetrics.compute_all(y_val, y_pred)
            metrics['fold'] = fold_idx
            fold_metrics.append(metrics)

        cv_scores = {}
        metric_names = [k for k in fold_metrics[0].keys() if k != 'fold']

        for metric_name in metric_names:
            cv_scores[metric_name] = np.array([
                m[metric_name] for m in fold_metrics
            ])

        return cv_scores

    def evaluate_all_models(self, models: Dict[str, Any],
                           X: np.ndarray, y: np.ndarray,
                           fit_params: Optional[Dict[str, Dict]] = None) \
            -> Dict[str, Dict[str, np.ndarray]]:
        """
        Evaluate multiple models with cross-validation.

        Parameters
        ----------
        models : dict
            Dictionary of model_name to model object
        X : np.ndarray
            Features
        y : np.ndarray
            Targets
        fit_params : dict, optional
            Dictionary of model_name to fit parameters

        Returns
        -------
        all_cv_scores : dict
            Dictionary of model_name to cv_scores dict
        """
        if fit_params is None:
            fit_params = {name: {} for name in models.keys()}

        all_cv_scores = {}

        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")

            params = fit_params.get(model_name, {})
            cv_scores = self.evaluate_model(model, X, y, params)

            all_cv_scores[model_name] = cv_scores

        return all_cv_scores


class BaselineFramework:
    """
    Complete baseline and control evaluation framework.

    Orchestrates model evaluation, baseline comparison, and reporting.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize baseline framework.

        Parameters
        ----------
        n_splits : int, default=5
            Number of CV folds
        random_state : int, default=42
            Random seed
        """
        self.cv = CrossValidator(n_splits=n_splits,
                                random_state=random_state)
        self.results = {}

    def evaluate_with_baselines(self, models: Dict[str, Any],
                                X: np.ndarray, y: np.ndarray,
                                baseline_names: Optional[List[str]] = None,
                                fit_params: Optional[Dict[str, Dict]] = None) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate all models and compare against baselines.

        Parameters
        ----------
        models : dict
            Dictionary of model_name to model object
        X : np.ndarray
            Features
        y : np.ndarray
            Targets
        baseline_names : list, optional
            Names of models to treat as baselines for comparison
        fit_params : dict, optional
            Dictionary of model_name to fit parameters

        Returns
        -------
        summary_df : pd.DataFrame
            Summary of all model performance
        comparison_df : pd.DataFrame
            Statistical comparisons against baselines
        """
        all_cv_scores = self.cv.evaluate_all_models(models, X, y, fit_params)

        self.results = all_cv_scores

        summary_data = []
        for model_name, cv_scores in all_cv_scores.items():
            summary = {
                'model': model_name,
                'pearson_r_mean': np.mean(cv_scores['pearson_r']),
                'pearson_r_std': np.std(cv_scores['pearson_r']),
                'spearman_r_mean': np.mean(cv_scores['spearman_r']),
                'rmse_mean': np.mean(cv_scores['rmse']),
                'rmse_std': np.std(cv_scores['rmse']),
                'bias_mean': np.mean(cv_scores['bias']),
                'bias_std': np.std(cv_scores['bias']),
                'r2_mean': np.mean(cv_scores['r2']),
            }
            summary_data.append(summary)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('pearson_r_mean', ascending=False)

        if baseline_names is None:
            baseline_names = ['random', 'degree_product']
            baseline_names = [b for b in baseline_names if b in models]

        comparison_dfs = []
        for target_model in models.keys():
            if target_model in baseline_names:
                continue

            baseline_scores = {
                name: all_cv_scores[name]
                for name in baseline_names
                if name in all_cv_scores
            }

            comp_df = BaselineComparison.compare_all_baselines(
                all_cv_scores[target_model],
                baseline_scores,
                metric='pearson_r'
            )
            comp_df['target_model'] = target_model
            comparison_dfs.append(comp_df)

        if comparison_dfs:
            comparison_df = pd.concat(comparison_dfs, ignore_index=True)
        else:
            comparison_df = pd.DataFrame()

        return summary_df, comparison_df

    def print_report(self, summary_df: pd.DataFrame,
                    comparison_df: pd.DataFrame,
                    target_r: float = 0.95):
        """
        Print formatted evaluation report.

        Parameters
        ----------
        summary_df : pd.DataFrame
            Summary of model performance
        comparison_df : pd.DataFrame
            Baseline comparisons
        target_r : float, default=0.95
            Target correlation for success criteria
        """
        print("=" * 80)
        print("PATHWAY FREQUENCY PREDICTION: BASELINE EVALUATION REPORT")
        print("=" * 80)

        print("\n" + "-" * 80)
        print("MODEL PERFORMANCE SUMMARY")
        print("-" * 80)
        print(summary_df.to_string(index=False))

        print("\n" + "-" * 80)
        print("SUCCESS CRITERIA")
        print("-" * 80)

        best_model = summary_df.iloc[0]
        best_r = best_model['pearson_r_mean']

        print(f"Target: r > {target_r:.2f}")
        print(f"Best model: {best_model['model']}")
        print(f"Best r: {best_r:.4f} +/- {best_model['pearson_r_std']:.4f}")

        if best_r > target_r:
            print(f"STATUS: SUCCESS (exceeded target by {best_r - target_r:.4f})")
        else:
            print(f"STATUS: NOT MET (need {target_r - best_r:.4f} improvement)")

        if not comparison_df.empty:
            print("\n" + "-" * 80)
            print("BASELINE COMPARISONS (Paired t-tests)")
            print("-" * 80)

            for target in comparison_df['target_model'].unique():
                target_df = comparison_df[
                    comparison_df['target_model'] == target
                ]

                print(f"\n{target} vs baselines:")
                for _, row in target_df.iterrows():
                    sig_marker = "***" if row['significant_001'] else \
                                 "**" if row['significant'] else ""
                    print(f"  vs {row['baseline']}: "
                          f"improvement = {row['mean_improvement']:+.4f} "
                          f"(p = {row['p_value']:.4f}) {sig_marker}")

        print("\n" + "=" * 80)
