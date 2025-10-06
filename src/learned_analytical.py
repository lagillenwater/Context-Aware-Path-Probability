"""
Learned Analytical Formula for Edge Probability Estimation

This module learns an improved analytical function that generalizes across
sparse and dense graphs by fitting to empirical data from multiple permutations.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

from model_comparison import filter_zero_degree_nodes


class LearnedAnalyticalFormula:
    """
    Learn parameterized analytical function for edge probability estimation.

    Formula: P(u, v | graph) = α × (u^β × v^γ) / (δ + ε×m + ζ×(u×v)^η + θ×density^κ)

    Where:
        u, v: source/target degrees
        m: total edges
        density: edge density
        α, β, γ, δ, ε, ζ, η, θ, κ: learnable parameters
    """

    def __init__(self, n_random_starts=10, regularization_lambda=0.001,
                 formula_type='original', bootstrap_samples=1, ensemble_size=1):
        self.params = None
        self.graph_stats = {}
        self.convergence_history = []
        self.param_names = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'κ']

        # Configuration
        self.n_random_starts = n_random_starts
        self.regularization_lambda = regularization_lambda
        self.formula_type = formula_type  # 'original', 'extended', 'polynomial'
        self.bootstrap_samples = bootstrap_samples
        self.ensemble_size = ensemble_size

        # For ensemble
        self.ensemble_params = []

    def find_minimum_permutations(
        self,
        graph_name: str,
        data_dir: Path,
        results_dir: Path,
        N_candidates: List[int] = None,
        convergence_threshold: float = 0.02,
        target_metric: str = 'correlation',
        min_metric_value: float = 0.95,
        degree_stratified: bool = False,
        small_graph_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Automatically find minimum N permutations needed for accurate learning.

        Enhanced Strategy:
        1. Train on increasing N: [2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
        2. For each N, validate against 200-perm empirical
        3. If degree_stratified=True, analyze convergence by degree combinations
        4. Stop when:
           a) Validation metric plateaus (improvement < threshold)
           b) Validation metric exceeds target
           c) All degree combinations converge (if degree_stratified=True)

        Parameters
        ----------
        graph_name : str
            Graph to analyze (e.g., 'CtD', 'AeG')
        data_dir : Path
            Path to data directory
        results_dir : Path
            Path to results directory
        N_candidates : List[int], optional
            Permutation counts to test
        convergence_threshold : float
            Stop if improvement < this (e.g., 0.02 = 2%)
        target_metric : str
            Metric to optimize ('correlation', 'mae', 'rmse')
        min_metric_value : float
            Target performance level
        degree_stratified : bool
            Whether to perform degree-stratified convergence analysis
        small_graph_mode : bool
            Whether to use small graph optimizations

        Returns
        -------
        results : dict
            {
                'N_min': minimum permutations needed,
                'best_params': learned parameters at N_min,
                'convergence_curve': performance vs N,
                'final_metrics': validation metrics at N_min,
                'degree_convergence': degree-stratified analysis (if enabled)
            }
        """

        if N_candidates is None:
            N_candidates = [2, 3, 5, 7, 10, 15, 20, 30, 40, 50]

        print(f'{"="*80}')
        print(f'FINDING MINIMUM PERMUTATIONS FOR {graph_name}')
        print(f'{"="*80}')
        print(f'Testing N values: {N_candidates}')
        print(f'Target: {target_metric} > {min_metric_value}')
        print(f'Convergence threshold: {convergence_threshold} ({convergence_threshold*100}%)')
        print(f'Degree stratified: {degree_stratified}')
        print(f'Small graph mode: {small_graph_mode}')
        print()

        # Load 200-permutation empirical (validation gold standard)
        print('Loading 200-permutation empirical frequencies (validation target)...')
        empirical_200 = self._load_200_perm_empirical(graph_name, results_dir)
        print(f'  Degree combinations: {len(empirical_200)}')

        # Load graph statistics
        edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{graph_name}.sparse.npz'
        edge_matrix = sp.load_npz(edge_file)
        edge_matrix, _, _ = filter_zero_degree_nodes(edge_matrix)

        m = edge_matrix.nnz
        n_sources, n_targets = edge_matrix.shape
        density = m / (n_sources * n_targets)

        self.graph_stats[graph_name] = {
            'm': m, 'density': density,
            'n_sources': n_sources, 'n_targets': n_targets
        }

        print(f'\nGraph statistics:')
        print(f'  Edges (m): {m}')
        print(f'  Density: {density:.4f}')
        print(f'  Nodes: {n_sources} × {n_targets}')
        print()

        # Track results across different N
        results_by_N = []
        previous_metric = None
        N_min = None
        best_params = None

        for N in N_candidates:
            print(f'{"-"*80}')
            print(f'Training with N = {N} permutations')
            print(f'{"-"*80}')

            # Train on first N permutations
            train_result = self._train_on_N_permutations(
                graph_name, N, m, density, data_dir
            )

            # Validate against 200-perm empirical with optional degree stratification
            val_metrics = self._validate_against_empirical_200(
                empirical_200, m, density, degree_stratified, small_graph_mode
            )

            # Store results
            result = {
                'N': N,
                'train_metrics': train_result['train_metrics'],
                'val_metrics': val_metrics,
                'params': self.params.copy(),
                'success': train_result['success'],
                'iterations': train_result['iterations']
            }
            results_by_N.append(result)

            # Print summary
            print(f'\nResults for N = {N}:')
            print(f'  Optimization: {"✓ Success" if train_result["success"] else "✗ Failed"} ({train_result["iterations"]} iterations)')
            print(f'  Training:')
            print(f'    MAE: {train_result["train_metrics"]["mae"]:.6f}')
            print(f'    Correlation: {train_result["train_metrics"]["correlation"]:.6f}')
            print(f'  Validation (vs 200-perm empirical):')
            print(f'    MAE: {val_metrics["mae"]:.6f}')
            print(f'    RMSE: {val_metrics["rmse"]:.6f}')
            print(f'    Correlation: {val_metrics["correlation"]:.6f}')

            # Print degree-stratified results if available
            if degree_stratified and 'degree_stratified' in val_metrics:
                degree_combos = val_metrics['degree_combinations_count']
                degree_corr_mean = val_metrics.get('degree_correlation_mean', np.nan)
                degree_corr_std = val_metrics.get('degree_correlation_std', np.nan)
                print(f'  Degree-stratified ({degree_combos} combinations):')
                if not np.isnan(degree_corr_mean):
                    print(f'    Avg correlation: {degree_corr_mean:.6f} ± {degree_corr_std:.6f}')
                else:
                    print(f'    Correlation: insufficient data')

            current_metric = val_metrics[target_metric]

            # Check convergence criteria
            if previous_metric is not None:
                # Calculate improvement
                if target_metric == 'mae' or target_metric == 'rmse':
                    # For error metrics, improvement = decrease
                    improvement = (previous_metric - current_metric) / previous_metric
                else:
                    # For correlation/R², improvement = increase
                    improvement = (current_metric - previous_metric) / previous_metric

                print(f'  Improvement over N={results_by_N[-2]["N"]}: {improvement:+.4f} ({improvement*100:+.2f}%)')

                # Convergence check 1: Plateau detection
                if abs(improvement) < convergence_threshold:
                    print(f'\n✓ CONVERGENCE: Improvement < {convergence_threshold*100}%')
                    N_min = N
                    best_params = self.params.copy()
                    break

            # Convergence check 2: Target metric achieved
            if target_metric in ['correlation', 'r2']:
                if current_metric >= min_metric_value:
                    print(f'\n✓ TARGET ACHIEVED: {target_metric} = {current_metric:.6f} >= {min_metric_value}')
                    N_min = N
                    best_params = self.params.copy()
                    break
            elif target_metric in ['mae', 'rmse']:
                if current_metric <= min_metric_value:
                    print(f'\n✓ TARGET ACHIEVED: {target_metric} = {current_metric:.6f} <= {min_metric_value}')
                    N_min = N
                    best_params = self.params.copy()
                    break

            previous_metric = current_metric
            print()

        # If no convergence, use best N from candidates
        if N_min is None:
            print(f'\n⚠ No convergence detected. Using largest N = {N_candidates[-1]}')
            N_min = N_candidates[-1]
            best_params = results_by_N[-1]['params']

        # Final summary
        print(f'\n{"="*80}')
        print(f'MINIMUM PERMUTATIONS FOUND: N = {N_min}')
        print(f'{"="*80}')

        final_result_idx = [r['N'] for r in results_by_N].index(N_min)
        final_metrics = results_by_N[final_result_idx]['val_metrics']

        print(f'\nFinal validation metrics (N={N_min} vs 200-perm empirical):')
        print(f'  MAE: {final_metrics["mae"]:.6f}')
        print(f'  RMSE: {final_metrics["rmse"]:.6f}')
        print(f'  R²: {final_metrics["r2"]:.6f}')
        print(f'  Correlation: {final_metrics["correlation"]:.6f}')

        # Compare to baseline
        print(f'\nBaseline (current analytical vs 200-perm empirical):')
        baseline_metrics = self._compute_baseline_metrics(empirical_200, m)
        print(f'  MAE: {baseline_metrics["mae"]:.6f}')
        print(f'  RMSE: {baseline_metrics["rmse"]:.6f}')
        print(f'  Correlation: {baseline_metrics["correlation"]:.6f}')

        # Improvement
        mae_improvement = (baseline_metrics['mae'] - final_metrics['mae']) / baseline_metrics['mae'] * 100
        corr_improvement = (final_metrics['correlation'] - baseline_metrics['correlation']) / baseline_metrics['correlation'] * 100

        print(f'\nImprovement over current analytical:')
        print(f'  MAE: {mae_improvement:+.1f}%')
        print(f'  Correlation: {corr_improvement:+.1f}%')
        print(f'{"="*80}')

        # Create convergence curve
        self._plot_convergence_curve(results_by_N, N_min, graph_name, results_dir)

        # Restore best parameters
        self.params = best_params

        return {
            'N_min': N_min,
            'best_params': best_params,
            'convergence_curve': results_by_N,
            'final_metrics': final_metrics,
            'baseline_metrics': baseline_metrics,
            'graph_name': graph_name,
            'graph_stats': self.graph_stats[graph_name]
        }

    def _train_on_N_permutations(self, graph_name: str, N: int, m: int,
                                  density: float, data_dir: Path) -> Dict[str, Any]:
        """Train analytical formula on N permutations"""

        # Compute empirical from first N permutations
        train_empirical = self._compute_empirical_from_permutations(
            graph_name,
            perm_ids=list(range(N)),
            data_dir=data_dir
        )

        # Prepare training data
        X_train = []
        y_train = []

        for (u, v), freq in train_empirical.items():
            X_train.append([u, v, m, density])
            y_train.append(freq)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Define loss function
        def loss_function(params):
            predictions = []
            for u, v, m_val, d in X_train:
                p = self._formula(u, v, m_val, d, params)
                predictions.append(p)

            predictions = np.array(predictions)
            mse = np.mean((predictions - y_train) ** 2)

            # L2 regularization (configurable)
            l2_penalty = self.regularization_lambda * np.sum(params ** 2)

            return mse + l2_penalty

        # Bounds and initial parameters based on formula type
        if self.formula_type == 'extended':
            # Extended formula: 11 parameters
            bounds = [
                (0.001, 100.0),   # α
                (0.1, 3.0),       # β
                (0.1, 3.0),       # γ
                (0.001, 1000.0),  # δ
                (1e-10, 1.0),     # ε
                (0.001, 1000.0),  # ζ
                (0.5, 3.0),       # η
                (0.0, 1000.0),    # θ
                (0.1, 3.0),       # κ
                (0.0, 1000.0),    # λ (log term for u)
                (0.0, 1000.0)     # μ (log term for v)
            ]
            default_params = np.array([1.0, 1.0, 1.0, 1.0, 1e-6, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0])
            self.param_names = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'κ', 'λ', 'μ']

        elif self.formula_type == 'polynomial':
            # Polynomial formula: 9 parameters (different meaning)
            bounds = [
                (0.001, 100.0),   # α (coef for u×v in numerator)
                (0.0, 100.0),     # β (coef for u in numerator)
                (0.0, 100.0),     # γ (coef for v in numerator)
                (0.0, 100.0),     # δ (constant in numerator)
                (1e-10, 1.0),     # ε (coef for m in denominator)
                (0.001, 100.0),   # ζ (coef for u×v in denominator)
                (0.0, 100.0),     # η (coef for u+v in denominator)
                (0.001, 1000.0),  # θ (constant in denominator)
                (0.0, 1000.0)     # ι (coef for density in denominator)
            ]
            default_params = np.array([1.0, 0.1, 0.1, 0.01, 1e-6, 1.0, 0.1, 1.0, 0.0])
            self.param_names = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι']

        else:  # original
            # Original formula: 9 parameters
            bounds = [
                (0.001, 100.0),   # α
                (0.1, 3.0),       # β
                (0.1, 3.0),       # γ
                (0.001, 1000.0),  # δ
                (1e-10, 1.0),     # ε
                (0.001, 1000.0),  # ζ
                (0.5, 3.0),       # η
                (0.0, 1000.0),    # θ
                (0.1, 3.0)        # κ
            ]
            default_params = np.array([1.0, 1.0, 1.0, 1.0, 1e-6, 1.0, 2.0, 0.0, 1.0])
            # param_names already set in __init__

        # Bootstrap + multi-start optimization
        all_bootstrap_params = []

        if self.bootstrap_samples > 1:
            print(f"  Running bootstrap training ({self.bootstrap_samples} bootstrap samples, {self.n_random_starts} starts each)...")
        else:
            print(f"  Running multi-start optimization ({self.n_random_starts} starts, formula={self.formula_type})...")

        for bootstrap_idx in range(self.bootstrap_samples):
            # Bootstrap sampling (with replacement)
            if self.bootstrap_samples > 1:
                n_samples = len(X_train)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X_train[bootstrap_indices]
                y_bootstrap = y_train[bootstrap_indices]

                # Create bootstrap-specific loss function
                def bootstrap_loss_function(params):
                    predictions = []
                    for u, v, m_val, d in X_bootstrap:
                        p = self._formula(u, v, m_val, d, params)
                        predictions.append(p)

                    predictions = np.array(predictions)
                    mse = np.mean((predictions - y_bootstrap) ** 2)
                    l2_penalty = self.regularization_lambda * np.sum(params ** 2)
                    return mse + l2_penalty

                current_loss_function = bootstrap_loss_function
            else:
                X_bootstrap = X_train
                y_bootstrap = y_train
                current_loss_function = loss_function

            # Multi-start optimization for this bootstrap sample
            best_result = None
            best_loss = float('inf')

            for start_idx in range(self.n_random_starts):
                # Generate random initial parameters
                if start_idx == 0:
                    # First start: use sensible defaults
                    initial_params = default_params.copy()
                else:
                    # Random starts: sample uniformly from bounds
                    initial_params = np.array([
                        np.random.uniform(bounds[i][0], bounds[i][1])
                        for i in range(len(bounds))
                    ])

                # Optimize
                result = minimize(
                    current_loss_function,
                    x0=initial_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
                )

                # Keep track of best result for this bootstrap
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result

            # Store parameters from this bootstrap sample
            all_bootstrap_params.append(best_result.x)

            if self.bootstrap_samples > 1:
                print(f"    Bootstrap {bootstrap_idx + 1}/{self.bootstrap_samples}: loss = {best_loss:.6f}")

        # Average parameters across bootstrap samples (or use single if bootstrap_samples=1)
        self.params = np.mean(all_bootstrap_params, axis=0)

        # Store ensemble params for ensemble prediction
        if self.ensemble_size > 1:
            # Keep top ensemble_size models by loss on full training set
            losses = []
            for params in all_bootstrap_params:
                loss = loss_function(params)
                losses.append(loss)

            # Sort by loss and keep top ensemble_size
            sorted_indices = np.argsort(losses)[:self.ensemble_size]
            self.ensemble_params = [all_bootstrap_params[i] for i in sorted_indices]

            avg_loss = np.mean([losses[i] for i in sorted_indices])
            print(f"  Ensemble: using top {self.ensemble_size} models (avg loss = {avg_loss:.6f})")
        else:
            self.ensemble_params = [self.params]

        if self.bootstrap_samples > 1:
            final_loss = loss_function(self.params)
            print(f"  Averaged parameters: loss = {final_loss:.6f}")
        else:
            print(f"  Best loss: {best_loss:.6f} (from {self.n_random_starts} starts)")

        # Compute training metrics using averaged/final parameters
        train_predictions = []
        for u, v, m_val, d in X_train:
            p = self._formula(u, v, m_val, d, self.params)
            train_predictions.append(p)

        train_metrics = self._compute_metrics(train_predictions, y_train)

        return {
            'train_metrics': train_metrics,
            'success': True,  # Always true if we completed all bootstraps
            'iterations': self.n_random_starts * self.bootstrap_samples  # Total optimization runs
        }

    def _validate_against_empirical_200(self, empirical_200: Dict[Tuple[int, int], float],
                                         m: int, density: float,
                                         degree_stratified: bool = False,
                                         small_graph_mode: bool = False) -> Dict[str, float]:
        """
        Enhanced validation against 200-permutation empirical with optional degree stratification.

        Parameters
        ----------
        empirical_200 : dict
            200-permutation empirical frequencies
        m : int
            Number of edges
        density : float
            Graph density
        degree_stratified : bool
            Whether to include degree-stratified metrics
        small_graph_mode : bool
            Whether to use small graph optimizations

        Returns
        -------
        metrics : dict
            Validation metrics, optionally including degree-stratified results
        """

        val_predictions = []
        val_targets = []
        degree_data = []

        # Initialize degree analyzer if needed
        if degree_stratified:
            from degree_analysis import DegreeAnalyzer
            analyzer = DegreeAnalyzer(small_graph_mode=small_graph_mode)

        for (u, v), freq_200 in empirical_200.items():
            # Use ensemble prediction if ensemble_size > 1
            if self.ensemble_size > 1 and len(self.ensemble_params) > 1:
                # Average predictions from ensemble members
                ensemble_preds = []
                for params in self.ensemble_params:
                    p = self._formula(u, v, m, density, params)
                    ensemble_preds.append(p)
                p_pred = np.mean(ensemble_preds)
            else:
                # Single model prediction
                p_pred = self._formula(u, v, m, density, self.params)

            val_predictions.append(p_pred)
            val_targets.append(freq_200)

            # Store degree information if needed
            if degree_stratified:
                u_category = analyzer.categorize_degrees(np.array([u]))[0]
                v_category = analyzer.categorize_degrees(np.array([v]))[0]
                degree_combination = analyzer.create_degree_combination_labels(
                    np.array([u_category]), np.array([v_category])
                )[0]

                degree_data.append({
                    'u': u, 'v': v,
                    'predicted': p_pred,
                    'empirical': freq_200,
                    'degree_combination': degree_combination
                })

        # Compute overall metrics
        overall_metrics = self._compute_metrics(val_predictions, val_targets)

        # Add degree-stratified metrics if requested
        if degree_stratified and degree_data:
            degree_df = pd.DataFrame(degree_data)

            # Compute metrics by degree combination
            degree_metrics = {}
            for combo in degree_df['degree_combination'].unique():
                combo_data = degree_df[degree_df['degree_combination'] == combo]
                if len(combo_data) >= 2:  # Need at least 2 points
                    combo_metrics = self._compute_metrics(
                        combo_data['predicted'].values,
                        combo_data['empirical'].values
                    )
                    degree_metrics[combo] = combo_metrics

            overall_metrics['degree_stratified'] = degree_metrics
            overall_metrics['degree_combinations_count'] = len(degree_metrics)

            # Compute degree-stratified summary statistics
            if degree_metrics:
                degree_correlations = [metrics['correlation'] for metrics in degree_metrics.values()
                                     if not np.isnan(metrics['correlation'])]
                degree_maes = [metrics['mae'] for metrics in degree_metrics.values()]

                overall_metrics['degree_correlation_mean'] = np.mean(degree_correlations) if degree_correlations else np.nan
                overall_metrics['degree_correlation_std'] = np.std(degree_correlations) if degree_correlations else np.nan
                overall_metrics['degree_mae_mean'] = np.mean(degree_maes) if degree_maes else np.nan
                overall_metrics['degree_mae_std'] = np.std(degree_maes) if degree_maes else np.nan

        return overall_metrics

    def _compute_baseline_metrics(self, empirical_200: Dict[Tuple[int, int], float],
                                    m: int) -> Dict[str, float]:
        """Compute baseline (current analytical) metrics"""

        baseline_predictions = []
        targets = []

        for (u, v), freq_200 in empirical_200.items():
            p_baseline = self._current_analytical(u, v, m)
            baseline_predictions.append(p_baseline)
            targets.append(freq_200)

        return self._compute_metrics(baseline_predictions, targets)

    def predict_all_edges(self, graph_name: str, data_dir: Path) -> pd.DataFrame:
        """
        Generate predictions for all possible source-target combinations.

        Parameters
        ----------
        graph_name : str
            Graph name
        data_dir : Path
            Data directory

        Returns
        -------
        predictions_df : pd.DataFrame
            Columns: source_index, target_index, source_degree, target_degree,
                    learned_probability, analytical_probability
        """

        if self.params is None:
            raise ValueError("Must train model before making predictions. Call find_minimum_permutations first.")

        # Load graph
        edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{graph_name}.sparse.npz'
        edge_matrix = sp.load_npz(edge_file)
        edge_matrix, source_mapping, target_mapping = filter_zero_degree_nodes(edge_matrix)

        stats = self.graph_stats.get(graph_name)
        if stats is None:
            # Compute stats
            m = edge_matrix.nnz
            n_sources, n_targets = edge_matrix.shape
            density = m / (n_sources * n_targets)
            stats = {'m': m, 'density': density, 'n_sources': n_sources, 'n_targets': n_targets}

        m = stats['m']
        density = stats['density']

        # Get degrees
        source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
        target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

        # Generate predictions for all combinations
        predictions = []

        for i in range(len(source_degrees)):
            for j in range(len(target_degrees)):
                u = source_degrees[i]
                v = target_degrees[j]

                # Use ensemble prediction if ensemble_size > 1
                if self.ensemble_size > 1 and len(self.ensemble_params) > 1:
                    # Average predictions from ensemble members
                    ensemble_preds = []
                    for params in self.ensemble_params:
                        p = self._formula(u, v, m, density, params)
                        ensemble_preds.append(p)
                    p_learned = np.mean(ensemble_preds)
                else:
                    # Single model prediction
                    p_learned = self._formula(u, v, m, density, self.params)

                p_analytical = self._current_analytical(u, v, m)

                predictions.append({
                    'source_index': source_mapping[i],
                    'target_index': target_mapping[j],
                    'source_degree': int(u),
                    'target_degree': int(v),
                    'learned_probability': p_learned,
                    'analytical_probability': p_analytical
                })

        return pd.DataFrame(predictions)

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save learned parameters and results"""

        output_dir.mkdir(parents=True, exist_ok=True)

        graph_name = results['graph_name']

        # Save parameters
        params_dict = {
            name: float(value)
            for name, value in zip(self.param_names, results['best_params'])
        }
        params_dict['N_min'] = results['N_min']
        params_dict['graph_name'] = graph_name
        params_dict['graph_stats'] = results['graph_stats']

        params_file = output_dir / f'{graph_name}_learned_parameters.json'
        with open(params_file, 'w') as f:
            json.dump(params_dict, f, indent=2)

        print(f'Parameters saved to: {params_file}')

        # Save metrics
        metrics_data = {
            'N_min': results['N_min'],
            'final_metrics': results['final_metrics'],
            'baseline_metrics': results['baseline_metrics'],
            'convergence_curve': [
                {
                    'N': r['N'],
                    'train_mae': r['train_metrics']['mae'],
                    'train_correlation': r['train_metrics']['correlation'],
                    'val_mae': r['val_metrics']['mae'],
                    'val_correlation': r['val_metrics']['correlation']
                }
                for r in results['convergence_curve']
            ]
        }

        metrics_file = output_dir / f'{graph_name}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(f'Metrics saved to: {metrics_file}')

    def _plot_convergence_curve(self, results_by_N: List[Dict], N_min: int,
                                 graph_name: str, results_dir: Path):
        """Plot validation metric vs N permutations"""

        N_values = [r['N'] for r in results_by_N]
        val_mae = [r['val_metrics']['mae'] for r in results_by_N]
        val_corr = [r['val_metrics']['correlation'] for r in results_by_N]
        train_mae = [r['train_metrics']['mae'] for r in results_by_N]
        train_corr = [r['train_metrics']['correlation'] for r in results_by_N]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # MAE plot
        axes[0].plot(N_values, train_mae, 'o-', label='Training', linewidth=2, markersize=8)
        axes[0].plot(N_values, val_mae, 's-', label='Validation (200-perm)', linewidth=2, markersize=8)
        axes[0].axvline(N_min, color='red', linestyle='--', linewidth=2, label=f'N_min = {N_min}')
        axes[0].set_xlabel('Number of Training Permutations', fontsize=12)
        axes[0].set_ylabel('MAE', fontsize=12)
        axes[0].set_title(f'{graph_name}: MAE vs N Permutations', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')

        # Correlation plot
        axes[1].plot(N_values, train_corr, 'o-', label='Training', linewidth=2, markersize=8)
        axes[1].plot(N_values, val_corr, 's-', label='Validation (200-perm)', linewidth=2, markersize=8)
        axes[1].axvline(N_min, color='red', linestyle='--', linewidth=2, label=f'N_min = {N_min}')
        axes[1].set_xlabel('Number of Training Permutations', fontsize=12)
        axes[1].set_ylabel('Correlation', fontsize=12)
        axes[1].set_title(f'{graph_name}: Correlation vs N Permutations', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')

        plt.tight_layout()

        output_dir = results_dir / 'learned_analytical'
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_file = output_dir / f'{graph_name}_convergence_curve.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f'Convergence plot saved to: {plot_file}')
        plt.close()

    def _formula(self, u: float, v: float, m: float, density: float,
                 params: np.ndarray) -> float:
        """
        Parameterized analytical function (dispatches to specific formula type).
        """
        if self.formula_type == 'original':
            return self._formula_original(u, v, m, density, params)
        elif self.formula_type == 'extended':
            return self._formula_extended(u, v, m, density, params)
        elif self.formula_type == 'polynomial':
            return self._formula_polynomial(u, v, m, density, params)
        else:
            raise ValueError(f"Unknown formula type: {self.formula_type}")

    def _formula_original(self, u: float, v: float, m: float, density: float,
                          params: np.ndarray) -> float:
        """
        Original parameterized analytical function (9 parameters).

        P = α × (u^β × v^γ) / (δ + ε×m + ζ×(u×v)^η + θ×density^κ)
        """
        α, β, γ, δ, ε, ζ, η, θ, κ = params

        numerator = α * (u**β * v**γ)
        denominator = δ + ε*m + ζ*(u*v)**η + θ*density**κ

        if denominator < 1e-10:
            return 0.0

        p = numerator / denominator
        return np.clip(p, 0.0, 1.0)

    def _formula_extended(self, u: float, v: float, m: float, density: float,
                          params: np.ndarray) -> float:
        """
        Extended formula with log terms (11 parameters).

        P = α × (u^β × v^γ) / (δ + ε×m + ζ×(u×v)^η + θ×density^κ + λ×log(u+1) + μ×log(v+1))
        """
        if len(params) == 9:
            # Fallback if called with 9 params
            return self._formula_original(u, v, m, density, params)

        α, β, γ, δ, ε, ζ, η, θ, κ, λ, μ = params

        numerator = α * (u**β * v**γ)
        denominator = δ + ε*m + ζ*(u*v)**η + θ*density**κ + λ*np.log(u + 1) + μ*np.log(v + 1)

        if denominator < 1e-10:
            return 0.0

        p = numerator / denominator
        return np.clip(p, 0.0, 1.0)

    def _formula_polynomial(self, u: float, v: float, m: float, density: float,
                            params: np.ndarray) -> float:
        """
        Polynomial formula (9 parameters).

        P = (α×u×v + β×u + γ×v + δ) / (ε×m + ζ×u×v + η×(u+v) + θ + ι×density)
        """
        α, β, γ, δ, ε, ζ, η, θ, ι = params

        numerator = α*u*v + β*u + γ*v + δ
        denominator = ε*m + ζ*u*v + η*(u+v) + θ + ι*density

        if denominator < 1e-10:
            return 0.0

        p = numerator / denominator
        return np.clip(p, 0.0, 1.0)

    def _current_analytical(self, u: float, v: float, m: float) -> float:
        """Current analytical approximation"""
        uv = u * v
        denominator = np.sqrt(uv**2 + (m - u - v + 1)**2)
        return uv / denominator if denominator > 0 else 0.0

    def _compute_metrics(self, predictions: List[float], targets: List[float]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions = np.array(predictions)
        targets = np.array(targets)

        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))

        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        correlation = np.corrcoef(predictions, targets)[0, 1]

        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'correlation': correlation}

    def _compute_empirical_from_permutations(self, graph_name: str, perm_ids: List[int],
                                              data_dir: Path) -> Dict[Tuple[int, int], float]:
        """
        Compute empirical frequencies from specific permutations.

        Frequency = (# permutations with edge) / (# nodes with that degree combination)
        """
        empirical_counts = {}  # Count occurrences across permutations
        degree_pair_counts = {}  # Count how many (i,j) pairs have each (u,v)

        # Get reference degrees from perm 000 (UNFILTERED for consistent indexing)
        ref_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{graph_name}.sparse.npz'
        ref_matrix_orig = sp.load_npz(ref_file)

        # Calculate degrees from ORIGINAL matrix
        source_degrees_orig = np.array(ref_matrix_orig.sum(axis=1)).flatten()
        target_degrees_orig = np.array(ref_matrix_orig.sum(axis=0)).flatten()

        # Count how many (source, target) pairs have each (u, v) degree combination
        for i in range(len(source_degrees_orig)):
            for j in range(len(target_degrees_orig)):
                u = int(source_degrees_orig[i])
                v = int(target_degrees_orig[j])

                # Only consider pairs where both have non-zero degree
                if u > 0 and v > 0:
                    key = (u, v)
                    degree_pair_counts[key] = degree_pair_counts.get(key, 0) + 1

        # Count edge occurrences across permutations
        for perm_id in perm_ids:
            perm_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{graph_name}.sparse.npz'
            edge_matrix_orig = sp.load_npz(perm_file)

            # Get edges from original unfiltered matrix
            for i, j in zip(*edge_matrix_orig.nonzero()):
                u = int(source_degrees_orig[i])
                v = int(target_degrees_orig[j])

                # Only count edges where both nodes have non-zero degree
                if u > 0 and v > 0:
                    key = (u, v)
                    empirical_counts[key] = empirical_counts.get(key, 0) + 1

        # Compute frequencies: (total edges with this degree pair) / (N_perms × N_pairs_with_this_degree)
        empirical = {}
        N = len(perm_ids)

        for key in empirical_counts:
            # Frequency = (# edges observed) / (# permutations × # possible pairs)
            empirical[key] = empirical_counts[key] / (N * degree_pair_counts[key])

        return empirical

    def _load_200_perm_empirical(self, graph_name: str, results_dir: Path) -> Dict[Tuple[int, int], float]:
        """Load precomputed 200-permutation empirical frequencies"""
        file_path = results_dir / 'empirical_edge_frequencies' / f'edge_frequency_by_degree_{graph_name}.csv'
        df = pd.read_csv(file_path)

        empirical = {}
        for _, row in df.iterrows():
            u = int(row['source_degree'])
            v = int(row['target_degree'])
            freq = float(row['frequency'])
            empirical[(u, v)] = freq

        return empirical

    def analyze_residuals(self, empirical_200: Dict[Tuple[int, int], float],
                         m: int, density: float, results_dir: Path,
                         graph_name: str, small_graph_mode: bool = False):
        """
        Enhanced residual analysis with degree-based error decomposition.

        Parameters
        ----------
        empirical_200 : dict
            200-permutation empirical frequencies
        m : int
            Number of edges
        density : float
            Graph density
        results_dir : Path
            Directory to save plots
        graph_name : str
            Graph name for plot titles
        small_graph_mode : bool
            Whether to use small graph optimizations
        """
        if self.params is None:
            raise ValueError("Must train model first")

        # Import degree analysis utilities
        from degree_analysis import DegreeAnalyzer

        # Initialize degree analyzer
        analyzer = DegreeAnalyzer(small_graph_mode=small_graph_mode)

        # Compute predictions and residuals
        residuals_data = []

        for (u, v), freq_empirical in empirical_200.items():
            # Get predictions
            if self.ensemble_size > 1 and len(self.ensemble_params) > 1:
                ensemble_preds = [self._formula(u, v, m, density, p) for p in self.ensemble_params]
                p_learned = np.mean(ensemble_preds)
            else:
                p_learned = self._formula(u, v, m, density, self.params)

            p_analytical = self._current_analytical(u, v, m)

            # Enhanced degree categorization
            u_category = analyzer.categorize_degrees(np.array([u]))[0]
            v_category = analyzer.categorize_degrees(np.array([v]))[0]
            degree_combination = analyzer.create_degree_combination_labels(
                np.array([u_category]), np.array([v_category])
            )[0]

            residuals_data.append({
                'u': u,
                'v': v,
                'uv_product': u * v,
                'empirical': freq_empirical,
                'learned': p_learned,
                'analytical': p_analytical,
                'residual_learned': p_learned - freq_empirical,
                'residual_analytical': p_analytical - freq_empirical,
                'relative_error_learned': (p_learned - freq_empirical) / freq_empirical if freq_empirical > 0 else 0,
                'relative_error_analytical': (p_analytical - freq_empirical) / freq_empirical if freq_empirical > 0 else 0,
                'degree_category_old': self._categorize_degrees(u, v),  # Keep old for compatibility
                'source_degree_category': str(u_category),
                'target_degree_category': str(v_category),
                'degree_combination': degree_combination
            })

        residuals_df = pd.DataFrame(residuals_data)

        # Create enhanced residual plots with degree-based analysis
        fig, axes = plt.subplots(3, 2, figsize=(18, 20))

        # Plot 1: Residuals vs u×v product (Enhanced)
        ax = axes[0, 0]
        ax.scatter(residuals_df['uv_product'], residuals_df['residual_learned'],
                  alpha=0.5, s=20, label='Learned', color='green')
        ax.scatter(residuals_df['uv_product'], residuals_df['residual_analytical'],
                  alpha=0.5, s=20, label='Current Analytical', color='orange')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('u × v (Degree Product)', fontsize=12)
        ax.set_ylabel('Residual (Predicted - Empirical)', fontsize=12)
        ax.set_title(f'{graph_name} - Residuals vs Degree Product', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log')

        # Plot 2: Enhanced residuals by degree combination
        ax = axes[0, 1]
        combinations = sorted(residuals_df['degree_combination'].unique())
        learned_by_comb = [residuals_df[residuals_df['degree_combination'] == comb]['residual_learned'].values
                          for comb in combinations]
        analytical_by_comb = [residuals_df[residuals_df['degree_combination'] == comb]['residual_analytical'].values
                             for comb in combinations]

        positions = np.arange(len(combinations))
        width = 0.35

        bp1 = ax.boxplot(learned_by_comb, positions=positions - width/2, widths=width,
                        patch_artist=True, showmeans=True)
        bp2 = ax.boxplot(analytical_by_comb, positions=positions + width/2, widths=width,
                        patch_artist=True, showmeans=True)

        for patch in bp1['boxes']:
            patch.set_facecolor('green')
            patch.set_alpha(0.6)
        for patch in bp2['boxes']:
            patch.set_facecolor('orange')
            patch.set_alpha(0.6)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(combinations, rotation=45, ha='right')
        ax.set_ylabel('Residual', fontsize=12)
        ax.set_title(f'{graph_name} - Residuals by Degree Combination', fontsize=14, fontweight='bold')
        ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Learned', 'Current Analytical'])
        ax.grid(axis='y', alpha=0.3)

        # Plot 3: Relative error analysis by degree combination
        ax = axes[1, 0]
        relative_errors = residuals_df.groupby('degree_combination').agg({
            'relative_error_learned': ['mean', 'std'],
            'relative_error_analytical': ['mean', 'std']
        })

        x_pos = np.arange(len(combinations))
        ax.bar(x_pos - 0.2, relative_errors[('relative_error_learned', 'mean')],
               width=0.4, label='Learned', alpha=0.7, color='green')
        ax.bar(x_pos + 0.2, relative_errors[('relative_error_analytical', 'mean')],
               width=0.4, label='Analytical', alpha=0.7, color='orange')

        ax.set_xlabel('Degree Combination', fontsize=12)
        ax.set_ylabel('Mean Relative Error', fontsize=12)
        ax.set_title(f'{graph_name} - Relative Error by Degree Combination', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(combinations, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Plot 4: Error magnitude heatmap
        ax = axes[1, 1]
        error_pivot = residuals_df.pivot_table(
            values='residual_learned',
            index='source_degree_category',
            columns='target_degree_category',
            aggfunc=lambda x: np.sqrt(np.mean(x**2))  # RMSE
        )

        import seaborn as sns
        sns.heatmap(error_pivot, annot=True, fmt='.4f', cmap='Reds', ax=ax,
                   cbar_kws={'label': 'RMSE'})
        ax.set_title(f'{graph_name} - RMSE Heatmap by Degree Categories', fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Degree Category', fontsize=12)
        ax.set_ylabel('Source Degree Category', fontsize=12)

        # Plot 5: Q-Q plot for learned residuals
        ax = axes[2, 0]
        from scipy import stats
        stats.probplot(residuals_df['residual_learned'], dist="norm", plot=ax)
        ax.set_title(f'{graph_name} - Q-Q Plot (Learned Formula)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # Plot 6: Bias analysis by degree combination
        ax = axes[2, 1]
        bias_analysis = residuals_df.groupby('degree_combination').agg({
            'residual_learned': 'mean',
            'residual_analytical': 'mean'
        })

        bias_analysis.plot(kind='bar', ax=ax, color=['green', 'orange'], alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Degree Combination', fontsize=12)
        ax.set_ylabel('Mean Residual (Bias)', fontsize=12)
        ax.set_title(f'{graph_name} - Bias by Degree Combination', fontsize=14, fontweight='bold')
        ax.legend(['Learned', 'Analytical'])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        output_path = results_dir / f'enhanced_residual_analysis_{graph_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Enhanced residual analysis saved to: {output_path}")

        # Generate detailed degree-based error metrics
        degree_error_metrics = residuals_df.groupby('degree_combination').agg({
            'residual_learned': ['count', 'mean', 'std', lambda x: np.sqrt(np.mean(x**2))],
            'residual_analytical': ['mean', 'std', lambda x: np.sqrt(np.mean(x**2))],
            'relative_error_learned': ['mean', 'std'],
            'relative_error_analytical': ['mean', 'std'],
            'empirical': 'mean',
            'learned': 'mean',
            'analytical': 'mean'
        }).round(6)

        degree_error_metrics.columns = [
            'n_samples', 'bias_learned', 'std_learned', 'rmse_learned',
            'bias_analytical', 'std_analytical', 'rmse_analytical',
            'rel_error_mean_learned', 'rel_error_std_learned',
            'rel_error_mean_analytical', 'rel_error_std_analytical',
            'mean_empirical', 'mean_learned', 'mean_analytical'
        ]

        # Save detailed metrics
        metrics_file = results_dir / f'degree_based_error_metrics_{graph_name}.csv'
        degree_error_metrics.to_csv(metrics_file)
        print(f"Degree-based error metrics saved to: {metrics_file}")

        # Print enhanced summary statistics
        print(f"\n{'='*80}")
        print(f"ENHANCED RESIDUAL ANALYSIS SUMMARY - {graph_name}")
        print(f"{'='*80}")

        print(f"\nOverall Performance:")
        print(f"  Learned Formula:")
        print(f"    Mean residual: {residuals_df['residual_learned'].mean():.6f}")
        print(f"    Std residual: {residuals_df['residual_learned'].std():.6f}")
        print(f"    RMSE: {np.sqrt(np.mean(residuals_df['residual_learned']**2)):.6f}")

        print(f"  Current Analytical:")
        print(f"    Mean residual: {residuals_df['residual_analytical'].mean():.6f}")
        print(f"    Std residual: {residuals_df['residual_analytical'].std():.6f}")
        print(f"    RMSE: {np.sqrt(np.mean(residuals_df['residual_analytical']**2)):.6f}")

        print(f"\nDegree-Based Error Analysis:")
        print(f"  Degree combinations analyzed: {len(degree_error_metrics)}")

        # Identify best and worst performing degree combinations
        best_combination = degree_error_metrics['rmse_learned'].idxmin()
        worst_combination = degree_error_metrics['rmse_learned'].idxmax()

        print(f"  Best performing combination: {best_combination}")
        print(f"    RMSE: {degree_error_metrics.loc[best_combination, 'rmse_learned']:.6f}")
        print(f"    Samples: {int(degree_error_metrics.loc[best_combination, 'n_samples'])}")

        print(f"  Worst performing combination: {worst_combination}")
        print(f"    RMSE: {degree_error_metrics.loc[worst_combination, 'rmse_learned']:.6f}")
        print(f"    Samples: {int(degree_error_metrics.loc[worst_combination, 'n_samples'])}")

        # Identify bias patterns
        high_bias_combinations = degree_error_metrics[
            np.abs(degree_error_metrics['bias_learned']) > 0.01
        ]

        if len(high_bias_combinations) > 0:
            print(f"\n  High bias combinations (|bias| > 0.01):")
            for combo in high_bias_combinations.index:
                bias = high_bias_combinations.loc[combo, 'bias_learned']
                print(f"    {combo}: {bias:+.6f}")

        return residuals_df, degree_error_metrics

    def _categorize_degrees(self, u: int, v: int) -> str:
        """Categorize degree pairs for analysis"""
        uv = u * v
        if uv < 10:
            return 'Very Low (<10)'
        elif uv < 100:
            return 'Low (10-100)'
        elif uv < 1000:
            return 'Medium (100-1k)'
        elif uv < 10000:
            return 'High (1k-10k)'
        else:
            return 'Very High (>10k)'

    def analyze_parameter_importance(self, empirical_200: Dict[Tuple[int, int], float],
                                    m: int, density: float, graph_name: str,
                                    results_dir: Path):
        """
        Phase 4.11: Analyze parameter importance via sensitivity analysis.

        Perturbs each parameter by ±10% and measures impact on predictions.
        """
        if self.params is None:
            raise ValueError("Must train model first")

        print(f"\n{'='*60}")
        print(f"PARAMETER IMPORTANCE ANALYSIS - {graph_name}")
        print(f"{'='*60}")

        # Use averaged params for analysis
        params = self.params.copy()

        # Baseline predictions
        baseline_preds = []
        targets = []
        for (u, v), freq in empirical_200.items():
            p = self._formula(u, v, m, density, params)
            baseline_preds.append(p)
            targets.append(freq)

        baseline_preds = np.array(baseline_preds)
        targets = np.array(targets)
        baseline_corr = np.corrcoef(baseline_preds, targets)[0, 1]

        # Sensitivity analysis: perturb each parameter
        sensitivities = []

        for i, param_name in enumerate(self.param_names):
            # Perturb +10%
            params_plus = params.copy()
            params_plus[i] *= 1.1

            preds_plus = []
            for (u, v), _ in empirical_200.items():
                p = self._formula(u, v, m, density, params_plus)
                preds_plus.append(p)
            preds_plus = np.array(preds_plus)
            corr_plus = np.corrcoef(preds_plus, targets)[0, 1]

            # Perturb -10%
            params_minus = params.copy()
            params_minus[i] *= 0.9

            preds_minus = []
            for (u, v), _ in empirical_200.items():
                p = self._formula(u, v, m, density, params_minus)
                preds_minus.append(p)
            preds_minus = np.array(preds_minus)
            corr_minus = np.corrcoef(preds_minus, targets)[0, 1]

            # Sensitivity = average change in correlation
            sensitivity = (abs(corr_plus - baseline_corr) + abs(corr_minus - baseline_corr)) / 2

            sensitivities.append({
                'parameter': param_name,
                'value': params[i],
                'sensitivity': sensitivity,
                'corr_plus': corr_plus,
                'corr_minus': corr_minus
            })

        sens_df = pd.DataFrame(sensitivities)
        sens_df = sens_df.sort_values('sensitivity', ascending=False)

        # Print results
        print(f"\nBaseline correlation: {baseline_corr:.6f}\n")
        print(f"{'Parameter':<12} {'Value':<12} {'Sensitivity':<12} {'Impact'}")
        print(f"{'-'*60}")
        for _, row in sens_df.iterrows():
            impact = 'HIGH' if row['sensitivity'] > 0.01 else 'MEDIUM' if row['sensitivity'] > 0.001 else 'LOW'
            print(f"{row['parameter']:<12} {row['value']:<12.4f} {row['sensitivity']:<12.6f} {impact}")

        # Visualize
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = ['red' if s > 0.01 else 'orange' if s > 0.001 else 'green'
                 for s in sens_df['sensitivity']]

        ax.barh(sens_df['parameter'], sens_df['sensitivity'], color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Sensitivity (Avg Δ Correlation)', fontsize=12)
        ax.set_ylabel('Parameter', fontsize=12)
        ax.set_title(f'{graph_name} - Parameter Importance ({self.formula_type})',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='High (>0.01)'),
            Patch(facecolor='orange', alpha=0.7, label='Medium (>0.001)'),
            Patch(facecolor='green', alpha=0.7, label='Low (≤0.001)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        output_path = results_dir / f'parameter_importance_{graph_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nParameter importance plot saved to: {output_path}")

        return sens_df

    def create_comparison_plots(self, results_by_phase: Dict[str, Dict],
                               results_dir: Path, graph_name: str):
        """
        Phase 4.12: Create side-by-side comparison visualizations across phases.

        Parameters
        ----------
        results_by_phase : dict
            {
                'baseline': {...},
                'phase1': {...},
                'phase2_original': {...},
                'phase2_extended': {...},
                'phase2_polynomial': {...},
                'phase3': {...}
            }
        results_dir : Path
            Output directory
        graph_name : str
            Graph name
        """
        print(f"\n{'='*60}")
        print(f"CREATING COMPARISON VISUALIZATIONS - {graph_name}")
        print(f"{'='*60}")

        # Extract correlations
        phase_names = list(results_by_phase.keys())
        correlations = [results_by_phase[phase]['correlation'] for phase in phase_names]
        maes = [results_by_phase[phase]['mae'] for phase in phase_names]

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Correlation comparison
        ax = axes[0]
        colors = ['orange'] + ['green'] * (len(phase_names) - 1)
        bars = ax.bar(range(len(phase_names)), correlations, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{corr:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(range(len(phase_names)))
        ax.set_xticklabels(phase_names, rotation=45, ha='right')
        ax.set_ylabel('Correlation with Empirical', fontsize=12)
        ax.set_title(f'{graph_name} - Correlation Improvement by Phase', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([min(correlations) * 0.95, 1.0])

        # Plot 2: MAE comparison
        ax = axes[1]
        bars = ax.bar(range(len(phase_names)), maes, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for i, (bar, mae) in enumerate(zip(bars, maes)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                   f'{mae:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(range(len(phase_names)))
        ax.set_xticklabels(phase_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Absolute Error', fontsize=12)
        ax.set_title(f'{graph_name} - MAE Reduction by Phase', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = results_dir / f'phase_comparison_{graph_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comparison plot saved to: {output_path}")

        # Print summary table
        print(f"\n{'Phase':<20} {'Correlation':<15} {'MAE':<15} {'vs Baseline'}")
        print(f"{'-'*70}")
        baseline_corr = correlations[0]
        baseline_mae = maes[0]

        for phase, corr, mae in zip(phase_names, correlations, maes):
            if phase == 'baseline':
                vs_baseline = '-'
            else:
                corr_improvement = (corr - baseline_corr) / baseline_corr * 100
                mae_improvement = (baseline_mae - mae) / baseline_mae * 100
                vs_baseline = f'+{corr_improvement:.1f}% corr, +{mae_improvement:.1f}% MAE'

            print(f"{phase:<20} {corr:<15.6f} {mae:<15.6f} {vs_baseline}")
