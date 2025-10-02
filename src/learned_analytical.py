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

    def __init__(self):
        self.params = None
        self.graph_stats = {}
        self.convergence_history = []
        self.param_names = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'κ']

    def find_minimum_permutations(
        self,
        graph_name: str,
        data_dir: Path,
        results_dir: Path,
        N_candidates: List[int] = None,
        convergence_threshold: float = 0.02,
        target_metric: str = 'correlation',
        min_metric_value: float = 0.95
    ) -> Dict[str, Any]:
        """
        Automatically find minimum N permutations needed for accurate learning.

        Strategy:
        1. Train on increasing N: [2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
        2. For each N, validate against 200-perm empirical
        3. Stop when:
           a) Validation metric plateaus (improvement < threshold)
           b) Validation metric exceeds target

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

        Returns
        -------
        results : dict
            {
                'N_min': minimum permutations needed,
                'best_params': learned parameters at N_min,
                'convergence_curve': performance vs N,
                'final_metrics': validation metrics at N_min
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

            # Validate against 200-perm empirical
            val_metrics = self._validate_against_empirical_200(
                empirical_200, m, density
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

            # L2 regularization
            l2_penalty = 0.001 * np.sum(params ** 2)

            return mse + l2_penalty

        # Initial parameters
        initial_params = np.array([1.0, 1.0, 1.0, 1.0, 1e-6, 1.0, 2.0, 0.0, 1.0])

        # Bounds
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

        # Optimize with L-BFGS-B
        result = minimize(
            loss_function,
            x0=initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
        )

        self.params = result.x

        # Compute training metrics
        train_predictions = []
        for u, v, m_val, d in X_train:
            p = self._formula(u, v, m_val, d, self.params)
            train_predictions.append(p)

        train_metrics = self._compute_metrics(train_predictions, y_train)

        return {
            'train_metrics': train_metrics,
            'success': result.success,
            'iterations': result.nit
        }

    def _validate_against_empirical_200(self, empirical_200: Dict[Tuple[int, int], float],
                                         m: int, density: float) -> Dict[str, float]:
        """Validate learned formula against 200-permutation empirical"""

        val_predictions = []
        val_targets = []

        for (u, v), freq_200 in empirical_200.items():
            p_pred = self._formula(u, v, m, density, self.params)
            val_predictions.append(p_pred)
            val_targets.append(freq_200)

        return self._compute_metrics(val_predictions, val_targets)

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
        """Parameterized analytical function"""
        α, β, γ, δ, ε, ζ, η, θ, κ = params

        numerator = α * (u**β * v**γ)
        denominator = δ + ε*m + ζ*(u*v)**η + θ*density**κ

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
        """Compute empirical frequencies from specific permutations"""
        empirical = {}

        # Get reference degrees from perm 000
        ref_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{graph_name}.sparse.npz'
        ref_matrix = sp.load_npz(ref_file)
        ref_matrix, _, _ = filter_zero_degree_nodes(ref_matrix)

        source_degrees = np.array(ref_matrix.sum(axis=1)).flatten()
        target_degrees = np.array(ref_matrix.sum(axis=0)).flatten()

        for perm_id in perm_ids:
            perm_file = data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{graph_name}.sparse.npz'
            edge_matrix = sp.load_npz(perm_file)
            edge_matrix, source_map, target_map = filter_zero_degree_nodes(edge_matrix)

            # Map back to reference degrees
            for i, j in zip(*edge_matrix.nonzero()):
                orig_i = source_map[i]
                orig_j = target_map[j]
                u = int(source_degrees[orig_i])
                v = int(target_degrees[orig_j])
                key = (u, v)
                empirical[key] = empirical.get(key, 0) + 1

        # Convert to frequencies
        N = len(perm_ids)
        for key in empirical:
            empirical[key] /= N

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
