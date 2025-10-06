"""
Degree-Based Error Analysis Utilities

This module provides utilities for analyzing model performance and errors
stratified by node degree combinations. Designed for both local testing
on small graphs and HPC deployment on full datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class DegreeAnalyzer:
    """
    Analyzes model performance and errors stratified by node degree combinations.

    Key Features:
    - Configurable degree binning for different graph sizes
    - Comprehensive error metrics by degree category
    - Visualization tools for degree-based analysis
    - Scalable design for both local and HPC execution
    """

    def __init__(self,
                 degree_bins: Optional[List[int]] = None,
                 small_graph_mode: bool = True):
        """
        Initialize DegreeAnalyzer.

        Parameters
        ----------
        degree_bins : List[int], optional
            Custom degree bin edges. If None, uses adaptive binning.
        small_graph_mode : bool
            If True, uses settings optimized for small graphs (<10k edges)
        """
        self.small_graph_mode = small_graph_mode

        if degree_bins is None:
            if small_graph_mode:
                # Optimized for small graphs
                self.degree_bins = [1, 5, 20, 100, np.inf]
                self.degree_labels = ['Very Low (1-4)', 'Low (5-19)', 'Medium (20-99)', 'High (100+)']
            else:
                # Full-scale analysis
                self.degree_bins = [1, 10, 100, 1000, np.inf]
                self.degree_labels = ['Low (1-9)', 'Medium (10-99)', 'High (100-999)', 'Hub (1000+)']
        else:
            self.degree_bins = degree_bins
            self.degree_labels = [f'Bin_{i}' for i in range(len(degree_bins)-1)]

    def load_graph_degrees(self, edge_type: str, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load source and target degrees for a given edge type.

        Parameters
        ----------
        edge_type : str
            Edge type identifier (e.g., 'CtD', 'AeG')
        data_dir : Path
            Path to data directory

        Returns
        -------
        source_degrees : np.ndarray
            Source node degrees
        target_degrees : np.ndarray
            Target node degrees
        """
        edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

        if not edge_file.exists():
            raise FileNotFoundError(f"Edge file not found: {edge_file}")

        edge_matrix = sp.load_npz(edge_file)

        # Calculate degrees
        source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
        target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

        return source_degrees, target_degrees

    def categorize_degrees(self, degrees: np.ndarray) -> np.ndarray:
        """
        Categorize degrees into bins.

        Parameters
        ----------
        degrees : np.ndarray
            Array of degree values

        Returns
        -------
        categories : np.ndarray
            Array of category labels
        """
        return pd.cut(degrees, bins=self.degree_bins, labels=self.degree_labels,
                     include_lowest=True, right=False)

    def create_degree_combination_labels(self, source_categories: np.ndarray,
                                       target_categories: np.ndarray) -> np.ndarray:
        """
        Create combined degree category labels for source-target pairs.

        Parameters
        ----------
        source_categories : np.ndarray
            Source degree categories
        target_categories : np.ndarray
            Target degree categories

        Returns
        -------
        combination_labels : np.ndarray
            Combined category labels (e.g., 'Low-Medium')
        """
        combinations = []
        for src, tgt in zip(source_categories, target_categories):
            if pd.isna(src) or pd.isna(tgt):
                combinations.append('Unknown')
            else:
                # Shorten labels for combinations
                src_short = str(src).split('(')[0].strip()
                tgt_short = str(tgt).split('(')[0].strip()
                combinations.append(f'{src_short}-{tgt_short}')

        return np.array(combinations)

    def analyze_predictions_by_degree(self,
                                    predictions_df: pd.DataFrame,
                                    source_degrees: np.ndarray,
                                    target_degrees: np.ndarray,
                                    empirical_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Analyze model predictions stratified by degree combinations.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Model predictions with columns: source_index, target_index, predicted_prob
        source_degrees : np.ndarray
            Source node degrees
        target_degrees : np.ndarray
            Target node degrees
        empirical_df : pd.DataFrame, optional
            Empirical frequencies for comparison

        Returns
        -------
        analysis_df : pd.DataFrame
            Detailed analysis by degree combination
        """
        # Create degree categories for each prediction
        pred_df = predictions_df.copy()

        # Map degrees to predictions
        pred_df['source_degree'] = source_degrees[pred_df['source_index']]
        pred_df['target_degree'] = target_degrees[pred_df['target_index']]

        # Categorize degrees
        pred_df['source_category'] = self.categorize_degrees(pred_df['source_degree'])
        pred_df['target_category'] = self.categorize_degrees(pred_df['target_degree'])
        pred_df['degree_combination'] = self.create_degree_combination_labels(
            pred_df['source_category'], pred_df['target_category']
        )

        # Add empirical frequencies if available
        if empirical_df is not None:
            pred_df = pred_df.merge(
                empirical_df[['source_degree', 'target_degree', 'frequency']],
                on=['source_degree', 'target_degree'],
                how='left'
            )
            pred_df['empirical_freq'] = pred_df['frequency'].fillna(0)

        return pred_df

    def compute_degree_error_metrics(self, analysis_df: pd.DataFrame,
                                   prediction_col: str = 'predicted_prob',
                                   empirical_col: str = 'empirical_freq') -> pd.DataFrame:
        """
        Compute comprehensive error metrics by degree combination.

        Parameters
        ----------
        analysis_df : pd.DataFrame
            Analysis dataframe from analyze_predictions_by_degree
        prediction_col : str
            Column name for predictions
        empirical_col : str
            Column name for empirical frequencies

        Returns
        -------
        metrics_df : pd.DataFrame
            Error metrics by degree combination
        """
        if empirical_col not in analysis_df.columns:
            raise ValueError(f"Empirical column '{empirical_col}' not found")

        # Filter out missing empirical data
        valid_data = analysis_df.dropna(subset=[prediction_col, empirical_col])

        if len(valid_data) == 0:
            return pd.DataFrame()

        # Group by degree combination
        metrics = []

        for combination in valid_data['degree_combination'].unique():
            if combination == 'Unknown':
                continue

            subset = valid_data[valid_data['degree_combination'] == combination]

            if len(subset) < 2:  # Need at least 2 points for meaningful metrics
                continue

            pred = subset[prediction_col].values
            emp = subset[empirical_col].values

            # Calculate comprehensive metrics
            n_samples = len(subset)

            # Error metrics
            absolute_error = np.abs(pred - emp)
            squared_error = (pred - emp) ** 2
            relative_error = np.where(emp > 0, absolute_error / emp, np.nan)

            # Bias and variance
            bias = np.mean(pred - emp)
            variance = np.var(pred - emp)

            # Performance metrics
            correlation = np.corrcoef(pred, emp)[0, 1] if len(pred) > 1 else np.nan

            # Robust statistics
            mae = np.mean(absolute_error)
            rmse = np.sqrt(np.mean(squared_error))
            median_ae = np.median(absolute_error)
            q75_ae = np.percentile(absolute_error, 75)
            q95_ae = np.percentile(absolute_error, 95)

            metrics.append({
                'degree_combination': combination,
                'n_samples': n_samples,
                'mean_source_degree': subset['source_degree'].mean(),
                'mean_target_degree': subset['target_degree'].mean(),
                'mean_predicted': np.mean(pred),
                'mean_empirical': np.mean(emp),
                'bias': bias,
                'variance': variance,
                'mae': mae,
                'rmse': rmse,
                'median_ae': median_ae,
                'q75_ae': q75_ae,
                'q95_ae': q95_ae,
                'correlation': correlation,
                'mean_relative_error': np.nanmean(relative_error),
                'median_relative_error': np.nanmedian(relative_error)
            })

        return pd.DataFrame(metrics)

    def plot_error_by_degree(self, metrics_df: pd.DataFrame,
                           metric: str = 'mae',
                           title_suffix: str = '',
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create visualization of error metrics by degree combination.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Error metrics from compute_degree_error_metrics
        metric : str
            Metric to plot ('mae', 'rmse', 'correlation', etc.)
        title_suffix : str
            Additional text for plot title
        figsize : Tuple[int, int]
            Figure size

        Returns
        -------
        fig : plt.Figure
            Generated figure
        """
        if metrics_df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{metric.upper()} by Degree Combination {title_suffix}')
            return fig

        fig, ax = plt.subplots(figsize=figsize)

        # Sort by metric value for better visualization
        plot_data = metrics_df.sort_values(metric, ascending=(metric not in ['correlation']))

        # Create bar plot
        bars = ax.bar(range(len(plot_data)), plot_data[metric],
                     color=sns.color_palette('viridis', len(plot_data)),
                     alpha=0.7, edgecolor='black')

        # Customize plot
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data['degree_combination'], rotation=45, ha='right')
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} by Degree Combination {title_suffix}',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, plot_data[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10)

        # Add sample size as secondary info
        ax2 = ax.twinx()
        ax2.plot(range(len(plot_data)), plot_data['n_samples'], 'ro-', alpha=0.6,
                label='Sample Size')
        ax2.set_ylabel('Sample Size', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        return fig

    def create_degree_heatmap(self, analysis_df: pd.DataFrame,
                            value_col: str = 'predicted_prob',
                            title_suffix: str = '',
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create heatmap of values by source and target degree categories.

        Parameters
        ----------
        analysis_df : pd.DataFrame
            Analysis dataframe with degree categories
        value_col : str
            Column to visualize
        title_suffix : str
            Additional text for plot title
        figsize : Tuple[int, int]
            Figure size

        Returns
        -------
        fig : plt.Figure
            Generated figure
        """
        # Create pivot table
        pivot_data = analysis_df.groupby(['source_category', 'target_category'])[value_col].mean()
        pivot_df = pivot_data.unstack(fill_value=0)

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='viridis',
                   cbar_kws={'label': value_col}, ax=ax)

        ax.set_title(f'{value_col} by Degree Categories {title_suffix}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Degree Category', fontsize=12)
        ax.set_ylabel('Source Degree Category', fontsize=12)

        plt.tight_layout()
        return fig

    def generate_degree_analysis_report(self,
                                      edge_type: str,
                                      analysis_df: pd.DataFrame,
                                      metrics_df: pd.DataFrame,
                                      output_dir: Path) -> Dict[str, str]:
        """
        Generate comprehensive degree-based analysis report.

        Parameters
        ----------
        edge_type : str
            Edge type identifier
        analysis_df : pd.DataFrame
            Full analysis dataframe
        metrics_df : pd.DataFrame
            Error metrics dataframe
        output_dir : Path
            Output directory for files

        Returns
        -------
        file_paths : Dict[str, str]
            Dictionary of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_paths = {}

        # Save detailed analysis
        analysis_file = output_dir / f'{edge_type}_degree_analysis.csv'
        analysis_df.to_csv(analysis_file, index=False)
        file_paths['analysis'] = str(analysis_file)

        # Save metrics summary
        metrics_file = output_dir / f'{edge_type}_degree_metrics.csv'
        metrics_df.to_csv(metrics_file, index=False)
        file_paths['metrics'] = str(metrics_file)

        # Create visualizations
        if not metrics_df.empty:
            # MAE plot
            fig_mae = self.plot_error_by_degree(metrics_df, 'mae', f'- {edge_type}')
            mae_file = output_dir / f'{edge_type}_mae_by_degree.png'
            fig_mae.savefig(mae_file, dpi=300, bbox_inches='tight')
            plt.close(fig_mae)
            file_paths['mae_plot'] = str(mae_file)

            # Correlation plot
            fig_corr = self.plot_error_by_degree(metrics_df, 'correlation', f'- {edge_type}')
            corr_file = output_dir / f'{edge_type}_correlation_by_degree.png'
            fig_corr.savefig(corr_file, dpi=300, bbox_inches='tight')
            plt.close(fig_corr)
            file_paths['correlation_plot'] = str(corr_file)

        # Heatmap if we have empirical data
        if 'empirical_freq' in analysis_df.columns:
            fig_heatmap = self.create_degree_heatmap(analysis_df, 'empirical_freq',
                                                   f'- {edge_type} Empirical')
            heatmap_file = output_dir / f'{edge_type}_empirical_heatmap.png'
            fig_heatmap.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close(fig_heatmap)
            file_paths['heatmap'] = str(heatmap_file)

        return file_paths


def identify_small_graphs(data_dir: Path, max_edges: int = 10000) -> List[Dict[str, Union[str, int]]]:
    """
    Identify edge types with small graphs suitable for local testing.

    Parameters
    ----------
    data_dir : Path
        Path to data directory
    max_edges : int
        Maximum number of edges for "small" classification

    Returns
    -------
    small_graphs : List[Dict]
        List of dictionaries with edge_type and n_edges
    """
    edge_types = [
        "AdG", "AeG", "AuG", "CbG", "CcSE", "CdG", "CpD", "CrC", "CtD", "CuG",
        "DaG", "DdG", "DlA", "DpS", "DrD", "DuG", "GcG", "GiG", "GpBP", "GpCC",
        "GpMF", "GpPW", "Gr>G", "PCiC"
    ]

    small_graphs = []

    for edge_type in edge_types:
        edge_file = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

        if edge_file.exists():
            try:
                edge_matrix = sp.load_npz(edge_file)
                n_edges = edge_matrix.nnz

                if n_edges <= max_edges:
                    small_graphs.append({
                        'edge_type': edge_type,
                        'n_edges': n_edges,
                        'shape': edge_matrix.shape,
                        'density': n_edges / (edge_matrix.shape[0] * edge_matrix.shape[1])
                    })
            except Exception as e:
                print(f"Error loading {edge_type}: {e}")

    # Sort by number of edges
    small_graphs.sort(key=lambda x: x['n_edges'])

    return small_graphs


def run_degree_analysis_pipeline(edge_type: str,
                                data_dir: Path,
                                results_dir: Path,
                                output_dir: Path,
                                small_graph_mode: bool = True) -> Dict[str, str]:
    """
    Run complete degree-based analysis pipeline for a single edge type.

    Parameters
    ----------
    edge_type : str
        Edge type to analyze
    data_dir : Path
        Path to data directory
    results_dir : Path
        Path to model results directory
    output_dir : Path
        Output directory for degree analysis
    small_graph_mode : bool
        Whether to use small graph optimizations

    Returns
    -------
    file_paths : Dict[str, str]
        Generated file paths
    """
    # Initialize analyzer
    analyzer = DegreeAnalyzer(small_graph_mode=small_graph_mode)

    # Load graph degrees
    try:
        source_degrees, target_degrees = analyzer.load_graph_degrees(edge_type, data_dir)
    except FileNotFoundError:
        print(f"Edge file not found for {edge_type}")
        return {}

    # Check for model predictions file
    pred_file = results_dir / f'{edge_type}_results' / f'{edge_type}_all_model_predictions.csv'

    if not pred_file.exists():
        # Try alternative approaches if main predictions file doesn't exist
        print(f"Main predictions file not found: {pred_file}")

        # Check what files are available in the results directory
        results_subdir = results_dir / f'{edge_type}_results'
        if results_subdir.exists():
            available_files = list(results_subdir.glob('*.csv'))
            print(f"Available CSV files in {results_subdir.name}:")
            for file in available_files:
                print(f"  - {file.name}")

            # Try to generate basic analysis from available data
            return _run_basic_degree_analysis(
                edge_type, source_degrees, target_degrees,
                results_subdir, output_dir, analyzer
            )
        else:
            print(f"Results directory not found: {results_subdir}")
            return {}

    # Load model predictions
    try:
        predictions_df = pd.read_csv(pred_file)
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        return {}

    # Load empirical frequencies if available
    empirical_file = results_dir.parent / 'empirical_edge_frequencies' / f'edge_frequency_by_degree_{edge_type}.csv'
    empirical_df = None
    if empirical_file.exists():
        empirical_df = pd.read_csv(empirical_file)

    # Run analysis for each model
    all_file_paths = {}

    for model in predictions_df['Model'].unique():
        model_preds = predictions_df[predictions_df['Model'] == model].copy()

        # Analyze predictions by degree
        analysis_df = analyzer.analyze_predictions_by_degree(
            model_preds, source_degrees, target_degrees, empirical_df
        )

        # Compute error metrics if empirical data available
        if empirical_df is not None:
            metrics_df = analyzer.compute_degree_error_metrics(analysis_df)
        else:
            metrics_df = pd.DataFrame()

        # Generate report
        model_output_dir = output_dir / f'{edge_type}_{model.replace(" ", "_")}'
        file_paths = analyzer.generate_degree_analysis_report(
            f'{edge_type}_{model}', analysis_df, metrics_df, model_output_dir
        )

        all_file_paths[model] = file_paths

    return all_file_paths


def _run_basic_degree_analysis(edge_type: str,
                              source_degrees: np.ndarray,
                              target_degrees: np.ndarray,
                              results_subdir: Path,
                              output_dir: Path,
                              analyzer: DegreeAnalyzer) -> Dict[str, str]:
    """
    Run basic degree analysis when full predictions are not available.

    This fallback function analyzes graph structure and available metrics
    without detailed model predictions.
    """
    print(f"Running basic degree analysis for {edge_type}...")

    # Create degree distribution analysis
    analysis_data = []

    # Analyze all possible source-target combinations
    for i in range(len(source_degrees)):
        for j in range(len(target_degrees)):
            if source_degrees[i] > 0 and target_degrees[j] > 0:  # Skip zero-degree nodes
                u = source_degrees[i]
                v = target_degrees[j]

                # Categorize degrees
                u_category = analyzer.categorize_degrees(np.array([u]))[0]
                v_category = analyzer.categorize_degrees(np.array([v]))[0]
                degree_combination = analyzer.create_degree_combination_labels(
                    np.array([u_category]), np.array([v_category])
                )[0]

                analysis_data.append({
                    'source_index': i,
                    'target_index': j,
                    'source_degree': u,
                    'target_degree': v,
                    'source_category': str(u_category),
                    'target_category': str(v_category),
                    'degree_combination': degree_combination
                })

    analysis_df = pd.DataFrame(analysis_data)

    # Create basic statistics by degree combination
    degree_stats = analysis_df.groupby('degree_combination').agg({
        'source_degree': ['count', 'mean', 'std'],
        'target_degree': ['mean', 'std']
    }).round(4)

    degree_stats.columns = [
        'n_pairs', 'source_degree_mean', 'source_degree_std',
        'target_degree_mean', 'target_degree_std'
    ]

    # Save basic analysis results
    basic_output_dir = output_dir / f'{edge_type}_basic_analysis'
    basic_output_dir.mkdir(parents=True, exist_ok=True)

    # Save degree statistics
    stats_file = basic_output_dir / f'{edge_type}_degree_statistics.csv'
    degree_stats.to_csv(stats_file)

    # Save full analysis data
    analysis_file = basic_output_dir / f'{edge_type}_degree_combinations.csv'
    analysis_df.to_csv(analysis_file, index=False)

    # Create basic visualizations
    file_paths = {'basic': {}}

    try:
        # Create degree combination distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Sample size by degree combination
        degree_counts = analysis_df['degree_combination'].value_counts()
        degree_counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title(f'{edge_type} - Node Pairs by Degree Combination', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Degree Combination', fontsize=12)
        axes[0].set_ylabel('Number of Node Pairs', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

        # Plot 2: Degree distribution heatmap
        degree_pivot = analysis_df.groupby(['source_category', 'target_category']).size().unstack(fill_value=0)
        import seaborn as sns
        sns.heatmap(degree_pivot, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                   cbar_kws={'label': 'Number of Pairs'})
        axes[1].set_title(f'{edge_type} - Degree Category Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Target Degree Category', fontsize=12)
        axes[1].set_ylabel('Source Degree Category', fontsize=12)

        plt.tight_layout()

        # Save plot
        plot_file = basic_output_dir / f'{edge_type}_basic_degree_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        file_paths['basic']['plot'] = str(plot_file)

    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")

    file_paths['basic']['statistics'] = str(stats_file)
    file_paths['basic']['analysis'] = str(analysis_file)

    print(f"Basic degree analysis completed for {edge_type}")
    print(f"  Generated files: {len(file_paths['basic'])}")
    print(f"  Degree combinations found: {len(degree_stats)}")
    print(f"  Total node pairs analyzed: {len(analysis_df):,}")

    return file_paths