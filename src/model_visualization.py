"""
Visualization functions for model comparison and edge probability heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, precision_recall_curve
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional


class ModelVisualizer:
    """Comprehensive visualization for model comparison and predictions."""

    def __init__(self, figsize_default: Tuple[int, int] = (12, 8)):
        self.figsize_default = figsize_default
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Default colors for 4 models

    def plot_roc_curves(self, evaluation_results: Dict[str, Dict[str, Any]],
                       figsize: Optional[Tuple[int, int]] = None, save_path: str = None):
        """
        Plot ROC curves for all models.

        Parameters:
        -----------
        evaluation_results : Dict[str, Dict[str, Any]]
            Evaluation results from all models
        figsize : Optional[Tuple[int, int]]
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if figsize is None:
            figsize = self.figsize_default

        plt.figure(figsize=figsize)

        for i, (model_name, results) in enumerate(evaluation_results.items()):
            fpr, tpr = results['classification']['roc_curve']
            auc = results['classification']['auc']

            if fpr is not None and tpr is not None:
                color = self.colors[i % len(self.colors)]
                plt.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{model_name} (AUC = {auc:.3f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_precision_recall_curves(self, evaluation_results: Dict[str, Dict[str, Any]],
                                   figsize: Optional[Tuple[int, int]] = None, save_path: str = None):
        """
        Plot Precision-Recall curves for all models.

        Parameters:
        -----------
        evaluation_results : Dict[str, Dict[str, Any]]
            Evaluation results from all models
        figsize : Optional[Tuple[int, int]]
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if figsize is None:
            figsize = self.figsize_default

        plt.figure(figsize=figsize)

        for i, (model_name, results) in enumerate(evaluation_results.items()):
            precision, recall = results['classification']['pr_curve']
            avg_precision = results['classification']['average_precision']

            if precision is not None and recall is not None:
                color = self.colors[i % len(self.colors)]
                plt.plot(recall, precision, color=color, linewidth=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_performance_comparison(self, evaluation_results: Dict[str, Dict[str, Any]],
                                  metrics: List[str] = None,
                                  figsize: Optional[Tuple[int, int]] = None, save_path: str = None):
        """
        Plot bar chart comparing model performance across metrics.

        Parameters:
        -----------
        evaluation_results : Dict[str, Dict[str, Any]]
            Evaluation results from all models
        metrics : List[str], optional
            List of metrics to compare
        figsize : Optional[Tuple[int, int]]
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if metrics is None:
            metrics = ['auc', 'accuracy', 'f1_score', 'average_precision']

        if figsize is None:
            figsize = (15, 8)

        n_metrics = len(metrics)
        n_models = len(evaluation_results)

        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        model_names = list(evaluation_results.keys())

        for i, metric in enumerate(metrics):
            values = []
            for model_name in model_names:
                if metric in ['auc', 'accuracy', 'f1_score', 'average_precision', 'precision', 'recall']:
                    value = evaluation_results[model_name]['classification'][metric]
                else:
                    value = evaluation_results[model_name]['regression'][metric]
                values.append(value if not np.isnan(value) else 0)

            bars = axes[i].bar(model_names, values, color=self.colors[:n_models])
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylim(0, 1.0 if metric != 'rmse' else max(values) * 1.1)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')

            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def create_prediction_heatmap(self, model: Any, model_name: str,
                                source_bins: np.ndarray, target_bins: np.ndarray,
                                scaler: Any = None, figsize: Optional[Tuple[int, int]] = None,
                                save_path: str = None):
        """
        Create a heatmap showing edge probability predictions across degree combinations.

        Parameters:
        -----------
        model : Any
            Trained model
        model_name : str
            Name of the model
        source_bins : np.ndarray
            Source degree bins
        target_bins : np.ndarray
            Target degree bins
        scaler : Any, optional
            Feature scaler
        figsize : Optional[Tuple[int, int]]
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        from model_training import predict_with_model

        if figsize is None:
            figsize = (10, 8)

        # Create meshgrid
        source_grid, target_grid = np.meshgrid(source_bins, target_bins)
        grid_features = np.column_stack([source_grid.ravel(), target_grid.ravel()])

        # Make predictions
        predictions = predict_with_model(model, grid_features, model_name, scaler)
        prediction_matrix = predictions.reshape(source_grid.shape)

        # Create heatmap
        plt.figure(figsize=figsize)

        # Use a color map that emphasizes the probability range
        cmap = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed

        im = plt.imshow(prediction_matrix, cmap=cmap, aspect='auto',
                       extent=[source_bins.min(), source_bins.max(),
                              target_bins.min(), target_bins.max()],
                       origin='lower', vmin=0, vmax=1)

        plt.colorbar(im, label='Edge Probability')
        plt.xlabel('Source Degree')
        plt.ylabel('Target Degree')
        plt.title(f'Edge Probability Predictions - {model_name}')

        # Add contour lines for better visualization
        contour_levels = np.arange(0.1, 1.0, 0.2)
        plt.contour(source_grid, target_grid, prediction_matrix,
                   levels=contour_levels, colors='black', alpha=0.3, linewidths=0.5)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def create_all_prediction_heatmaps(self, models_results: Dict[str, Dict[str, Any]],
                                     source_bins: np.ndarray, target_bins: np.ndarray,
                                     figsize_individual: Optional[Tuple[int, int]] = None,
                                     save_dir: str = None):
        """
        Create prediction heatmaps for all models.

        Parameters:
        -----------
        models_results : Dict[str, Dict[str, Any]]
            Results from model training
        source_bins : np.ndarray
            Source degree bins
        target_bins : np.ndarray
            Target degree bins
        figsize_individual : Optional[Tuple[int, int]]
            Figure size for individual heatmaps
        save_dir : str, optional
            Directory to save figures
        """
        if figsize_individual is None:
            figsize_individual = (10, 8)

        print("Creating prediction heatmaps for all models...")

        for model_name, model_result in models_results.items():
            if model_name == 'data_splits':
                continue

            model = model_result['model']
            scaler = model_result['training_result'].get('scaler')

            save_path = None
            if save_dir:
                save_path = f"{save_dir}/{model_name.replace(' ', '_')}_heatmap.png"

            print(f"Creating heatmap for {model_name}...")
            self.create_prediction_heatmap(
                model, model_name, source_bins, target_bins,
                scaler, figsize_individual, save_path
            )

    def create_combined_heatmap_grid(self, models_results: Dict[str, Dict[str, Any]],
                                   source_bins: np.ndarray, target_bins: np.ndarray,
                                   figsize: Optional[Tuple[int, int]] = None, save_path: str = None):
        """
        Create a grid of heatmaps showing all model predictions side by side.

        Parameters:
        -----------
        models_results : Dict[str, Dict[str, Any]]
            Results from model training
        source_bins : np.ndarray
            Source degree bins
        target_bins : np.ndarray
            Target degree bins
        figsize : Optional[Tuple[int, int]]
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        from model_training import predict_with_model

        # Filter out data_splits
        model_items = [(name, result) for name, result in models_results.items()
                      if name != 'data_splits']

        n_models = len(model_items)

        if figsize is None:
            figsize = (5 * n_models, 4)

        # Create meshgrid
        source_grid, target_grid = np.meshgrid(source_bins, target_bins)
        grid_features = np.column_stack([source_grid.ravel(), target_grid.ravel()])

        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]

        for i, (model_name, model_result) in enumerate(model_items):
            model = model_result['model']
            scaler = model_result['training_result'].get('scaler')

            # Make predictions
            predictions = predict_with_model(model, grid_features, model_name, scaler)
            prediction_matrix = predictions.reshape(source_grid.shape)

            # Create heatmap
            im = axes[i].imshow(prediction_matrix, cmap='RdYlBu_r', aspect='auto',
                               extent=[source_bins.min(), source_bins.max(),
                                      target_bins.min(), target_bins.max()],
                               origin='lower', vmin=0, vmax=1)

            axes[i].set_xlabel('Source Degree')
            if i == 0:
                axes[i].set_ylabel('Target Degree')
            axes[i].set_title(f'{model_name}')

            # Add contour lines
            contour_levels = np.arange(0.1, 1.0, 0.2)
            axes[i].contour(source_grid, target_grid, prediction_matrix,
                           levels=contour_levels, colors='black', alpha=0.3, linewidths=0.5)

        # Add colorbar
        fig.colorbar(im, ax=axes, label='Edge Probability', shrink=0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_training_history(self, training_results: Dict[str, Dict[str, Any]],
                            figsize: Optional[Tuple[int, int]] = None, save_path: str = None):
        """
        Plot training history for neural network models.

        Parameters:
        -----------
        training_results : Dict[str, Dict[str, Any]]
            Training results from all models
        figsize : Optional[Tuple[int, int]]
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if figsize is None:
            figsize = (15, 5)

        # Find neural network models with training history
        nn_models = {}
        for model_name, result in training_results.items():
            if model_name == 'data_splits':
                continue
            if 'history' in result['training_result']:
                nn_models[model_name] = result['training_result']['history']

        if not nn_models:
            print("No neural network training history found.")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for model_name, history in nn_models.items():
            epochs = range(1, len(history['train_loss']) + 1)

            # Training and validation loss
            axes[0].plot(epochs, history['train_loss'], label=f'{model_name} (Train)', linewidth=2)
            axes[0].plot(epochs, history['val_loss'], label=f'{model_name} (Val)', linewidth=2, linestyle='--')

            # Training and validation accuracy
            axes[1].plot(epochs, history['train_acc'], label=f'{model_name} (Train)', linewidth=2)
            axes[1].plot(epochs, history['val_acc'], label=f'{model_name} (Val)', linewidth=2, linestyle='--')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training/Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training/Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Model comparison summary
        axes[2].axis('off')
        summary_text = "Model Training Summary:\n\n"
        for model_name, result in training_results.items():
            if model_name == 'data_splits':
                continue
            train_result = result['training_result']
            summary_text += f"{model_name}:\n"
            summary_text += f"  Training Time: {train_result['training_time']:.2f}s\n"
            if 'epochs_trained' in train_result:
                summary_text += f"  Epochs: {train_result['epochs_trained']}\n"
            if 'best_val_loss' in train_result:
                summary_text += f"  Best Val Loss: {train_result['best_val_loss']:.4f}\n"
            summary_text += "\n"

        axes[2].text(0.1, 0.9, summary_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def create_comprehensive_report(self, models_results: Dict[str, Dict[str, Any]],
                                  evaluation_results: Dict[str, Dict[str, Any]],
                                  source_bins: np.ndarray, target_bins: np.ndarray,
                                  save_dir: str = None):
        """
        Create a comprehensive visualization report including all plots.

        Parameters:
        -----------
        models_results : Dict[str, Dict[str, Any]]
            Results from model training
        evaluation_results : Dict[str, Dict[str, Any]]
            Evaluation results from all models
        source_bins : np.ndarray
            Source degree bins
        target_bins : np.ndarray
            Target degree bins
        save_dir : str, optional
            Directory to save all figures
        """
        print("="*60)
        print("CREATING COMPREHENSIVE VISUALIZATION REPORT")
        print("="*60)

        # 1. ROC Curves
        print("1. Creating ROC curves...")
        roc_save_path = f"{save_dir}/roc_curves.png" if save_dir else None
        self.plot_roc_curves(evaluation_results, save_path=roc_save_path)

        # 2. Precision-Recall Curves
        print("2. Creating Precision-Recall curves...")
        pr_save_path = f"{save_dir}/precision_recall_curves.png" if save_dir else None
        self.plot_precision_recall_curves(evaluation_results, save_path=pr_save_path)

        # 3. Performance comparison
        print("3. Creating performance comparison...")
        perf_save_path = f"{save_dir}/performance_comparison.png" if save_dir else None
        self.plot_performance_comparison(evaluation_results, save_path=perf_save_path)

        # 4. Training history
        print("4. Creating training history...")
        history_save_path = f"{save_dir}/training_history.png" if save_dir else None
        self.plot_training_history(models_results, save_path=history_save_path)

        # 5. Individual heatmaps
        print("5. Creating individual prediction heatmaps...")
        self.create_all_prediction_heatmaps(models_results, source_bins, target_bins,
                                          save_dir=save_dir)

        # 6. Combined heatmap grid
        print("6. Creating combined heatmap grid...")
        combined_save_path = f"{save_dir}/combined_heatmaps.png" if save_dir else None
        self.create_combined_heatmap_grid(models_results, source_bins, target_bins,
                                        save_path=combined_save_path)

        print("\nVisualization report complete!")
        if save_dir:
            print(f"All figures saved to: {save_dir}")

    def plot_analytical_validation(self, validation_results: Dict[str, Any],
                                 figsize: Optional[Tuple[int, int]] = None, save_path: str = None):
        """
        Plot analytical approximation validation against empirical frequencies.

        Parameters:
        -----------
        validation_results : Dict[str, Any]
            Results from validate_analytical_approximation_vs_empirical
        figsize : Optional[Tuple[int, int]]
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if figsize is None:
            figsize = (12, 5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        analytical_probs = validation_results['analytical_probabilities']
        empirical_freqs = validation_results['empirical_frequencies']
        correlation = validation_results['correlation_vs_empirical']
        mae = validation_results['mae_vs_empirical']
        r2 = validation_results['r2_vs_empirical']

        # Scatter plot
        ax1.scatter(empirical_freqs, analytical_probs, alpha=0.6, s=20)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Match')
        ax1.set_xlabel('Empirical Frequency')
        ax1.set_ylabel('Analytical Approximation')
        ax1.set_title(f'Analytical vs Empirical\nr = {correlation:.3f}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Residuals plot
        residuals = analytical_probs - empirical_freqs
        ax2.scatter(empirical_freqs, residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Empirical Frequency')
        ax2.set_ylabel('Residuals (Analytical - Empirical)')
        ax2.set_title(f'Residuals\nMAE = {mae:.4f}, R² = {r2:.3f}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def create_analytical_heatmap(self, source_bins: np.ndarray, target_bins: np.ndarray,
                                total_edges_m: int, figsize: Optional[Tuple[int, int]] = None,
                                save_path: str = None):
        """
        Create heatmap of analytical approximation probabilities.

        Parameters:
        -----------
        source_bins : np.ndarray
            Source degree bins
        target_bins : np.ndarray
            Target degree bins
        total_edges_m : int
            Total number of edges
        figsize : Optional[Tuple[int, int]]
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if figsize is None:
            figsize = (10, 8)

        # Create grid
        source_grid, target_grid = np.meshgrid(source_bins, target_bins)

        # Calculate analytical approximation
        u_i = source_grid.astype(np.float64)
        v_j = target_grid.astype(np.float64)
        m = float(total_edges_m)

        numerator = u_i * v_j
        denominator_term = (m - u_i - v_j + 1.0)
        denominator = np.sqrt(numerator**2 + denominator_term**2)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        probabilities = numerator / denominator
        probabilities = np.clip(probabilities, 0.0, 1.0)

        plt.figure(figsize=figsize)
        plt.imshow(probabilities, extent=[source_bins.min(), source_bins.max(),
                                        target_bins.min(), target_bins.max()],
                  aspect='auto', origin='lower', cmap='viridis', interpolation='bilinear')

        plt.colorbar(label='Edge Probability')
        plt.xlabel('Source Degree')
        plt.ylabel('Target Degree')
        plt.title('Analytical Approximation Heatmap')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_models_vs_analytical_comparison(self, analytical_comparison: Dict[str, Dict[str, Any]],
                                           figsize: Optional[Tuple[int, int]] = None, save_path: str = None):
        """
        Plot model predictions vs analytical approximation comparison.

        Parameters:
        -----------
        analytical_comparison : Dict[str, Dict[str, Any]]
            Results from compare_models_vs_analytical_approximation
        figsize : Optional[Tuple[int, int]]
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if figsize is None:
            figsize = (15, 10)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for i, (model_name, results) in enumerate(analytical_comparison.items()):
            if i >= 4:  # Only plot first 4 models
                break

            model_predictions = results['model_predictions']
            analytical_probs = results['analytical_probabilities']
            correlation = results['correlation_vs_analytical']

            axes[i].scatter(analytical_probs, model_predictions, alpha=0.6, s=20)
            axes[i].plot([0, 1], [0, 1], 'r--', alpha=0.8)
            axes[i].set_xlabel('Analytical Approximation')
            axes[i].set_ylabel('Model Prediction')
            axes[i].set_title(f'{model_name}\nr = {correlation:.3f}')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()



def create_comparison_table_plot(comparison_df: pd.DataFrame,
                               figsize: Optional[Tuple[int, int]] = None, save_path: str = None):
    """
    Create a visual table showing model comparison metrics.

    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison DataFrame from ModelEvaluator
    figsize : Optional[Tuple[int, int]]
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if figsize is None:
        figsize = (14, 6)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=comparison_df.round(4).values,
                    colLabels=comparison_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Color code the header
    for i in range(len(comparison_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code the best values in each column
    for col_idx in range(1, len(comparison_df.columns)):  # Skip model name column
        col_values = comparison_df.iloc[:, col_idx].values
        if comparison_df.columns[col_idx] in ['RMSE', 'MAE']:
            # Lower is better
            best_idx = np.nanargmin(col_values)
        else:
            # Higher is better
            best_idx = np.nanargmax(col_values)

        table[(best_idx + 1, col_idx)].set_facecolor('#90EE90')  # Light green

    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()