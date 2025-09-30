"""
Helper functions for creating colored scatter plots based on node features.

This module provides functions to color scatter plots by various node features
such as source degree, target degree, and their combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from typing import Tuple, Optional, Dict, Any
import seaborn as sns


def create_degree_color_mapping(degrees: np.ndarray, color_scheme: str = 'viridis',
                               log_scale: bool = False) -> Tuple[np.ndarray, Any]:
    """
    Create color mapping for degrees.

    Parameters:
    -----------
    degrees : np.ndarray
        Array of degree values
    color_scheme : str
        Matplotlib colormap name
    log_scale : bool
        Whether to use log scale for coloring

    Returns:
    --------
    Tuple[np.ndarray, Any]
        Colors array and colormap normalization object
    """
    if log_scale:
        # Add 1 to avoid log(0) and use log scale
        norm_values = np.log10(degrees + 1)
        norm = mcolors.LogNorm(vmin=degrees.min() + 1, vmax=degrees.max() + 1)
    else:
        norm_values = degrees
        norm = mcolors.Normalize(vmin=degrees.min(), vmax=degrees.max())

    cmap = plt.cm.get_cmap(color_scheme)
    colors = cmap(norm(degrees))

    return colors, norm


def plot_colored_scatter_by_source_degree(x_values: np.ndarray, y_values: np.ndarray,
                                         source_degrees: np.ndarray,
                                         title: str = "Scatter Plot Colored by Source Degree",
                                         xlabel: str = "X Values", ylabel: str = "Y Values",
                                         color_scheme: str = 'viridis', log_scale: bool = False,
                                         figsize: Tuple[int, int] = (10, 8),
                                         save_path: Optional[str] = None) -> None:
    """
    Create scatter plot colored by source degree.

    Parameters:
    -----------
    x_values, y_values : np.ndarray
        Scatter plot coordinates
    source_degrees : np.ndarray
        Source degree values for coloring
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    color_scheme : str
        Matplotlib colormap name
    log_scale : bool
        Whether to use log scale for coloring
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save the figure
    """
    colors, norm = create_degree_color_mapping(source_degrees, color_scheme, log_scale)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(x_values, y_values, c=source_degrees, cmap=color_scheme,
                         norm=norm, alpha=0.6, s=20)

    plt.colorbar(scatter, label='Source Degree')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_colored_scatter_by_target_degree(x_values: np.ndarray, y_values: np.ndarray,
                                         target_degrees: np.ndarray,
                                         title: str = "Scatter Plot Colored by Target Degree",
                                         xlabel: str = "X Values", ylabel: str = "Y Values",
                                         color_scheme: str = 'plasma', log_scale: bool = False,
                                         figsize: Tuple[int, int] = (10, 8),
                                         save_path: Optional[str] = None) -> None:
    """
    Create scatter plot colored by target degree.

    Parameters:
    -----------
    x_values, y_values : np.ndarray
        Scatter plot coordinates
    target_degrees : np.ndarray
        Target degree values for coloring
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    color_scheme : str
        Matplotlib colormap name
    log_scale : bool
        Whether to use log scale for coloring
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save the figure
    """
    colors, norm = create_degree_color_mapping(target_degrees, color_scheme, log_scale)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(x_values, y_values, c=target_degrees, cmap=color_scheme,
                         norm=norm, alpha=0.6, s=20)

    plt.colorbar(scatter, label='Target Degree')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_colored_scatter_by_degree_product(x_values: np.ndarray, y_values: np.ndarray,
                                          source_degrees: np.ndarray, target_degrees: np.ndarray,
                                          title: str = "Scatter Plot Colored by Source × Target Degree",
                                          xlabel: str = "X Values", ylabel: str = "Y Values",
                                          color_scheme: str = 'coolwarm', log_scale: bool = True,
                                          figsize: Tuple[int, int] = (10, 8),
                                          save_path: Optional[str] = None) -> None:
    """
    Create scatter plot colored by source degree × target degree.

    Parameters:
    -----------
    x_values, y_values : np.ndarray
        Scatter plot coordinates
    source_degrees, target_degrees : np.ndarray
        Degree values for computing product
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    color_scheme : str
        Matplotlib colormap name
    log_scale : bool
        Whether to use log scale for coloring
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save the figure
    """
    degree_product = source_degrees * target_degrees
    colors, norm = create_degree_color_mapping(degree_product, color_scheme, log_scale)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(x_values, y_values, c=degree_product, cmap=color_scheme,
                         norm=norm, alpha=0.6, s=20)

    plt.colorbar(scatter, label='Source × Target Degree')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_model_comparison_grid_with_coloring(models_predictions: Dict[str, np.ndarray],
                                             empirical_values: np.ndarray,
                                             source_degrees: np.ndarray,
                                             target_degrees: np.ndarray,
                                             color_by: str = 'source_degree',
                                             figsize: Tuple[int, int] = (15, 12),
                                             save_path: Optional[str] = None) -> None:
    """
    Create a grid of model comparison plots with degree-based coloring.

    Parameters:
    -----------
    models_predictions : Dict[str, np.ndarray]
        Dictionary mapping model names to prediction arrays
    empirical_values : np.ndarray
        Empirical frequency values
    source_degrees, target_degrees : np.ndarray
        Degree values for coloring
    color_by : str
        What to color by: 'source_degree', 'target_degree', or 'degree_product'
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save the figure
    """
    n_models = len(models_predictions)
    cols = 2
    rows = (n_models + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]

    # Determine coloring values
    if color_by == 'source_degree':
        color_values = source_degrees
        color_label = 'Source Degree'
        cmap = 'viridis'
    elif color_by == 'target_degree':
        color_values = target_degrees
        color_label = 'Target Degree'
        cmap = 'plasma'
    elif color_by == 'degree_product':
        color_values = source_degrees * target_degrees
        color_label = 'Source × Target Degree'
        cmap = 'coolwarm'
    else:
        raise ValueError("color_by must be 'source_degree', 'target_degree', or 'degree_product'")

    # Create normalization for consistent coloring across subplots
    norm = mcolors.LogNorm(vmin=color_values.min() + 1, vmax=color_values.max() + 1) if color_by == 'degree_product' else mcolors.Normalize(vmin=color_values.min(), vmax=color_values.max())

    for i, (model_name, predictions) in enumerate(models_predictions.items()):
        if i >= len(axes):
            break

        ax = axes[i]

        # Ensure arrays are the same length
        min_len = min(len(empirical_values), len(predictions), len(color_values))
        emp_subset = empirical_values[:min_len]
        pred_subset = predictions[:min_len]
        color_subset = color_values[:min_len]

        scatter = ax.scatter(emp_subset, pred_subset, c=color_subset,
                           cmap=cmap, norm=norm, alpha=0.6, s=20)

        ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        ax.set_xlabel('Empirical Frequency')
        ax.set_ylabel('Model Prediction')
        ax.set_title(f'{model_name} vs Empirical\n(colored by {color_label})')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Hide unused subplots
    for i in range(len(models_predictions), len(axes)):
        axes[i].set_visible(False)

    # Add shared colorbar
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(color_label)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_degree_distribution_comparison(source_degrees: np.ndarray, target_degrees: np.ndarray,
                                      figsize: Tuple[int, int] = (12, 5),
                                      save_path: Optional[str] = None) -> None:
    """
    Plot comparison of source and target degree distributions.

    Parameters:
    -----------
    source_degrees, target_degrees : np.ndarray
        Degree arrays to compare
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Source degree distribution
    axes[0].hist(source_degrees, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Source Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Source Degree Distribution\nMean: {source_degrees.mean():.1f}, Max: {source_degrees.max()}')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Target degree distribution
    axes[1].hist(target_degrees, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_xlabel('Target Degree')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Target Degree Distribution\nMean: {target_degrees.mean():.1f}, Max: {target_degrees.max()}')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()