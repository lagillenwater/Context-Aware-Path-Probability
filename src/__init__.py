"""
DWPC RNN Analysis Package

This package contains utilities and models for analyzing degree-weighted path counts (DWPC)
using recurrent neural networks and other machine learning approaches.

Modules:
--------
models : Neural network architectures
data_processing : Data preparation and processing utilities  
training : Training loops and model management
visualization : Plotting and analysis visualization tools
download_utils : File download and organization utilities
"""

from .models import EdgePredictionNN
from .data_processing import prepare_edge_prediction_data
from .training import train_edge_prediction_model, train_across_permutations
from .visualization import (
    plot_training_history, 
    evaluate_model_performance, 
    create_probability_heatmap,
    plot_permutation_comparison
)
from .download_utils import (
    download_file,
    extract_zip,
    organize_permutations,
    download_hetionet_permutations
)

__version__ = "0.1.0"
__author__ = "DWPC RNN Team"

__all__ = [
    # Models
    "EdgePredictionNN",
    
    # Data Processing
    "prepare_edge_prediction_data",
    
    # Training
    "train_edge_prediction_model",
    "train_across_permutations",
    
    # Visualization
    "plot_training_history",
    "evaluate_model_performance", 
    "create_probability_heatmap",
    "plot_permutation_comparison",
    
    # Download utilities
    "download_file",
    "extract_zip",
    "organize_permutations",
    "download_hetionet_permutations",
]
