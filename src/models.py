"""
Neural Network Models for DWPC Analysis

This module contains the neural network architectures used for edge prediction
based on source and target node degrees.
"""

import torch
import torch.nn as nn


class EdgePredictionNN(nn.Module):
    """
    Neural Network for predicting edge probability based on source and target degrees.
    """
    
    def __init__(self, input_dim=2, hidden_dims=[64, 32], dropout_rate=0.2):
        """
        Initialize the neural network.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (default: 2 for source and target degrees)
        hidden_dims : list
            List of hidden layer dimensions
        dropout_rate : float
            Dropout rate for regularization
        """
        super(EdgePredictionNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer with sigmoid activation for probability
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x).squeeze()


def get_model_info(model):
    """
    Extract model architecture information safely.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to analyze
        
    Returns:
    --------
    dict
        Dictionary containing model architecture information including:
        - model_class: Name of the model class
        - total_parameters: Total number of parameters
        - trainable_parameters: Number of trainable parameters
        - layers: List of layer information (name, type, parameters)
    """
    architecture_info = {
        'model_class': model.__class__.__name__,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    # Try to get layer information
    try:
        layer_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                })
        architecture_info['layers'] = layer_info
    except Exception as e:
        architecture_info['layers'] = f"Could not extract layer info: {str(e)}"
    
    return architecture_info
