"""
Simple model architectures for edge probability prediction.

This module contains lightweight model definitions used for baseline comparisons
and optimizer experiments.
"""

import torch
import torch.nn as nn


class SingleLayerNN(nn.Module):
    """
    Single-layer neural network for edge probability prediction.

    This model is mathematically equivalent to logistic regression when trained
    with the same optimizer and loss function. Used for baseline comparisons
    and optimizer benchmarking.

    Architecture:
        Input (2 features) -> Linear -> Output (1 logit)

    Returns raw logits for use with BCEWithLogitsLoss.
    """

    def __init__(self):
        """Initialize single-layer neural network with Kaiming initialization."""
        super(SingleLayerNN, self).__init__()
        self.linear = nn.Linear(2, 1)

        # Kaiming initialization for ReLU networks
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        nn.init.constant_(self.linear.bias, 0)

        # Add network attribute for compatibility with explicit prediction method
        self.network = self.linear

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 2) containing [source_degree, target_degree]

        Returns:
            Raw logits of shape (batch_size, 1)
        """
        return self.linear(x)
