"""
Model definitions for pathway prediction (v2).

This module defines neural network models for predicting pathway counts
based on degree-binned features from the original graph.

Classes
-------
DegreeSignatureNN
    Feed-forward neural network with degree and intermediate signatures
"""

import torch
import torch.nn as nn


class DegreeSignatureNN(nn.Module):
    """
    Degree Signature Neural Network.

    Architecture:
    Input (102) -> Linear(128) -> ReLU -> Dropout(0.1)
                -> Linear(64) -> ReLU -> Dropout(0.1)
                -> Linear(32) -> ReLU -> Dropout(0.1)
                -> Linear(1) -> Softplus

    Parameters
    ----------
    input_dim : int
        Number of input features (default: 102 for Set A)
    hidden_dims : list of int
        Hidden layer dimensions (default: [128, 64, 32])
    dropout : float
        Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        input_dim=102,
        hidden_dims=None,
        dropout=0.1
    ):
        """
        Initialize DegreeSignatureNN.
        """
        super(DegreeSignatureNN, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)

        # Count parameters
        self.n_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Predicted pathway counts (batch_size, 1)
        """
        return self.network(x)

    def __repr__(self):
        """
        String representation.
        """
        return (
            f"DegreeSignatureNN(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  dropout={self.dropout_rate},\n"
            f"  n_parameters={self.n_parameters:,}\n"
            f")"
        )
