"""
Theory-Guided Neural Network for Edge Probability Prediction

Architecture designed to:
1. Learn corrections to the analytical formula
2. Produce unbiased residuals
3. Adapt to edge-type-specific patterns
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Dict
from pathlib import Path


class TheoryGuidedNN(nn.Module):
    """
    Neural network that learns corrections to the analytical formula.

    Architecture:
    - Takes theory-guided features as input
    - Includes analytical formula as a feature
    - Learns residual corrections
    - Output: P(edge | degrees)
    """

    def __init__(self,
                 n_features: int,
                 hidden_dims: tuple = (64, 32, 16),
                 dropout: float = 0.2,
                 use_batch_norm: bool = True):
        """
        Initialize theory-guided neural network.

        Parameters
        ----------
        n_features : int
            Number of input features
        hidden_dims : tuple
            Hidden layer dimensions
        dropout : float
            Dropout probability
        use_batch_norm : bool
            Whether to use batch normalization
        """
        super().__init__()

        self.n_features = n_features
        self.hidden_dims = hidden_dims

        # Build architecture
        layers = []
        input_dim = n_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        # Output layer: single probability value
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # Ensure output in [0, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x).squeeze()


class ResidualCorrectionNN(nn.Module):
    """
    Neural network that learns residual corrections to analytical formula.

    Architecture:
    - Analytical formula provides base prediction
    - NN learns additive or multiplicative correction
    - Ensures predictions stay in [0, 1]
    """

    def __init__(self,
                 n_features: int,
                 correction_type: str = 'additive',
                 hidden_dims: tuple = (32, 16),
                 dropout: float = 0.1):
        """
        Initialize residual correction network.

        Parameters
        ----------
        n_features : int
            Number of input features
        correction_type : str
            'additive' or 'multiplicative' correction
        hidden_dims : tuple
            Hidden layer dimensions (smaller than full network)
        dropout : float
            Dropout probability
        """
        super().__init__()

        self.correction_type = correction_type

        # Build correction network
        layers = []
        input_dim = n_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        if correction_type == 'additive':
            # Tanh for additive correction in [-1, 1]
            layers.append(nn.Tanh())
        elif correction_type == 'multiplicative':
            # Softplus for multiplicative correction > 0
            layers.append(nn.Softplus())
        else:
            raise ValueError(f"Unknown correction_type: {correction_type}")

        self.correction_network = nn.Sequential(*layers)

    def forward(self, x, analytical_pred):
        """
        Forward pass with analytical baseline.

        Parameters
        ----------
        x : tensor
            Input features
        analytical_pred : tensor
            Analytical formula predictions (baseline)

        Returns
        -------
        corrected_pred : tensor
            Corrected predictions in [0, 1]
        """
        correction = self.correction_network(x).squeeze()

        if self.correction_type == 'additive':
            # Additive correction with clamping
            corrected = analytical_pred + 0.5 * correction  # Scale correction
            corrected = torch.clamp(corrected, 0, 1)
        else:  # multiplicative
            # Multiplicative correction
            corrected = analytical_pred * correction
            corrected = torch.clamp(corrected, 0, 1)

        return corrected


def train_theory_guided_model(X_train: pd.DataFrame,
                              y_train: np.ndarray,
                              X_val: Optional[pd.DataFrame] = None,
                              y_val: Optional[np.ndarray] = None,
                              analytical_baseline: Optional[np.ndarray] = None,
                              model_type: str = 'full',
                              hidden_dims: tuple = (64, 32, 16),
                              learning_rate: float = 0.001,
                              n_epochs: int = 200,
                              batch_size: int = 256,
                              patience: int = 20,
                              device: str = 'cpu') -> Dict:
    """
    Train theory-guided neural network.

    Parameters
    ----------
    X_train : DataFrame
        Training features
    y_train : array
        Training targets (empirical frequencies)
    X_val : DataFrame, optional
        Validation features
    y_val : array, optional
        Validation targets
    analytical_baseline : array, optional
        Analytical formula predictions (for residual correction)
    model_type : str
        'full' or 'residual'
    hidden_dims : tuple
        Hidden layer dimensions
    learning_rate : float
        Learning rate
    n_epochs : int
        Maximum epochs
    batch_size : int
        Batch size
    patience : int
        Early stopping patience
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    results : dict
        Trained model, history, and metrics
    """
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train.values).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val.values).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
    else:
        X_val_t, y_val_t = None, None

    # Initialize model
    n_features = X_train.shape[1]

    if model_type == 'full':
        model = TheoryGuidedNN(n_features, hidden_dims=hidden_dims).to(device)
    elif model_type == 'residual':
        model = ResidualCorrectionNN(n_features, hidden_dims=(32, 16)).to(device)
        if analytical_baseline is None:
            raise ValueError("analytical_baseline required for residual model")
        analytical_train_t = torch.FloatTensor(analytical_baseline).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Loss and optimizer
    criterion = nn.MSELoss()  # MSE for probability regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_corr': [],
        'val_corr': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []

        # Mini-batch training
        n_samples = len(X_train_t)
        indices = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()

            if model_type == 'full':
                pred = model(X_batch)
            else:  # residual
                analytical_batch = analytical_train_t[batch_idx]
                pred = model(X_batch, analytical_batch)

            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Calculate epoch metrics
        model.eval()
        with torch.no_grad():
            if model_type == 'full':
                train_pred = model(X_train_t)
            else:
                train_pred = model(X_train_t, analytical_train_t)

            train_loss = criterion(train_pred, y_train_t).item()
            train_corr = np.corrcoef(train_pred.cpu().numpy(), y_train.flatten())[0, 1]

            history['train_loss'].append(train_loss)
            history['train_corr'].append(train_corr)

            # Validation
            if X_val_t is not None:
                if model_type == 'full':
                    val_pred = model(X_val_t)
                else:
                    analytical_val_t = torch.FloatTensor(
                        analytical_baseline[len(y_train):]
                    ).to(device)
                    val_pred = model(X_val_t, analytical_val_t)

                val_loss = criterion(val_pred, y_val_t).item()
                val_corr = np.corrcoef(val_pred.cpu().numpy(), y_val.flatten())[0, 1]

                history['val_loss'].append(val_loss)
                history['val_corr'].append(val_corr)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                # Learning rate scheduling
                scheduler.step(val_loss)

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, "
                          f"val_loss={val_loss:.6f}, train_r={train_corr:.4f}, "
                          f"val_r={val_corr:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, "
                          f"train_r={train_corr:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss,
        'final_train_corr': history['train_corr'][-1],
        'final_val_corr': history['val_corr'][-1] if X_val_t is not None else None
    }
