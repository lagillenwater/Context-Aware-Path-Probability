"""
Training utilities for pathway prediction models (v2).

This module provides training functions for pathway prediction models.

Functions
---------
train_model
    Train a model with early stopping and learning rate scheduling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import time


def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=500,
    batch_size=32,
    learning_rate=0.001,
    early_stopping_patience=50,
    device='cpu',
    verbose=True,
    random_state=789
):
    """
    Train a pathway prediction model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray, optional
        Validation features
    y_val : np.ndarray, optional
        Validation targets
    epochs : int
        Maximum number of epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Initial learning rate
    early_stopping_patience : int
        Number of epochs without improvement before stopping
    device : str
        Device to use ('cpu' or 'cuda')
    verbose : bool
        Print training progress
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Training history and metrics
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    model = model.to(device)
    model.train()

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)

    if X_val is not None and y_val is not None:
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
        has_validation = True
    else:
        has_validation = False

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'epoch_time': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Mini-batch training
        n_samples = len(X_train)
        indices = np.random.permutation(n_samples)

        train_losses = []

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train_t[batch_indices]
            y_batch = y_train_t[batch_indices]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Compute epoch metrics
        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)
        history['learning_rate'].append(
            optimizer.param_groups[0]['lr']
        )
        history['epoch_time'].append(time.time() - epoch_start)

        # Validation
        if has_validation:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            model.train()

            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Print progress
        if verbose and (epoch + 1) % 50 == 0:
            msg = f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.6f}"
            if has_validation:
                msg += f" - Val Loss: {val_loss:.6f}"
            msg += f" - LR: {optimizer.param_groups[0]['lr']:.6f}"
            print(msg)

    total_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose:
        print(f"\nTraining complete in {total_time:.2f}s")
        print(f"Best validation loss: {best_val_loss:.6f}")

    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'final_epoch': len(history['train_loss']),
        'best_model_state': best_model_state
    }
