"""
Training functions for different edge probability prediction models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional
import time
from model_comparison import SimpleNN


class ModelTrainer:
    """Unified trainer for different types of models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scalers = {}
        self.training_history = {}

    def train_neural_network(self, model: SimpleNN, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           epochs: int = 100, batch_size: int = 512,
                           learning_rate: float = 0.001, patience: int = 10) -> Dict[str, Any]:
        """
        Train a neural network model.

        Parameters:
        -----------
        model : SimpleNN
            The neural network model to train
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray
            Validation data
        epochs : int
            Maximum number of epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        patience : int
            Early stopping patience

        Returns:
        --------
        Dict[str, Any]
            Training history and metrics
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Setup training
        criterion = nn.BCELoss()  # Binary cross-entropy for probability prediction
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"Training Neural Network:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max epochs: {epochs}")

        start_time = time.time()

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_predicted = (val_outputs > 0.5).float()
                val_acc = (val_predicted == y_val_tensor).float().mean().item()

            # Record metrics
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")
        print(f"  Best validation loss: {best_val_loss:.4f}")

        return {
            'history': history,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'epochs_trained': epoch + 1
        }

    def train_sklearn_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                          model_name: str, scale_features: bool = True) -> Dict[str, Any]:
        """
        Train a scikit-learn model.

        Parameters:
        -----------
        model : Any
            Scikit-learn model to train
        X_train, y_train : np.ndarray
            Training data
        model_name : str
            Name of the model for tracking
        scale_features : bool
            Whether to scale features

        Returns:
        --------
        Dict[str, Any]
            Training metrics
        """
        print(f"Training {model_name}:")
        print(f"  Training samples: {len(X_train)}")

        start_time = time.time()

        # Scale features if requested
        if scale_features and model_name != 'Random Forest':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train
            self.scalers[model_name] = None

        # Train the model
        model.fit(X_train_scaled, y_train)

        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")

        return {
            'training_time': training_time,
            'scaler': self.scalers[model_name]
        }

    def train_all_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """
        Train all models and return results.

        Parameters:
        -----------
        models : Dict[str, Any]
            Dictionary of models to train
        X, y : np.ndarray
            Features and labels
        test_size : float
            Proportion of data for testing
        val_size : float
            Proportion of training data for validation (NN only)

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Training results for all models
        """
        # Split data into train and test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # For neural network, further split training into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size,
            random_state=self.random_state, stratify=y_train_full
        )

        results = {}
        results['data_splits'] = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_full': X_train_full,
            'y_train_full': y_train_full
        }

        print("="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        print(f"Total samples: {len(X)}")
        print(f"Training samples: {len(X_train_full)} (NN uses {len(X_train)} for training, {len(X_val)} for validation)")
        print(f"Test samples: {len(X_test)}")
        print()

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            if isinstance(model, SimpleNN):
                # Train neural network with validation split
                training_result = self.train_neural_network(
                    model, X_train, y_train, X_val, y_val
                )
            else:
                # Train sklearn models with full training data
                training_result = self.train_sklearn_model(
                    model, X_train_full, y_train_full, model_name
                )

            results[model_name] = {
                'model': model,
                'training_result': training_result
            }
            print()

        self.training_history = results
        return results

    def get_scaler(self, model_name: str) -> Optional[StandardScaler]:
        """Get the scaler for a specific model."""
        return self.scalers.get(model_name)


def predict_with_model(model: Any, X: np.ndarray, model_name: str,
                      scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """
    Make predictions with any model type.

    Parameters:
    -----------
    model : Any
        Trained model
    X : np.ndarray
        Features to predict
    model_name : str
        Name of the model
    scaler : Optional[StandardScaler]
        Scaler to apply to features

    Returns:
    --------
    np.ndarray
        Predictions
    """
    # Scale features if scaler is provided
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    if isinstance(model, SimpleNN):
        # Neural network prediction
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            predictions = model(X_tensor).numpy()
    else:
        # Scikit-learn prediction
        if hasattr(model, 'predict_proba'):
            # For models that support probability prediction
            predictions = model.predict_proba(X_scaled)[:, 1]
        else:
            # For regression models
            predictions = model.predict(X_scaled)
            # Clip to [0, 1] range for probability interpretation
            predictions = np.clip(predictions, 0, 1)

    return predictions