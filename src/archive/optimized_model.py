"""
Optimized model and trainer for distribution-aware edge prediction.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

class DistributionAwareNN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dims: List[int] = [64, 32, 16], dropout_rate: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.network(x).squeeze()

class OptimizedModelTrainer:
    def __init__(self, model_type: str, random_seed: int = 42, use_regression: bool = True, use_distribution_loss: bool = True):
        self.model_type = model_type
        self.random_seed = random_seed
        self.use_regression = use_regression
        self.use_distribution_loss = use_distribution_loss
        self.model = None
        self.scaler = None
    def train(self, features: np.ndarray, targets: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        if self.use_regression:
            target_bins = pd.qcut(targets, q=5, labels=False, duplicates='drop')
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=self.random_seed, stratify=target_bins)
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=self.random_seed, stratify=targets.astype(int))
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        if self.model_type == 'NN':
            self.model, train_metrics = self._train_improved_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return {'model': self.model, 'scaler': self.scaler, 'metrics': train_metrics, 'model_type': self.model_type, 'use_regression': self.use_regression}
    def _train_improved_neural_network(self, X_train, y_train, X_test, y_test):
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        input_dim = X_train.shape[1]
        model = DistributionAwareNN(input_dim=input_dim)
        def distribution_aware_loss(predictions, targets):
            mse_loss = nn.MSELoss()(predictions, targets)
            if len(predictions) > 1:
                pred_var = torch.var(predictions)
                target_var = torch.var(targets)
                consistency_loss = torch.abs(pred_var - target_var)
            else:
                consistency_loss = torch.tensor(0.0)
            return mse_loss + 0.1 * consistency_loss
        criterion = distribution_aware_loss
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        batch_size = max(32, len(X_train) // 20)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        model.train()
        for epoch in range(200):
            epoch_loss = 0.0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                if len(batch_X) == 1:
                    continue
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                scheduler.step(avg_loss)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).numpy()
            test_pred = model(X_test_tensor).numpy()
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        metrics = {'train_mse': train_mse, 'test_mse': test_mse, 'train_mae': train_mae, 'test_mae': test_mae, 'train_r2': r2_score(y_train, train_pred), 'test_r2': r2_score(y_test, test_pred)}
        return model, metrics
    def predict_probabilities(self, features):
        if hasattr(self.model, 'eval'):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(self.scaler.transform(features))
                return self.model(X_tensor).cpu().numpy()
        else:
            raise NotImplementedError("Model does not support probability prediction.")
