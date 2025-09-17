"""
Enhanced experiment functions for stable Neural Network training and edge prediction experiments.
Contains the improved NN training logic with proper seed control and multiple initialization averaging.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import time

from sampling import create_representative_dataset

# Additional imports for compatibility
import pandas as pd
import matplotlib.pyplot as plt


class StableEdgeNN(nn.Module):
    """Enhanced Neural Network with stable initialization and architecture."""
    
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super(StableEdgeNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Only add BatchNorm for larger networks to avoid instability in small ones
            if len(hidden_dims) > 2 or hidden_dim >= 64:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for stability
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


def train_enhanced_neural_network(X_train, y_train, X_test, y_test, sample_size, run_id):
    """
    Train enhanced Neural Network with proper stability measures.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        sample_size: Size of the training sample (for adaptive hyperparameters)
        run_id: Run identifier for seed control
        
    Returns:
        dict: Results containing test_auc, test_ap, and other metrics
    """
    # CRITICAL: Proper seed control for reproducible NN training
    torch.manual_seed(42 + run_id)
    torch.cuda.manual_seed_all(42 + run_id)
    np.random.seed(42 + run_id)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ENHANCED ADAPTIVE HYPERPARAMETERS
    # More conservative scaling to ensure stability
    base_epochs = 30  # Increased base for better convergence
    adaptive_epochs = max(base_epochs, min(200, int(sample_size / 25)))  # More epochs
    
    # More stable learning rate scaling
    base_lr = 0.001
    if sample_size <= 1000:
        adaptive_lr = 0.002  # Higher for small datasets
    elif sample_size <= 5000:
        adaptive_lr = 0.001  # Standard for medium
    else:
        adaptive_lr = 0.0005  # Lower for large datasets
    
    # More conservative architecture scaling
    if sample_size <= 1000:
        hidden_dims = [32, 16]
        dropout_rate = 0.1  # Lower dropout for small data
    elif sample_size <= 5000:
        hidden_dims = [64, 32]
        dropout_rate = 0.2
    else:
        hidden_dims = [128, 64, 32]
        dropout_rate = 0.3
    
    # MULTIPLE INITIALIZATION AVERAGING for ultra-stability
    n_inits = 3  # Average across multiple random initializations
    all_test_preds = []
    
    for init_id in range(n_inits):
        # Seed each initialization differently
        torch.manual_seed(42 + run_id * 10 + init_id)
        
        model = StableEdgeNN(X_train_scaled.shape[1], hidden_dims, dropout_rate)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=adaptive_lr, weight_decay=1e-5)  # Added weight decay
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        # Enhanced training with validation split
        val_size = int(0.2 * len(X_train_tensor))
        train_size = len(X_train_tensor) - val_size
        
        X_train_sub = X_train_tensor[:train_size]
        y_train_sub = y_train_tensor[:train_size]
        X_val = X_train_tensor[train_size:]
        y_val = y_train_tensor[train_size:]
        
        model.train()
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(adaptive_epochs):
            # Training
            optimizer.zero_grad()
            train_outputs = model(X_train_sub).squeeze()
            train_loss = criterion(train_outputs, y_train_sub)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val)
            model.train()
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch > 30:  # Minimum 30 epochs
                    break
        
        # Restore best model and evaluate
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor).squeeze().numpy()
            all_test_preds.append(test_pred)
    
    # Average predictions across all initializations for maximum stability
    final_test_pred = np.mean(all_test_preds, axis=0)
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test, final_test_pred)
    test_ap = average_precision_score(y_test, final_test_pred)
    
    return {
        'test_auc': test_auc,
        'test_ap': test_ap,
        'predictions': final_test_pred,
        'n_inits_averaged': n_inits,
        'epochs_used': epoch + 1,
        'final_lr': adaptive_lr,
        'architecture': hidden_dims
    }


def run_enhanced_experiment(sample_size, run_id, edges, degrees_dict, verbose=False):
    """
    Enhanced experiment function with stable Neural Network training.
    
    Args:
        sample_size: Number of samples to use
        run_id: Run identifier for reproducibility
        edges: Edge matrix
        degrees_dict: Dictionary of node degrees
        verbose: Whether to print progress
        
    Returns:
        dict: Experiment results for all models
    """
    if verbose:
        print(f"  Enhanced Experiment Run {run_id+1}: Sample size {sample_size}")
    
    # Create dataset with proper seed control
    n_positive = sample_size // 2
    n_negative = sample_size // 2
    
    X_exp, y_exp, _ = create_representative_dataset(
        edges, degrees_dict,
        n_positive=n_positive,
        n_negative=n_negative,
        pos_method='stratified',
        neg_method='degree_matched',
        random_state=42 + run_id
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_exp, y_exp, test_size=0.2, random_state=42+run_id, stratify=y_exp
    )
    
    start_time = time.time()
    
    # Enhanced Neural Network
    nn_start_time = time.time()
    nn_result = train_enhanced_neural_network(X_train, y_train, X_test, y_test, sample_size, run_id)
    nn_training_time = time.time() - nn_start_time
    
    # Standard models for comparison
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    lr_start_time = time.time()
    lr_model = LogisticRegression(random_state=42+run_id, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_pred)
    lr_ap = average_precision_score(y_test, lr_pred)
    lr_training_time = time.time() - lr_start_time
    
    # Polynomial Logistic Regression
    plr_start_time = time.time()
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train_scaled)
    X_test_poly = poly_features.transform(X_test_scaled)
    
    plr_model = LogisticRegression(random_state=42+run_id, max_iter=1000)
    plr_model.fit(X_train_poly, y_train)
    plr_pred = plr_model.predict_proba(X_test_poly)[:, 1]
    plr_auc = roc_auc_score(y_test, plr_pred)
    plr_ap = average_precision_score(y_test, plr_pred)
    plr_training_time = time.time() - plr_start_time
    
    # Random Forest
    rf_start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42+run_id)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred)
    rf_ap = average_precision_score(y_test, rf_pred)
    rf_training_time = time.time() - rf_start_time
    
    training_time = time.time() - start_time
    
    return {
        'sample_size': sample_size,
        'run': run_id,
        'training_time': training_time,
        'models': {
            'Enhanced Stable NN': {
                'test_auc': nn_result['test_auc'],
                'test_ap': nn_result['test_ap'],
                'predictions': nn_result['predictions'],
                'training_time': nn_training_time,
                'model_info': {
                    'n_inits_averaged': nn_result['n_inits_averaged'],
                    'epochs_used': nn_result['epochs_used'],
                    'learning_rate': nn_result['final_lr'],
                    'architecture': nn_result['architecture']
                }
            },
            'Logistic Regression': {
                'test_auc': lr_auc,
                'test_ap': lr_ap,
                'predictions': lr_pred,
                'training_time': lr_training_time
            },
            'Polynomial Logistic Regression': {
                'test_auc': plr_auc,
                'test_ap': plr_ap,
                'predictions': plr_pred,
                'training_time': plr_training_time
            },
            'Random Forest': {
                'test_auc': rf_auc,
                'test_ap': rf_ap,
                'predictions': rf_pred,
                'training_time': rf_training_time
            }
        }
    }


def calculate_prediction_stability(predictions_list):
    """
    Calculate the coefficient of variation of predictions across multiple runs.
    
    Args:
        predictions_list: List of prediction arrays from different runs
        
    Returns:
        float: Coefficient of variation (std/mean)
    """
    if len(predictions_list) < 2:
        return 0.0
    
    # Check if all prediction arrays have the same shape
    shapes = [pred.shape for pred in predictions_list]
    if len(set(shapes)) > 1:
        # Handle different shapes by finding minimum length and truncating
        min_length = min(len(pred) for pred in predictions_list)
        if min_length == 0:
            return 0.0
        
        # Truncate all arrays to the minimum length
        truncated_preds = [pred[:min_length] for pred in predictions_list]
        pred_matrix = np.stack(truncated_preds, axis=0)
    else:
        # All arrays have the same shape, stack normally
        pred_matrix = np.stack(predictions_list, axis=0)
    
    # Calculate mean and std across runs for each prediction
    pred_means = np.mean(pred_matrix, axis=0)
    pred_stds = np.std(pred_matrix, axis=0)
    
    # Calculate overall coefficient of variation
    overall_mean = np.mean(pred_means)
    overall_std = np.mean(pred_stds)
    
    cv = overall_std / overall_mean if overall_mean > 0 else 0.0
    
    return cv


def analyze_enhanced_experiment_results(experiment_results, model_names=None):
    """
    Analyze enhanced experiment results with focus on stability metrics.
    
    Args:
        experiment_results: List of experiment result dictionaries
        model_names: List of model names to analyze
        
    Returns:
        dict: Analysis results including stability metrics
    """
    if model_names is None:
        model_names = ['Enhanced Stable NN', 'Logistic Regression', 
                      'Polynomial Logistic Regression', 'Random Forest']
    
    # Group results by sample size
    size_groups = {}
    for result in experiment_results:
        size = result['sample_size']
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(result)
    
    analysis = {}
    
    for size, results in size_groups.items():
        analysis[size] = {}
        
        for model_name in model_names:
            if model_name not in analysis[size]:
                analysis[size][model_name] = {}
            
            # Extract metrics for this model
            aucs = []
            aps = []
            predictions = []
            
            for result in results:
                if model_name in result['models']:
                    model_result = result['models'][model_name]
                    aucs.append(model_result['test_auc'])
                    aps.append(model_result['test_ap'])
                    predictions.append(model_result['predictions'])
            
            if aucs:
                # Calculate stability metrics
                stability = calculate_prediction_stability(predictions)
                
                analysis[size][model_name] = {
                    'mean_auc': np.mean(aucs),
                    'std_auc': np.std(aucs),
                    'cv_auc': np.std(aucs) / np.mean(aucs) if np.mean(aucs) > 0 else 0,
                    'mean_ap': np.mean(aps),
                    'std_ap': np.std(aps),
                    'cv_ap': np.std(aps) / np.mean(aps) if np.mean(aps) > 0 else 0,
                    'prediction_stability': stability,
                    'n_runs': len(aucs)
                }
    
    return analysis