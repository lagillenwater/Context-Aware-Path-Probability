"""
Edge Prediction Training and Evaluation Script

This script consolidates all the notebook functionality for training and evaluating
edge prediction models on single permutations. It handles data loading, model training,
baseline comparisons, and comprehensive visualization.

Usage:
    python run_edge_prediction.py --permutation 001.hetmat --edge-type AeG
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import local modules
from models import EdgePredictionNN, get_model_info
from data_processing import (
    prepare_edge_prediction_data,
    load_permutation_data,
)
from training import train_edge_prediction_model
from visualization import create_probability_heatmap


class EdgePredictionRunner:
    """Main class for running edge prediction experiments."""
    
    def __init__(self, 
                 permutations_dir: Path,
                 output_dir: Path,
                 edge_type: str = "AeG",
                 source_node_type: str = "Anatomy", 
                 target_node_type: str = "Gene"):
        """
        Initialize the edge prediction runner.
        
        Args:
            permutations_dir: Directory containing permutation data
            output_dir: Directory to save results and models
            edge_type: Type of edges to predict (e.g., "AeG", "CbG", "DaG")
            source_node_type: Type of source nodes (e.g., "Anatomy", "Compound")
            target_node_type: Type of target nodes (e.g., "Gene", "Disease")
        """
        self.permutations_dir = Path(permutations_dir)
        self.output_dir = Path(output_dir)
        self.edge_type = edge_type
        self.source_node_type = source_node_type
        self.target_node_type = target_node_type
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        self.training_times = {}
        
    def get_available_permutations(self) -> list:
        """Get list of available permutation directories."""
        if not self.permutations_dir.exists():
            raise ValueError(f"Permutations directory not found: {self.permutations_dir}")
        
        available = [p.name for p in self.permutations_dir.iterdir() if p.is_dir()]
        if not available:
            raise ValueError(f"No permutations found in {self.permutations_dir}")
        
        return sorted(available)
    
    def load_permutation_data(self, permutation_name: str) -> Dict[str, Any]:
        """Load data for a specific permutation."""
        print(f"Loading permutation: {permutation_name}")
        print(f"Edge type: {self.edge_type} ({self.source_node_type} -> {self.target_node_type})")
        
        perm_data = load_permutation_data(
            permutation_name,
            self.permutations_dir,
            edge_type=self.edge_type,
            source_node_type=self.source_node_type,
            target_node_type=self.target_node_type
        )
        
        if not perm_data:
            raise ValueError(f"Failed to load permutation data for: {permutation_name}")
        
        print(f"Successfully loaded permutation: {permutation_name}")
        
        # Extract data components
        edges = perm_data["edges"]
        source_nodes = perm_data["source_nodes"]
        target_nodes = perm_data["target_nodes"]
        
        print(f"Permutation {permutation_name} data summary:")
        print(f"  {self.edge_type} edges matrix shape: {edges.shape}")
        print(f"  Number of edges: {edges.nnz}")
        print(f"  {self.source_node_type} nodes: {len(source_nodes)}")
        print(f"  {self.target_node_type} nodes: {len(target_nodes)}")
        print(f"  Matrix density: {edges.nnz / (edges.shape[0] * edges.shape[1]):.6f}")
        
        return perm_data
    
    def prepare_features(self, perm_data: Dict[str, Any], sample_negative_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for model training."""
        print(f"Preparing edge prediction data for {self.edge_type} relationships...")
        
        features, labels = prepare_edge_prediction_data(
            perm_data,
            sample_negative_ratio=sample_negative_ratio
        )
        
        print(f"Prepared {len(features)} samples for {self.edge_type} edge prediction")
        print(f"Feature vector dimension: {features.shape[1]}")
        print(f"Positive samples: {labels.sum()}")
        print(f"Negative samples: {len(labels) - labels.sum()}")
        print(f"Class balance: {labels.mean():.3f}")
        
        return features, labels
    
    def train_neural_network(self, features: np.ndarray, labels: np.ndarray, permutation_name: str) -> Dict[str, Any]:
        """Train neural network model."""
        print("\n" + "=" * 60)
        print(f"TRAINING NEURAL NETWORK ON PERMUTATION {permutation_name}")
        print("=" * 60)
        
        # Track training time
        start_time = time.time()
        
        model, train_history, test_metrics = train_edge_prediction_model(
            features,
            labels,
            epochs=50,
            batch_size=512,
            learning_rate=0.001,
            patience=10
        )
        
        training_time = time.time() - start_time
        self.training_times['neural_network'] = training_time
        
        print(f"\nTraining completed for permutation {permutation_name}!")
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Final Test AUC: {test_metrics['auc']:.4f}")
        print(f"Final Test Average Precision: {test_metrics['average_precision']:.4f}")
        
        # Save model
        self._save_neural_network_model(model, train_history, test_metrics, permutation_name, training_time, features.shape[1])
        
        return {
            'model': model,
            'train_history': train_history,
            'test_metrics': test_metrics,
            'training_time': training_time
        }
    
    def _save_neural_network_model(self, model, train_history, test_metrics, permutation_name, training_time, input_features):
        """Save neural network model and metrics."""
        # Save model
        model_filename = f"edge_prediction_model_{permutation_name}.pt"
        model_path = self.output_dir / model_filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': get_model_info(model),
            'test_metrics': test_metrics,
            'permutation_name': permutation_name,
            'train_history': train_history,
            'input_features': input_features,
            'training_time_seconds': training_time,
            'edge_type': self.edge_type,
            'source_node_type': self.source_node_type,
            'target_node_type': self.target_node_type
        }, model_path)
        
        print(f"Model saved to: {model_path}")
        
        # Save training metrics separately
        metrics_filename = f"training_metrics_{permutation_name}.json"
        metrics_path = self.output_dir / metrics_filename
        
        metrics_to_save = {
            'permutation_name': permutation_name,
            'edge_type': self.edge_type,
            'source_node_type': self.source_node_type,
            'target_node_type': self.target_node_type,
            'test_auc': float(test_metrics['auc']),
            'test_average_precision': float(test_metrics['average_precision']),
            'final_train_loss': float(train_history['train_losses'][-1]),
            'final_val_loss': float(train_history['val_losses'][-1]),
            'best_val_loss': float(test_metrics['best_val_loss']),
            'final_learning_rate': float(test_metrics['final_learning_rate']),
            'early_stopped': bool(train_history.get('early_stopped', False)),
            'best_epoch': int(train_history.get('best_epoch', len(train_history['train_losses']))),
            'total_epochs': int(len(train_history['train_losses'])),
            'input_features': int(input_features),
            'training_time_seconds': float(training_time),
            'training_time_minutes': float(training_time / 60)
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"Training metrics saved to: {metrics_path}")
    
    def train_baseline_models(self, features: np.ndarray, labels: np.ndarray, permutation_name: str) -> Dict[str, Any]:
        """Train baseline models for comparison."""
        print("\n" + "=" * 60)
        print(f"COMPREHENSIVE BASELINE COMPARISON FOR PERMUTATION {permutation_name}")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler_baseline = StandardScaler()
        X_train_scaled = scaler_baseline.fit_transform(X_train)
        X_test_scaled = scaler_baseline.transform(X_test)
        
        models_results = {}
        
        print(f"Training baseline models for permutation {permutation_name}...")
        
        # 1. Logistic Regression
        print("  Training Logistic Regression...")
        lr_start_time = time.time()
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        lr_model.fit(X_train_scaled, y_train)
        lr_test_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
        lr_test_auc = roc_auc_score(y_test, lr_test_pred)
        lr_test_ap = average_precision_score(y_test, lr_test_pred)
        
        lr_training_time = time.time() - lr_start_time
        self.training_times['logistic_regression'] = lr_training_time
        
        models_results['logistic_regression'] = {
            'model': lr_model,
            'scaler': scaler_baseline,
            'test_auc': lr_test_auc,
            'test_ap': lr_test_ap,
            'predictions': lr_test_pred,
            'true_labels': y_test,
            'name': 'Logistic Regression',
            'training_time': lr_training_time
        }
        
        print(f"    Logistic Regression training time: {lr_training_time:.3f} seconds")
        
        # 2. Polynomial Logistic Regression
        print("  Training Polynomial Logistic Regression...")
        poly_start_time = time.time()
        
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train_scaled)
        X_test_poly = poly_features.transform(X_test_scaled)
        
        poly_scaler = StandardScaler()
        X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
        X_test_poly_scaled = poly_scaler.transform(X_test_poly)
        
        poly_lr_model = LogisticRegression(random_state=42, max_iter=2000, C=0.1)
        poly_lr_model.fit(X_train_poly_scaled, y_train)
        poly_lr_test_pred = poly_lr_model.predict_proba(X_test_poly_scaled)[:, 1]
        poly_lr_test_auc = roc_auc_score(y_test, poly_lr_test_pred)
        poly_lr_test_ap = average_precision_score(y_test, poly_lr_test_pred)
        
        poly_lr_training_time = time.time() - poly_start_time
        self.training_times['polynomial_logistic'] = poly_lr_training_time
        
        models_results['polynomial_logistic'] = {
            'model': poly_lr_model,
            'scaler': scaler_baseline,
            'poly_features': poly_features,
            'poly_scaler': poly_scaler,
            'test_auc': poly_lr_test_auc,
            'test_ap': poly_lr_test_ap,
            'predictions': poly_lr_test_pred,
            'true_labels': y_test,
            'name': 'Polynomial Logistic Regression',
            'training_time': poly_lr_training_time
        }
        
        print(f"    Polynomial LR training time: {poly_lr_training_time:.3f} seconds")
        
        # 3. Random Forest
        print("  Training Random Forest...")
        rf_start_time = time.time()
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            min_samples_split=10,
            min_samples_leaf=5
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_test_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        rf_test_auc = roc_auc_score(y_test, rf_test_pred)
        rf_test_ap = average_precision_score(y_test, rf_test_pred)
        
        rf_training_time = time.time() - rf_start_time
        self.training_times['random_forest'] = rf_training_time
        
        models_results['random_forest'] = {
            'model': rf_model,
            'scaler': scaler_baseline,
            'test_auc': rf_test_auc,
            'test_ap': rf_test_ap,
            'predictions': rf_test_pred,
            'true_labels': y_test,
            'name': 'Random Forest',
            'training_time': rf_training_time
        }
        
        print(f"    Random Forest training time: {rf_training_time:.3f} seconds")
        
        # Save baseline models
        self._save_baseline_models(models_results, permutation_name)
        
        return models_results
    
    def _save_baseline_models(self, models_results: Dict[str, Any], permutation_name: str):
        """Save baseline models to disk."""
        for model_name, model_data in models_results.items():
            model_save_path = self.output_dir / f"{model_name}_model_{permutation_name}.pkl"
            
            save_data = {
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'test_metrics': {
                    'auc': model_data['test_auc'],
                    'ap': model_data['test_ap']
                },
                'training_time_seconds': model_data['training_time'],
                'permutation_name': permutation_name,
                'edge_type': self.edge_type,
                'source_node_type': self.source_node_type,
                'target_node_type': self.target_node_type
            }
            
            if model_name == 'polynomial_logistic':
                save_data['poly_features'] = model_data['poly_features']
                save_data['poly_scaler'] = model_data['poly_scaler']
            
            with open(model_save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"  {model_data['name']} model saved to: {model_save_path}")
    
    def create_visualizations(self, nn_results: Dict[str, Any], baseline_results: Dict[str, Any], 
                            features: np.ndarray, permutation_name: str):
        """Create comprehensive visualizations."""
        print("\n" + "=" * 60)
        print(f"CREATING VISUALIZATIONS FOR PERMUTATION {permutation_name}")
        print("=" * 60)
        
        # 1. Training history plot
        self._plot_training_history(nn_results['train_history'], permutation_name)
        
        # 2. ROC comparison plot
        self._plot_roc_comparison(nn_results, baseline_results, permutation_name)
        
        # 3. Probability heatmaps
        self._create_probability_heatmaps(nn_results, baseline_results, features, permutation_name)
    
    def _plot_training_history(self, train_history: Dict[str, Any], permutation_name: str):
        """Plot training and validation loss over epochs."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(train_history['train_losses']) + 1)
        ax.plot(epochs, train_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, train_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        
        if train_history.get('early_stopped', False) and 'best_epoch' in train_history:
            best_epoch = train_history['best_epoch']
            ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                      label=f'Best Epoch ({best_epoch})')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training and Validation Loss - Permutation {permutation_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"training_history_{permutation_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history plot saved to: {plot_path}")
    
    def _plot_roc_comparison(self, nn_results: Dict[str, Any], baseline_results: Dict[str, Any], permutation_name: str):
        """Plot ROC curves for all models."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Neural Network ROC
        nn_test_metrics = nn_results['test_metrics']
        nn_fpr, nn_tpr, _ = roc_curve(nn_test_metrics["true_labels"], nn_test_metrics["predictions"])
        ax.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_test_metrics["auc"]:.3f})', 
                linewidth=2, color='red')
        
        # Baseline ROCs
        colors = ['blue', 'green', 'orange']
        for i, (model_name, model_data) in enumerate(baseline_results.items()):
            fpr, tpr, _ = roc_curve(model_data['true_labels'], model_data['predictions'])
            ax.plot(fpr, tpr, label=f"{model_data['name']} (AUC = {model_data['test_auc']:.3f})", 
                   linewidth=2, color=colors[i])
        
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves Comparison - Permutation {permutation_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"roc_comparison_{permutation_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"ROC comparison plot saved to: {plot_path}")
    
    def _create_probability_heatmaps(self, nn_results: Dict[str, Any], baseline_results: Dict[str, Any], 
                                   features: np.ndarray, permutation_name: str):
        """Create probability heatmaps for all models."""
        print("Creating edge probability heatmaps for all models...")
        
        # Define degree ranges
        source_degrees = np.linspace(features[:, 0].min(), features[:, 0].max(), 30)
        target_degrees = np.linspace(features[:, 1].min(), features[:, 1].max(), 30)
        source_grid, target_grid = np.meshgrid(source_degrees, target_degrees)
        heatmap_features = np.column_stack([source_grid.ravel(), target_grid.ravel()])
        
        # Scale features for baseline models
        baseline_scaler = baseline_results['logistic_regression']['scaler']
        heatmap_features_scaled = baseline_scaler.transform(heatmap_features)
        
        # Create heatmaps for baseline models
        lr_heatmap = self._create_model_heatmap(heatmap_features_scaled, baseline_results['logistic_regression'], source_grid.shape)
        poly_heatmap = self._create_model_heatmap(heatmap_features_scaled, baseline_results['polynomial_logistic'], source_grid.shape)
        rf_heatmap = self._create_model_heatmap(heatmap_features_scaled, baseline_results['random_forest'], source_grid.shape)
        
        # Create Neural Network heatmap
        nn_scaler = nn_results['test_metrics']["scaler"]
        heatmap_features_nn_scaled = nn_scaler.transform(heatmap_features)
        
        model = nn_results['model']
        model.eval()
        with torch.no_grad():
            heatmap_tensor = torch.FloatTensor(heatmap_features_nn_scaled)
            nn_probs = model(heatmap_tensor).cpu().numpy()
        
        nn_prob_matrix = nn_probs.reshape(source_grid.shape)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot all heatmaps
        extent = [source_degrees.min(), source_degrees.max(), target_degrees.min(), target_degrees.max()]
        
        im1 = axes[0, 0].imshow(nn_prob_matrix, extent=extent, origin="lower", aspect="auto", cmap="viridis", alpha=0.8)
        axes[0, 0].set_xlabel(f"Source Degree ({self.source_node_type})")
        axes[0, 0].set_ylabel(f"Target Degree ({self.target_node_type})")
        axes[0, 0].set_title(f"Neural Network\nPermutation {permutation_name}")
        plt.colorbar(im1, ax=axes[0, 0]).set_label("Edge Probability")
        
        im2 = axes[0, 1].imshow(lr_heatmap, extent=extent, origin="lower", aspect="auto", cmap="viridis", alpha=0.8)
        axes[0, 1].set_xlabel(f"Source Degree ({self.source_node_type})")
        axes[0, 1].set_ylabel(f"Target Degree ({self.target_node_type})")
        axes[0, 1].set_title(f"Logistic Regression\nPermutation {permutation_name}")
        plt.colorbar(im2, ax=axes[0, 1]).set_label("Edge Probability")
        
        im3 = axes[1, 0].imshow(poly_heatmap, extent=extent, origin="lower", aspect="auto", cmap="viridis", alpha=0.8)
        axes[1, 0].set_xlabel(f"Source Degree ({self.source_node_type})")
        axes[1, 0].set_ylabel(f"Target Degree ({self.target_node_type})")
        axes[1, 0].set_title(f"Polynomial Logistic Regression\nPermutation {permutation_name}")
        plt.colorbar(im3, ax=axes[1, 0]).set_label("Edge Probability")
        
        im4 = axes[1, 1].imshow(rf_heatmap, extent=extent, origin="lower", aspect="auto", cmap="viridis", alpha=0.8)
        axes[1, 1].set_xlabel(f"Source Degree ({self.source_node_type})")
        axes[1, 1].set_ylabel(f"Target Degree ({self.target_node_type})")
        axes[1, 1].set_title(f"Random Forest\nPermutation {permutation_name}")
        plt.colorbar(im4, ax=axes[1, 1]).set_label("Edge Probability")
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"probability_heatmaps_{permutation_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save heatmap data
        heatmaps_path = self.output_dir / f"all_models_heatmaps_{permutation_name}.npz"
        np.savez(heatmaps_path,
                 source_degrees=source_degrees,
                 target_degrees=target_degrees,
                 neural_network=nn_prob_matrix,
                 logistic_regression=lr_heatmap,
                 polynomial_logistic=poly_heatmap,
                 random_forest=rf_heatmap,
                 permutation_name=permutation_name)
        
        print(f"Probability heatmaps plot saved to: {plot_path}")
        print(f"Heatmap data saved to: {heatmaps_path}")
    
    def _create_model_heatmap(self, features_for_pred: np.ndarray, model_data: Dict[str, Any], grid_shape: tuple) -> np.ndarray:
        """Create heatmap for a specific model."""
        if 'poly_features' in model_data:
            features_poly = model_data['poly_features'].transform(features_for_pred)
            features_poly_scaled = model_data['poly_scaler'].transform(features_poly)
            probs = model_data['model'].predict_proba(features_poly_scaled)[:, 1]
        else:
            probs = model_data['model'].predict_proba(features_for_pred)[:, 1]
        
        return probs.reshape(grid_shape)
    
    def save_comprehensive_results(self, nn_results: Dict[str, Any], baseline_results: Dict[str, Any], 
                                 permutation_name: str):
        """Save comprehensive comparison results."""
        print("\n" + "=" * 60)
        print(f"SAVING COMPREHENSIVE RESULTS FOR PERMUTATION {permutation_name}")
        print("=" * 60)
        
        # Print comparison results
        nn_test_metrics = nn_results['test_metrics']
        print(f"\nResults for Permutation {permutation_name}:")
        print(f"Neural Network       - Test AUC: {nn_test_metrics['auc']:.4f}, Test AP: {nn_test_metrics['average_precision']:.4f}, Training time: {nn_results['training_time']:.2f}s")
        
        for model_name, model_data in baseline_results.items():
            print(f"{model_data['name']:<20} - Test AUC: {model_data['test_auc']:.4f}, Test AP: {model_data['test_ap']:.4f}, Training time: {model_data['training_time']:.3f}s")
        
        # Calculate performance differences
        best_baseline_auc = max(model_data['test_auc'] for model_data in baseline_results.values())
        best_baseline_ap = max(model_data['test_ap'] for model_data in baseline_results.values())
        
        print(f"\nPerformance vs Best Baseline:")
        print(f"NN AUC improvement: {nn_test_metrics['auc'] - best_baseline_auc:+.4f}")
        print(f"NN AP improvement: {nn_test_metrics['average_precision'] - best_baseline_ap:+.4f}")
        
        # Create comprehensive results dictionary
        comparison_results = {
            'permutation_name': permutation_name,
            'edge_type': self.edge_type,
            'source_node_type': self.source_node_type,
            'target_node_type': self.target_node_type,
            'neural_network': {
                'test_auc': float(nn_test_metrics['auc']),
                'test_ap': float(nn_test_metrics['average_precision']),
                'training_time_seconds': float(nn_results['training_time']),
                'architecture': get_model_info(nn_results['model']),
                'training_params': {
                    'epochs': len(nn_results['train_history']['train_losses']),
                    'early_stopped': nn_results['train_history'].get('early_stopped', False),
                    'best_epoch': nn_results['train_history'].get('best_epoch', None),
                    'final_learning_rate': float(nn_test_metrics['final_learning_rate'])
                }
            },
            'performance_summary': {
                'best_auc': float(max(nn_test_metrics['auc'], best_baseline_auc)),
                'best_ap': float(max(nn_test_metrics['average_precision'], best_baseline_ap)),
                'nn_vs_best_baseline_auc': float(nn_test_metrics['auc'] - best_baseline_auc),
                'nn_vs_best_baseline_ap': float(nn_test_metrics['average_precision'] - best_baseline_ap)
            },
            'training_times': self.training_times.copy()
        }
        
        # Add baseline results
        for model_name, model_data in baseline_results.items():
            comparison_results[model_name] = {
                'test_auc': float(model_data['test_auc']),
                'test_ap': float(model_data['test_ap']),
                'training_time_seconds': float(model_data['training_time']),
                'model_params': self._get_model_params(model_data)
            }
        
        # Save results
        comparison_path = self.output_dir / f"comprehensive_model_comparison_{permutation_name}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Comprehensive comparison results saved to: {comparison_path}")
        
        return comparison_results
    
    def _get_model_params(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model parameters for saving."""
        model = model_data['model']
        
        if hasattr(model, 'C'):  # Logistic Regression
            return {
                'C': model.C,
                'max_iter': model.max_iter,
                'n_features': len(model.coef_[0]),
                'intercept': float(model.intercept_[0])
            }
        elif hasattr(model, 'n_estimators'):  # Random Forest
            return {
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'min_samples_split': model.min_samples_split,
                'min_samples_leaf': model.min_samples_leaf,
                'feature_importances': model.feature_importances_.tolist(),
                'n_features': model.n_features_in_
            }
        else:
            return {}
    
    def run_experiment(self, permutation_name: Optional[str] = None, sample_negative_ratio: float = 1.0):
        """Run complete experiment for a single permutation."""
        # Get available permutations
        available_permutations = self.get_available_permutations()
        
        # Select permutation
        if permutation_name:
            if permutation_name not in available_permutations:
                raise ValueError(f"Permutation '{permutation_name}' not found. Available: {available_permutations[:5]}...")
            selected_permutation = permutation_name
        else:
            selected_permutation = available_permutations[0]
            print(f"No permutation specified, using first available: {selected_permutation}")
        
        print(f"\n{'='*80}")
        print(f"STARTING EDGE PREDICTION EXPERIMENT")
        print(f"Permutation: {selected_permutation}")
        print(f"Edge Type: {self.edge_type} ({self.source_node_type} -> {self.target_node_type})")
        print(f"{'='*80}")
        
        # Load data
        perm_data = self.load_permutation_data(selected_permutation)
        
        # Prepare features
        features, labels = self.prepare_features(perm_data, sample_negative_ratio)
        
        # Train neural network
        nn_results = self.train_neural_network(features, labels, selected_permutation)
        
        # Train baseline models
        baseline_results = self.train_baseline_models(features, labels, selected_permutation)
        
        # Create visualizations
        self.create_visualizations(nn_results, baseline_results, features, selected_permutation)
        
        # Save comprehensive results
        final_results = self.save_comprehensive_results(nn_results, baseline_results, selected_permutation)
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"All results saved to: {self.output_dir}")
        print(f"{'='*80}")
        
        return final_results


def main():
    """Main function to run edge prediction experiments."""
    parser = argparse.ArgumentParser(description='Run edge prediction experiments')
    
    parser.add_argument('--permutations-dir', type=str, default='../data/permutations',
                       help='Directory containing permutation data')
    parser.add_argument('--output-dir', type=str, default='../models',
                       help='Directory to save results and models')
    parser.add_argument('--permutation', type=str, default=None,
                       help='Specific permutation to process (e.g., "001.hetmat")')
    parser.add_argument('--edge-type', type=str, default='AeG',
                       help='Type of edges to predict (e.g., AeG, CbG, DaG)')
    parser.add_argument('--source-node-type', type=str, default='Anatomy',
                       help='Type of source nodes (e.g., Anatomy, Compound, Disease)')
    parser.add_argument('--target-node-type', type=str, default='Gene',
                       help='Type of target nodes (e.g., Gene, Disease, Compound)')
    parser.add_argument('--sample-negative-ratio', type=float, default=1.0,
                       help='Ratio of negative to positive samples')
    
    args = parser.parse_args()
    
    # Create runner
    runner = EdgePredictionRunner(
        permutations_dir=args.permutations_dir,
        output_dir=args.output_dir,
        edge_type=args.edge_type,
        source_node_type=args.source_node_type,
        target_node_type=args.target_node_type
    )
    
    # Run experiment
    results = runner.run_experiment(
        permutation_name=args.permutation,
        sample_negative_ratio=args.sample_negative_ratio
    )
    
    print(f"\nExperiment completed! Results: {results['performance_summary']}")


if __name__ == "__main__":
    main()
