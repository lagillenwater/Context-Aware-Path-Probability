"""
Visualization Utilities for Neural Network Analysis

This module contains functions for plotting training history, model performance,
and creating various visualizations for analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import torch


def plot_training_history(train_history):
    """Plot training history including loss curves, validation AUC, and learning rate."""
    # Determine number of subplots based on available data
    has_lr = 'learning_rates' in train_history
    n_plots = 4 if has_lr else 3
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Loss curves
    axes[0].plot(train_history['train_losses'], label='Training Loss', alpha=0.8)
    axes[0].plot(train_history['val_losses'], label='Validation Loss', alpha=0.8)
    
    # Mark best epoch if available
    if 'best_epoch' in train_history and train_history['best_epoch'] is not None:
        best_epoch = train_history['best_epoch']
        axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, 
                       label=f'Best Epoch ({best_epoch})')
    
    # Mark early stopping if it occurred
    if train_history.get('early_stopped', False):
        axes[0].text(0.02, 0.98, 'Early Stopped', transform=axes[0].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation AUC
    axes[1].plot(train_history['val_aucs'], label='Validation AUC', color='green', alpha=0.8)
    
    # Mark best epoch
    if 'best_epoch' in train_history and train_history['best_epoch'] is not None:
        best_epoch = train_history['best_epoch']
        axes[1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Validation AUC Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Learning rate (if available)
    if has_lr:
        axes[2].plot(train_history['learning_rates'], label='Learning Rate', color='orange', alpha=0.8)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')  # Log scale for better visualization
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Final metrics summary
    final_plot_idx = 3 if has_lr else 2
    final_auc = train_history['val_aucs'][-1]
    final_train_loss = train_history['train_losses'][-1]
    final_val_loss = train_history['val_losses'][-1]
    
    metrics = ['Val AUC', 'Train Loss', 'Val Loss']
    values = [final_auc, final_train_loss, final_val_loss]
    colors = ['green', 'blue', 'red']
    
    bars = axes[final_plot_idx].bar(metrics, values, color=colors, alpha=0.7)
    axes[final_plot_idx].set_ylabel('Value')
    axes[final_plot_idx].set_title('Final Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[final_plot_idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                 f'{value:.4f}', ha='center', va='bottom')
    
    # Adjust y-axis for better visualization
    if final_auc > 0.1:  # Only adjust if AUC is reasonable
        axes[final_plot_idx].set_ylim(0, max(1.0, max(values) * 1.1))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {len(train_history['train_losses'])}")
    print(f"  Early stopped: {'Yes' if train_history.get('early_stopped', False) else 'No'}")
    if 'best_epoch' in train_history and train_history['best_epoch'] is not None:
        print(f"  Best epoch: {train_history['best_epoch']}")
    print(f"  Final validation AUC: {final_auc:.4f}")
    print(f"  Final validation loss: {final_val_loss:.4f}")
    if has_lr:
        print(f"  Final learning rate: {train_history['learning_rates'][-1]:.2e}")


def evaluate_model_performance(test_metrics):
    """Evaluate and visualize model performance."""
    predictions = test_metrics['predictions']
    true_labels = test_metrics['true_labels']
    auc = test_metrics['auc']
    ap = test_metrics['average_precision']
    
    print(f"Test AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    # Create evaluation plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    axes[0, 1].plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Prediction distribution
    axes[1, 0].hist(predictions[true_labels == 0], bins=50, alpha=0.7, label='Negative', density=True)
    axes[1, 0].hist(predictions[true_labels == 1], bins=50, alpha=0.7, label='Positive', density=True)
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Confusion matrix at different thresholds
    thresholds = [0.3, 0.5, 0.7]
    threshold_results = []
    
    for thresh in thresholds:
        pred_binary = (predictions >= thresh).astype(int)
        cm = confusion_matrix(true_labels, pred_binary)
        threshold_results.append({'threshold': thresh, 'confusion_matrix': cm})
    
    # Plot confusion matrix for threshold = 0.5
    cm = threshold_results[1]['confusion_matrix']
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].figure.colorbar(im, ax=axes[1, 1])
    axes[1, 1].set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   title='Confusion Matrix (threshold=0.5)',
                   ylabel='True label',
                   xlabel='Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()
    
    return threshold_results


def create_probability_heatmap(model, scaler, source_degrees, target_degrees, resolution=100):
    """
    Create a heatmap showing edge prediction probabilities across degree ranges.
    
    Parameters:
    -----------
    model : EdgePredictionNN
        Trained neural network model
    scaler : StandardScaler
        Fitted scaler from training
    source_degrees : array-like
        Range of source node degrees
    target_degrees : array-like
        Range of target node degrees
    resolution : int
        Resolution of the heatmap grid
    
    Returns:
    --------
    probability_matrix : numpy.ndarray
        Matrix of prediction probabilities
    source_degrees : numpy.ndarray
        Source degree grid values
    target_degrees : numpy.ndarray
        Target degree grid values
    """
    # Create degree grids
    source_grid, target_grid = np.meshgrid(source_degrees, target_degrees)
    
    # Flatten grids to create feature matrix
    features_grid = np.column_stack([
        source_grid.flatten(),
        target_grid.flatten()
    ])
    
    # Scale features using the same scaler from training
    features_scaled = scaler.transform(features_grid)
    
    # Convert to PyTorch tensor
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        probabilities = model(features_tensor).cpu().numpy()
    
    # Reshape back to grid
    probability_matrix = probabilities.reshape(resolution, resolution)
    
    return probability_matrix, source_degrees, target_degrees


def plot_permutation_comparison(all_results):
    """
    Plot comparison of results across different permutations.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for each permutation
    """
    successful_results = {k: v for k, v in all_results.items() if v is not None}
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    # Extract metrics
    perm_names = list(successful_results.keys())
    aucs = [result['test_metrics']['auc'] for result in successful_results.values()]
    aps = [result['test_metrics']['average_precision'] for result in successful_results.values()]
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC comparison
    x_pos = np.arange(len(perm_names))
    axes[0].bar(x_pos, aucs, alpha=0.7, color='blue')
    axes[0].set_xlabel('Permutation')
    axes[0].set_ylabel('AUC Score')
    axes[0].set_title('AUC Scores Across Permutations')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(perm_names, rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Average Precision comparison
    axes[1].bar(x_pos, aps, alpha=0.7, color='green')
    axes[1].set_xlabel('Permutation')
    axes[1].set_ylabel('Average Precision')
    axes[1].set_title('Average Precision Across Permutations')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(perm_names, rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Mean AP: {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"Min AUC: {np.min(aucs):.4f}, Max AUC: {np.max(aucs):.4f}")
    print(f"Min AP: {np.min(aps):.4f}, Max AP: {np.max(aps):.4f}")
