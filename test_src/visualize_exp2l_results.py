"""
Create visualizations for Experiment 2L results.

Shows correlations and residuals for all models tested.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'

print("Loading and retraining models for visualization...")

# Recreate the experiment (abbreviated - just load key results)
exec(open('test_src/test_nonlinear_aggregation_exp2l.py').read())

# Now create visualizations
print("\nCreating visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

model_names = ['Linear-Baseline', 'Linear-All', 'RandomForest', 'GradientBoosting']
model_results = [results[name] for name in model_names]

# Row 1: Scatter plots (predicted vs actual for validation)
for idx, (name, res) in enumerate(zip(model_names, model_results)):
    ax = fig.add_subplot(gs[0, idx])

    y_pred = res['predictions']
    r = res['r']
    mae = res['mae']

    ax.scatter(y_val_test, y_pred, alpha=0.4, s=20, c='blue', edgecolors='none')

    # Perfect prediction line
    max_val = max(y_val_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('True Count (mean perms 11-20)', fontsize=10)
    ax.set_ylabel('Predicted Count', fontsize=10)
    ax.set_title(f'{name}\nr={r:.4f}, MAE={mae:.3f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Row 2: Residual plots
for idx, (name, res) in enumerate(zip(model_names, model_results)):
    ax = fig.add_subplot(gs[1, idx])

    y_pred = res['predictions']
    residuals = y_val_test - y_pred

    ax.scatter(y_pred, residuals, alpha=0.4, s=20, c='green', edgecolors='none')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)

    ax.set_xlabel('Predicted Count', fontsize=10)
    ax.set_ylabel('Residual (true - pred)', fontsize=10)
    ax.set_title(f'{name} Residuals', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

# Row 3: Residual histograms
for idx, (name, res) in enumerate(zip(model_names, model_results)):
    ax = fig.add_subplot(gs[2, idx])

    y_pred = res['predictions']
    residuals = y_val_test - y_pred

    ax.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)

    ax.set_xlabel('Residual', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{name} Residual Distribution', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Add statistics
    mean_res = residuals.mean()
    std_res = residuals.std()
    ax.text(0.05, 0.95, f'Mean: {mean_res:.3f}\nStd: {std_res:.3f}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(results_dir / 'experiment2l_visualizations.png', dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'experiment2l_visualizations.png'}")

# Create additional analysis plot
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model comparison bar chart
ax = axes[0, 0]
models = list(results.keys())
r_values = [results[m]['r'] for m in models]
colors = ['red' if r < 0.8 else 'orange' if r < 0.9 else 'green' for r in r_values]

bars = ax.bar(range(len(models)), r_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('Correlation (r)', fontsize=11)
ax.set_title('Model Performance Comparison\n(K=1, validate on mean perms 11-20)',
             fontsize=12, fontweight='bold')
ax.axhline(y=0.95, color='blue', linestyle='--', linewidth=2, label='Target r=0.95')
ax.axhline(y=0.80, color='gray', linestyle='--', linewidth=1, label='Previous ceiling r=0.80')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, r) in enumerate(zip(bars, r_values)):
    ax.text(i, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: MAE comparison
ax = axes[0, 1]
mae_values = [results[m]['mae'] for m in models]
ax.bar(range(len(models)), mae_values, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('Mean Absolute Error', fontsize=11)
ax.set_title('Model Error Comparison', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Residuals by true value (best model)
ax = axes[1, 0]
best_model_name = max(results.items(), key=lambda x: x[1]['r'])[0]
best_pred = results[best_model_name]['predictions']
residuals = y_val_test - best_pred

scatter = ax.scatter(y_val_test, residuals, alpha=0.4, s=20,
                    c=np.abs(residuals), cmap='RdYlGn_r', edgecolors='none')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('True Count', fontsize=11)
ax.set_ylabel('Residual', fontsize=11)
ax.set_title(f'Best Model ({best_model_name}) Residuals vs True Value',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='|Residual|')

# Plot 4: Q-Q plot for best model
ax = axes[1, 1]
from scipy import stats
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title(f'Q-Q Plot: {best_model_name} Residuals', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'experiment2l_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'experiment2l_analysis.png'}")

print("\nVisualization complete!")
