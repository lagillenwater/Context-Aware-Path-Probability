import json
from pathlib import Path

# Create comprehensive notebook 22 with all tests 1-6
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def create_cell(cell_type, source):
    """Create a notebook cell."""
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]
        if source and not source[-1].endswith('\n'):
            source[-1] = source[-1].rstrip('\n')

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source
    }

    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

    return cell

# Cell 0: Title
notebook["cells"].append(create_cell("markdown", "# Empirical Frequency Validation - All Tests (1-6)\n\nComprehensive comparison of all model predictions against empirical edge frequencies."))

# Cell 1: Imports
imports_code = """import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Add src to path
sys.path.insert(0, '../src')
sys.path.insert(0, 'src')

# Import project modules
from simple_models import SingleLayerNN
from model_comparison import SimpleNN, prepare_edge_features_and_labels, filter_zero_degree_nodes
from model_training import predict_with_model

# Setup paths
repo_dir = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'empirical_frequency_validation'
results_dir.mkdir(parents=True, exist_ok=True)

print('Imports complete')
print(f'Results directory: {results_dir}')"""

notebook["cells"].append(create_cell("code", imports_code))

# Cell 2: Load and prepare data
data_code = """# Load CbG edge file
edge_type = 'CbG'
edge_file_path = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

print(f'Loading edge data: {edge_file_path}')
edge_matrix = sp.load_npz(str(edge_file_path))
print(f'Original edge matrix: {edge_matrix.shape} with {edge_matrix.nnz} edges')

# Apply zero-degree filtering
filtered_edge_matrix, source_mapping, target_mapping = filter_zero_degree_nodes(edge_matrix)
print(f'Filtered edge matrix: {filtered_edge_matrix.shape} with {filtered_edge_matrix.nnz} edges')

# Save filtered matrix temporarily
filtered_edge_path = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'filtered_{edge_type}_temp.sparse.npz'
sp.save_npz(str(filtered_edge_path), filtered_edge_matrix)

# Prepare features and labels
print('\\nPreparing edge features and labels...')
X, y = prepare_edge_features_and_labels(
    str(filtered_edge_path),
    sample_ratio=0.01,
    adaptive_sampling=True,
    enhanced_features=False
)

filtered_edge_path.unlink()

print(f'Loaded {X.shape[0]} samples with {X.shape[1]} features')
print(f'Positive samples: {y.sum()}, Negative samples: {(1-y).sum()}')

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'\\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}')"""

notebook["cells"].append(create_cell("code", data_code))

# Cell 3: Load empirical frequencies
empirical_code = """# Load empirical frequencies
empirical_freq_file = repo_dir / 'results' / 'empirical_edge_frequencies' / f'edge_frequency_by_degree_{edge_type}.csv'
print(f'Loading empirical frequencies from: {empirical_freq_file}')

empirical_df = pd.read_csv(empirical_freq_file)
print(f'Loaded {len(empirical_df)} unique degree combinations')

# Create empirical lookup dictionary
empirical_lookup = {}
for _, row in empirical_df.iterrows():
    key = (int(row['source_degree']), int(row['target_degree']))
    empirical_lookup[key] = row['frequency']

print(f'Empirical frequency range: {empirical_df["frequency"].min():.6f} - {empirical_df["frequency"].max():.6f}')"""

notebook["cells"].append(create_cell("code", empirical_code))

# Cell 4: Load ALL models
load_models_code = """# Load ALL models from Tests 1-6
model_dir = repo_dir / 'results' / 'nn_optimizer_comparison'
models = {}
model_predictions = {}

print('Loading models from all tests...\\n')
print('=' * 80)

# Test 1: Baseline (BCE, No Weighting)
print('\\nTEST 1: Baseline (BCE, No Class Weighting)')
print('-' * 60)
with open(model_dir / 'single_layer_nn_adam.pkl', 'rb') as f:
    models['Test1-Adam'] = pickle.load(f)
print('Loaded: Adam NN')

with open(model_dir / 'single_layer_nn_lbfgs.pkl', 'rb') as f:
    models['Test1-LBFGS'] = pickle.load(f)
print('Loaded: L-BFGS NN')

with open(model_dir / 'logistic_regression.pkl', 'rb') as f:
    models['Test1-LogReg'] = pickle.load(f)
print('Loaded: LogReg')

# Test 3: Class Weighting
print('\\nTEST 3: BCE with Class Weighting')
print('-' * 60)
with open(model_dir / 'test3_single_layer_nn_adam_weighted.pkl', 'rb') as f:
    models['Test3-Adam-Weighted'] = pickle.load(f)
print('Loaded: Adam NN (Weighted)')

with open(model_dir / 'test3_single_layer_nn_lbfgs_weighted.pkl', 'rb') as f:
    models['Test3-LBFGS-Weighted'] = pickle.load(f)
print('Loaded: L-BFGS NN (Weighted)')

with open(model_dir / 'test3_logreg_weighted.pkl', 'rb') as f:
    models['Test3-LogReg-Weighted'] = pickle.load(f)
print('Loaded: LogReg (Weighted)')

with open(model_dir / 'test3_simple_nn.pkl', 'rb') as f:
    models['Test3-SimpleNN'] = pickle.load(f)
print('Loaded: SimpleNN')

# Test 4: Class Weighting + StandardScaler
print('\\nTEST 4: BCE with Class Weighting + StandardScaler')
print('-' * 60)
with open(model_dir / 'test4_single_layer_nn_adam_weighted_scaled.pkl', 'rb') as f:
    models['Test4-Adam-Weighted-Scaled'] = pickle.load(f)
print('Loaded: Adam NN (Weighted+Scaled)')

with open(model_dir / 'test4_single_layer_nn_lbfgs_weighted_scaled.pkl', 'rb') as f:
    models['Test4-LBFGS-Weighted-Scaled'] = pickle.load(f)
print('Loaded: L-BFGS NN (Weighted+Scaled)')

with open(model_dir / 'test4_logreg_weighted_scaled.pkl', 'rb') as f:
    models['Test4-LogReg-Weighted-Scaled'] = pickle.load(f)
print('Loaded: LogReg (Weighted+Scaled)')

with open(model_dir / 'test4_simple_nn_weighted_scaled.pkl', 'rb') as f:
    models['Test4-SimpleNN-Weighted-Scaled'] = pickle.load(f)
print('Loaded: SimpleNN (Weighted+Scaled)')

# Test 5: MSE Loss
print('\\nTEST 5: MSE Loss (No Class Weighting)')
print('-' * 60)
with open(model_dir / 'test5_single_layer_nn_adam_mse.pkl', 'rb') as f:
    models['Test5-Adam-MSE'] = pickle.load(f)
print('Loaded: Adam NN (MSE)')

with open(model_dir / 'test5_single_layer_nn_lbfgs_mse.pkl', 'rb') as f:
    models['Test5-LBFGS-MSE'] = pickle.load(f)
print('Loaded: L-BFGS NN (MSE)')

with open(model_dir / 'test5_simple_nn_mse.pkl', 'rb') as f:
    models['Test5-SimpleNN-MSE'] = pickle.load(f)
print('Loaded: SimpleNN (MSE)')

with open(model_dir / 'test5_ridge.pkl', 'rb') as f:
    models['Test5-Ridge'] = pickle.load(f)
print('Loaded: Ridge')

# Test 6: Weighted MSE Loss
print('\\nTEST 6: Weighted MSE Loss')
print('-' * 60)
with open(model_dir / 'test6_single_layer_nn_adam_weighted_mse.pkl', 'rb') as f:
    models['Test6-Adam-WeightedMSE'] = pickle.load(f)
print('Loaded: Adam NN (Weighted MSE)')

with open(model_dir / 'test6_single_layer_nn_lbfgs_weighted_mse.pkl', 'rb') as f:
    models['Test6-LBFGS-WeightedMSE'] = pickle.load(f)
print('Loaded: L-BFGS NN (Weighted MSE)')

with open(model_dir / 'test6_simple_nn_weighted_mse.pkl', 'rb') as f:
    models['Test6-SimpleNN-WeightedMSE'] = pickle.load(f)
print('Loaded: SimpleNN (Weighted MSE)')

with open(model_dir / 'test6_ridge_weighted.pkl', 'rb') as f:
    models['Test6-Ridge-Weighted'] = pickle.load(f)
print('Loaded: Ridge (Weighted)')

print('\\n' + '=' * 80)
print(f'Total models loaded: {len(models)}')"""

notebook["cells"].append(create_cell("code", load_models_code))

# Cell 5: Generate predictions and calculate correlations
predict_code = """# Generate predictions for all models
results = {}

print('\\nGenerating predictions and matching to empirical frequencies...')
print('=' * 80)

for model_name, model_data in models.items():
    print(f'\\n{model_name}:')

    model = model_data['model']
    scaler = model_data.get('scaler')
    predictions = predict_with_model(model, X_test, model_name, scaler)

    print(f'  Predictions: {len(predictions)}, Range: [{predictions.min():.4f}, {predictions.max():.4f}]')

    # Match to empirical frequencies
    matched_pred = []
    matched_emp = []

    for i in range(len(X_test)):
        key = (int(X_test[i, 0]), int(X_test[i, 1]))
        if key in empirical_lookup:
            matched_pred.append(predictions[i])
            matched_emp.append(empirical_lookup[key])

    matched_pred = np.array(matched_pred)
    matched_emp = np.array(matched_emp)

    # Calculate metrics
    pearson_r, _ = pearsonr(matched_emp, matched_pred)
    spearman_r, _ = spearmanr(matched_emp, matched_pred)
    rmse = np.sqrt(mean_squared_error(matched_emp, matched_pred))
    mae = mean_absolute_error(matched_emp, matched_pred)
    r2 = r2_score(matched_emp, matched_pred)

    results[model_name] = {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'matched_pred': matched_pred,
        'matched_emp': matched_emp,
        'n_matched': len(matched_pred)
    }

    print(f'  Pearson r: {pearson_r:.4f}, Spearman r: {spearman_r:.4f}, RMSE: {rmse:.4f}')

print('\\n' + '=' * 80)"""

notebook["cells"].append(create_cell("code", predict_code))

# Cell 6: Summary table
summary_code = """# Create summary table
summary_df = pd.DataFrame({
    name: {
        'pearson_r': res['pearson_r'],
        'spearman_r': res['spearman_r'],
        'rmse': res['rmse'],
        'mae': res['mae'],
        'r2': res['r2'],
        'n_matched': res['n_matched']
    }
    for name, res in results.items()
}).T

summary_df = summary_df.sort_values('pearson_r', ascending=False)

print('\\nSUMMARY: All Tests (1-6) vs Empirical Frequencies')
print('=' * 100)
print(summary_df.to_string())

# Save summary
summary_path = results_dir / 'all_tests_correlation_summary.csv'
summary_df.to_csv(summary_path)
print(f'\\nSaved summary to: {summary_path}')"""

notebook["cells"].append(create_cell("code", summary_code))

# Cell 7: Individual scatter plots (grid)
scatter_grid_code = """# Create structured grid: rows = model type, columns = configuration
row_models = ['LogReg', 'Ridge', 'Adam', 'LBFGS', 'SimpleNN']
col_configs = ['Baseline', 'Weighted', 'Weighted+Scaled', 'MSE', 'Weighted MSE']

# Create mapping of (row, col) to model name
grid_map = {}
for model_name in results.keys():
    row_idx = None
    col_idx = None

    # Determine row (model type)
    if 'LogReg' in model_name:
        row_idx = 0
    elif 'Ridge' in model_name:
        row_idx = 1
    elif 'Adam' in model_name and 'LBFGS' not in model_name:
        row_idx = 2
    elif 'LBFGS' in model_name:
        row_idx = 3
    elif 'SimpleNN' in model_name:
        row_idx = 4

    # Determine column (configuration)
    if model_name.startswith('Test1'):
        col_idx = 0  # Baseline
    elif model_name.startswith('Test3'):
        col_idx = 1  # Weighted
    elif model_name.startswith('Test4'):
        col_idx = 2  # Weighted+Scaled
    elif model_name.startswith('Test5') and 'Ridge' not in model_name:
        col_idx = 3  # MSE
    elif model_name.startswith('Test5') and 'Ridge' in model_name:
        col_idx = 0  # Ridge baseline is MSE
    elif model_name.startswith('Test6'):
        col_idx = 4 if 'Ridge' not in model_name else 1  # Weighted MSE or Ridge weighted

    if row_idx is not None and col_idx is not None:
        grid_map[(row_idx, col_idx)] = model_name

nrows = len(row_models)
ncols = len(col_configs)
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

for row in range(nrows):
    for col in range(ncols):
        ax = axes[row, col]

        if (row, col) in grid_map:
            model_name = grid_map[(row, col)]
            res = results[model_name]

            emp = res['matched_emp']
            pred = res['matched_pred']
            r = res['pearson_r']

            # Scatter plot with density coloring
            from scipy.stats import gaussian_kde
            xy = np.vstack([emp, pred])
            z = gaussian_kde(xy)(xy)
            scatter = ax.scatter(emp, pred, c=z, s=15, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Density')

            # Perfect correlation line
            ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')

            # Labels and title
            ax.set_xlabel('Empirical Frequency', fontsize=9)
            ax.set_ylabel('Predicted Frequency', fontsize=9)
            ax.set_title(f'{model_name}\\nr = {r:.4f}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=7, loc='upper left')
        else:
            # Empty cell
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])

# Add row and column labels
for row, model_type in enumerate(row_models):
    axes[row, 0].set_ylabel(f'{model_type}\\nPredicted Freq', fontsize=11, fontweight='bold')

for col, config in enumerate(col_configs):
    axes[0, col].set_title(f'{config}\\n{axes[0, col].get_title()}', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(results_dir / 'all_tests_scatter_grid.png', dpi=300, bbox_inches='tight')
print(f'Saved scatter grid to: {results_dir / "all_tests_scatter_grid.png"}')
plt.show()"""

notebook["cells"].append(create_cell("code", scatter_grid_code))

# Cell 8: High-resolution individual plots for top models
top_models_code = """# Create high-resolution plots for top 6 models
from scipy.stats import gaussian_kde

top_n = 6
top_models = sorted(results.items(), key=lambda x: x[1]['pearson_r'], reverse=True)[:top_n]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (model_name, res) in enumerate(top_models):
    ax = axes[idx]

    emp = res['matched_emp']
    pred = res['matched_pred']
    r_pearson = res['pearson_r']
    r_spearman = res['spearman_r']
    rmse = res['rmse']

    # Scatter plot with density-based coloring using gaussian_kde
    xy = np.vstack([emp, pred])
    z = gaussian_kde(xy)(xy)
    scatter = ax.scatter(emp, pred, c=z, s=20, cmap='viridis', alpha=0.7, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label='Point Density')

    # Perfect correlation line
    ax.plot([0, 1], [0, 1], 'r--', linewidth=3, label='y=x')

    # Labels and title
    ax.set_xlabel('Empirical Frequency', fontsize=12)
    ax.set_ylabel('Predicted Frequency', fontsize=12)
    ax.set_title(f'{model_name}\\nPearson r={r_pearson:.4f}, Spearman r={r_spearman:.4f}, RMSE={rmse:.4f}',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(results_dir / 'top_models_detailed_scatter.png', dpi=300, bbox_inches='tight')
print(f'Saved top models scatter to: {results_dir / "top_models_detailed_scatter.png"}')
plt.show()"""

notebook["cells"].append(create_cell("code", top_models_code))

# Cell 9: Comparison by test
test_comparison_code = """# Compare performance across test configurations
test_groups = {
    'Test 1 (Baseline)': [k for k in results.keys() if k.startswith('Test1')],
    'Test 3 (Weighted)': [k for k in results.keys() if k.startswith('Test3')],
    'Test 4 (Weighted+Scaled)': [k for k in results.keys() if k.startswith('Test4')],
    'Test 5 (MSE)': [k for k in results.keys() if k.startswith('Test5')],
    'Test 6 (Weighted MSE)': [k for k in results.keys() if k.startswith('Test6')],
}

test_summary = {}
for test_name, model_list in test_groups.items():
    if model_list:
        pearson_values = [results[m]['pearson_r'] for m in model_list]
        test_summary[test_name] = {
            'mean_pearson': np.mean(pearson_values),
            'max_pearson': np.max(pearson_values),
            'min_pearson': np.min(pearson_values),
            'n_models': len(model_list)
        }

test_summary_df = pd.DataFrame(test_summary).T
print('\\nSummary by Test Configuration:')
print('=' * 80)
print(test_summary_df.to_string())

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
x = np.arange(len(test_summary))
means = [test_summary[t]['mean_pearson'] for t in test_summary.keys()]
ax.bar(x, means, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax.set_xticks(x)
ax.set_xticklabels(test_summary.keys(), rotation=15, ha='right')
ax.set_ylabel('Mean Pearson Correlation', fontsize=12)
ax.set_title('Average Model Performance by Test Configuration', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

for i, (test_name, vals) in enumerate(test_summary.items()):
    ax.text(i, vals['mean_pearson'] + 0.02, f"{vals['mean_pearson']:.3f}",
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(results_dir / 'test_configuration_comparison.png', dpi=300, bbox_inches='tight')
print(f'\\nSaved test comparison to: {results_dir / "test_configuration_comparison.png"}')
plt.show()"""

notebook["cells"].append(create_cell("code", test_comparison_code))

# Save notebook
output_path = Path('notebooks/22_empirical_frequency_validation.ipynb')
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f'Created comprehensive notebook: {output_path}')
print(f'Total cells: {len(notebook["cells"])}')
print('Cells:')
print('  0: Title')
print('  1: Imports')
print('  2: Load and prepare data')
print('  3: Load empirical frequencies')
print('  4: Load ALL models (Tests 1-6)')
print('  5: Generate predictions and calculate correlations')
print('  6: Summary table')
print('  7: Scatter plot grid (all models)')
print('  8: Detailed scatter plots (top 6 models)')
print('  9: Comparison by test configuration')
