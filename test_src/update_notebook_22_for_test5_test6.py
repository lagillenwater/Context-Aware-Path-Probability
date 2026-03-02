import json
from pathlib import Path

nb_path = Path('notebooks/22_empirical_frequency_validation.ipynb')

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Update cell 0 (markdown header)
nb['cells'][0]['source'] = [
    "# Empirical Frequency Validation - Test 5 and Test 6\n",
    "\n",
    "Compare Test 5 (MSE) and Test 6 (Weighted MSE) model predictions against empirical frequencies using notebook 04's approach."
]

# Update cell 1 (imports) - add Ridge
imports_cell_source = nb['cells'][1]['source']
# Insert Ridge import after sklearn.model_selection line
new_imports = []
for line in imports_cell_source:
    new_imports.append(line)
    if 'from sklearn.model_selection import' in line:
        new_imports.append('from sklearn.linear_model import Ridge\n')
nb['cells'][1]['source'] = new_imports

# Update cell 4 (load models)
cell_4_new_source = """# Load Test 5 and Test 6 models
model_dir = repo_dir / 'results' / 'nn_optimizer_comparison'
print(f'Loading Test 5 and Test 6 models from: {model_dir}\\n')

models = {}

# Test 5: MSE Loss models
print('TEST 5: MSE Loss (No Class Weighting)')
print('=' * 60)

# Load Adam NN (MSE)
with open(model_dir / 'test5_single_layer_nn_adam_mse.pkl', 'rb') as f:
    adam_mse_results = pickle.load(f)
models['Test5-Adam-MSE'] = {'model': adam_mse_results['model'], 'scaler': adam_mse_results.get('scaler')}
print('Loaded Adam NN (MSE)')

# Load L-BFGS NN (MSE)
with open(model_dir / 'test5_single_layer_nn_lbfgs_mse.pkl', 'rb') as f:
    lbfgs_mse_results = pickle.load(f)
models['Test5-LBFGS-MSE'] = {'model': lbfgs_mse_results['model'], 'scaler': lbfgs_mse_results.get('scaler')}
print('Loaded L-BFGS NN (MSE)')

# Load SimpleNN (MSE)
with open(model_dir / 'test5_simple_nn_mse.pkl', 'rb') as f:
    simplenn_mse_results = pickle.load(f)
models['Test5-SimpleNN-MSE'] = {'model': simplenn_mse_results['model'], 'scaler': simplenn_mse_results.get('scaler')}
print('Loaded SimpleNN (MSE)')

# Load Ridge
with open(model_dir / 'test5_ridge.pkl', 'rb') as f:
    ridge_results = pickle.load(f)
models['Test5-Ridge'] = {'model': ridge_results['model'], 'scaler': ridge_results.get('scaler')}
print('Loaded Ridge')

print()

# Test 6: Weighted MSE Loss models
print('TEST 6: Weighted MSE Loss')
print('=' * 60)

# Load Adam NN (Weighted MSE)
with open(model_dir / 'test6_single_layer_nn_adam_weighted_mse.pkl', 'rb') as f:
    adam_wmse_results = pickle.load(f)
models['Test6-Adam-WeightedMSE'] = {'model': adam_wmse_results['model'], 'scaler': adam_wmse_results.get('scaler')}
print('Loaded Adam NN (Weighted MSE)')

# Load L-BFGS NN (Weighted MSE)
with open(model_dir / 'test6_single_layer_nn_lbfgs_weighted_mse.pkl', 'rb') as f:
    lbfgs_wmse_results = pickle.load(f)
models['Test6-LBFGS-WeightedMSE'] = {'model': lbfgs_wmse_results['model'], 'scaler': lbfgs_wmse_results.get('scaler')}
print('Loaded L-BFGS NN (Weighted MSE)')

# Load SimpleNN (Weighted MSE)
with open(model_dir / 'test6_simple_nn_weighted_mse.pkl', 'rb') as f:
    simplenn_wmse_results = pickle.load(f)
models['Test6-SimpleNN-WeightedMSE'] = {'model': simplenn_wmse_results['model'], 'scaler': simplenn_wmse_results.get('scaler')}
print('Loaded SimpleNN (Weighted MSE)')

# Load Ridge (Weighted)
with open(model_dir / 'test6_ridge_weighted.pkl', 'rb') as f:
    ridge_weighted_results = pickle.load(f)
models['Test6-Ridge-Weighted'] = {'model': ridge_weighted_results['model'], 'scaler': ridge_weighted_results.get('scaler')}
print('Loaded Ridge (Weighted)')

print(f'\\nLoaded {len(models)} models total (4 Test 5 + 4 Test 6)')
"""

nb['cells'][4]['source'] = cell_4_new_source.split('\n')

# Update cell 6 (summary)
cell_6_source = nb['cells'][6]['source']
new_cell_6 = []
for line in cell_6_source:
    if "print('\\nSUMMARY: Test 3 Models vs Empirical Frequencies')" in line:
        new_cell_6.append("print('\\nSUMMARY: Test 5 and Test 6 Models vs Empirical Frequencies')\n")
    elif "'test4_correlation_summary.csv'" in line:
        new_cell_6.append("summary_path = results_dir / 'test5_test6_correlation_summary.csv'\n")
    else:
        new_cell_6.append(line)
nb['cells'][6]['source'] = new_cell_6

# Save updated notebook
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Updated {nb_path}')
print('Changes:')
print('  - Cell 0: Updated title to "Test 5 and Test 6"')
print('  - Cell 1: Added Ridge import')
print('  - Cell 4: Updated to load 8 models (4 Test 5 + 4 Test 6)')
print('  - Cell 6: Updated summary title and filename')
