import json
from pathlib import Path

nb_path = Path('notebooks/22_empirical_frequency_validation.ipynb')

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Fix cell 4 with proper newlines
cell_4_source = [
    "# Load Test 5 and Test 6 models\n",
    "model_dir = repo_dir / 'results' / 'nn_optimizer_comparison'\n",
    "print(f'Loading Test 5 and Test 6 models from: {model_dir}\\n')\n",
    "\n",
    "models = {}\n",
    "\n",
    "# Test 5: MSE Loss models\n",
    "print('TEST 5: MSE Loss (No Class Weighting)')\n",
    "print('=' * 60)\n",
    "\n",
    "# Load Adam NN (MSE)\n",
    "with open(model_dir / 'test5_single_layer_nn_adam_mse.pkl', 'rb') as f:\n",
    "    adam_mse_results = pickle.load(f)\n",
    "models['Test5-Adam-MSE'] = {'model': adam_mse_results['model'], 'scaler': adam_mse_results.get('scaler')}\n",
    "print('Loaded Adam NN (MSE)')\n",
    "\n",
    "# Load L-BFGS NN (MSE)\n",
    "with open(model_dir / 'test5_single_layer_nn_lbfgs_mse.pkl', 'rb') as f:\n",
    "    lbfgs_mse_results = pickle.load(f)\n",
    "models['Test5-LBFGS-MSE'] = {'model': lbfgs_mse_results['model'], 'scaler': lbfgs_mse_results.get('scaler')}\n",
    "print('Loaded L-BFGS NN (MSE)')\n",
    "\n",
    "# Load SimpleNN (MSE)\n",
    "with open(model_dir / 'test5_simple_nn_mse.pkl', 'rb') as f:\n",
    "    simplenn_mse_results = pickle.load(f)\n",
    "models['Test5-SimpleNN-MSE'] = {'model': simplenn_mse_results['model'], 'scaler': simplenn_mse_results.get('scaler')}\n",
    "print('Loaded SimpleNN (MSE)')\n",
    "\n",
    "# Load Ridge\n",
    "with open(model_dir / 'test5_ridge.pkl', 'rb') as f:\n",
    "    ridge_results = pickle.load(f)\n",
    "models['Test5-Ridge'] = {'model': ridge_results['model'], 'scaler': ridge_results.get('scaler')}\n",
    "print('Loaded Ridge')\n",
    "\n",
    "print()\n",
    "\n",
    "# Test 6: Weighted MSE Loss models\n",
    "print('TEST 6: Weighted MSE Loss')\n",
    "print('=' * 60)\n",
    "\n",
    "# Load Adam NN (Weighted MSE)\n",
    "with open(model_dir / 'test6_single_layer_nn_adam_weighted_mse.pkl', 'rb') as f:\n",
    "    adam_wmse_results = pickle.load(f)\n",
    "models['Test6-Adam-WeightedMSE'] = {'model': adam_wmse_results['model'], 'scaler': adam_wmse_results.get('scaler')}\n",
    "print('Loaded Adam NN (Weighted MSE)')\n",
    "\n",
    "# Load L-BFGS NN (Weighted MSE)\n",
    "with open(model_dir / 'test6_single_layer_nn_lbfgs_weighted_mse.pkl', 'rb') as f:\n",
    "    lbfgs_wmse_results = pickle.load(f)\n",
    "models['Test6-LBFGS-WeightedMSE'] = {'model': lbfgs_wmse_results['model'], 'scaler': lbfgs_wmse_results.get('scaler')}\n",
    "print('Loaded L-BFGS NN (Weighted MSE)')\n",
    "\n",
    "# Load SimpleNN (Weighted MSE)\n",
    "with open(model_dir / 'test6_simple_nn_weighted_mse.pkl', 'rb') as f:\n",
    "    simplenn_wmse_results = pickle.load(f)\n",
    "models['Test6-SimpleNN-WeightedMSE'] = {'model': simplenn_wmse_results['model'], 'scaler': simplenn_wmse_results.get('scaler')}\n",
    "print('Loaded SimpleNN (Weighted MSE)')\n",
    "\n",
    "# Load Ridge (Weighted)\n",
    "with open(model_dir / 'test6_ridge_weighted.pkl', 'rb') as f:\n",
    "    ridge_weighted_results = pickle.load(f)\n",
    "models['Test6-Ridge-Weighted'] = {'model': ridge_weighted_results['model'], 'scaler': ridge_weighted_results.get('scaler')}\n",
    "print('Loaded Ridge (Weighted)')\n",
    "\n",
    "print(f'\\nLoaded {len(models)} models total (4 Test 5 + 4 Test 6)')"
]

nb['cells'][4]['source'] = cell_4_source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed cell 4 in notebook 22')
