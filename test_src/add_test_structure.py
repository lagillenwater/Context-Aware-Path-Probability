"""
Add Test 1 and Test 2 structure to notebook 21.

This script reorganizes the notebook into clear test sections with
result persistence and visualization code.
"""

import json
import sys


def load_notebook(filepath):
    """Load a Jupyter notebook from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(notebook, filepath):
    """Save a Jupyter notebook to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write('\n')


def create_markdown_cell(content):
    """Create a markdown cell with given content."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content if isinstance(content, list) else [content]
    }


def create_code_cell(content, outputs=None):
    """Create a code cell with given content."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": content if isinstance(content, list) else [content]
    }


def find_cell_by_content(cells, search_text):
    """Find index of cell containing search text."""
    for i, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if search_text in source:
            return i
    return -1


def get_cells_up_to(cells, search_text):
    """Get all cells up to (not including) cell with search_text."""
    idx = find_cell_by_content(cells, search_text)
    if idx == -1:
        return cells
    return cells[:idx]


def main():
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/21_nn_architecture_exploration.ipynb'

    print(f"Loading notebook: {notebook_path}")
    notebook = load_notebook(notebook_path)

    cells = notebook['cells']

    # Find key sections
    sanity_check_idx = find_cell_by_content(cells, "## 2. Sanity Check")
    empirical_freq_idx = find_cell_by_content(cells, "## 3. Empirical Frequency")
    recommendations_idx = find_cell_by_content(cells, "## 8. Recommendations")

    print(f"Found sections:")
    print(f"  Sanity Check: {sanity_check_idx}")
    print(f"  Empirical Frequency: {empirical_freq_idx}")
    print(f"  Recommendations: {recommendations_idx}")

    # Keep introduction cells (before sanity check)
    intro_cells = cells[:sanity_check_idx] if sanity_check_idx > 0 else []

    # Get cells from sanity check to empirical frequency
    # (these contain data loading and initial model definitions)
    if empirical_freq_idx > sanity_check_idx > 0:
        data_prep_cells = cells[sanity_check_idx:empirical_freq_idx]
    else:
        print("Error: Could not find expected section boundaries")
        return 1

    # Create new Test 1 section
    test1_header = create_markdown_cell([
        "## Test 1: Single Layer NN (Adam) vs Logistic Regression\n",
        "\n",
        "This test compares:\n",
        "1. Single-layer neural network trained with Adam optimizer\n",
        "2. Logistic regression (exact configuration from notebook 04)\n",
        "\n",
        "Results are saved for reuse in Test 2.\n"
    ])

    test1_model_defs = create_code_cell([
        "# Model definitions for Test 1\n",
        "\n",
        "class SingleLayerNN(nn.Module):\n",
        "    \"\"\"\n",
        "    Single-layer neural network for edge probability prediction.\n",
        "    \n",
        "    Uses zero initialization to match sklearn LogisticRegression defaults.\n",
        "    Returns raw logits for use with BCEWithLogitsLoss.\n",
        "    \"\"\"\n",
        "    def __init__(self):\n",
        "        super(SingleLayerNN, self).__init__()\n",
        "        self.linear = nn.Linear(2, 1)\n",
        "        \n",
        "        # Zero initialization to match sklearn defaults\n",
        "        nn.init.zeros_(self.linear.weight)\n",
        "        nn.init.zeros_(self.linear.bias)\n",
        "    \n",
        "    def forward(self, x):\n",
        "        return self.linear(x)\n",
        "\n",
        "print(\"SingleLayerNN defined with zero initialization\")\n"
    ])

    test1_train_adam = create_code_cell([
        "# Train single-layer NN with Adam optimizer\n",
        "import time\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Training Single-Layer NN with Adam Optimizer\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "model_adam = SingleLayerNN()\n",
        "optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)\n",
        "criterion = nn.BCEWithLogitsLoss()\n",
        "\n",
        "num_epochs = 200\n",
        "train_losses_adam = []\n",
        "test_losses_adam = []\n",
        "\n",
        "start_time = time.time()\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "    # Training\n",
        "    model_adam.train()\n",
        "    optimizer_adam.zero_grad()\n",
        "    outputs = model_adam(X_train_tensor)\n",
        "    loss = criterion(outputs.squeeze(), y_train_tensor)\n",
        "    loss.backward()\n",
        "    optimizer_adam.step()\n",
        "    train_losses_adam.append(loss.item())\n",
        "    \n",
        "    # Testing\n",
        "    model_adam.eval()\n",
        "    with torch.no_grad():\n",
        "        test_outputs = model_adam(X_test_tensor)\n",
        "        test_loss = criterion(test_outputs.squeeze(), y_test_tensor)\n",
        "        test_losses_adam.append(test_loss.item())\n",
        "    \n",
        "    if (epoch + 1) % 50 == 0:\n",
        "        print(f\"Epoch {epoch+1}/{num_epochs} - \"\n",
        "              f\"Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}\")\n",
        "\n",
        "train_time_adam = time.time() - start_time\n",
        "print(f\"\\nTraining completed in {train_time_adam:.2f} seconds\")\n",
        "\n",
        "# Get predictions\n",
        "model_adam.eval()\n",
        "with torch.no_grad():\n",
        "    train_pred_adam = torch.sigmoid(model_adam(X_train_tensor)).numpy().flatten()\n",
        "    test_pred_adam = torch.sigmoid(model_adam(X_test_tensor)).numpy().flatten()\n",
        "\n",
        "print(f\"Predictions shape - Train: {train_pred_adam.shape}, Test: {test_pred_adam.shape}\")\n"
    ])

    test1_train_logreg = create_code_cell([
        "# Train logistic regression (exact config from notebook 04)\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import roc_auc_score, average_precision_score\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Training Logistic Regression (Notebook 04 Configuration)\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "logreg = LogisticRegression(\n",
        "    random_state=42,\n",
        "    max_iter=1000\n",
        "    # Uses sklearn defaults: solver='lbfgs', penalty='l2', C=1.0\n",
        ")\n",
        "\n",
        "start_time = time.time()\n",
        "logreg.fit(X_train, y_train)\n",
        "train_time_logreg = time.time() - start_time\n",
        "\n",
        "# Get predictions (probabilities)\n",
        "train_pred_logreg = logreg.predict_proba(X_train)[:, 1]\n",
        "test_pred_logreg = logreg.predict_proba(X_test)[:, 1]\n",
        "\n",
        "print(f\"Training completed in {train_time_logreg:.2f} seconds\")\n",
        "print(f\"Predictions shape - Train: {train_pred_logreg.shape}, Test: {test_pred_logreg.shape}\")\n"
    ])

    test1_evaluate = create_code_cell([
        "# Evaluate Test 1 models\n",
        "from scipy.stats import pearsonr\n",
        "from sklearn.metrics import mean_squared_error\n",
        "\n",
        "def evaluate_model(y_true, y_pred, model_name):\n",
        "    \"\"\"\n",
        "    Compute evaluation metrics for a model.\n",
        "    \n",
        "    Args:\n",
        "        y_true: True binary labels\n",
        "        y_pred: Predicted probabilities\n",
        "        model_name: Name of the model for display\n",
        "    \n",
        "    Returns:\n",
        "        Dictionary of metrics\n",
        "    \"\"\"\n",
        "    auc = roc_auc_score(y_true, y_pred)\n",
        "    ap = average_precision_score(y_true, y_pred)\n",
        "    corr, _ = pearsonr(y_true, y_pred)\n",
        "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
        "    \n",
        "    metrics = {\n",
        "        'model': model_name,\n",
        "        'auc': auc,\n",
        "        'average_precision': ap,\n",
        "        'correlation': corr,\n",
        "        'rmse': rmse\n",
        "    }\n",
        "    \n",
        "    return metrics\n",
        "\n",
        "# Evaluate on test set\n",
        "metrics_adam = evaluate_model(y_test, test_pred_adam, 'Single Layer NN (Adam)')\n",
        "metrics_logreg = evaluate_model(y_test, test_pred_logreg, 'Logistic Regression')\n",
        "\n",
        "# Print results\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Test 1 Evaluation Results\")\n",
        "print(\"=\"*60)\n",
        "for metrics in [metrics_adam, metrics_logreg]:\n",
        "    print(f\"\\n{metrics['model']}:\")\n",
        "    print(f\"  AUC: {metrics['auc']:.4f}\")\n",
        "    print(f\"  Average Precision: {metrics['average_precision']:.4f}\")\n",
        "    print(f\"  Correlation: {metrics['correlation']:.4f}\")\n",
        "    print(f\"  RMSE: {metrics['rmse']:.4f}\")\n"
    ])

    test1_save_results = create_code_cell([
        "# Save Test 1 results\n",
        "import pickle\n",
        "import os\n",
        "\n",
        "results_dir = 'results/nn_optimizer_comparison'\n",
        "os.makedirs(results_dir, exist_ok=True)\n",
        "\n",
        "# Save Single Layer NN (Adam) results\n",
        "adam_results = {\n",
        "    'model': model_adam,\n",
        "    'train_pred': train_pred_adam,\n",
        "    'test_pred': test_pred_adam,\n",
        "    'train_losses': train_losses_adam,\n",
        "    'test_losses': test_losses_adam,\n",
        "    'metrics': metrics_adam,\n",
        "    'train_time': train_time_adam,\n",
        "    'num_epochs': num_epochs\n",
        "}\n",
        "\n",
        "with open(f'{results_dir}/single_layer_nn_adam.pkl', 'wb') as f:\n",
        "    pickle.dump(adam_results, f)\n",
        "print(f\"Saved: {results_dir}/single_layer_nn_adam.pkl\")\n",
        "\n",
        "# Save Logistic Regression results\n",
        "logreg_results = {\n",
        "    'model': logreg,\n",
        "    'train_pred': train_pred_logreg,\n",
        "    'test_pred': test_pred_logreg,\n",
        "    'metrics': metrics_logreg,\n",
        "    'train_time': train_time_logreg\n",
        "}\n",
        "\n",
        "with open(f'{results_dir}/logistic_regression.pkl', 'wb') as f:\n",
        "    pickle.dump(logreg_results, f)\n",
        "print(f\"Saved: {results_dir}/logistic_regression.pkl\")\n",
        "\n",
        "print(\"\\nTest 1 results saved successfully!\")\n"
    ])

    test1_viz = create_code_cell([
        "# Visualize Test 1 loss curves\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "fig, ax = plt.subplots(1, 1, figsize=(10, 6))\n",
        "\n",
        "ax.plot(train_losses_adam, label='Train Loss', linewidth=2)\n",
        "ax.plot(test_losses_adam, label='Test Loss', linewidth=2)\n",
        "ax.set_xlabel('Epoch', fontsize=12)\n",
        "ax.set_ylabel('Loss (BCEWithLogitsLoss)', fontsize=12)\n",
        "ax.set_title('Single Layer NN (Adam) - Training History', fontsize=14, fontweight='bold')\n",
        "ax.legend(fontsize=11)\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{results_dir}/single_layer_nn_adam_loss.png', dpi=300, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f\"Saved: {results_dir}/single_layer_nn_adam_loss.png\")\n"
    ])

    # Create Test 2 section
    test2_header = create_markdown_cell([
        "## Test 2: Add L-BFGS Optimizer Comparison\n",
        "\n",
        "This test:\n",
        "1. Loads results from Test 1 (Adam NN and Logistic Regression)\n",
        "2. Trains single-layer NN with L-BFGS optimizer\n",
        "3. Compares all three approaches\n",
        "\n",
        "No redundant calculations are performed.\n"
    ])

    test2_load_results = create_code_cell([
        "# Load Test 1 results\n",
        "import pickle\n",
        "\n",
        "results_dir = 'results/nn_optimizer_comparison'\n",
        "\n",
        "print(\"Loading Test 1 results...\")\n",
        "\n",
        "with open(f'{results_dir}/single_layer_nn_adam.pkl', 'rb') as f:\n",
        "    adam_results = pickle.load(f)\n",
        "\n",
        "with open(f'{results_dir}/logistic_regression.pkl', 'rb') as f:\n",
        "    logreg_results = pickle.load(f)\n",
        "\n",
        "print(\"Test 1 results loaded successfully!\")\n",
        "print(f\"  Adam NN - Test AUC: {adam_results['metrics']['auc']:.4f}\")\n",
        "print(f\"  LogReg - Test AUC: {logreg_results['metrics']['auc']:.4f}\")\n"
    ])

    test2_train_lbfgs = create_code_cell([
        "# Train single-layer NN with L-BFGS optimizer\n",
        "import time\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Training Single-Layer NN with L-BFGS Optimizer\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "model_lbfgs = SingleLayerNN()\n",
        "optimizer_lbfgs = torch.optim.LBFGS(\n",
        "    model_lbfgs.parameters(),\n",
        "    lr=0.01,\n",
        "    max_iter=20\n",
        ")\n",
        "criterion = nn.BCEWithLogitsLoss()\n",
        "\n",
        "max_epochs = 500\n",
        "patience = 50\n",
        "min_delta = 1e-6\n",
        "\n",
        "train_losses_lbfgs = []\n",
        "test_losses_lbfgs = []\n",
        "best_loss = float('inf')\n",
        "epochs_no_improve = 0\n",
        "converged_epoch = max_epochs\n",
        "\n",
        "start_time = time.time()\n",
        "\n",
        "for epoch in range(max_epochs):\n",
        "    model_lbfgs.train()\n",
        "    \n",
        "    def closure():\n",
        "        optimizer_lbfgs.zero_grad()\n",
        "        outputs = model_lbfgs(X_train_tensor)\n",
        "        loss = criterion(outputs.squeeze(), y_train_tensor)\n",
        "        loss.backward()\n",
        "        return loss\n",
        "    \n",
        "    loss = optimizer_lbfgs.step(closure)\n",
        "    train_losses_lbfgs.append(loss.item())\n",
        "    \n",
        "    # Testing\n",
        "    model_lbfgs.eval()\n",
        "    with torch.no_grad():\n",
        "        test_outputs = model_lbfgs(X_test_tensor)\n",
        "        test_loss = criterion(test_outputs.squeeze(), y_test_tensor)\n",
        "        test_losses_lbfgs.append(test_loss.item())\n",
        "    \n",
        "    # Check convergence\n",
        "    if loss.item() < best_loss - min_delta:\n",
        "        best_loss = loss.item()\n",
        "        epochs_no_improve = 0\n",
        "    else:\n",
        "        epochs_no_improve += 1\n",
        "    \n",
        "    if (epoch + 1) % 50 == 0:\n",
        "        print(f\"Epoch {epoch+1}/{max_epochs} - \"\n",
        "              f\"Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}\")\n",
        "    \n",
        "    if epochs_no_improve >= patience:\n",
        "        converged_epoch = epoch + 1\n",
        "        print(f\"\\nConverged at epoch {converged_epoch}\")\n",
        "        break\n",
        "\n",
        "train_time_lbfgs = time.time() - start_time\n",
        "print(f\"\\nTraining completed in {train_time_lbfgs:.2f} seconds\")\n",
        "\n",
        "# Get predictions\n",
        "model_lbfgs.eval()\n",
        "with torch.no_grad():\n",
        "    train_pred_lbfgs = torch.sigmoid(model_lbfgs(X_train_tensor)).numpy().flatten()\n",
        "    test_pred_lbfgs = torch.sigmoid(model_lbfgs(X_test_tensor)).numpy().flatten()\n",
        "\n",
        "print(f\"Predictions shape - Train: {train_pred_lbfgs.shape}, Test: {test_pred_lbfgs.shape}\")\n"
    ])

    test2_evaluate = create_code_cell([
        "# Evaluate all three models\n",
        "\n",
        "metrics_lbfgs = evaluate_model(y_test, test_pred_lbfgs, 'Single Layer NN (L-BFGS)')\n",
        "\n",
        "# Print comparison\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Test 2 - Complete Model Comparison\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "all_metrics = [\n",
        "    adam_results['metrics'],\n",
        "    metrics_lbfgs,\n",
        "    logreg_results['metrics']\n",
        "]\n",
        "\n",
        "for metrics in all_metrics:\n",
        "    print(f\"\\n{metrics['model']}:\")\n",
        "    print(f\"  AUC: {metrics['auc']:.4f}\")\n",
        "    print(f\"  Average Precision: {metrics['average_precision']:.4f}\")\n",
        "    print(f\"  Correlation: {metrics['correlation']:.4f}\")\n",
        "    print(f\"  RMSE: {metrics['rmse']:.4f}\")\n",
        "\n",
        "# Create comparison dataframe\n",
        "import pandas as pd\n",
        "comparison_df = pd.DataFrame(all_metrics)\n",
        "print(\"\\nComparison Table:\")\n",
        "print(comparison_df.to_string(index=False))\n"
    ])

    test2_save_results = create_code_cell([
        "# Save Test 2 results\n",
        "\n",
        "# Save L-BFGS results\n",
        "lbfgs_results = {\n",
        "    'model': model_lbfgs,\n",
        "    'train_pred': train_pred_lbfgs,\n",
        "    'test_pred': test_pred_lbfgs,\n",
        "    'train_losses': train_losses_lbfgs,\n",
        "    'test_losses': test_losses_lbfgs,\n",
        "    'metrics': metrics_lbfgs,\n",
        "    'train_time': train_time_lbfgs,\n",
        "    'converged_epoch': converged_epoch\n",
        "}\n",
        "\n",
        "with open(f'{results_dir}/single_layer_nn_lbfgs.pkl', 'wb') as f:\n",
        "    pickle.dump(lbfgs_results, f)\n",
        "print(f\"Saved: {results_dir}/single_layer_nn_lbfgs.pkl\")\n",
        "\n",
        "# Save combined comparison\n",
        "comparison_results = {\n",
        "    'models': ['Single Layer NN (Adam)', 'Single Layer NN (L-BFGS)', 'Logistic Regression'],\n",
        "    'metrics_df': comparison_df,\n",
        "    'test_predictions': {\n",
        "        'adam': adam_results['test_pred'],\n",
        "        'lbfgs': test_pred_lbfgs,\n",
        "        'logreg': logreg_results['test_pred']\n",
        "    },\n",
        "    'true_labels': y_test\n",
        "}\n",
        "\n",
        "with open(f'{results_dir}/comparison_summary.pkl', 'wb') as f:\n",
        "    pickle.dump(comparison_results, f)\n",
        "print(f\"Saved: {results_dir}/comparison_summary.pkl\")\n",
        "\n",
        "# Save comparison table as CSV\n",
        "comparison_df.to_csv(f'{results_dir}/model_comparison.csv', index=False)\n",
        "print(f\"Saved: {results_dir}/model_comparison.csv\")\n",
        "\n",
        "print(\"\\nTest 2 results saved successfully!\")\n"
    ])

    test2_viz_individual = create_code_cell([
        "# Visualize L-BFGS loss curves\n",
        "\n",
        "fig, ax = plt.subplots(1, 1, figsize=(10, 6))\n",
        "\n",
        "ax.plot(train_losses_lbfgs, label='Train Loss', linewidth=2)\n",
        "ax.plot(test_losses_lbfgs, label='Test Loss', linewidth=2)\n",
        "ax.axvline(x=converged_epoch-1, color='red', linestyle='--', \n",
        "           label=f'Converged (Epoch {converged_epoch})', alpha=0.7)\n",
        "ax.set_xlabel('Epoch', fontsize=12)\n",
        "ax.set_ylabel('Loss (BCEWithLogitsLoss)', fontsize=12)\n",
        "ax.set_title('Single Layer NN (L-BFGS) - Training History', fontsize=14, fontweight='bold')\n",
        "ax.legend(fontsize=11)\n",
        "ax.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{results_dir}/single_layer_nn_lbfgs_loss.png', dpi=300, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f\"Saved: {results_dir}/single_layer_nn_lbfgs_loss.png\")\n"
    ])

    test2_viz_combined = create_code_cell([
        "# Create combined comparison plot\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "# Adam optimizer\n",
        "axes[0].plot(adam_results['train_losses'], label='Train Loss', linewidth=2)\n",
        "axes[0].plot(adam_results['test_losses'], label='Test Loss', linewidth=2)\n",
        "axes[0].set_xlabel('Epoch', fontsize=12)\n",
        "axes[0].set_ylabel('Loss (BCEWithLogitsLoss)', fontsize=12)\n",
        "axes[0].set_title('Single Layer NN (Adam)', fontsize=13, fontweight='bold')\n",
        "axes[0].legend(fontsize=11)\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "# L-BFGS optimizer\n",
        "axes[1].plot(train_losses_lbfgs, label='Train Loss', linewidth=2)\n",
        "axes[1].plot(test_losses_lbfgs, label='Test Loss', linewidth=2)\n",
        "axes[1].axvline(x=converged_epoch-1, color='red', linestyle='--', \n",
        "                label=f'Converged (Epoch {converged_epoch})', alpha=0.7)\n",
        "axes[1].set_xlabel('Epoch', fontsize=12)\n",
        "axes[1].set_ylabel('Loss (BCEWithLogitsLoss)', fontsize=12)\n",
        "axes[1].set_title('Single Layer NN (L-BFGS)', fontsize=13, fontweight='bold')\n",
        "axes[1].legend(fontsize=11)\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "\n",
        "fig.suptitle('Optimizer Comparison - Training Histories', \n",
        "             fontsize=15, fontweight='bold', y=1.02)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{results_dir}/optimizer_comparison_losses.png', dpi=300, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f\"Saved: {results_dir}/optimizer_comparison_losses.png\")\n"
    ])

    test2_viz_predictions = create_code_cell([
        "# Create scatter plots comparing predictions\n",
        "\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "\n",
        "models_data = [\n",
        "    ('Single Layer NN (Adam)', adam_results['test_pred'], adam_results['metrics']),\n",
        "    ('Single Layer NN (L-BFGS)', test_pred_lbfgs, metrics_lbfgs),\n",
        "    ('Logistic Regression', logreg_results['test_pred'], logreg_results['metrics'])\n",
        "]\n",
        "\n",
        "for idx, (name, preds, metrics) in enumerate(models_data):\n",
        "    ax = axes[idx]\n",
        "    ax.scatter(y_test, preds, alpha=0.3, s=10)\n",
        "    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')\n",
        "    ax.set_xlabel('True Labels', fontsize=11)\n",
        "    ax.set_ylabel('Predicted Probabilities', fontsize=11)\n",
        "    ax.set_title(f\"{name}\\nCorr: {metrics['correlation']:.4f}, AUC: {metrics['auc']:.4f}\",\n",
        "                fontsize=11, fontweight='bold')\n",
        "    ax.legend(fontsize=9)\n",
        "    ax.grid(True, alpha=0.3)\n",
        "    ax.set_xlim(-0.05, 1.05)\n",
        "    ax.set_ylim(-0.05, 1.05)\n",
        "\n",
        "fig.suptitle('Model Predictions Comparison - Test Set', \n",
        "             fontsize=14, fontweight='bold', y=1.02)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{results_dir}/prediction_comparison.png', dpi=300, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f\"Saved: {results_dir}/prediction_comparison.png\")\n"
    ])

    # Build new cell list
    new_cells = (
        intro_cells +
        data_prep_cells +
        [test1_header, test1_model_defs, test1_train_adam, test1_train_logreg,
         test1_evaluate, test1_save_results, test1_viz] +
        [test2_header, test2_load_results, test2_train_lbfgs, test2_evaluate,
         test2_save_results, test2_viz_individual, test2_viz_combined, test2_viz_predictions]
    )

    # Add recommendations if they exist
    if recommendations_idx > 0:
        new_cells.extend(cells[recommendations_idx:])

    notebook['cells'] = new_cells

    # Save modified notebook
    print(f"\nSaving restructured notebook to: {notebook_path}")
    save_notebook(notebook, notebook_path)

    print(f"Added {len(new_cells) - len(cells)} new cells")
    print("Notebook restructured successfully!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
