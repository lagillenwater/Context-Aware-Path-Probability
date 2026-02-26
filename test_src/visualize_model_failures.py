"""
Visualize where models fail and how it relates to topology-specific outliers.

Key questions:
1. Are the 3.4% topology-specific outliers where models fail?
2. What's the prediction error distribution for different pair types?
3. Can we visualize the relationship between outliers and errors?

Usage:
    python test_src/visualize_model_failures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

from test_src.validate_mean_variance_prediction import (
    load_permuted_edge_matrices,
    sample_pairs,
    compute_pathway_counts,
    extract_degree_features
)


class HeteroscedasticNN(nn.Module):
    """Neural network that jointly predicts mean and variance."""

    def __init__(self, input_dim=5, hidden_dims=[64, 32, 16], dropout=0.2):
        super(HeteroscedasticNN, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev_dim, 1)
        self.logvar_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = self.mean_head(h).squeeze()
        logvar = self.logvar_head(h).squeeze()
        return mean, logvar


def train_best_models(X_train, y_train, counts_train):
    """Train Linear, RF, and Heteroscedastic NN models."""
    print("Training models...")

    # Linear
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Heteroscedastic NN
    hetero_nn = train_heteroscedastic_nn(X_train, counts_train)

    return {'Linear': lr, 'RandomForest': rf, 'HeteroscedasticNN': hetero_nn}


def train_heteroscedastic_nn(X, counts_train):
    """Train heteroscedastic neural network."""
    print("  Training Heteroscedastic NN...")

    # Flatten data
    counts_flat = []
    X_expanded = []
    for i in range(len(X)):
        for j in range(counts_train.shape[1]):
            counts_flat.append(counts_train[i, j])
            X_expanded.append(X[i])

    counts_flat = np.array(counts_flat)
    X_expanded = np.array(X_expanded)
    X_expanded_t = torch.FloatTensor(X_expanded)
    counts_flat_t = torch.FloatTensor(counts_flat)

    model = HeteroscedasticNN(input_dim=X.shape[1], hidden_dims=[64, 32, 16], dropout=0.2)

    def heteroscedastic_loss(y_true, y_pred_mean, y_pred_logvar):
        var = torch.exp(y_pred_logvar)
        loss = 0.5 * (y_pred_logvar + (y_true - y_pred_mean)**2 / var)
        return loss.mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = np.inf
    patience = 20
    patience_counter = 0
    best_state = None

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        mu_pred, logvar_pred = model(X_expanded_t)
        loss = heteroscedastic_loss(counts_flat_t, mu_pred, logvar_pred)

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    print(f"    Best loss: {best_loss:.4f}")

    return model


def analyze_failures(edge1_type='CbG', edge2_type='GpPW', n_samples=10000, random_state=42):
    """Analyze where models fail and relate to topology-specific outliers."""
    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'model_failures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Model Failure Analysis")
    print("="*70)

    print(f"\nLoading data for {edge1_type}+{edge2_type}...")
    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(edge1_type, edge2_type, 0, data_dir)
    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples=n_samples, random_state=random_state)
    X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)

    print(f"  Sampled {len(pairs)} pairs")

    train_perms = list(range(5))
    test_perms = list(range(15, 21))

    # Compute training counts
    print("\nComputing training counts (perms 0-4)...")
    counts_train = []
    for perm in train_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_train.append(counts)
    counts_train = np.column_stack(counts_train)
    mean_train = counts_train.mean(axis=1)

    # Identify topology-specific outliers from training
    high_threshold = np.percentile(mean_train, 99)
    high_in_mean = mean_train > high_threshold
    any_high_in_train = (counts_train > high_threshold).any(axis=1)
    topology_specific = any_high_in_train & ~high_in_mean
    consistent_high = high_in_mean
    never_high = ~any_high_in_train

    print(f"\nTraining pair categorization:")
    print(f"  Consistent high (high in mean): {consistent_high.sum()} ({100*consistent_high.mean():.1f}%)")
    print(f"  Topology-specific (high in some, not mean): {topology_specific.sum()} ({100*topology_specific.mean():.1f}%)")
    print(f"  Never high: {never_high.sum()} ({100*never_high.mean():.1f}%)")

    # Train models
    models = train_best_models(X, mean_train, counts_train)

    # Compute test counts
    print("\nComputing test counts (perms 15-20)...")
    counts_test = []
    for perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)
    counts_test = np.column_stack(counts_test)

    # Make predictions and compute errors
    print("\nComputing prediction errors...")
    results = []

    for model_name, model in models.items():
        if model_name == 'HeteroscedasticNN':
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                pred, _ = model(X_t)
                pred = pred.numpy()
        else:
            pred = model.predict(X)

        # For each test permutation
        for test_idx, perm in enumerate(test_perms):
            actual = counts_test[:, test_idx]

            # Errors
            errors = actual - pred
            abs_errors = np.abs(errors)
            squared_errors = errors ** 2

            # Correlation
            r = np.corrcoef(actual, pred)[0, 1]

            # Error statistics by pair type
            for pair_type, mask in [
                ('Consistent High', consistent_high),
                ('Topology-Specific', topology_specific),
                ('Never High', never_high)
            ]:
                if mask.sum() > 0:
                    results.append({
                        'model': model_name,
                        'test_perm': perm,
                        'pair_type': pair_type,
                        'n_pairs': mask.sum(),
                        'mean_error': errors[mask].mean(),
                        'mean_abs_error': abs_errors[mask].mean(),
                        'rmse': np.sqrt(squared_errors[mask].mean()),
                        'mean_actual': actual[mask].mean(),
                        'mean_pred': pred[mask].mean(),
                        'r': r
                    })

    results_df = pd.DataFrame(results)

    # Average across test perms
    summary = results_df.groupby(['model', 'pair_type']).agg({
        'n_pairs': 'first',
        'mean_error': 'mean',
        'mean_abs_error': 'mean',
        'rmse': 'mean',
        'mean_actual': 'mean',
        'mean_pred': 'mean',
        'r': 'mean'
    }).reset_index()

    print("\n" + "="*70)
    print("Error Analysis by Pair Type")
    print("="*70)

    for model_name in ['Linear', 'RandomForest']:
        print(f"\n{model_name}:")
        model_summary = summary[summary['model'] == model_name]

        for _, row in model_summary.iterrows():
            print(f"\n  {row['pair_type']} ({int(row['n_pairs'])} pairs):")
            print(f"    Mean actual count: {row['mean_actual']:.3f}")
            print(f"    Mean predicted: {row['mean_pred']:.3f}")
            print(f"    Mean error: {row['mean_error']:+.3f}")
            print(f"    Mean abs error: {row['mean_abs_error']:.3f}")
            print(f"    RMSE: {row['rmse']:.3f}")

    # Create visualizations
    create_failure_visualizations(
        X, mean_train, counts_test, models,
        consistent_high, topology_specific, never_high,
        output_dir
    )

    # Detailed analysis: Are topology-specific pairs the largest errors?
    print("\n" + "="*70)
    print("Are Topology-Specific Outliers the Largest Errors?")
    print("="*70)

    for model_name, model in models.items():
        if model_name == 'HeteroscedasticNN':
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                pred, _ = model(X_t)
                pred = pred.numpy()
        else:
            pred = model.predict(X)

        # Average absolute error across test perms
        avg_abs_error = np.abs(counts_test - pred.reshape(-1, 1)).mean(axis=1)

        # Top 10% largest errors
        error_threshold = np.percentile(avg_abs_error, 90)
        large_errors = avg_abs_error > error_threshold

        print(f"\n{model_name}:")
        print(f"  Pairs with large errors (top 10%): {large_errors.sum()}")
        print(f"  Of those, how many are:")
        print(f"    Consistent high: {(large_errors & consistent_high).sum()} ({100*(large_errors & consistent_high).sum()/large_errors.sum():.1f}%)")
        print(f"    Topology-specific: {(large_errors & topology_specific).sum()} ({100*(large_errors & topology_specific).sum()/large_errors.sum():.1f}%)")
        print(f"    Never high: {(large_errors & never_high).sum()} ({100*(large_errors & never_high).sum()/large_errors.sum():.1f}%)")

        print(f"\n  Expected if errors were random:")
        print(f"    Consistent high: {100*consistent_high.mean():.1f}%")
        print(f"    Topology-specific: {100*topology_specific.mean():.1f}%")
        print(f"    Never high: {100*never_high.mean():.1f}%")

    return results_df, summary


def create_failure_visualizations(X, mean_train, counts_test, models,
                                  consistent_high, topology_specific, never_high,
                                  output_dir):
    """Create comprehensive failure visualizations."""

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # Color scheme
    colors = {
        'Consistent High': '#d62728',
        'Topology-Specific': '#ff7f0e',
        'Never High': '#1f77b4'
    }

    pair_types = {
        'Consistent High': consistent_high,
        'Topology-Specific': topology_specific,
        'Never High': never_high
    }

    deg_product = X[:, 2]

    # Plot 1: Degree distribution by pair type
    ax = fig.add_subplot(gs[0, 0])
    for name, mask in pair_types.items():
        ax.hist(np.log10(deg_product[mask] + 1), bins=50, alpha=0.5,
               label=name, color=colors[name], density=True)
    ax.set_xlabel('log10(Degree Product + 1)')
    ax.set_ylabel('Density')
    ax.set_title('Degree Distribution by Pair Type')
    ax.legend()

    # Plot 2: Mean count distribution by pair type
    ax = fig.add_subplot(gs[0, 1])
    for name, mask in pair_types.items():
        ax.hist(mean_train[mask], bins=50, alpha=0.5,
               label=name, color=colors[name], density=True)
    ax.set_xlabel('Mean Count (train perms 0-4)')
    ax.set_ylabel('Density')
    ax.set_title('Mean Count Distribution by Pair Type')
    ax.set_xlim(0, 5)
    ax.legend()

    # Plot 3: Variance across training perms
    ax = fig.add_subplot(gs[0, 2])
    var_train = counts_test.var(axis=1)  # Actually using test for visualization
    for name, mask in pair_types.items():
        if mask.sum() > 0:
            ax.scatter(deg_product[mask], var_train[mask],
                      alpha=0.3, s=10, label=name, color=colors[name])
    ax.set_xlabel('Degree Product')
    ax.set_ylabel('Variance (test perms)')
    ax.set_title('Variance vs Degree')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    # Plots 4-6: Prediction errors for Linear model
    model_name = 'Linear'
    model = models[model_name]
    pred = model.predict(X)

    # Average actual and error across test perms
    avg_actual = counts_test.mean(axis=1)
    avg_error = (counts_test - pred.reshape(-1, 1)).mean(axis=1)
    avg_abs_error = np.abs(counts_test - pred.reshape(-1, 1)).mean(axis=1)

    # Plot 4: Predicted vs Actual (colored by pair type)
    ax = fig.add_subplot(gs[1, 0])
    for name, mask in pair_types.items():
        ax.scatter(pred[mask], avg_actual[mask], alpha=0.3, s=5,
                  label=name, color=colors[name])
    max_val = max(pred.max(), avg_actual.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Actual Count (avg test)')
    ax.set_title(f'{model_name}: Predicted vs Actual')
    ax.legend()

    # Plot 5: Residuals vs Predicted
    ax = fig.add_subplot(gs[1, 1])
    for name, mask in pair_types.items():
        ax.scatter(pred[mask], avg_error[mask], alpha=0.3, s=5,
                  label=name, color=colors[name])
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Error (Actual - Predicted)')
    ax.set_title(f'{model_name}: Residuals')
    ax.legend()

    # Plot 6: Absolute error vs Degree
    ax = fig.add_subplot(gs[1, 2])
    for name, mask in pair_types.items():
        ax.scatter(deg_product[mask], avg_abs_error[mask], alpha=0.3, s=5,
                  label=name, color=colors[name])
    ax.set_xlabel('Degree Product')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(f'{model_name}: Error vs Degree')
    ax.set_xscale('log')
    ax.legend()

    # Plots 7-9: Same for Random Forest
    model_name = 'RandomForest'
    model = models[model_name]
    pred = model.predict(X)

    avg_error = (counts_test - pred.reshape(-1, 1)).mean(axis=1)
    avg_abs_error = np.abs(counts_test - pred.reshape(-1, 1)).mean(axis=1)

    # Plot 7: Predicted vs Actual
    ax = fig.add_subplot(gs[2, 0])
    for name, mask in pair_types.items():
        ax.scatter(pred[mask], avg_actual[mask], alpha=0.3, s=5,
                  label=name, color=colors[name])
    max_val = max(pred.max(), avg_actual.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Actual Count (avg test)')
    ax.set_title(f'{model_name}: Predicted vs Actual')
    ax.legend()

    # Plot 8: Residuals vs Predicted
    ax = fig.add_subplot(gs[2, 1])
    for name, mask in pair_types.items():
        ax.scatter(pred[mask], avg_error[mask], alpha=0.3, s=5,
                  label=name, color=colors[name])
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Error (Actual - Predicted)')
    ax.set_title(f'{model_name}: Residuals')
    ax.legend()

    # Plot 9: Absolute error vs Degree
    ax = fig.add_subplot(gs[2, 2])
    for name, mask in pair_types.items():
        ax.scatter(deg_product[mask], avg_abs_error[mask], alpha=0.3, s=5,
                  label=name, color=colors[name])
    ax.set_xlabel('Degree Product')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(f'{model_name}: Error vs Degree')
    ax.set_xscale('log')
    ax.legend()

    # Plots 10-12: Same for Heteroscedastic NN
    model_name = 'HeteroscedasticNN'
    model = models[model_name]
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        pred, _ = model(X_t)
        pred = pred.numpy()

    avg_error = (counts_test - pred.reshape(-1, 1)).mean(axis=1)
    avg_abs_error = np.abs(counts_test - pred.reshape(-1, 1)).mean(axis=1)

    # Plot 10: Predicted vs Actual
    ax = fig.add_subplot(gs[3, 0])
    for name, mask in pair_types.items():
        ax.scatter(pred[mask], avg_actual[mask], alpha=0.3, s=5,
                  label=name, color=colors[name])
    max_val = max(pred.max(), avg_actual.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Actual Count (avg test)')
    ax.set_title(f'{model_name}: Predicted vs Actual')
    ax.legend()

    # Plot 11: Residuals vs Predicted
    ax = fig.add_subplot(gs[3, 1])
    for name, mask in pair_types.items():
        ax.scatter(pred[mask], avg_error[mask], alpha=0.3, s=5,
                  label=name, color=colors[name])
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Error (Actual - Predicted)')
    ax.set_title(f'{model_name}: Residuals')
    ax.legend()

    # Plot 12: Absolute error vs Degree
    ax = fig.add_subplot(gs[3, 2])
    for name, mask in pair_types.items():
        ax.scatter(deg_product[mask], avg_abs_error[mask], alpha=0.3, s=5,
                  label=name, color=colors[name])
    ax.set_xlabel('Degree Product')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(f'{model_name}: Error vs Degree')
    ax.set_xscale('log')
    ax.legend()

    output_file = output_dir / 'model_failure_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_file}")
    plt.close()

    # Additional plot: Error distribution by pair type
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (model_name, model) in enumerate(models.items()):
        ax = axes[idx]
        if model_name == 'HeteroscedasticNN':
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                pred, _ = model(X_t)
                pred = pred.numpy()
        else:
            pred = model.predict(X)
        avg_abs_error = np.abs(counts_test - pred.reshape(-1, 1)).mean(axis=1)

        error_data = []
        labels = []
        for name, mask in pair_types.items():
            if mask.sum() > 0:
                error_data.append(avg_abs_error[mask])
                labels.append(f"{name}\n(n={mask.sum()})")

        bp = ax.boxplot(error_data, labels=labels, patch_artist=True)
        for patch, name in zip(bp['boxes'], pair_types.keys()):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.6)

        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'{model_name}: Error Distribution by Pair Type')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'error_distributions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_file}")
    plt.close()


def main():
    results_df, summary = analyze_failures()

    output_dir = repo_dir / 'results' / 'model_failures'
    summary.to_csv(output_dir / 'error_summary.csv', index=False)
    print(f"\nSaved summary: {output_dir / 'error_summary.csv'}")


if __name__ == '__main__':
    main()
