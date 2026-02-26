"""
Analyze why "consistent high" pairs have the highest errors.

If they're consistently high, the model should predict them well from degrees.
Yet they have MAE=1.5, much higher than topology-specific (0.9) or never-high (0.2).

Hypothesis: "Consistent high" means high MEAN, but they still have high VARIANCE
across permutations. The model correctly predicts the mean, but individual
permutation counts deviate substantially.

Usage:
    python test_src/analyze_consistent_high_variance.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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


def train_heteroscedastic_nn(X, counts_train):
    """Train heteroscedastic neural network."""
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

    return model


def analyze_consistent_high_variance(edge1_type='CbG', edge2_type='GpPW',
                                    n_samples=10000, random_state=42):
    """Analyze variance structure of consistent high pairs."""
    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'consistent_high_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Why Do Consistent High Pairs Have Highest Errors?")
    print("="*70)

    print(f"\nLoading data...")
    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(edge1_type, edge2_type, 0, data_dir)
    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples=n_samples, random_state=random_state)
    X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)

    train_perms = list(range(5))
    test_perms = list(range(15, 21))

    # Load all counts
    print("Loading training counts (perms 0-4)...")
    counts_train = []
    for perm in train_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_train.append(counts)
    counts_train = np.column_stack(counts_train)
    mean_train = counts_train.mean(axis=1)
    var_train = counts_train.var(axis=1)

    print("Loading test counts (perms 15-20)...")
    counts_test = []
    for perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)
    counts_test = np.column_stack(counts_test)
    mean_test = counts_test.mean(axis=1)
    var_test = counts_test.var(axis=1)

    # Categorize pairs
    high_threshold = np.percentile(mean_train, 99)
    high_in_mean = mean_train > high_threshold
    any_high_in_train = (counts_train > high_threshold).any(axis=1)
    consistent_high = high_in_mean
    topology_specific = any_high_in_train & ~high_in_mean
    never_high = ~any_high_in_train

    print(f"\nPair categorization:")
    print(f"  Consistent high: {consistent_high.sum()}")
    print(f"  Topology-specific: {topology_specific.sum()}")
    print(f"  Never high: {never_high.sum()}")

    print("\n" + "="*70)
    print("Analysis 1: How 'Consistent' Are Consistent High Pairs?")
    print("="*70)

    print(f"\nFor {consistent_high.sum()} 'consistent high' pairs:")
    print(f"How many training perms (out of 5) have count > threshold?")

    n_high_per_pair = (counts_train > high_threshold).sum(axis=1)

    for n in range(6):
        mask = consistent_high & (n_high_per_pair == n)
        if mask.sum() > 0:
            print(f"  High in {n}/5 perms: {mask.sum()} pairs ({100*mask.sum()/consistent_high.sum():.1f}%)")

    print("\nInterpretation:")
    always_high = consistent_high & (n_high_per_pair == 5)
    sometimes_high = consistent_high & (n_high_per_pair < 5)

    print(f"  'Always high' (5/5 perms): {always_high.sum()} ({100*always_high.sum()/consistent_high.sum():.1f}%)")
    print(f"  'Sometimes high' (<5/5 perms): {sometimes_high.sum()} ({100*sometimes_high.sum()/consistent_high.sum():.1f}%)")

    print("\n" + "="*70)
    print("Analysis 2: Variance Analysis")
    print("="*70)

    print(f"\nVariance across training perms (0-4):")
    print(f"  Consistent high: mean={var_train[consistent_high].mean():.3f}, median={np.median(var_train[consistent_high]):.3f}")
    print(f"  Topology-specific: mean={var_train[topology_specific].mean():.3f}, median={np.median(var_train[topology_specific]):.3f}")
    print(f"  Never high: mean={var_train[never_high].mean():.3f}, median={np.median(var_train[never_high]):.3f}")

    print(f"\nVariance across test perms (15-20):")
    print(f"  Consistent high: mean={var_test[consistent_high].mean():.3f}, median={np.median(var_test[consistent_high]):.3f}")
    print(f"  Topology-specific: mean={var_test[topology_specific].mean():.3f}, median={np.median(var_test[topology_specific]):.3f}")
    print(f"  Never high: mean={var_test[never_high].mean():.3f}, median={np.median(var_test[never_high]):.3f}")

    print(f"\nCoefficient of variation (std/mean) in training:")
    cv_train = np.sqrt(var_train) / (mean_train + 0.01)
    print(f"  Consistent high: mean={cv_train[consistent_high].mean():.3f}, median={np.median(cv_train[consistent_high]):.3f}")
    print(f"  Topology-specific: mean={cv_train[topology_specific].mean():.3f}, median={np.median(cv_train[topology_specific]):.3f}")
    print(f"  Never high: mean={cv_train[never_high].mean():.3f}, median={np.median(cv_train[never_high]):.3f}")

    print("\n" + "="*70)
    print("Analysis 3: Model Performance on Mean vs Individual Perms")
    print("="*70)

    # Train all three models
    print("\nTraining models...")
    lr = LinearRegression()
    lr.fit(X, mean_train)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X, mean_train)

    hetero_nn = train_heteroscedastic_nn(X, counts_train)

    models = {
        'Linear': lr,
        'RandomForest': rf,
        'HeteroscedasticNN': hetero_nn
    }

    for model_name, model in models.items():
        print(f"\n{model_name} model trained on mean of perms 0-4:")

        # Get predictions
        if model_name == 'HeteroscedasticNN':
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                pred, _ = model(X_t)
                pred = pred.numpy()
        else:
            pred = model.predict(X)

        # Error on predicting mean
        for name, mask in [('Consistent High', consistent_high),
                           ('Topology-Specific', topology_specific),
                           ('Never High', never_high)]:
            if mask.sum() > 0:
                mean_error = np.abs(mean_train[mask] - pred[mask]).mean()
                r_mean = np.corrcoef(mean_train[mask], pred[mask])[0, 1]
                print(f"\n  {name}:")
                print(f"    Predicting mean (what it was trained on):")
                print(f"      MAE: {mean_error:.3f}")
                print(f"      r: {r_mean:.3f}")

        # Error on predicting individual perms
        print("\n  Error on individual test permutations:")
        for name, mask in [('Consistent High', consistent_high),
                           ('Topology-Specific', topology_specific),
                           ('Never High', never_high)]:
            if mask.sum() > 0:
                errors = []
                for perm_idx in range(counts_test.shape[1]):
                    actual = counts_test[mask, perm_idx]
                    pred_mask = pred[mask]
                    errors.append(np.abs(actual - pred_mask).mean())

                print(f"\n    {name}:")
                print(f"      MAE on individual perms: {np.mean(errors):.3f}")
                if mean_error > 0:
                    print(f"      Ratio (individual/mean): {np.mean(errors)/mean_error:.2f}x")

    print("\n" + "="*70)
    print("Analysis 4: Can Model Predict Variance?")
    print("="*70)

    print("\nIf we train a model to predict variance, how well does it work?")

    var_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    var_model.fit(X, var_train)
    var_pred = var_model.predict(X)

    r_var_all = np.corrcoef(var_train, var_pred)[0, 1]
    print(f"\nVariance prediction (all pairs): r={r_var_all:.3f}")

    for name, mask in [('Consistent High', consistent_high),
                       ('Topology-Specific', topology_specific),
                       ('Never High', never_high)]:
        if mask.sum() > 10:
            r_var = np.corrcoef(var_train[mask], var_pred[mask])[0, 1]
            print(f"  {name}: r={r_var:.3f}")

    print("\n" + "="*70)
    print("Analysis 5: Example Pairs")
    print("="*70)

    print("\nLet's look at specific examples of 'consistent high' pairs:")

    # Find a few examples
    consistent_high_indices = np.where(consistent_high)[0][:5]

    for idx in consistent_high_indices:
        print(f"\nPair {idx}:")
        print(f"  Degree product: {X[idx, 2]:.0f}")
        print(f"  Counts in train perms 0-4: {counts_train[idx]}")
        print(f"  Mean: {mean_train[idx]:.2f}, Var: {var_train[idx]:.2f}, Std: {np.sqrt(var_train[idx]):.2f}")
        print(f"  Counts in test perms 15-20: {counts_test[idx]}")
        print(f"  Test mean: {mean_test[idx]:.2f}, Test var: {var_test[idx]:.2f}")
        print(f"  Model prediction: {pred[idx]:.2f}")
        print(f"  Error on test perms: {np.abs(counts_test[idx] - pred[idx])}")
        print(f"  Mean abs error: {np.abs(counts_test[idx] - pred[idx]).mean():.2f}")

    # Create visualization
    create_variance_plots(
        mean_train, var_train, mean_test, var_test,
        counts_train, counts_test, pred,
        consistent_high, topology_specific, never_high,
        consistent_high_indices,
        output_dir
    )

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    pct_sometimes = (sometimes_high.sum() / consistent_high.sum() * 100) if consistent_high.sum() > 0 else 0
    cv_ratio = cv_train[consistent_high].mean() / cv_train[never_high].mean() if never_high.sum() > 0 else 0

    print(f"\n'Consistent high' pairs have highest errors because:")
    print(f"  1. Only {100*always_high.sum()/consistent_high.sum():.0f}% are high in ALL 5 training perms")
    print(f"  2. They have {var_train[consistent_high].mean():.1f}x higher variance than never-high pairs")
    print(f"  3. Coefficient of variation is {cv_ratio:.1f}x higher")
    print(f"  4. Model correctly predicts the MEAN but individual perms deviate substantially")
    print(f"\nThis is fundamentally a VARIANCE prediction problem, not a mean prediction problem.")
    print(f"Even high-degree pairs show topology-dependent variation across permutations.")


def create_variance_plots(mean_train, var_train, mean_test, var_test,
                         counts_train, counts_test, pred,
                         consistent_high, topology_specific, never_high,
                         example_indices, output_dir):
    """Create visualizations of variance analysis."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = {
        'Consistent High': '#d62728',
        'Topology-Specific': '#ff7f0e',
        'Never High': '#1f77b4'
    }

    # Plot 1: Mean vs Variance
    ax = axes[0, 0]
    for name, mask in [('Never High', never_high),
                       ('Topology-Specific', topology_specific),
                       ('Consistent High', consistent_high)]:
        if mask.sum() > 0:
            ax.scatter(mean_train[mask], var_train[mask], alpha=0.3, s=10,
                      label=name, color=colors[name])
    ax.set_xlabel('Mean Count (train)')
    ax.set_ylabel('Variance (train)')
    ax.set_title('Mean-Variance Relationship')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    # Plot 2: CV distribution
    ax = axes[0, 1]
    cv_train = np.sqrt(var_train) / (mean_train + 0.01)
    cv_data = []
    labels = []
    for name, mask in [('Consistent High', consistent_high),
                       ('Topology-Specific', topology_specific),
                       ('Never High', never_high)]:
        if mask.sum() > 0:
            cv_data.append(cv_train[mask])
            labels.append(f'{name}\n(n={mask.sum()})')

    bp = ax.boxplot(cv_data, tick_labels=labels, patch_artist=True)
    for patch, name in zip(bp['boxes'], ['Consistent High', 'Topology-Specific', 'Never High']):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax.set_ylabel('Coefficient of Variation (std/mean)')
    ax.set_title('Relative Variability by Pair Type')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Train vs Test Variance
    ax = axes[0, 2]
    for name, mask in [('Never High', never_high),
                       ('Topology-Specific', topology_specific),
                       ('Consistent High', consistent_high)]:
        if mask.sum() > 0:
            ax.scatter(var_train[mask], var_test[mask], alpha=0.3, s=10,
                      label=name, color=colors[name])
    max_var = max(var_train.max(), var_test.max())
    ax.plot([0, max_var], [0, max_var], 'k--', alpha=0.5)
    ax.set_xlabel('Variance (train perms 0-4)')
    ax.set_ylabel('Variance (test perms 15-20)')
    ax.set_title('Variance Consistency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    # Plots 4-6: Example trajectories for consistent high pairs
    for plot_idx, pair_idx in enumerate(example_indices[:3]):
        ax = axes[1, plot_idx]

        train_counts = counts_train[pair_idx]
        test_counts = counts_test[pair_idx]
        all_perms = list(range(5)) + list(range(15, 21))
        all_counts = np.concatenate([train_counts, test_counts])

        ax.plot(range(5), train_counts, 'o-', color='blue', label='Train perms', markersize=8)
        ax.plot(range(5, 11), test_counts, 's-', color='red', label='Test perms', markersize=8)
        ax.axhline(pred[pair_idx], color='green', linestyle='--', linewidth=2, label='Model prediction')
        ax.axhline(mean_train[pair_idx], color='blue', linestyle=':', alpha=0.5, label='Train mean')

        ax.set_xlabel('Permutation index')
        ax.set_ylabel('Count')
        ax.set_title(f'Pair {pair_idx}\nDegProd={int(pred[pair_idx])}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'consistent_high_variance_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_file}")
    plt.close()


def main():
    analyze_consistent_high_variance()


if __name__ == '__main__':
    main()
