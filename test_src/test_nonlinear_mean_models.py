"""
Test non-linear models for pathway count mean prediction.

Compares multiple approaches to improve mean prediction beyond baseline linear
regression (r=0.787). Uses proper train/validation/test split:
- Train: perms 0-4 (K=5)
- Validation: perms 10-14 (5 perms) for hyperparameter tuning
- Test: perms 15-20 (6 perms) for final evaluation

Models tested:
1. Baseline: Linear regression on 5 degree features
2. Random Forest: Regression with tuned hyperparameters
3. Gradient Boosting: Regression with tuned hyperparameters
4. Neural Network: Multi-layer with early stopping
5. Polynomial (degree=3): Higher-order polynomial features
6. Log Features: Log-transformed degrees

Goal: Determine if non-linear models can improve r > 0.82 and reduce Q-Q heavy tail.

Usage:
    python test_src/test_nonlinear_mean_models.py CbGpPW --model rf
    python test_src/test_nonlinear_mean_models.py CbGpPW --all

References:
    - docs/2025-11-11_MEAN_VARIANCE_VALIDATION_RESULTS.md (baseline r=0.787)
    - docs/2025-11-11_VARIANCE_ALTERNATIVES.md (Q-Q calibration issues)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

from test_src.validate_mean_variance_prediction import (
    load_permuted_edge_matrices,
    sample_pairs,
    extract_degree_features,
    compute_pathway_counts,
    evaluate_z_scores
)


class SimpleNN(nn.Module):
    """Multi-layer neural network for pathway count regression."""

    def __init__(self, input_dim=5, hidden_dims=[64, 32, 16], dropout=0.2):
        super(SimpleNN, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


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


def train_linear_mean(X, mu_train):
    """Train baseline linear regression."""
    model = LinearRegression()
    model.fit(X, mu_train)
    return model, 'linear'


def train_rf_mean(X_train, mu_train, X_val, mu_val):
    """Train Random Forest with hyperparameter tuning on validation set."""
    best_score = -np.inf
    best_params = None
    best_model = None

    param_grid = [
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 10},
        {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 10},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 5},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_leaf': 3},
    ]

    print("  Tuning Random Forest hyperparameters...")
    for params in param_grid:
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )
        model.fit(X_train, mu_train)
        mu_pred_val = model.predict(X_val)
        score = np.corrcoef(mu_val, mu_pred_val)[0, 1]

        print(f"    {params}: r={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    print(f"  Best params: {best_params}, val r={best_score:.4f}")
    return best_model, 'rf'


def train_gb_mean(X_train, mu_train, X_val, mu_val):
    """Train Gradient Boosting with hyperparameter tuning on validation set."""
    best_score = -np.inf
    best_params = None
    best_model = None

    param_grid = [
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5},
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5},
    ]

    print("  Tuning Gradient Boosting hyperparameters...")
    for params in param_grid:
        model = GradientBoostingRegressor(
            random_state=42,
            **params
        )
        model.fit(X_train, mu_train)
        mu_pred_val = model.predict(X_val)
        score = np.corrcoef(mu_val, mu_pred_val)[0, 1]

        print(f"    {params}: r={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    print(f"  Best params: {best_params}, val r={best_score:.4f}")
    return best_model, 'gb'


def train_nn_mean(X_train, mu_train, X_val, mu_val):
    """Train Neural Network with early stopping on validation set."""
    X_train_t = torch.FloatTensor(X_train)
    mu_train_t = torch.FloatTensor(mu_train)
    X_val_t = torch.FloatTensor(X_val)
    mu_val_t = torch.FloatTensor(mu_val)

    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim=input_dim, hidden_dims=[64, 32, 16], dropout=0.2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = np.inf
    patience = 20
    patience_counter = 0
    best_state = None

    print("  Training Neural Network with early stopping...")
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        mu_pred_train = model(X_train_t)
        loss = criterion(mu_pred_train, mu_train_t)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_pred_val = model(X_val_t)
            val_loss = criterion(mu_pred_val, mu_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        mu_pred_val = model(X_val_t).numpy()
    val_r = np.corrcoef(mu_val, mu_pred_val)[0, 1]
    print(f"  Best validation r={val_r:.4f}")

    return model, 'nn'


def train_poly_mean(X_train, mu_train, degree=3):
    """Train linear regression on polynomial features."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)

    print(f"  Polynomial degree={degree}: {X_train.shape[1]} -> {X_poly.shape[1]} features")

    model = LinearRegression()
    model.fit(X_poly, mu_train)

    return (model, poly), f'poly{degree}'


def train_log_mean(X_train, mu_train):
    """Train linear regression on log-transformed features."""
    X_log = np.log1p(X_train)

    model = LinearRegression()
    model.fit(X_log, mu_train)

    return model, 'log'


def train_negbin_glm(X_train, counts_train, X_val, counts_val):
    """
    Train Negative Binomial GLM for count data.

    Negative binomial regression is appropriate for count data with overdispersion
    (variance > mean). Unlike normal regression, it respects the discrete nature
    of counts and naturally models heteroscedasticity through the mean-variance
    relationship: var = mu + mu^2 / alpha.

    Args:
        X_train: Training features (degree features)
        counts_train: Individual count observations from training perms (n_pairs × n_perms)
        X_val: Validation features (same as X_train)
        counts_val: Validation counts for model selection

    Returns:
        model: Tuple of (NegativeBinomialResults, scaler)
        model_type: String 'negbin_glm'
    """
    from sklearn.preprocessing import StandardScaler

    counts_train_flat = []
    X_train_expanded = []
    for i in range(len(X_train)):
        for j in range(counts_train.shape[1]):
            counts_train_flat.append(counts_train[i, j])
            X_train_expanded.append(X_train[i])

    counts_train_flat = np.array(counts_train_flat)
    X_train_expanded = np.array(X_train_expanded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_expanded)
    X_with_intercept = sm.add_constant(X_train_scaled)

    print("  Fitting Negative Binomial GLM (this may take a few minutes)...")

    nb_model = NegativeBinomial(counts_train_flat, X_with_intercept)

    try:
        nb_results = nb_model.fit(method='newton', maxiter=100, disp=False)

        print(f"  Converged: {nb_results.mle_retvals['converged']}")

        alpha = 1.0 / nb_results.params[-1]
        print(f"  Alpha (dispersion): {alpha:.4f}")

        counts_val_mean = np.mean(counts_val, axis=1)
        X_val_scaled = scaler.transform(X_val)
        X_val_with_intercept = sm.add_constant(X_val_scaled)
        mu_pred_val = nb_results.predict(X_val_with_intercept)
        val_r = np.corrcoef(counts_val_mean, mu_pred_val)[0, 1]
        print(f"  Validation r={val_r:.4f}")

        return (nb_results, scaler), 'negbin_glm'

    except Exception as e:
        print(f"  WARNING: NB GLM fitting failed: {e}")
        print(f"  Falling back to Poisson GLM...")

        from statsmodels.discrete.discrete_model import Poisson
        poisson_model = Poisson(counts_train_flat, X_with_intercept)
        poisson_results = poisson_model.fit(method='newton', maxiter=100, disp=False)

        print(f"  Poisson converged: {poisson_results.mle_retvals['converged']}")

        return (poisson_results, scaler), 'poisson_glm'


def train_weighted_linear(X, mu_train, sigma_train):
    """Train weighted linear regression using empirical variance."""
    weights = 1.0 / (sigma_train ** 2 + 1e-6)

    model = LinearRegression()
    model.fit(X, mu_train, sample_weight=weights)

    print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")

    return model, 'weighted_linear'


def train_weighted_rf(X_train, mu_train, sigma_train, X_val, mu_val):
    """Train weighted Random Forest using empirical variance."""
    weights = 1.0 / (sigma_train ** 2 + 1e-6)

    best_score = -np.inf
    best_params = None
    best_model = None

    param_grid = [
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 10},
        {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 10},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 5},
    ]

    print("  Tuning Weighted Random Forest hyperparameters...")
    for params in param_grid:
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )
        model.fit(X_train, mu_train, sample_weight=weights)
        mu_pred_val = model.predict(X_val)
        score = np.corrcoef(mu_val, mu_pred_val)[0, 1]

        print(f"    {params}: r={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    print(f"  Best params: {best_params}, val r={best_score:.4f}")
    return best_model, 'weighted_rf'


def train_weighted_gb(X_train, mu_train, sigma_train, X_val, mu_val):
    """Train weighted Gradient Boosting using empirical variance."""
    weights = 1.0 / (sigma_train ** 2 + 1e-6)

    best_score = -np.inf
    best_params = None
    best_model = None

    param_grid = [
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
    ]

    print("  Tuning Weighted Gradient Boosting hyperparameters...")
    for params in param_grid:
        model = GradientBoostingRegressor(
            random_state=42,
            **params
        )
        model.fit(X_train, mu_train, sample_weight=weights)
        mu_pred_val = model.predict(X_val)
        score = np.corrcoef(mu_val, mu_pred_val)[0, 1]

        print(f"    {params}: r={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    print(f"  Best params: {best_params}, val r={best_score:.4f}")
    return best_model, 'weighted_gb'


def calibrate_variance_isotonic(model_std, X_val, counts_val):
    """
    Calibrate variance predictions using isotonic regression on validation set.

    Args:
        model_std: Trained variance model
        X_val: Validation features
        counts_val: Validation counts, shape (n_pairs, n_val_perms)

    Returns:
        IsotonicRegression model mapping predicted_std → calibrated_std
    """
    sigma_pred_val = model_std.predict(X_val)
    sigma_pred_val = np.maximum(sigma_pred_val, 0.1)

    sigma_empirical_val = np.std(counts_val, axis=1, ddof=1)

    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(sigma_pred_val, sigma_empirical_val)

    print(f"  Isotonic calibration:")
    print(f"    Predicted std range: [{sigma_pred_val.min():.3f}, {sigma_pred_val.max():.3f}]")
    print(f"    Empirical std range: [{sigma_empirical_val.min():.3f}, {sigma_empirical_val.max():.3f}]")

    sigma_calibrated_val = iso_reg.predict(sigma_pred_val)
    r_calib = np.corrcoef(sigma_empirical_val, sigma_calibrated_val)[0, 1]
    r_uncalib = np.corrcoef(sigma_empirical_val, sigma_pred_val)[0, 1]
    print(f"    Correlation uncalibrated: {r_uncalib:.4f}")
    print(f"    Correlation calibrated: {r_calib:.4f}")

    return iso_reg


def train_heteroscedastic_nn(X_train, counts_train, X_val, counts_val):
    """
    Train Heteroscedastic NN on individual observations.

    Flattens training data to (pair, perm) observations and trains with
    negative log-likelihood loss that jointly optimizes mean and variance.
    """
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)

    counts_train_flat = []
    X_train_expanded = []
    for i in range(len(X_train)):
        for j in range(counts_train.shape[1]):
            counts_train_flat.append(counts_train[i, j])
            X_train_expanded.append(X_train[i])

    counts_train_flat = np.array(counts_train_flat)
    X_train_expanded = np.array(X_train_expanded)
    X_train_expanded_t = torch.FloatTensor(X_train_expanded)
    counts_train_flat_t = torch.FloatTensor(counts_train_flat)

    counts_val_flat = []
    X_val_expanded = []
    for i in range(len(X_val)):
        for j in range(counts_val.shape[1]):
            counts_val_flat.append(counts_val[i, j])
            X_val_expanded.append(X_val[i])

    counts_val_flat = np.array(counts_val_flat)
    X_val_expanded = np.array(X_val_expanded)
    X_val_expanded_t = torch.FloatTensor(X_val_expanded)
    counts_val_flat_t = torch.FloatTensor(counts_val_flat)

    input_dim = X_train.shape[1]
    model = HeteroscedasticNN(input_dim=input_dim, hidden_dims=[64, 32, 16], dropout=0.2)

    def heteroscedastic_loss(y_true, y_pred_mean, y_pred_logvar):
        var = torch.exp(y_pred_logvar)
        loss = 0.5 * (y_pred_logvar + (y_true - y_pred_mean)**2 / var)
        return loss.mean()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = np.inf
    patience = 20
    patience_counter = 0
    best_state = None

    print("  Training Heteroscedastic NN with early stopping...")
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        mu_pred_train, logvar_pred_train = model(X_train_expanded_t)
        loss = heteroscedastic_loss(counts_train_flat_t, mu_pred_train, logvar_pred_train)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_pred_val, logvar_pred_val = model(X_val_expanded_t)
            val_loss = heteroscedastic_loss(counts_val_flat_t, mu_pred_val, logvar_pred_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        mu_pred_val, _ = model(X_val_t)
    val_r = np.corrcoef(np.mean(counts_val, axis=1), mu_pred_val.numpy())[0, 1]
    print(f"  Best validation r={val_r:.4f}")

    return model, 'hetero_nn'


def predict_mean(model, model_type, X):
    """Predict mean with various model types."""
    if model_type in ['linear', 'weighted_linear']:
        return model.predict(X)
    elif model_type in ['rf', 'gb', 'weighted_rf', 'weighted_gb']:
        return model.predict(X)
    elif model_type == 'nn':
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            return model(X_t).numpy()
    elif model_type == 'hetero_nn':
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            mu, _ = model(X_t)
            return mu.numpy()
    elif model_type.startswith('poly'):
        model_lr, poly = model
        X_poly = poly.transform(X)
        return model_lr.predict(X_poly)
    elif model_type == 'log':
        X_log = np.log1p(X)
        return model.predict(X_log)
    elif model_type in ['negbin_glm', 'poisson_glm']:
        glm_model, scaler = model
        X_scaled = scaler.transform(X)
        X_with_intercept = sm.add_constant(X_scaled)
        return glm_model.predict(X_with_intercept)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_test_statistics(pairs, edge1_type, edge2_type, data_dir,
                            test_perms, model_mean, model_std, model_type, X,
                            iso_calibration=None):
    """Compute test statistics on individual permutations."""
    if model_type == 'hetero_nn':
        model_mean.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            mu_pred, logvar_pred = model_mean(X_t)
            mu_pred = mu_pred.numpy()
            sigma_pred = np.sqrt(np.exp(logvar_pred.numpy()))
            sigma_pred = np.maximum(sigma_pred, 0.1)
    elif model_type == 'negbin_glm':
        glm_model, scaler = model_mean
        X_scaled = scaler.transform(X)
        X_with_intercept = sm.add_constant(X_scaled)
        mu_pred = glm_model.predict(X_with_intercept)
        alpha = 1.0 / glm_model.params[-1]
        var_pred = mu_pred + mu_pred**2 / alpha
        sigma_pred = np.sqrt(var_pred)
        sigma_pred = np.maximum(sigma_pred, 0.1)
    elif model_type == 'poisson_glm':
        glm_model, scaler = model_mean
        X_scaled = scaler.transform(X)
        X_with_intercept = sm.add_constant(X_scaled)
        mu_pred = glm_model.predict(X_with_intercept)
        var_pred = mu_pred
        sigma_pred = np.sqrt(var_pred)
        sigma_pred = np.maximum(sigma_pred, 0.1)
    else:
        mu_pred = predict_mean(model_mean, model_type, X)
        sigma_pred = model_std.predict(X)
        sigma_pred = np.maximum(sigma_pred, 0.1)

        if iso_calibration is not None:
            sigma_pred = iso_calibration.predict(sigma_pred)
            sigma_pred = np.maximum(sigma_pred, 0.1)

    counts_test = []
    for perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)
    counts_test = np.column_stack(counts_test)

    n_test = len(test_perms)
    z_scores = []
    counts_flat = []
    mu_pred_expanded = []
    sigma_pred_expanded = []

    for i in range(len(pairs)):
        for j in range(n_test):
            count = counts_test[i, j]
            z = (count - mu_pred[i]) / sigma_pred[i]
            z_scores.append(z)
            counts_flat.append(count)
            mu_pred_expanded.append(mu_pred[i])
            sigma_pred_expanded.append(sigma_pred[i])

    z_scores = np.array(z_scores)
    counts_flat = np.array(counts_flat)
    mu_pred_expanded = np.array(mu_pred_expanded)
    sigma_pred_expanded = np.array(sigma_pred_expanded)

    return {
        'mu_pred_per_pair': mu_pred,
        'sigma_pred_per_pair': sigma_pred,
        'mu_pred': mu_pred_expanded,
        'sigma_pred': sigma_pred_expanded,
        'counts_test': counts_test,
        'counts_flat': counts_flat,
        'z_scores': z_scores
    }


def create_diagnostic_plot(stats_dict, metapath, model_type, output_dir):
    """Create 6-panel diagnostic visualization."""
    z_scores = stats_dict['z_scores']
    mu_pred = stats_dict['mu_pred']
    sigma_pred = stats_dict['sigma_pred']
    counts_flat = stats_dict['counts_flat']
    mu_pred_per_pair = stats_dict['mu_pred_per_pair']
    sigma_pred_per_pair = stats_dict['sigma_pred_per_pair']
    counts_test = stats_dict['counts_test']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Distribution Diagnostics: {metapath}, {model_type} mean model',
                 fontsize=16, y=0.995)

    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    z_mean = np.mean(np.abs(z_scores))
    z_std = np.std(z_scores)

    x = np.linspace(-4, 4, 100)
    ax1.hist(z_scores, bins=50, density=True, alpha=0.7,
             label=f'Empirical (n={len(z_scores)})')
    ax1.plot(x, sp_stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Z-score')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Z-Score Distribution\nmean={z_mean:.3f}, std={z_std:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    qq_result = sp_stats.probplot(z_scores, dist="norm", plot=ax2)
    qq_corr = np.corrcoef(qq_result[0][0], qq_result[0][1])[0, 1]
    ax2.set_title(f'Q-Q Plot\nr={qq_corr:.4f}')
    ax2.grid(True, alpha=0.3)

    ax3.scatter(counts_flat, mu_pred, alpha=0.1, s=5)
    r_mean = np.corrcoef(counts_flat, mu_pred)[0, 1]
    max_val = max(counts_flat.max(), mu_pred.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect calibration')
    ax3.set_xlabel('Observed count (individual perms)')
    ax3.set_ylabel('Predicted mean')
    ax3.set_title(f'Mean Calibration\nr={r_mean:.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    abs_residuals = np.abs(counts_flat - mu_pred)
    ax4.scatter(sigma_pred, abs_residuals, alpha=0.1, s=5)
    r_std = np.corrcoef(sigma_pred, abs_residuals)[0, 1]
    max_sigma = sigma_pred.max()
    ax4.plot([0, max_sigma], [0, max_sigma], 'r--', linewidth=2,
             label='Perfect calibration')
    ax4.set_xlabel('Predicted std')
    ax4.set_ylabel('|Observed - Predicted mean|')
    ax4.set_title(f'Std Calibration\nr={r_std:.4f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    example_indices = [0, len(counts_test) // 2, len(counts_test) - 1]
    for idx in example_indices:
        counts = counts_test[idx]
        mu = mu_pred_per_pair[idx]
        sigma = sigma_pred_per_pair[idx]

        x_range = np.linspace(max(0, mu - 3*sigma), mu + 3*sigma, 100)
        ax5.plot(x_range, sp_stats.norm.pdf(x_range, mu, sigma),
                label=f'N({mu:.2f},{sigma:.2f})', linewidth=2)
        ax5.hist(counts, bins=20, alpha=0.3, density=True)

    ax5.set_xlabel('Pathway count')
    ax5.set_ylabel('Density')
    ax5.set_title('Example Pair Distributions')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    residuals = (counts_flat - mu_pred) / sigma_pred
    ax6.scatter(mu_pred, residuals, alpha=0.1, s=5)
    ax6.axhline(0, color='r', linestyle='--', linewidth=2)
    ax6.axhline(2, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax6.axhline(-2, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Predicted mean')
    ax6.set_ylabel('Standardized residual (z-score)')
    ax6.set_title(f'Residual Analysis\nmean={np.mean(residuals):.3f}, std={np.std(residuals):.3f}')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f'distribution_diagnostics_{model_type}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved diagnostic plot: {output_path}")

    print(f"\nDiagnostic Summary ({model_type}):")
    print(f"  Z-score mean: {z_mean:.4f} (target: 0.8)")
    print(f"  Z-score std: {z_std:.4f} (target: 1.0)")
    print(f"  Mean r (individual perms): {r_mean:.4f}")
    print(f"  Std r (absolute residuals): {r_std:.4f}")
    print(f"  Q-Q corr: {qq_corr:.4f} (target: >0.95)")


def run_experiment(metapath, edge1_type, edge2_type, data_dir, output_dir,
                   model_type='linear', use_isotonic_calibration=False,
                   n_samples=10000, random_state=42):
    """Run mean model experiment with proper train/val/test split."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {metapath} - {model_type} mean model")
    print(f"{'='*70}\n")

    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(
        edge1_type, edge2_type, 0, data_dir
    )

    print(f"Sampling {n_samples} node pairs...")
    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples, random_state)
    print(f"  Sampled pairs: {len(pairs)}")

    print(f"Extracting features...")
    X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)
    print(f"  Feature shape: {X.shape}")

    print(f"Computing TRAINING targets (perms 0-4)...")
    train_perms = list(range(5))
    counts_train = []
    for perm in train_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_train.append(counts)
    counts_train = np.column_stack(counts_train)
    mu_train = np.mean(counts_train, axis=1)
    sigma_train = np.std(counts_train, axis=1, ddof=1)

    print(f"Computing VALIDATION targets (perms 10-14)...")
    val_perms = list(range(10, 15))
    counts_val = []
    for perm in val_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_val.append(counts)
    counts_val = np.column_stack(counts_val)
    mu_val = np.mean(counts_val, axis=1)

    print(f"Training {model_type} mean model...")
    if model_type == 'linear':
        model_mean, _ = train_linear_mean(X, mu_train)
    elif model_type == 'weighted_linear':
        model_mean, _ = train_weighted_linear(X, mu_train, sigma_train)
    elif model_type == 'rf':
        model_mean, _ = train_rf_mean(X, mu_train, X, mu_val)
    elif model_type == 'weighted_rf':
        model_mean, _ = train_weighted_rf(X, mu_train, sigma_train, X, mu_val)
    elif model_type == 'gb':
        model_mean, _ = train_gb_mean(X, mu_train, X, mu_val)
    elif model_type == 'weighted_gb':
        model_mean, _ = train_weighted_gb(X, mu_train, sigma_train, X, mu_val)
    elif model_type == 'nn':
        model_mean, _ = train_nn_mean(X, mu_train, X, mu_val)
    elif model_type == 'hetero_nn':
        model_mean, _ = train_heteroscedastic_nn(X, counts_train, X, counts_val)
    elif model_type.startswith('poly'):
        degree = int(model_type.replace('poly', ''))
        model_mean, _ = train_poly_mean(X, mu_train, degree=degree)
    elif model_type == 'log':
        model_mean, _ = train_log_mean(X, mu_train)
    elif model_type == 'negbin_glm':
        model_mean, model_type = train_negbin_glm(X, counts_train, X, counts_val)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type not in ['hetero_nn', 'negbin_glm', 'poisson_glm']:
        print(f"Training linear variance model...")
        model_std = LinearRegression()
        model_std.fit(X, sigma_train)

        iso_calibration = None
        if use_isotonic_calibration:
            print(f"Training isotonic calibration on validation set...")
            iso_calibration = calibrate_variance_isotonic(model_std, X, counts_val)
    else:
        model_std = None
        iso_calibration = None

    print(f"Evaluating on VALIDATION set (perms 10-14, individual observations)...")
    mu_pred_val = predict_mean(model_mean, model_type, X)

    counts_val_flat = []
    mu_pred_val_expanded = []
    for i in range(len(pairs)):
        for j in range(counts_val.shape[1]):
            counts_val_flat.append(counts_val[i, j])
            mu_pred_val_expanded.append(mu_pred_val[i])

    r_val = np.corrcoef(counts_val_flat, mu_pred_val_expanded)[0, 1]
    print(f"  Validation r (individual obs): {r_val:.4f}")

    print(f"Computing test statistics (TEST perms 15-20)...")
    test_perms = list(range(15, 21))

    actual_model_type = model_type
    if use_isotonic_calibration and model_type != 'hetero_nn':
        actual_model_type = model_type + '_calibrated'

    stats_dict = compute_test_statistics(
        pairs, edge1_type, edge2_type, data_dir, test_perms,
        model_mean, model_std, model_type, X,
        iso_calibration=iso_calibration
    )

    print(f"Generating diagnostic plots...")
    create_diagnostic_plot(stats_dict, metapath, actual_model_type, output_dir)

    results = []
    counts_test = stats_dict['counts_test']
    mu_pred_per_pair = stats_dict['mu_pred_per_pair']
    sigma_pred_per_pair = stats_dict['sigma_pred_per_pair']

    for idx, perm in enumerate(test_perms):
        counts = counts_test[:, idx]
        z_scores = (counts - mu_pred_per_pair) / sigma_pred_per_pair

        r_mean = np.corrcoef(counts, mu_pred_per_pair)[0, 1]
        z_metrics = evaluate_z_scores(z_scores)

        results.append({
            'test_perm': perm,
            'model_type': actual_model_type,
            'r_mean': r_mean,
            'z_mean': z_metrics['z_mean'],
            'z_std': z_metrics['z_std'],
            'z_outliers': z_metrics['z_outliers']
        })

    results_df = pd.DataFrame(results)

    output_file = output_dir / f'{metapath}_{actual_model_type}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results: {output_file}")

    return results_df, r_val


def main():
    parser = argparse.ArgumentParser(description='Test non-linear mean models')
    parser.add_argument('metapath', help='Metapath name (e.g., CbGpPW)')
    parser.add_argument('--model', choices=['linear', 'rf', 'gb', 'nn', 'poly3', 'log',
                                            'weighted_linear', 'weighted_rf', 'weighted_gb', 'hetero_nn',
                                            'negbin_glm'],
                       help='Mean model type')
    parser.add_argument('--variance_models', action='store_true',
                       help='Run variance-aware models (weighted + heteroscedastic)')
    parser.add_argument('--calibrate', action='store_true',
                       help='Use isotonic regression to calibrate variance predictions')
    parser.add_argument('--all', action='store_true',
                       help='Run all basic models')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of node pairs to sample')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    metapath_configs = {
        'CbGpPW': ('CbG', 'GpPW')
    }

    if args.metapath not in metapath_configs:
        print(f"Error: Unknown metapath {args.metapath}")
        print(f"Available: {list(metapath_configs.keys())}")
        sys.exit(1)

    edge1_type, edge2_type = metapath_configs[args.metapath]

    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'nonlinear_mean_models'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.variance_models:
        models = ['linear', 'weighted_linear', 'rf', 'weighted_rf', 'gb', 'weighted_gb', 'hetero_nn']
    elif args.all:
        models = ['linear', 'rf', 'gb', 'nn', 'poly3', 'log']
    elif args.model:
        models = [args.model]
    else:
        print("Error: Must specify --model or --all")
        sys.exit(1)

    all_results = []
    val_scores = {}

    for model_type in models:
        results_df, r_val = run_experiment(
            args.metapath, edge1_type, edge2_type, data_dir, output_dir,
            model_type=model_type, use_isotonic_calibration=args.calibrate,
            n_samples=args.n_samples, random_state=args.random_state
        )
        all_results.append(results_df)
        actual_model_type = model_type + '_calibrated' if args.calibrate and model_type != 'hetero_nn' else model_type
        val_scores[actual_model_type] = r_val

    if len(all_results) > 1:
        comparison_df = pd.concat(all_results, ignore_index=True)
        comparison_file = output_dir / f'{args.metapath}_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nSaved comparison: {comparison_file}")

        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print("\nValidation Performance (perms 10-14):")
        for model, r_val in val_scores.items():
            print(f"  {model:10s}: r = {r_val:.4f}")

        print("\nTest Performance (perms 15-20, averaged):")
        summary = comparison_df.groupby('model_type')[['r_mean', 'z_mean', 'z_std', 'z_outliers']].mean()
        print(summary)


if __name__ == '__main__':
    main()
