"""
Theoretical correction formulas for pathway count prediction bias.

This module implements theoretically-motivated correction formulas to address
the systematic bias between original graph predictions and permutation averages.

The key insight from Phase 5b diagnostics: bias is heteroscedastic (varies with
pathway count magnitude), requiring multiplicative or ratio-based correction
rather than simple additive offset.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin


class MultiplicativeCorrectionModel(BaseEstimator, RegressorMixin):
    """
    Multiplicative correction model.

    Hypothesis: Original graph pathway counts are systematically lower than
    permutation averages by a scaling factor that depends on pathway magnitude.

    Model: P_perm = P_original × exp(alpha + beta × log(P_original))

    This captures the observed pattern where bias increases with pathway count.
    """

    def __init__(self, base_model=None):
        """
        Initialize multiplicative correction model.

        Parameters
        ----------
        base_model : sklearn estimator
            Base model for initial predictions (e.g., LinearRegression)
        """
        self.base_model = base_model
        self.alpha_ = None
        self.beta_ = None

    def fit(self, X, y):
        """
        Fit base model and learn multiplicative correction.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets (permutation averages)

        Returns
        -------
        self
        """
        self.base_model.fit(X, y)

        y_pred_base = self.base_model.predict(X)

        y_pred_safe = np.clip(y_pred_base, 1e-8, None)
        y_safe = np.clip(y, 1e-8, None)

        log_ratio = np.log(y_safe / y_pred_safe)
        log_pred = np.log(y_pred_safe)

        from sklearn.linear_model import LinearRegression
        correction_model = LinearRegression()
        correction_model.fit(log_pred.reshape(-1, 1), log_ratio)

        self.beta_ = correction_model.coef_[0]
        self.alpha_ = correction_model.intercept_

        return self

    def predict(self, X):
        """
        Predict with multiplicative correction.

        Parameters
        ----------
        X : np.ndarray
            Test features

        Returns
        -------
        np.ndarray
            Corrected predictions
        """
        y_pred_base = self.base_model.predict(X)

        y_pred_safe = np.clip(y_pred_base, 1e-8, None)
        log_pred = np.log(y_pred_safe)

        correction_factor = np.exp(self.alpha_ + self.beta_ * log_pred)

        return y_pred_base * correction_factor


class RatioCorrectionModel(BaseEstimator, RegressorMixin):
    """
    Ratio-based correction model.

    Hypothesis: The ratio P_perm / P_original depends on pathway magnitude
    and can be modeled as a smooth function.

    Model: P_perm = P_original × f(P_original)
    where f is learned from data.
    """

    def __init__(self, base_model=None, method='polynomial', degree=2):
        """
        Initialize ratio correction model.

        Parameters
        ----------
        base_model : sklearn estimator
            Base model for initial predictions
        method : str
            Correction method: 'polynomial', 'exponential', 'sigmoid'
        degree : int
            Polynomial degree (if method='polynomial')
        """
        self.base_model = base_model
        self.method = method
        self.degree = degree
        self.correction_params_ = None

    def fit(self, X, y):
        """
        Fit base model and learn ratio correction.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets

        Returns
        -------
        self
        """
        self.base_model.fit(X, y)

        y_pred_base = self.base_model.predict(X)

        y_pred_safe = np.clip(y_pred_base, 1e-8, None)
        y_safe = np.clip(y, 1e-8, None)

        ratio = y_safe / y_pred_safe

        if self.method == 'polynomial':
            coeffs = np.polyfit(y_pred_safe, ratio, self.degree)
            self.correction_params_ = coeffs

        elif self.method == 'exponential':
            def exp_model(params, x):
                a, b, c = params
                return a + b * np.exp(c * x)

            def loss(params):
                return np.mean((ratio - exp_model(params, y_pred_safe)) ** 2)

            result = minimize(loss, [1.0, 0.1, -1.0], method='L-BFGS-B')
            self.correction_params_ = result.x

        elif self.method == 'sigmoid':
            def sigmoid_model(params, x):
                a, b, c, d = params
                return a + (b - a) / (1 + np.exp(-c * (x - d)))

            def loss(params):
                return np.mean((ratio - sigmoid_model(params, y_pred_safe)) ** 2)

            result = minimize(
                loss,
                [0.8, 1.2, 1.0, np.median(y_pred_safe)],
                method='L-BFGS-B'
            )
            self.correction_params_ = result.x

        return self

    def predict(self, X):
        """
        Predict with ratio correction.

        Parameters
        ----------
        X : np.ndarray
            Test features

        Returns
        -------
        np.ndarray
            Corrected predictions
        """
        y_pred_base = self.base_model.predict(X)

        y_pred_safe = np.clip(y_pred_base, 1e-8, None)

        if self.method == 'polynomial':
            ratio_pred = np.polyval(self.correction_params_, y_pred_safe)

        elif self.method == 'exponential':
            a, b, c = self.correction_params_
            ratio_pred = a + b * np.exp(c * y_pred_safe)

        elif self.method == 'sigmoid':
            a, b, c, d = self.correction_params_
            ratio_pred = a + (b - a) / (1 + np.exp(-c * (y_pred_safe - d)))

        ratio_pred = np.clip(ratio_pred, 0.1, 10.0)

        return y_pred_base * ratio_pred


class QuantileCorrectionModel(BaseEstimator, RegressorMixin):
    """
    Quantile-based correction model.

    Divides prediction range into quantiles and learns separate correction
    for each quantile. Non-parametric alternative to polynomial/exponential.
    """

    def __init__(self, base_model=None, n_quantiles=4):
        """
        Initialize quantile correction model.

        Parameters
        ----------
        base_model : sklearn estimator
            Base model for initial predictions
        n_quantiles : int
            Number of quantiles
        """
        self.base_model = base_model
        self.n_quantiles = n_quantiles
        self.quantile_bounds_ = None
        self.quantile_corrections_ = None

    def fit(self, X, y):
        """
        Fit base model and learn quantile corrections.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets

        Returns
        -------
        self
        """
        self.base_model.fit(X, y)

        y_pred_base = self.base_model.predict(X)

        self.quantile_bounds_ = np.quantile(
            y_pred_base,
            np.linspace(0, 1, self.n_quantiles + 1)
        )

        self.quantile_corrections_ = []

        for i in range(self.n_quantiles):
            mask = (y_pred_base >= self.quantile_bounds_[i]) & \
                   (y_pred_base < self.quantile_bounds_[i + 1])

            if i == self.n_quantiles - 1:
                mask = (y_pred_base >= self.quantile_bounds_[i]) & \
                       (y_pred_base <= self.quantile_bounds_[i + 1])

            if mask.sum() > 0:
                correction = np.mean(y[mask] - y_pred_base[mask])
            else:
                correction = 0.0

            self.quantile_corrections_.append(correction)

        return self

    def predict(self, X):
        """
        Predict with quantile correction.

        Parameters
        ----------
        X : np.ndarray
            Test features

        Returns
        -------
        np.ndarray
            Corrected predictions
        """
        y_pred_base = self.base_model.predict(X)

        corrections = np.zeros_like(y_pred_base)

        for i in range(self.n_quantiles):
            mask = (y_pred_base >= self.quantile_bounds_[i]) & \
                   (y_pred_base < self.quantile_bounds_[i + 1])

            if i == self.n_quantiles - 1:
                mask = (y_pred_base >= self.quantile_bounds_[i]) & \
                       (y_pred_base <= self.quantile_bounds_[i + 1])

            corrections[mask] = self.quantile_corrections_[i]

        return y_pred_base + corrections


class TheoryGuidedCorrectionModel(BaseEstimator, RegressorMixin):
    """
    Theory-guided correction based on degree product model.

    Hypothesis: Permutation average pathway counts follow degree product model
    more closely than original graph, which has biological constraints.

    Model: P_perm = P_original + gamma × (P_degree_product - P_original)

    where gamma is the "biologica constraint relaxation factor".
    """

    def __init__(self, base_model=None):
        """
        Initialize theory-guided correction model.

        Parameters
        ----------
        base_model : sklearn estimator
            Base model for initial predictions
        """
        self.base_model = base_model
        self.gamma_ = None

    def fit(self, X, y):
        """
        Fit base model and learn relaxation factor.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets

        Returns
        -------
        self
        """
        self.base_model.fit(X, y)

        y_pred_base = self.base_model.predict(X)

        source_bin = X[:, 0]
        target_bin = X[:, 1]
        degree_product_proxy = source_bin * target_bin

        from sklearn.linear_model import LinearRegression
        dp_model = LinearRegression()
        dp_model.fit(degree_product_proxy.reshape(-1, 1), y)
        y_degree_product = dp_model.predict(
            degree_product_proxy.reshape(-1, 1)
        )

        residual = y - y_pred_base
        delta = y_degree_product - y_pred_base

        mask = np.abs(delta) > 1e-8
        if mask.sum() > 0:
            self.gamma_ = np.mean(residual[mask] / delta[mask])
        else:
            self.gamma_ = 0.0

        self.gamma_ = np.clip(self.gamma_, -2.0, 2.0)

        return self

    def predict(self, X):
        """
        Predict with theory-guided correction.

        Parameters
        ----------
        X : np.ndarray
            Test features

        Returns
        -------
        np.ndarray
            Corrected predictions
        """
        y_pred_base = self.base_model.predict(X)

        source_bin = X[:, 0]
        target_bin = X[:, 1]
        degree_product_proxy = source_bin * target_bin

        from sklearn.linear_model import LinearRegression
        dp_model = LinearRegression()

        return y_pred_base + self.gamma_ * degree_product_proxy * 0.01
