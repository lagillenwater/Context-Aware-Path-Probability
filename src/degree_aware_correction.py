"""
Degree-aware correction model for pathway count predictions.

This module implements a correction model that learns the systematic difference
between original graph and permutation pathway counts as a function of both
degree features and pathway count magnitude.

Key insight: The correction needed varies by degree bin and scales with
pathway count, requiring a model that takes both into account.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, Lasso


class DegreeAwareCorrectionModel(BaseEstimator, RegressorMixin):
    """
    Two-stage model with degree-aware correction.

    Stage 1: Predict original graph pathway counts
    Stage 2: Predict correction based on degrees and Stage 1 predictions

    The correction is learned from (original graph, permutation 0) pairs.
    """

    def __init__(self, base_model=None, correction_model=None,
                 use_interaction=True, alpha=1.0):
        """
        Initialize degree-aware correction model.

        Parameters
        ----------
        base_model : sklearn estimator
            Model for predicting original graph counts
        correction_model : sklearn estimator
            Model for predicting correction (default: Ridge)
        use_interaction : bool
            Whether to include interaction terms (pred × degree features)
        alpha : float
            Regularization strength for correction model
        """
        self.base_model = base_model
        self.correction_model = correction_model
        self.use_interaction = use_interaction
        self.alpha = alpha

    def _extract_correction_features(self, X, y_pred):
        """
        Extract features for correction model.

        Parameters
        ----------
        X : np.ndarray
            Original degree features
        y_pred : np.ndarray
            Predictions from base model

        Returns
        -------
        np.ndarray
            Correction features
        """
        # Start with degree features (use first few features which are degree-related)
        # Assuming Feature Set E: first features are source_bin, target_bin
        source_bin = X[:, 0]
        target_bin = X[:, 1]

        features = [
            source_bin,
            target_bin,
            source_bin * target_bin,  # Degree product
            source_bin ** 2,
            target_bin ** 2,
            np.sqrt(source_bin + 1),
            np.sqrt(target_bin + 1),
            y_pred,  # Predicted pathway count
            y_pred ** 2,  # Squared for non-linearity
            np.log1p(y_pred)  # Log for scale
        ]

        if self.use_interaction:
            # Interaction terms: prediction × degree features
            features.extend([
                y_pred * source_bin,
                y_pred * target_bin,
                y_pred * source_bin * target_bin,
                np.sqrt(y_pred + 1) * source_bin,
                np.sqrt(y_pred + 1) * target_bin
            ])

        return np.column_stack(features)

    def fit(self, X, y_original, y_perm0):
        """
        Fit two-stage model.

        Parameters
        ----------
        X : np.ndarray
            Degree features
        y_original : np.ndarray
            Pathway counts from original graph
        y_perm0 : np.ndarray
            Pathway counts from permutation 0

        Returns
        -------
        self
        """
        # Stage 1: Fit base model on original graph
        self.base_model.fit(X, y_original)

        # Get predictions on training data
        y_pred = self.base_model.predict(X)

        # Compute needed correction: what we need to add to get from original to perm
        correction_target = y_perm0 - y_pred

        # Stage 2: Fit correction model
        correction_features = self._extract_correction_features(X, y_pred)

        if self.correction_model is None:
            self.correction_model = Ridge(alpha=self.alpha)

        self.correction_model.fit(correction_features, correction_target)

        return self

    def predict(self, X):
        """
        Predict with degree-aware correction.

        Parameters
        ----------
        X : np.ndarray
            Degree features

        Returns
        -------
        np.ndarray
            Corrected predictions (estimates of permutation counts)
        """
        # Stage 1: Predict original graph counts
        y_pred = self.base_model.predict(X)

        # Stage 2: Predict correction
        correction_features = self._extract_correction_features(X, y_pred)
        correction = self.correction_model.predict(correction_features)

        # Apply correction
        return y_pred + correction


class AdaptiveDegreeAwareCorrectionModel(BaseEstimator, RegressorMixin):
    """
    Adaptive correction that learns separate models for different degree regions.

    Divides degree space into regions and learns specialized corrections.
    """

    def __init__(self, base_model=None, n_regions=4, alpha=1.0):
        """
        Initialize adaptive correction model.

        Parameters
        ----------
        base_model : sklearn estimator
            Model for predicting original graph counts
        n_regions : int
            Number of degree regions (split by quantiles)
        alpha : float
            Regularization strength
        """
        self.base_model = base_model
        self.n_regions = n_regions
        self.alpha = alpha
        self.region_bounds_ = None
        self.region_models_ = None

    def _assign_regions(self, X):
        """
        Assign samples to degree regions.

        Parameters
        ----------
        X : np.ndarray
            Degree features

        Returns
        -------
        np.ndarray
            Region assignments (0 to n_regions-1)
        """
        source_bin = X[:, 0]
        target_bin = X[:, 1]

        # Use geometric mean of degrees to define regions
        degree_metric = np.sqrt(source_bin * target_bin)

        if self.region_bounds_ is None:
            # Learn region boundaries from training data
            self.region_bounds_ = np.quantile(
                degree_metric,
                np.linspace(0, 1, self.n_regions + 1)
            )

        # Assign to regions
        regions = np.digitize(degree_metric, self.region_bounds_[1:-1])

        return regions

    def fit(self, X, y_original, y_perm0):
        """
        Fit adaptive correction model.

        Parameters
        ----------
        X : np.ndarray
            Degree features
        y_original : np.ndarray
            Pathway counts from original graph
        y_perm0 : np.ndarray
            Pathway counts from permutation 0

        Returns
        -------
        self
        """
        # Fit base model
        self.base_model.fit(X, y_original)
        y_pred = self.base_model.predict(X)

        # Assign samples to regions
        regions = self._assign_regions(X)

        # Fit separate correction model for each region
        self.region_models_ = []

        for r in range(self.n_regions):
            mask = regions == r

            if mask.sum() > 3:  # Need at least 3 samples
                correction_target = y_perm0[mask] - y_pred[mask]

                # Simple features for region-specific model
                region_features = np.column_stack([
                    y_pred[mask],
                    y_pred[mask] ** 2,
                    np.log1p(y_pred[mask])
                ])

                region_model = Ridge(alpha=self.alpha)
                region_model.fit(region_features, correction_target)
            else:
                # Not enough samples, use zero correction
                region_model = None

            self.region_models_.append(region_model)

        return self

    def predict(self, X):
        """
        Predict with adaptive correction.

        Parameters
        ----------
        X : np.ndarray
            Degree features

        Returns
        -------
        np.ndarray
            Corrected predictions
        """
        y_pred = self.base_model.predict(X)
        regions = self._assign_regions(X)

        corrections = np.zeros_like(y_pred)

        for r in range(self.n_regions):
            mask = regions == r

            if mask.sum() > 0 and self.region_models_[r] is not None:
                region_features = np.column_stack([
                    y_pred[mask],
                    y_pred[mask] ** 2,
                    np.log1p(y_pred[mask])
                ])

                corrections[mask] = self.region_models_[r].predict(region_features)

        return y_pred + corrections


class MultiplicativeDegreeAwareCorrectionModel(BaseEstimator, RegressorMixin):
    """
    Multiplicative correction: P_perm = P_original × correction_factor(degrees, P).

    Models the ratio rather than the difference.
    """

    def __init__(self, base_model=None, alpha=1.0):
        """
        Initialize multiplicative correction model.

        Parameters
        ----------
        base_model : sklearn estimator
            Model for predicting original graph counts
        alpha : float
            Regularization strength
        """
        self.base_model = base_model
        self.alpha = alpha
        self.correction_model = None

    def _extract_correction_features(self, X, y_pred):
        """Extract features for multiplicative correction."""
        source_bin = X[:, 0]
        target_bin = X[:, 1]

        log_pred = np.log1p(y_pred)

        return np.column_stack([
            source_bin,
            target_bin,
            source_bin * target_bin,
            log_pred,
            log_pred ** 2,
            source_bin * log_pred,
            target_bin * log_pred
        ])

    def fit(self, X, y_original, y_perm0):
        """
        Fit multiplicative correction model.

        Parameters
        ----------
        X : np.ndarray
            Degree features
        y_original : np.ndarray
            Pathway counts from original graph
        y_perm0 : np.ndarray
            Pathway counts from permutation 0

        Returns
        -------
        self
        """
        self.base_model.fit(X, y_original)
        y_pred = self.base_model.predict(X)

        # Compute ratio (with safety for division)
        y_pred_safe = np.clip(y_pred, 1e-8, None)
        y_perm0_safe = np.clip(y_perm0, 1e-8, None)

        ratio_target = y_perm0_safe / y_pred_safe
        log_ratio = np.log(ratio_target)

        # Model log(ratio) as function of degrees and prediction
        correction_features = self._extract_correction_features(X, y_pred)

        self.correction_model = Ridge(alpha=self.alpha)
        self.correction_model.fit(correction_features, log_ratio)

        return self

    def predict(self, X):
        """
        Predict with multiplicative correction.

        Parameters
        ----------
        X : np.ndarray
            Degree features

        Returns
        -------
        np.ndarray
            Corrected predictions
        """
        y_pred = self.base_model.predict(X)

        correction_features = self._extract_correction_features(X, y_pred)
        log_ratio_pred = self.correction_model.predict(correction_features)

        # Convert back from log space
        ratio_pred = np.exp(log_ratio_pred)

        # Clip to reasonable range
        ratio_pred = np.clip(ratio_pred, 0.5, 2.0)

        return y_pred * ratio_pred
