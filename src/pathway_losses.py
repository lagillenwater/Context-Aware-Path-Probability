"""
Alternative loss functions for pathway frequency prediction.

This module implements loss functions designed to address the overprediction
bias observed with standard MSE loss. MSE heavily penalizes large errors,
causing the model to bias toward high counts. Alternative losses provide:

1. Huber loss: Robust to outliers with linear penalty for large errors
2. Log-scale MSE: Balances count ranges (reduces high-count bias)
3. Quantile loss: Median prediction (robust to outliers)
4. Negative binomial loss: Proper distribution for count data

All losses are implemented in PyTorch for gradient-based optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class HuberLoss(nn.Module):
    """
    Huber loss for robust regression.

    Combines MSE (for small errors) with MAE (for large errors), providing
    robustness to outliers while maintaining quadratic smoothness near zero.

    Loss(y, y_pred) = 0.5 * (y - y_pred)^2           if |y - y_pred| <= delta
                    = delta * (|y - y_pred| - 0.5*delta)  otherwise

    Parameters
    ----------
    delta : float, default=1.0
        Threshold for switching from quadratic to linear loss.
        Smaller delta = more robust to outliers
        Larger delta = closer to MSE
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'

    References
    ----------
    Huber, P. J. (1964). Robust Estimation of a Location Parameter.
    """

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y_true : torch.Tensor
            True values

        Returns
        -------
        loss : torch.Tensor
            Huber loss
        """
        error = y_true - y_pred
        abs_error = torch.abs(error)

        quadratic = 0.5 * error**2
        linear = self.delta * (abs_error - 0.5 * self.delta)

        loss = torch.where(abs_error <= self.delta, quadratic, linear)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class LogScaleMSE(nn.Module):
    """
    MSE in log-scale to balance count ranges.

    Computes MSE on log(1 + y) instead of y, reducing bias toward high counts.
    The log transformation compresses large values while preserving relative
    differences, leading to more balanced predictions across count ranges.

    Loss(y, y_pred) = mean((log(1 + y) - log(1 + y_pred))^2)

    Parameters
    ----------
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'
    epsilon : float, default=1e-8
        Small constant to prevent log(0)

    Notes
    -----
    The log(1 + y) transformation (log1p) is used instead of log(y) to handle
    zero counts gracefully.
    """

    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute log-scale MSE.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values (non-negative)
        y_true : torch.Tensor
            True values (non-negative)

        Returns
        -------
        loss : torch.Tensor
            Log-scale MSE
        """
        log_pred = torch.log1p(torch.clamp(y_pred, min=0.0) + self.epsilon)
        log_true = torch.log1p(y_true + self.epsilon)

        squared_error = (log_true - log_pred)**2

        if self.reduction == 'mean':
            return torch.mean(squared_error)
        elif self.reduction == 'sum':
            return torch.sum(squared_error)
        else:
            return squared_error


class QuantileLoss(nn.Module):
    """
    Quantile regression loss (pinball loss).

    Predicts a specific quantile of the target distribution. When quantile=0.5,
    this is equivalent to median regression (L1 loss), which is robust to
    outliers and produces unbiased predictions.

    Loss(y, y_pred) = quantile * max(y - y_pred, 0) +
                     (1 - quantile) * max(y_pred - y, 0)

    Parameters
    ----------
    quantile : float, default=0.5
        Target quantile to predict (0 < quantile < 1)
        0.5 = median (most robust)
        Lower values = conservative (underpredict)
        Higher values = aggressive (overpredict)
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'

    References
    ----------
    Koenker, R., & Bassett Jr, G. (1978). Regression quantiles.
    Econometrica, 46(1), 33-50.
    """

    def __init__(self, quantile: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        assert 0 < quantile < 1, "Quantile must be between 0 and 1"
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y_true : torch.Tensor
            True values

        Returns
        -------
        loss : torch.Tensor
            Quantile loss
        """
        error = y_true - y_pred

        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class NegativeBinomialLoss(nn.Module):
    """
    Negative binomial loss for count data with overdispersion.

    The negative binomial distribution is appropriate for count data where
    variance exceeds the mean (overdispersion), which is common in pathway
    counts. This loss maximizes the negative binomial log-likelihood.

    Var(y) = mu + alpha * mu^2

    where mu is the mean and alpha is the dispersion parameter.

    Parameters
    ----------
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'
    epsilon : float, default=1e-8
        Small constant for numerical stability
    init_alpha : float, default=1.0
        Initial value for dispersion parameter

    Notes
    -----
    The dispersion parameter alpha is learned during training. Higher alpha
    indicates greater overdispersion (variance >> mean).

    References
    ----------
    Hilbe, J. M. (2011). Negative binomial regression.
    Cambridge University Press.
    """

    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-8,
                 init_alpha: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon

        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_alpha)))

    @property
    def alpha(self) -> torch.Tensor:
        """Dispersion parameter (always positive via exp)."""
        return torch.exp(self.log_alpha)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute negative binomial negative log-likelihood.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted mean counts (mu)
        y_true : torch.Tensor
            True counts

        Returns
        -------
        loss : torch.Tensor
            Negative log-likelihood
        """
        mu = torch.clamp(y_pred, min=self.epsilon)
        alpha = self.alpha

        r = 1.0 / (alpha + self.epsilon)

        p = mu / (mu + r)
        p = torch.clamp(p, min=self.epsilon, max=1.0 - self.epsilon)

        nll = (
            torch.lgamma(y_true + 1)
            - torch.lgamma(r)
            - torch.lgamma(y_true - r + 1)
            + y_true * torch.log(p)
            + r * torch.log(1 - p)
        )

        loss = -nll

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of multiple losses.

    Allows mixing different loss functions with specified weights, enabling
    multi-objective optimization. For example:
    - Huber (primary) + log-MSE (scale balance) + quantile (robustness)

    Parameters
    ----------
    losses : dict
        Dictionary mapping loss names to (loss_fn, weight) tuples

    Example
    -------
    >>> combined = CombinedLoss({
    ...     'huber': (HuberLoss(delta=1.0), 0.7),
    ...     'log_mse': (LogScaleMSE(), 0.2),
    ...     'quantile': (QuantileLoss(0.5), 0.1)
    ... })
    """

    def __init__(self, losses: dict):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}

        for name, (loss_fn, weight) in losses.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight

        total_weight = sum(self.weights.values())
        assert abs(total_weight - 1.0) < 1e-6, "Weights must sum to 1.0"

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted combination of losses.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values
        y_true : torch.Tensor
            True values

        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of individual losses
        """
        total_loss = 0.0

        for name, loss_fn in self.losses.items():
            weight = self.weights[name]
            loss_value = loss_fn(y_pred, y_true)
            total_loss += weight * loss_value

        return total_loss


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create loss functions by name.

    Parameters
    ----------
    loss_name : str
        Loss function name: 'mse', 'huber', 'log_mse', 'quantile', 'negbin'
    **kwargs
        Additional arguments passed to loss constructor

    Returns
    -------
    loss_fn : nn.Module
        Configured loss function

    Example
    -------
    >>> loss_fn = get_loss_function('huber', delta=1.5)
    >>> loss_fn = get_loss_function('quantile', quantile=0.5)
    """
    loss_registry = {
        'mse': lambda: nn.MSELoss(**kwargs),
        'huber': lambda: HuberLoss(**kwargs),
        'log_mse': lambda: LogScaleMSE(**kwargs),
        'quantile': lambda: QuantileLoss(**kwargs),
        'negbin': lambda: NegativeBinomialLoss(**kwargs),
    }

    if loss_name not in loss_registry:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            f"Available: {list(loss_registry.keys())}"
        )

    return loss_registry[loss_name]()
