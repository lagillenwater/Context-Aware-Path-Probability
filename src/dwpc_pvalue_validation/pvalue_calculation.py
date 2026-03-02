"""
P-value calculation module for DWPC P-Value Validation

Calculates p-values using gamma-hurdle distribution following
Himmelstein et al. 2023 (GigaScience, giad047).
"""

from typing import Dict
import numpy as np
from scipy import stats

from .utils import setup_logging

logger = setup_logging(__name__)


def fit_gamma_hurdle(
    dwpcs: np.ndarray,
    use_bessel_correction: bool = True
) -> Dict[str, float]:
    """
    Fit gamma-hurdle distribution to null DWPC values.

    Uses method of moments with exact formulas from Himmelstein et al. 2023,
    page 8, equations for gamma-hurdle parameters.

    Parameters
    ----------
    dwpcs : np.ndarray
        Null DWPC values (from permuted networks).
    use_bessel_correction : bool
        Whether to use Bessel's correction (n-1 instead of n).
        Default True, following the paper.

    Returns
    -------
    params : dict
        Gamma-hurdle parameters:
        - lambda: Hurdle probability (proportion of zeros)
        - alpha: Gamma shape parameter
        - beta: Gamma rate parameter
    """
    total_samples = len(dwpcs)
    nonzero_dwpcs = dwpcs[dwpcs > 0]
    n_nonzero = len(nonzero_dwpcs)

    # Lambda: proportion of non-zero values (note: paper defines as P(X>0))
    lambda_hat = n_nonzero / total_samples if total_samples > 0 else 0

    # Fit gamma to non-zero values
    if n_nonzero == 0:
        # All zeros - degenerate case
        alpha_hat = 1.0
        beta_hat = 1.0
        logger.warning("All DWPC values are zero, using default gamma params")
    elif n_nonzero == 1:
        # Only one non-zero value - use simple estimate
        alpha_hat = 1.0
        beta_hat = 1.0 / nonzero_dwpcs[0]
        logger.warning("Only one non-zero DWPC, using simple gamma estimate")
    else:
        # Method of moments with Bessel's correction
        # From paper page 8:
        # alpha_hat = (n-1) * sum(x_i) / [n * sum(x_i^2) - (sum(x_i))^2]
        # beta_hat = (n-1)/n * [n * sum(x_i) / [n * sum(x_i^2) - (sum(x_i))^2]]

        sum_x = np.sum(nonzero_dwpcs)
        sum_x2 = np.sum(nonzero_dwpcs ** 2)

        denominator = n_nonzero * sum_x2 - sum_x ** 2

        if denominator <= 0:
            # Numerical issue - fall back to simple estimates
            mean_x = np.mean(nonzero_dwpcs)
            var_x = np.var(nonzero_dwpcs)
            if var_x > 0:
                alpha_hat = mean_x ** 2 / var_x
                beta_hat = mean_x / var_x
            else:
                alpha_hat = 1.0
                beta_hat = 1.0 / mean_x if mean_x > 0 else 1.0
            logger.warning(
                "Numerical issue in gamma fitting, using fallback estimates"
            )
        else:
            if use_bessel_correction:
                correction_factor = n_nonzero - 1
            else:
                correction_factor = n_nonzero

            alpha_hat = correction_factor * sum_x / denominator
            beta_hat = (correction_factor / n_nonzero) * (
                n_nonzero * sum_x / denominator
            )

    logger.debug(
        f"Gamma-hurdle params: lambda={lambda_hat:.4f}, "
        f"alpha={alpha_hat:.4f}, beta={beta_hat:.4f}"
    )

    return {
        'lambda': lambda_hat,
        'alpha': alpha_hat,
        'beta': beta_hat
    }


def calculate_pvalues(
    observed_dwpcs: np.ndarray,
    gamma_hurdle_params: Dict[str, float]
) -> np.ndarray:
    """
    Calculate p-values for observed DWPCs using gamma-hurdle distribution.

    P-value represents P(DWPC >= observed | null distribution).

    Parameters
    ----------
    observed_dwpcs : np.ndarray
        Observed DWPC values.
    gamma_hurdle_params : dict
        Gamma-hurdle parameters from fit_gamma_hurdle.

    Returns
    -------
    pvalues : np.ndarray
        P-values for each observed DWPC.
    """
    lambda_param = gamma_hurdle_params['lambda']
    alpha_param = gamma_hurdle_params['alpha']
    beta_param = gamma_hurdle_params['beta']

    pvalues = np.zeros(len(observed_dwpcs))

    for idx, dwpc in enumerate(observed_dwpcs):
        if dwpc == 0:
            # For zero DWPC, p-value = P(DWPC >= 0) = 1.0 (all values >= 0)
            pvalues[idx] = 1.0
        else:
            # P-value = P(DWPC >= observed)
            # In gamma-hurdle model:
            #   P(DWPC = 0) = 1 - lambda
            #   P(DWPC > 0) = lambda
            #   P(DWPC = x | x > 0) ~ Gamma(alpha, beta)
            #
            # Therefore:
            #   P(DWPC >= x) for x > 0:
            #   = P(DWPC > 0) * P(DWPC >= x | DWPC > 0)
            #   = lambda * survival_function(x)

            gamma_survival = stats.gamma.sf(
                dwpc,
                a=alpha_param,
                scale=1.0/beta_param
            )

            pvalues[idx] = lambda_param * gamma_survival

    logger.debug(
        f"Calculated {len(pvalues)} p-values: "
        f"min={np.min(pvalues):.6f}, max={np.max(pvalues):.6f}, "
        f"mean={np.mean(pvalues):.6f}"
    )

    return pvalues


def calculate_empirical_pvalue(
    observed: float,
    null_values: np.ndarray
) -> float:
    """
    Calculate empirical percentile p-value.

    P-value is calculated as the proportion of null values that are greater
    than or equal to the observed value.

    Parameters
    ----------
    observed : float
        Observed DWPC value.
    null_values : np.ndarray
        Null DWPC values from permutations.

    Returns
    -------
    pvalue : float
        Empirical p-value.
    """
    n_null = len(null_values)
    if n_null == 0:
        logger.warning("No null values provided, returning p-value = 1.0")
        return 1.0

    n_extreme = np.sum(null_values >= observed)
    pvalue = n_extreme / n_null

    return pvalue


def calculate_empirical_pvalues(
    observed_dwpcs: np.ndarray,
    null_dwpcs: np.ndarray
) -> np.ndarray:
    """
    Calculate empirical percentile p-values for multiple observations.

    Parameters
    ----------
    observed_dwpcs : np.ndarray
        Observed DWPC values.
    null_dwpcs : np.ndarray
        Null DWPC values from permutations.

    Returns
    -------
    pvalues : np.ndarray
        Empirical p-values for each observed DWPC.
    """
    pvalues = np.array([
        calculate_empirical_pvalue(obs, null_dwpcs)
        for obs in observed_dwpcs
    ])

    logger.debug(
        f"Calculated {len(pvalues)} empirical p-values: "
        f"min={np.min(pvalues):.6f}, max={np.max(pvalues):.6f}, "
        f"mean={np.mean(pvalues):.6f}"
    )

    return pvalues


def calculate_empirical_pvalues_stratified(
    observed_dwpcs: np.ndarray,
    null_dwpcs_by_category: Dict[tuple, np.ndarray],
    categories: np.ndarray
) -> np.ndarray:
    """
    Calculate degree-stratified empirical p-values.

    Each observation gets a p-value based on the empirical null distribution
    for its degree category.

    Parameters
    ----------
    observed_dwpcs : np.ndarray
        Observed DWPC values.
    null_dwpcs_by_category : dict
        Null DWPC distributions keyed by category tuple
        (e.g., ('Low', 'High')).
    categories : np.ndarray
        Category for each observed DWPC (same length as observed_dwpcs).

    Returns
    -------
    pvalues : np.ndarray
        Degree-stratified empirical p-values.
    """
    pvalues = np.zeros(len(observed_dwpcs))

    for category, null_dwpcs in null_dwpcs_by_category.items():
        # Get observations in this category
        mask = categories == category

        if not np.any(mask):
            continue

        # Calculate empirical p-values for this category
        observed_in_category = observed_dwpcs[mask]
        pvalues_in_category = calculate_empirical_pvalues(
            observed_in_category, null_dwpcs
        )

        # Store p-values
        pvalues[mask] = pvalues_in_category

    logger.info(
        f"Calculated stratified empirical p-values for {len(categories)} "
        f"observations across {len(null_dwpcs_by_category)} categories"
    )

    return pvalues


def calculate_pvalues_stratified(
    observed_dwpcs: np.ndarray,
    null_dwpcs_by_category: Dict[tuple, np.ndarray],
    categories: np.ndarray
) -> np.ndarray:
    """
    Calculate degree-stratified p-values using gamma-hurdle.

    Each observation gets a p-value based on the null distribution for its
    degree category.

    Parameters
    ----------
    observed_dwpcs : np.ndarray
        Observed DWPC values.
    null_dwpcs_by_category : dict
        Null DWPC distributions keyed by category tuple
        (e.g., ('Low', 'High')).
    categories : np.ndarray
        Category for each observed DWPC (same length as observed_dwpcs).

    Returns
    -------
    pvalues : np.ndarray
        Degree-stratified p-values.
    """
    pvalues = np.zeros(len(observed_dwpcs))

    for category, null_dwpcs in null_dwpcs_by_category.items():
        # Get observations in this category
        mask = categories == category

        if not np.any(mask):
            continue

        # Fit gamma-hurdle to this category's null
        params = fit_gamma_hurdle(null_dwpcs)

        # Calculate p-values for observations in this category
        observed_in_category = observed_dwpcs[mask]
        pvalues_in_category = calculate_pvalues(observed_in_category, params)

        # Store p-values
        pvalues[mask] = pvalues_in_category

    logger.info(
        f"Calculated stratified p-values for {len(categories)} observations "
        f"across {len(null_dwpcs_by_category)} categories"
    )

    return pvalues
