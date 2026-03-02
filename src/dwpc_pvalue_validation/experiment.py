"""
Experiment module for DWPC P-Value Validation

End-to-end workflow for testing DWPC p-value methodology following
Himmelstein et al. 2023.
"""

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from scipy import stats

from . import (
    config, sampling, dwpc_calculation,
    null_distribution, pvalue_calculation
)
from .utils import setup_logging

logger = setup_logging(__name__)


def run_scenario_a(
    metapath: str,
    n_samples_per_category: int,
    observed_perm: int = 0,
    null_perms: Optional[List[int]] = None,
    damping_exponent: float = 0.5,
    random_state: Optional[int] = None,
    require_connected: bool = False,
    method: str = 'gamma_hurdle'
) -> Dict:
    """
    Run Scenario A: Null test using permutation 0 vs permutations 1-20.

    This is a calibration test where observed and null come from the same
    distribution, so p-values should be uniformly distributed.

    Parameters
    ----------
    metapath : str
        Metapath abbreviation.
    n_samples_per_category : int
        Samples per degree category.
    observed_perm : int
        Permutation to treat as "observed" (default 0).
    null_perms : list of int, optional
        Permutations for null distribution (default 1-20).
    damping_exponent : float
        DWPC damping exponent.
    random_state : int, optional
        Random seed.
    require_connected : bool
        If True, filter to pairs with non-zero observed DWPC (default True).
        This prevents extreme zero-inflation in validation.
    method : str
        P-value calculation method: 'gamma_hurdle' or 'empirical' (default 'gamma_hurdle').

    Returns
    -------
    results : dict
        Experiment results with observed DWPCs, null distributions,
        p-values by category, and gamma-hurdle parameters.
    """
    if null_perms is None:
        null_perms = list(range(
            config.PERMUTATION_START,
            config.PERMUTATION_END + 1
        ))

    logger.info(
        f"Scenario A: {metapath} - "
        f"observed=perm{observed_perm}, null=perms{null_perms[0]}-{null_perms[-1]}"
    )

    # Sample observed paths from permutation 0
    observed_source = f"perm{observed_perm}"
    observed_samples = sampling.sample_metapath_pairs(
        metapath=metapath,
        source=observed_source,
        n_samples_per_category=n_samples_per_category,
        random_state=random_state
    )

    # Calculate observed DWPCs
    observed_dwpcs_all = dwpc_calculation.calculate_dwpc_for_samples(
        samples=observed_samples,
        metapath=metapath,
        source=observed_source,
        damping_exponent=damping_exponent
    )

    # Filter to connected pairs if requested
    if require_connected:
        nonzero_mask = observed_dwpcs_all > 0
        n_original = len(observed_samples)
        n_nonzero = np.sum(nonzero_mask)

        observed_samples = [s for s, keep in zip(observed_samples, nonzero_mask) if keep]
        observed_dwpcs_all = observed_dwpcs_all[nonzero_mask]

        logger.info(
            f"Filtered to connected pairs: kept {n_nonzero}/{n_original} "
            f"({100*n_nonzero/n_original:.1f}%)"
        )

    # Organize observed by EXACT degree (not category)
    observed_by_degree = {}
    degree_to_category = {}
    for sample, dwpc in zip(observed_samples, observed_dwpcs_all):
        degree_key = (sample['source_degree'], sample['target_degree'])
        if degree_key not in observed_by_degree:
            observed_by_degree[degree_key] = []
            degree_to_category[degree_key] = sample['category']
        observed_by_degree[degree_key].append(dwpc)

    # Build null distributions using the SAME sample pairs (grouped by exact degree)
    null_by_degree = null_distribution.build_null_distributions_for_samples(
        samples=observed_samples,
        metapath=metapath,
        perm_indices=null_perms,
        damping_exponent=damping_exponent
    )

    # Calculate p-values for each EXACT degree combination
    pvalues_by_degree = {}
    gamma_hurdle_params_by_degree = {}

    logger.info(f"Using p-value calculation method: {method}")

    for degree_key in observed_by_degree.keys():
        if degree_key not in null_by_degree:
            logger.warning(f"No null distribution for degrees {degree_key}")
            continue

        observed_deg = np.array(observed_by_degree[degree_key])
        null_deg = null_by_degree[degree_key]

        if method == 'empirical':
            # Use empirical percentile method
            pvalues = pvalue_calculation.calculate_empirical_pvalues(
                observed_deg, null_deg
            )
        elif method == 'gamma_hurdle':
            # Fit gamma-hurdle to null
            params = pvalue_calculation.fit_gamma_hurdle(null_deg)
            gamma_hurdle_params_by_degree[degree_key] = params

            # Calculate p-values
            pvalues = pvalue_calculation.calculate_pvalues(observed_deg, params)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gamma_hurdle' or 'empirical'")

        pvalues_by_degree[degree_key] = pvalues

    # Post-hoc: Group p-values by category for analysis output
    pvalues_by_category = {}
    gamma_hurdle_params = {}
    observed_by_category = {}

    for degree_key, pvals in pvalues_by_degree.items():
        category = degree_to_category[degree_key]
        if category not in pvalues_by_category:
            pvalues_by_category[category] = []
            observed_by_category[category] = []
        pvalues_by_category[category].extend(pvals)
        observed_by_category[category].extend(observed_by_degree[degree_key])

        if method == 'gamma_hurdle':
            if category not in gamma_hurdle_params:
                gamma_hurdle_params[category] = []
            gamma_hurdle_params[category].append(gamma_hurdle_params_by_degree[degree_key])

    # Convert lists to arrays
    for category in pvalues_by_category.keys():
        pvalues_by_category[category] = np.array(pvalues_by_category[category])
        observed_by_category[category] = np.array(observed_by_category[category])

    n_total = sum(len(p) for p in pvalues_by_category.values())

    logger.info(
        f"Scenario A complete: {len(pvalues_by_category)} categories, "
        f"{n_total} total p-values"
    )

    return {
        'metapath': metapath,
        'scenario': 'A',
        'observed_source': observed_source,
        'null_perms': null_perms,
        'observed_dwpcs': observed_by_category,
        'null_distributions': null_by_degree,
        'pvalues_by_category': pvalues_by_category,
        'gamma_hurdle_params': gamma_hurdle_params,
        'method': method
    }


def run_scenario_b(
    metapath: str,
    n_samples_per_category: int,
    null_perms: Optional[List[int]] = None,
    damping_exponent: float = 0.5,
    random_state: Optional[int] = None,
    require_connected: bool = False,
    method: str = 'gamma_hurdle'
) -> Dict:
    """
    Run Scenario B: Positive control using true Hetionet vs permutations.

    True biological network should show significant p-values for real
    disease-gene associations.

    Parameters
    ----------
    metapath : str
        Metapath abbreviation.
    n_samples_per_category : int
        Samples per degree category.
    null_perms : list of int, optional
        Permutations for null (default 1-20).
    damping_exponent : float
        DWPC damping exponent.
    random_state : int, optional
        Random seed.
    require_connected : bool
        If True, filter to pairs with non-zero observed DWPC (default True).
    method : str
        P-value calculation method: 'gamma_hurdle' or 'empirical' (default 'gamma_hurdle').

    Returns
    -------
    results : dict
        Experiment results.
    """
    if null_perms is None:
        null_perms = list(range(
            config.PERMUTATION_START,
            config.PERMUTATION_END + 1
        ))

    logger.info(
        f"Scenario B: {metapath} - "
        f"observed=true, null=perms{null_perms[0]}-{null_perms[-1]}"
    )

    # Sample observed paths from true Hetionet
    observed_samples = sampling.sample_metapath_pairs(
        metapath=metapath,
        source="true",
        n_samples_per_category=n_samples_per_category,
        random_state=random_state
    )

    # Calculate observed DWPCs
    observed_dwpcs_all = dwpc_calculation.calculate_dwpc_for_samples(
        samples=observed_samples,
        metapath=metapath,
        source="true",
        damping_exponent=damping_exponent
    )

    # Filter to connected pairs if requested
    if require_connected:
        nonzero_mask = observed_dwpcs_all > 0
        n_original = len(observed_samples)
        n_nonzero = np.sum(nonzero_mask)

        observed_samples = [s for s, keep in zip(observed_samples, nonzero_mask) if keep]
        observed_dwpcs_all = observed_dwpcs_all[nonzero_mask]

        logger.info(
            f"Filtered to connected pairs: kept {n_nonzero}/{n_original} "
            f"({100*n_nonzero/n_original:.1f}%)"
        )

    # Organize by EXACT degree (not category)
    observed_by_degree = {}
    degree_to_category = {}
    for sample, dwpc in zip(observed_samples, observed_dwpcs_all):
        degree_key = (sample['source_degree'], sample['target_degree'])
        if degree_key not in observed_by_degree:
            observed_by_degree[degree_key] = []
            degree_to_category[degree_key] = sample['category']
        observed_by_degree[degree_key].append(dwpc)

    # Build null distributions using the SAME sample pairs (grouped by exact degree)
    null_by_degree = null_distribution.build_null_distributions_for_samples(
        samples=observed_samples,
        metapath=metapath,
        perm_indices=null_perms,
        damping_exponent=damping_exponent
    )

    # Calculate p-values for each EXACT degree combination
    pvalues_by_degree = {}
    gamma_hurdle_params_by_degree = {}

    logger.info(f"Using p-value calculation method: {method}")

    for degree_key in observed_by_degree.keys():
        if degree_key not in null_by_degree:
            logger.warning(f"No null distribution for degrees {degree_key}")
            continue

        observed_deg = np.array(observed_by_degree[degree_key])
        null_deg = null_by_degree[degree_key]

        if method == 'empirical':
            # Use empirical percentile method
            pvalues = pvalue_calculation.calculate_empirical_pvalues(
                observed_deg, null_deg
            )
        elif method == 'gamma_hurdle':
            # Fit gamma-hurdle to null
            params = pvalue_calculation.fit_gamma_hurdle(null_deg)
            gamma_hurdle_params_by_degree[degree_key] = params

            # Calculate p-values
            pvalues = pvalue_calculation.calculate_pvalues(observed_deg, params)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gamma_hurdle' or 'empirical'")

        pvalues_by_degree[degree_key] = pvalues

    # Post-hoc: Group p-values by category for analysis output
    pvalues_by_category = {}
    gamma_hurdle_params = {}
    observed_by_category = {}

    for degree_key, pvals in pvalues_by_degree.items():
        category = degree_to_category[degree_key]
        if category not in pvalues_by_category:
            pvalues_by_category[category] = []
            observed_by_category[category] = []
        pvalues_by_category[category].extend(pvals)
        observed_by_category[category].extend(observed_by_degree[degree_key])

        if method == 'gamma_hurdle':
            if category not in gamma_hurdle_params:
                gamma_hurdle_params[category] = []
            gamma_hurdle_params[category].append(gamma_hurdle_params_by_degree[degree_key])

    # Convert lists to arrays
    for category in pvalues_by_category.keys():
        pvalues_by_category[category] = np.array(pvalues_by_category[category])
        observed_by_category[category] = np.array(observed_by_category[category])

    n_total = sum(len(p) for p in pvalues_by_category.values())

    logger.info(
        f"Scenario B complete: {len(pvalues_by_category)} categories, "
        f"{n_total} total p-values"
    )

    return {
        'metapath': metapath,
        'scenario': 'B',
        'observed_source': 'true',
        'null_perms': null_perms,
        'observed_dwpcs': observed_by_category,
        'null_distributions': null_by_degree,
        'pvalues_by_category': pvalues_by_category,
        'gamma_hurdle_params': gamma_hurdle_params,
        'method': method
    }


def analyze_calibration(pvalues_by_category: Dict[tuple, np.ndarray]) -> Dict:
    """
    Analyze p-value calibration.

    For null test (Scenario A), p-values should be uniformly distributed
    under correct calibration.

    Parameters
    ----------
    pvalues_by_category : dict
        P-values by degree category.

    Returns
    -------
    calibration : dict
        Calibration metrics:
        - n_total: Total p-values
        - n_sig_005: Number with p < 0.05
        - n_sig_001: Number with p < 0.01
        - mean_pvalue: Mean p-value
        - ks_statistic: Kolmogorov-Smirnov test vs uniform
        - ks_pvalue: KS test p-value
    """
    all_pvalues = np.concatenate(list(pvalues_by_category.values()))

    n_total = len(all_pvalues)
    n_sig_005 = np.sum(all_pvalues < 0.05)
    n_sig_001 = np.sum(all_pvalues < 0.01)

    # KS test vs uniform distribution
    ks_stat, ks_pval = stats.kstest(all_pvalues, 'uniform')

    calibration = {
        'n_total': int(n_total),
        'n_sig_005': int(n_sig_005),
        'n_sig_001': int(n_sig_001),
        'proportion_sig_005': float(n_sig_005 / n_total) if n_total > 0 else 0,
        'proportion_sig_001': float(n_sig_001 / n_total) if n_total > 0 else 0,
        'mean_pvalue': float(np.mean(all_pvalues)),
        'median_pvalue': float(np.median(all_pvalues)),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval)
    }

    return calibration


def compare_scenarios(results_a: Dict, results_b: Dict) -> Dict:
    """
    Compare Scenario A (null test) and Scenario B (positive control).

    Parameters
    ----------
    results_a : dict
        Scenario A results.
    results_b : dict
        Scenario B results.

    Returns
    -------
    comparison : dict
        Comparison metrics for both scenarios.
    """
    cal_a = analyze_calibration(results_a['pvalues_by_category'])
    cal_b = analyze_calibration(results_b['pvalues_by_category'])

    comparison = {
        'scenario_a': cal_a,
        'scenario_b': cal_b,
        'difference_mean_pvalue': cal_a['mean_pvalue'] - cal_b['mean_pvalue'],
        'difference_sig_005': cal_a['n_sig_005'] - cal_b['n_sig_005']
    }

    return comparison


def save_results(results: Dict, output_path: Path):
    """
    Save experiment results.

    Parameters
    ----------
    results : dict
        Experiment results.
    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'metapath': results['metapath'],
        'scenario': results['scenario'],
        'observed_source': results['observed_source']
    }

    # Save p-values by category
    for cat, pvals in results['pvalues_by_category'].items():
        key = f"pvalues_{cat[0]}_{cat[1]}"
        save_dict[key] = pvals

    # Save observed DWPCs by category
    for cat, dwpcs in results['observed_dwpcs'].items():
        key = f"observed_{cat[0]}_{cat[1]}"
        save_dict[key] = np.array(dwpcs)

    np.savez_compressed(output_path, **save_dict)

    logger.info(f"Saved experiment results to {output_path}")


def load_results(input_path: Path) -> Dict:
    """
    Load experiment results.

    Parameters
    ----------
    input_path : Path
        Input file path.

    Returns
    -------
    results : dict
        Loaded experiment results.
    """
    data = np.load(input_path, allow_pickle=True)

    results = {
        'metapath': str(data['metapath']),
        'scenario': str(data['scenario']),
        'observed_source': str(data['observed_source']),
        'pvalues_by_category': {},
        'observed_dwpcs': {}
    }

    for key in data.keys():
        if key.startswith('pvalues_'):
            parts = key.replace('pvalues_', '').split('_', 1)
            if len(parts) == 2:
                cat = tuple(parts)
                results['pvalues_by_category'][cat] = data[key]
        elif key.startswith('observed_'):
            parts = key.replace('observed_', '').split('_', 1)
            if len(parts) == 2:
                cat = tuple(parts)
                results['observed_dwpcs'][cat] = data[key]

    logger.info(f"Loaded experiment results from {input_path}")

    return results
