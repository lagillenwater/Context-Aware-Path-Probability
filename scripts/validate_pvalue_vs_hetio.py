"""
P-Value Validation Script: Compare our gamma-hurdle p-values to het.io ground truth.

This script validates our p-value calculation by comparing with het.io's
p-values from the Multi-DWPC repository output.

The Multi-DWPC CSV contains:
- p_value: het.io's calculated p-value
- dwpc: observed DWPC value
- dgp_n_dwpcs: total null samples (N)
- dgp_n_nonzero_dwpcs: non-zero null samples (n)
- dgp_nonzero_mean: mean of non-zero null DWPCs
- dgp_nonzero_sd: std of non-zero null DWPCs

We reconstruct gamma-hurdle parameters from the dgp statistics and compare
our p-value calculation to het.io's.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Paths
MULTI_DWPC_ROOT = Path(
    "/Users/lucas/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver"
    "/Repositories/Multi-DWPC/Multi-DWPC"
)
HETIO_OUTPUT = (
    MULTI_DWPC_ROOT /
    "output/dwpc_com/res_hetio_bp_go_2016_filt_com_go_w_g_50_250_add_1_25_pct_w_neoj4_ids.csv"
)


def get_metapath_length(metapath):
    """
    Get the length of a metapath (number of edges).

    Length is determined by counting edge abbreviations (lowercase letters
    and special characters like < and >).
    """
    # Simple heuristic: count transitions between node types
    # Each edge is represented by lowercase letters or symbols
    length = 0
    in_edge = False
    for char in metapath:
        if char.islower() or char in '<>':
            if not in_edge:
                length += 1
                in_edge = True
        else:
            in_edge = False
    return length


def load_hetio_data(n_samples=None, min_path_length=2):
    """
    Load het.io results with p-values.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to load. If None, load all.
    min_path_length : int
        Minimum metapath length to include (default 2).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with het.io p-values and null distribution statistics.
    """
    df = pd.read_csv(HETIO_OUTPUT)

    print(f"Loaded {len(df)} records from het.io output")
    print(f"Unique metapaths: {df['metapath_abbreviation'].nunique()}")

    # Filter by metapath length
    df['metapath_length'] = df['metapath_abbreviation'].apply(get_metapath_length)
    df = df[df['metapath_length'] >= min_path_length]
    print(f"Filtered to {len(df)} records with path length >= {min_path_length}")
    print(f"Unique metapaths after filter: {df['metapath_abbreviation'].nunique()}")

    if n_samples is not None and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
        print(f"Sampled {n_samples} records")

    return df


def reconstruct_gamma_params_method_of_moments(mean, sd, n):
    """
    Reconstruct gamma parameters from mean and std using method of moments.

    Parameters
    ----------
    mean : float
        Mean of non-zero DWPCs (dgp_nonzero_mean)
    sd : float
        Standard deviation of non-zero DWPCs (dgp_nonzero_sd)
    n : int
        Number of non-zero samples (dgp_n_nonzero_dwpcs)

    Returns
    -------
    alpha, beta : float
        Gamma shape and rate parameters
    """
    if sd <= 0 or np.isnan(sd) or mean <= 0 or np.isnan(mean):
        # Degenerate case - all values are the same
        return 1.0, 1.0 / mean if mean > 0 else 1.0

    # Standard method of moments for gamma:
    # mean = alpha / beta
    # var = alpha / beta^2
    # Therefore:
    # alpha = mean^2 / var
    # beta = mean / var

    var = sd ** 2

    alpha = (mean ** 2) / var
    beta = mean / var

    return alpha, beta


def calculate_pvalue_gamma_hurdle(observed_dwpc, lambda_param, alpha, beta):
    """
    Calculate p-value using gamma-hurdle distribution.

    P(DWPC >= observed) for the gamma-hurdle model.

    Parameters
    ----------
    observed_dwpc : float
        Observed DWPC value
    lambda_param : float
        Proportion of non-zero values (P(X > 0))
    alpha : float
        Gamma shape parameter
    beta : float
        Gamma rate parameter

    Returns
    -------
    pvalue : float
        Right-tail p-value
    """
    if observed_dwpc == 0:
        # P(DWPC >= 0) = 1.0 (all values are >= 0)
        return 1.0

    # P(DWPC >= x) for x > 0:
    # = P(DWPC > 0) * P(DWPC >= x | DWPC > 0)
    # = lambda * Gamma.sf(x)

    # scipy.stats.gamma uses shape a and scale (1/beta)
    gamma_survival = stats.gamma.sf(observed_dwpc, a=alpha, scale=1.0/beta)

    return lambda_param * gamma_survival


def calculate_our_pvalues(df):
    """
    Calculate p-values using our gamma-hurdle implementation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with het.io statistics

    Returns
    -------
    pvalues : np.ndarray
        Our calculated p-values
    """
    pvalues = []

    for _, row in df.iterrows():
        observed_dwpc = row['dwpc']
        n_total = row['dgp_n_dwpcs']
        n_nonzero = row['dgp_n_nonzero_dwpcs']
        dgp_mean = row['dgp_nonzero_mean']
        dgp_sd = row['dgp_nonzero_sd']

        # Handle edge cases
        if n_total == 0:
            pvalues.append(1.0)
            continue

        # Lambda = P(X > 0)
        lambda_param = n_nonzero / n_total if n_total > 0 else 0

        if n_nonzero == 0 or pd.isna(dgp_mean) or pd.isna(dgp_sd):
            # No non-zero values in null - all zeros
            if observed_dwpc == 0:
                pvalues.append(1.0)
            else:
                # Observed > 0 but null is all zeros -> very significant
                pvalues.append(0.0)
            continue

        # Reconstruct gamma parameters
        alpha, beta = reconstruct_gamma_params_method_of_moments(
            dgp_mean, dgp_sd, n_nonzero
        )

        # Calculate p-value
        pval = calculate_pvalue_gamma_hurdle(
            observed_dwpc, lambda_param, alpha, beta
        )
        pvalues.append(pval)

    return np.array(pvalues)


def analyze_pvalue_calibration(hetio_pvals, our_pvals, df):
    """
    Analyze calibration of our p-values vs het.io's.

    Parameters
    ----------
    hetio_pvals : np.ndarray
        Het.io p-values
    our_pvals : np.ndarray
        Our calculated p-values
    df : pd.DataFrame
        Original DataFrame for context
    """
    print("\n" + "=" * 70)
    print("P-VALUE VALIDATION RESULTS")
    print("=" * 70)

    # Basic statistics
    print(f"\nNumber of pairs compared: {len(hetio_pvals)}")

    # Filter to valid comparisons (both non-NaN)
    valid_mask = ~(np.isnan(hetio_pvals) | np.isnan(our_pvals))
    n_valid = valid_mask.sum()
    print(f"Valid comparisons: {n_valid}")

    if n_valid == 0:
        print("ERROR: No valid comparisons!")
        return

    hetio_valid = hetio_pvals[valid_mask]
    our_valid = our_pvals[valid_mask]
    df_valid = df[valid_mask].copy()

    # Summary statistics
    print(f"\nHet.io p-values:")
    print(f"  Mean:   {np.mean(hetio_valid):.6f}")
    print(f"  Median: {np.median(hetio_valid):.6f}")
    print(f"  Min:    {np.min(hetio_valid):.6f}")
    print(f"  Max:    {np.max(hetio_valid):.6f}")

    print(f"\nOur p-values:")
    print(f"  Mean:   {np.mean(our_valid):.6f}")
    print(f"  Median: {np.median(our_valid):.6f}")
    print(f"  Min:    {np.min(our_valid):.6f}")
    print(f"  Max:    {np.max(our_valid):.6f}")

    # Differences
    abs_diff = np.abs(hetio_valid - our_valid)
    rel_diff = np.where(
        hetio_valid > 0,
        abs_diff / hetio_valid,
        np.where(our_valid > 0, 1.0, 0.0)
    )

    print(f"\nAbsolute Difference:")
    print(f"  Mean: {np.mean(abs_diff):.6f}")
    print(f"  Max:  {np.max(abs_diff):.6f}")
    print(f"  Median: {np.median(abs_diff):.6f}")

    # Correlation
    # Avoid log of zero
    epsilon = 1e-300
    hetio_log = np.log10(np.maximum(hetio_valid, epsilon))
    our_log = np.log10(np.maximum(our_valid, epsilon))

    # Filter out -inf values for correlation
    finite_mask = np.isfinite(hetio_log) & np.isfinite(our_log)
    if finite_mask.sum() > 1:
        correlation = np.corrcoef(
            hetio_log[finite_mask],
            our_log[finite_mask]
        )[0, 1]
        print(f"\nLog10 p-value correlation: {correlation:.6f}")

    # Exact matches (within tolerance)
    rtol = 1e-4
    exact_matches = np.sum(np.isclose(hetio_valid, our_valid, rtol=rtol))
    print(f"\nExact matches (rtol={rtol}): {exact_matches}/{n_valid} "
          f"({100*exact_matches/n_valid:.1f}%)")

    # Close matches (within 1% relative error)
    close_matches = np.sum(rel_diff < 0.01)
    print(f"Close matches (<1% rel error): {close_matches}/{n_valid} "
          f"({100*close_matches/n_valid:.1f}%)")

    # Analyze by metapath length
    print("\n" + "-" * 70)
    print("Analysis by metapath:")
    print("-" * 70)

    df_valid['our_pval'] = our_valid
    df_valid['abs_diff'] = abs_diff

    metapath_stats = df_valid.groupby('metapath_abbreviation').agg({
        'p_value': 'mean',
        'our_pval': 'mean',
        'abs_diff': ['mean', 'max', 'count']
    }).round(6)
    metapath_stats.columns = [
        'het_pval_mean', 'our_pval_mean',
        'abs_diff_mean', 'abs_diff_max', 'count'
    ]

    print(metapath_stats.head(20).to_string())

    # Show examples of large discrepancies
    print("\n" + "-" * 70)
    print("Largest discrepancies:")
    print("-" * 70)

    df_valid['rel_diff'] = rel_diff
    worst = df_valid.nlargest(10, 'abs_diff')

    display_cols = [
        'metapath_abbreviation', 'path_count', 'dwpc',
        'p_value', 'our_pval', 'abs_diff',
        'dgp_n_dwpcs', 'dgp_n_nonzero_dwpcs'
    ]
    print(worst[display_cols].to_string(index=False))

    # Verdict
    print("\n" + "=" * 70)
    if exact_matches >= n_valid * 0.95:
        print("VALIDATION PASSED: P-values match het.io values")
    elif close_matches >= n_valid * 0.90:
        print("VALIDATION MOSTLY PASSED: Most p-values within 1% of het.io")
    else:
        print("VALIDATION FAILED: Significant p-value discrepancies detected")
        print("Investigating sources of calibration error...")

        # Analyze patterns in discrepancies
        analyze_discrepancy_patterns(df_valid, hetio_valid, our_valid)
    print("=" * 70)


def analyze_discrepancy_patterns(df, hetio_pvals, our_pvals):
    """Analyze patterns in p-value discrepancies."""
    abs_diff = np.abs(hetio_pvals - our_pvals)
    large_diff_mask = abs_diff > 0.01

    if large_diff_mask.sum() == 0:
        print("No large discrepancies to analyze")
        return

    large_diff_df = df[large_diff_mask].copy()
    large_diff_df['het_pval'] = hetio_pvals[large_diff_mask]
    large_diff_df['our_pval'] = our_pvals[large_diff_mask]
    large_diff_df['abs_diff'] = abs_diff[large_diff_mask]

    print(f"\nAnalyzing {len(large_diff_df)} pairs with >1% discrepancy:")

    # Check if discrepancies correlate with sample size
    if 'dgp_n_nonzero_dwpcs' in large_diff_df.columns:
        n_nonzero = large_diff_df['dgp_n_nonzero_dwpcs']
        if len(n_nonzero) > 1 and n_nonzero.std() > 0:
            corr = np.corrcoef(n_nonzero, large_diff_df['abs_diff'])[0, 1]
            print(f"  Correlation with n_nonzero: {corr:.4f}")

    # Check relationship with observed DWPC
    if 'dwpc' in large_diff_df.columns:
        dwpc = large_diff_df['dwpc']
        if len(dwpc) > 1 and dwpc.std() > 0:
            corr = np.corrcoef(dwpc, large_diff_df['abs_diff'])[0, 1]
            print(f"  Correlation with observed DWPC: {corr:.4f}")

    # Check relationship with lambda
    if 'dgp_n_dwpcs' in large_diff_df.columns:
        lambda_vals = (
            large_diff_df['dgp_n_nonzero_dwpcs'] /
            large_diff_df['dgp_n_dwpcs']
        )
        if len(lambda_vals) > 1 and lambda_vals.std() > 0:
            corr = np.corrcoef(lambda_vals, large_diff_df['abs_diff'])[0, 1]
            print(f"  Correlation with lambda: {corr:.4f}")


def main():
    print("=" * 70)
    print("P-Value Validation: Comparing our gamma-hurdle to het.io")
    print("=" * 70)

    # Load het.io data
    print("\nLoading het.io data...")
    df = load_hetio_data(n_samples=1000)

    # Calculate our p-values
    print("\nCalculating our p-values...")
    our_pvals = calculate_our_pvalues(df)

    # Get het.io p-values
    hetio_pvals = df['p_value'].values

    # Analyze calibration
    analyze_pvalue_calibration(hetio_pvals, our_pvals, df)

    return df, hetio_pvals, our_pvals


if __name__ == "__main__":
    main()
