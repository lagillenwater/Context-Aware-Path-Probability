#!/usr/bin/env python
"""
Diagnostic analysis of gamma-hurdle p-value calibration issues.

Tests three hypotheses:
1. Method-of-moments parameter estimation is biased
2. Degree binning creates heterogeneous within-bin distributions
3. High null variance categories show worse calibration

Greene Lab standards:
- No emojis
- PEP 8 compliant
- Comprehensive docstrings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import seaborn as sns

results_dir = Path("results/dwpc_pvalue_validation/dwpc_results")
output_dir = results_dir / "diagnostics"
output_dir.mkdir(exist_ok=True, parents=True)


def load_metapath_data(metapath, scenario='a'):
    """
    Load results for a metapath.

    Args:
        metapath: Metapath abbreviation (e.g., 'CbGpPW')
        scenario: 'a' for Permutation 0, 'b' for Hetionet

    Returns:
        dict: Keys are category names, values are dicts with
              'observed', 'pvalues', etc.
    """
    data = np.load(
        results_dir / f"{metapath}_scenario_{scenario}.npz",
        allow_pickle=True
    )

    results = {}
    for key in data.keys():
        if key.startswith('observed_'):
            cat = key.replace('observed_', '').replace('_', '-')
            pval_key = f"pvalues_{key.replace('observed_', '')}"

            if pval_key in data:
                results[cat] = {
                    'observed': data[key],
                    'pvalues': data[pval_key],
                    'n': len(data[key])
                }

    return results


def analyze_continuous_degrees(metapath):
    """
    Analyze p-values as function of continuous degree values.

    Tests whether binning into Low/Med/High creates artifacts.

    Args:
        metapath: Metapath abbreviation

    Returns:
        DataFrame with columns: source_degree, target_degree, pvalue, category
    """
    print(f"\n{'='*80}")
    print(f"Analyzing continuous degrees for {metapath}")
    print(f"{'='*80}")

    # Load data
    perm0_data = load_metapath_data(metapath, scenario='a')

    # For diagnostic purposes, we need actual degree values
    # These aren't stored in the npz files, so this is a limitation
    # We can only show patterns by category

    records = []
    for cat, vals in perm0_data.items():
        for pval in vals['pvalues']:
            records.append({
                'metapath': metapath,
                'category': cat,
                'pvalue': pval
            })

    df = pd.DataFrame(records)

    # Compute statistics by category
    summary = df.groupby('category').agg({
        'pvalue': ['mean', 'median', 'std', 'count']
    }).round(3)

    print("\nP-value statistics by degree category:")
    print(summary)

    return df


def analyze_null_variance(metapath):
    """
    Analyze relationship between null DWPC variance and calibration.

    Tests whether high-variance categories show worse calibration.

    Args:
        metapath: Metapath abbreviation

    Returns:
        DataFrame with variance metrics and calibration quality
    """
    print(f"\n{'='*80}")
    print(f"Analyzing null variance for {metapath}")
    print(f"{'='*80}")

    # Load Permutation 0 data
    perm0_data = load_metapath_data(metapath, scenario='a')

    records = []
    for cat, vals in perm0_data.items():
        observed = vals['observed']
        pvalues = vals['pvalues']

        # Compute statistics
        mean_obs = observed.mean()
        std_obs = observed.std()
        cv_obs = std_obs / mean_obs if mean_obs > 0 else np.nan

        mean_p = pvalues.mean()
        median_p = np.median(pvalues)
        std_p = pvalues.std()

        # Calibration quality: deviation from 0.50
        calibration_error = abs(mean_p - 0.50)

        records.append({
            'metapath': metapath,
            'category': cat,
            'n': len(observed),
            'mean_obs_dwpc': mean_obs,
            'std_obs_dwpc': std_obs,
            'cv_obs_dwpc': cv_obs,
            'mean_pvalue': mean_p,
            'median_pvalue': median_p,
            'std_pvalue': std_p,
            'calibration_error': calibration_error
        })

    df = pd.DataFrame(records)

    # Compute correlation between CV and calibration error
    if len(df) > 2:
        corr = df[['cv_obs_dwpc', 'calibration_error']].corr().iloc[0, 1]
        print(f"\nCorrelation(CV_observed, calibration_error): {corr:.3f}")

        if not np.isnan(corr):
            if abs(corr) > 0.5:
                print("  → Strong correlation: variance explains calibration!")
            elif abs(corr) > 0.3:
                print("  → Moderate correlation")
            else:
                print("  → Weak correlation")

    print("\nVariance and calibration by category:")
    print(df[['category', 'cv_obs_dwpc', 'mean_pvalue',
              'calibration_error']].to_string(index=False))

    return df


def test_method_of_moments_bias(metapath):
    """
    Test whether method-of-moments produces biased gamma parameters.

    Compares method-of-moments vs MLE parameter estimates.

    Args:
        metapath: Metapath abbreviation

    Returns:
        DataFrame comparing estimation methods
    """
    print(f"\n{'='*80}")
    print(f"Testing method-of-moments bias for {metapath}")
    print(f"{'='*80}")

    perm0_data = load_metapath_data(metapath, scenario='a')

    records = []
    for cat, vals in perm0_data.items():
        observed = vals['observed']
        pvalues = vals['pvalues']

        if len(observed) < 5:
            continue

        # Remove zeros for gamma fitting
        nonzero = observed[observed > 0]
        if len(nonzero) < 3:
            continue

        # Method-of-moments
        mean_obs = nonzero.mean()
        var_obs = nonzero.var(ddof=1)

        if var_obs > 0 and mean_obs > 0:
            beta_mom = var_obs / mean_obs
            alpha_mom = mean_obs / beta_mom
        else:
            alpha_mom = beta_mom = np.nan

        # Maximum likelihood estimation
        try:
            alpha_mle, loc, beta_mle = stats.gamma.fit(nonzero, floc=0)
        except Exception:
            alpha_mle = beta_mle = np.nan

        # Compare
        param_diff = abs(alpha_mom - alpha_mle) if not np.isnan(alpha_mom) else np.nan

        records.append({
            'metapath': metapath,
            'category': cat,
            'n': len(observed),
            'n_nonzero': len(nonzero),
            'lambda': len(nonzero) / len(observed),
            'alpha_mom': alpha_mom,
            'beta_mom': beta_mom,
            'alpha_mle': alpha_mle,
            'beta_mle': beta_mle,
            'alpha_diff': param_diff,
            'mean_pvalue': pvalues.mean()
        })

    df = pd.DataFrame(records)

    if len(df) > 0:
        print("\nMethod-of-moments vs MLE comparison:")
        print(df[['category', 'alpha_mom', 'alpha_mle', 'alpha_diff',
                  'mean_pvalue']].to_string(index=False))

        # Check if large parameter differences correlate with bad calibration
        valid = df[~df['alpha_diff'].isna()]
        if len(valid) > 2:
            corr = valid[['alpha_diff', 'mean_pvalue']].corr().iloc[0, 1]
            print(f"\nCorrelation(alpha_diff, mean_pvalue): {corr:.3f}")

    return df


def create_diagnostic_plots(all_variance_data, all_mom_data):
    """
    Create comprehensive diagnostic visualizations.

    Args:
        all_variance_data: Combined DataFrame from analyze_null_variance
        all_mom_data: Combined DataFrame from test_method_of_moments_bias
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: CV vs Calibration Error
    ax = axes[0, 0]
    for metapath in all_variance_data['metapath'].unique():
        subset = all_variance_data[all_variance_data['metapath'] == metapath]
        ax.scatter(subset['cv_obs_dwpc'], subset['calibration_error'],
                   label=metapath, alpha=0.7, s=50)
    ax.set_xlabel('Coefficient of Variation (DWPC)')
    ax.set_ylabel('Calibration Error (|mean_p - 0.5|)')
    ax.set_title('Variance vs Calibration Quality')
    ax.legend(fontsize=8)
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='Good calibration')
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean p-value by category
    ax = axes[0, 1]
    category_order = ['Low-Low', 'Low-Med', 'Low-High',
                      'Med-Low', 'Med-Med', 'Med-High',
                      'High-Low', 'High-Med', 'High-High']
    cat_means = all_variance_data.groupby('category')['mean_pvalue'].mean()
    cat_means = cat_means.reindex([c for c in category_order if c in cat_means.index])
    cat_means.plot(kind='bar', ax=ax, color='steelblue')
    ax.axhline(0.5, color='red', linestyle='--', label='Expected')
    ax.set_ylabel('Mean P-value')
    ax.set_title('Calibration by Degree Category')
    ax.set_xlabel('Degree Category')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Method-of-moments vs MLE alpha
    ax = axes[0, 2]
    valid = all_mom_data[~all_mom_data['alpha_diff'].isna()]
    if len(valid) > 0:
        ax.scatter(valid['alpha_mom'], valid['alpha_mle'], alpha=0.6, s=50)
        max_val = max(valid['alpha_mom'].max(), valid['alpha_mle'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect agreement')
        ax.set_xlabel('Alpha (Method-of-Moments)')
        ax.set_ylabel('Alpha (MLE)')
        ax.set_title('Parameter Estimation Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: P-value distribution by metapath
    ax = axes[1, 0]
    for metapath in all_variance_data['metapath'].unique()[:3]:  # First 3 for clarity
        subset = all_variance_data[all_variance_data['metapath'] == metapath]
        ax.hist(subset['mean_pvalue'], bins=20, alpha=0.5, label=metapath)
    ax.axvline(0.5, color='red', linestyle='--', label='Expected')
    ax.set_xlabel('Mean P-value')
    ax.set_ylabel('Count of Categories')
    ax.set_title('Distribution of Category Mean P-values')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Sample size vs calibration
    ax = axes[1, 1]
    ax.scatter(all_variance_data['n'], all_variance_data['calibration_error'],
               alpha=0.6, s=50)
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Calibration Error')
    ax.set_title('Sample Size vs Calibration Quality')
    ax.grid(True, alpha=0.3)

    # Plot 6: Lambda (proportion nonzero) vs calibration
    ax = axes[1, 2]
    valid = all_mom_data[~all_mom_data['lambda'].isna()]
    if len(valid) > 0:
        ax.scatter(valid['lambda'], valid['mean_pvalue'], alpha=0.6, s=50)
        ax.set_xlabel('Lambda (Proportion Non-zero)')
        ax.set_ylabel('Mean P-value')
        ax.set_title('Zero-Inflation vs Calibration')
        ax.axhline(0.5, color='red', linestyle='--')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'gamma_hurdle_diagnostics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved diagnostic plots to {output_file}")
    plt.close()


def main():
    """Run all diagnostic analyses."""
    metapaths = ['CbGpPW', 'CtDaG', 'GiGaD', 'CbGpPWpG', 'CtDaGiG', 'CbGpPWpGaD']

    all_variance_data = []
    all_mom_data = []

    for metapath in metapaths:
        print(f"\n\n{'='*80}")
        print(f"METAPATH: {metapath}")
        print(f"{'='*80}")

        # Analysis 1: Continuous degrees (limited by data availability)
        try:
            df_cont = analyze_continuous_degrees(metapath)
        except Exception as e:
            print(f"Error in continuous degree analysis: {e}")

        # Analysis 2: Null variance
        try:
            df_var = analyze_null_variance(metapath)
            all_variance_data.append(df_var)
        except Exception as e:
            print(f"Error in variance analysis: {e}")

        # Analysis 3: Method-of-moments bias
        try:
            df_mom = test_method_of_moments_bias(metapath)
            all_mom_data.append(df_mom)
        except Exception as e:
            print(f"Error in method-of-moments analysis: {e}")

    # Combine results
    if all_variance_data:
        combined_variance = pd.concat(all_variance_data, ignore_index=True)
        combined_variance.to_csv(
            output_dir / 'variance_analysis.csv',
            index=False
        )
        print(f"\n\nSaved variance analysis to {output_dir / 'variance_analysis.csv'}")

    if all_mom_data:
        combined_mom = pd.concat(all_mom_data, ignore_index=True)
        combined_mom.to_csv(
            output_dir / 'method_of_moments_analysis.csv',
            index=False
        )
        print(f"Saved MoM analysis to {output_dir / 'method_of_moments_analysis.csv'}")

    # Create diagnostic plots
    if all_variance_data and all_mom_data:
        create_diagnostic_plots(combined_variance, combined_mom)

    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)

    if all_variance_data:
        print("\n1. Variance vs Calibration:")
        corrs = []
        for metapath in combined_variance['metapath'].unique():
            subset = combined_variance[combined_variance['metapath'] == metapath]
            if len(subset) > 2:
                corr = subset[['cv_obs_dwpc', 'calibration_error']].corr().iloc[0, 1]
                if not np.isnan(corr):
                    corrs.append(corr)
                    print(f"  {metapath}: r = {corr:.3f}")

        if corrs:
            mean_corr = np.mean(corrs)
            print(f"  Mean correlation: {mean_corr:.3f}")

    if all_mom_data:
        print("\n2. Method-of-Moments vs MLE:")
        valid = combined_mom[~combined_mom['alpha_diff'].isna()]
        if len(valid) > 0:
            mean_diff = valid['alpha_diff'].mean()
            max_diff = valid['alpha_diff'].max()
            print(f"  Mean alpha difference: {mean_diff:.3f}")
            print(f"  Max alpha difference: {max_diff:.3f}")

    print("\n3. Degree Category Pattern:")
    cat_summary = combined_variance.groupby('category')['mean_pvalue'].agg(['mean', 'std', 'count'])
    print(cat_summary)


if __name__ == "__main__":
    main()
