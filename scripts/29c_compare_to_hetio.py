"""
Stage 2: Compare HPC-computed stats to het.io ground truth.

This script runs locally after downloading HPC results.
It computes p-values from HPC null stats and compares to het.io p-values.

Input: CSV from 29b_compute_hetio_pair_stats.py (downloaded from HPC)
Output: Comparison statistics and plots

Usage:
    python scripts/29c_compare_to_hetio.py --input_file results/pvalue_validation/hetio_pair_stats_20perms.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


PROJECT_ROOT = Path(__file__).parent.parent


def calculate_pvalue_gamma_hurdle(observed_dwpc, lambda_param, alpha, beta):
    """
    Calculate p-value using gamma-hurdle distribution.

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
        return 1.0

    if lambda_param == 0:
        # Null is all zeros, observed is positive -> very significant
        return 0.0

    # P(DWPC >= x) = lambda * Gamma.sf(x)
    gamma_survival = stats.gamma.sf(observed_dwpc, a=alpha, scale=1.0 / beta)
    return lambda_param * gamma_survival


def main(input_file, output_dir=None):
    """
    Compare HPC results to het.io ground truth.

    Parameters
    ----------
    input_file : str
        Path to CSV from HPC (29b output)
    output_dir : str
        Directory for output files
    """
    input_file = Path(input_file)

    if output_dir is None:
        output_dir = input_file.parent
    else:
        output_dir = Path(output_dir)

    print("=" * 80)
    print("Comparing HPC results to het.io ground truth")
    print("=" * 80)
    print(f"Input file: {input_file}")

    # Load HPC results
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} pairs")

    # Check required columns
    required_cols = [
        "dwpc_hetionet", "dwpc_perm0",
        "null_lambda", "null_alpha", "null_beta"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return

    # Calculate p-values using our null stats
    print("\nCalculating p-values from HPC null distributions...")

    our_pvalues_hetionet = []
    our_pvalues_perm0 = []

    for _, row in df.iterrows():
        # P-value for Hetionet DWPC
        pval_het = calculate_pvalue_gamma_hurdle(
            row["dwpc_hetionet"],
            row["null_lambda"],
            row["null_alpha"],
            row["null_beta"]
        )
        our_pvalues_hetionet.append(pval_het)

        # P-value for perm0 DWPC
        pval_p0 = calculate_pvalue_gamma_hurdle(
            row["dwpc_perm0"],
            row["null_lambda"],
            row["null_alpha"],
            row["null_beta"]
        )
        our_pvalues_perm0.append(pval_p0)

    df["our_pvalue_hetionet"] = our_pvalues_hetionet
    df["our_pvalue_perm0"] = our_pvalues_perm0

    # Compare to het.io if available
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # =========================================================================
    # 1. DWPC Comparison
    # =========================================================================
    print("\n--- DWPC Comparison ---")

    if "hetio_dwpc" in df.columns:
        # Compare Hetionet DWPCs
        valid_mask = df["hetio_dwpc"].notna() & (df["dwpc_hetionet"] > 0)
        if valid_mask.sum() > 0:
            hetio_dwpc = df.loc[valid_mask, "hetio_dwpc"]
            our_dwpc = df.loc[valid_mask, "dwpc_hetionet"]

            corr = np.corrcoef(hetio_dwpc, our_dwpc)[0, 1]
            abs_diff = np.abs(hetio_dwpc - our_dwpc)

            print(f"\nHetionet DWPC (our vs het.io):")
            print(f"  N pairs with both non-zero: {valid_mask.sum()}")
            print(f"  Correlation: {corr:.6f}")
            print(f"  Mean absolute diff: {abs_diff.mean():.6f}")
            print(f"  Max absolute diff: {abs_diff.max():.6f}")

            # Check for exact matches
            rtol = 1e-4
            exact = np.isclose(hetio_dwpc, our_dwpc, rtol=rtol).sum()
            print(f"  Exact matches (rtol={rtol}): {exact}/{valid_mask.sum()} ({100*exact/valid_mask.sum():.1f}%)")
    else:
        print("  (het.io DWPC not available in input file)")

    # =========================================================================
    # 2. P-Value Comparison (Hetionet)
    # =========================================================================
    print("\n--- P-Value Comparison (Hetionet) ---")

    if "hetio_pvalue" in df.columns:
        valid_mask = df["hetio_pvalue"].notna() & df["our_pvalue_hetionet"].notna()
        valid_mask &= (df["hetio_pvalue"] > 0) & (df["our_pvalue_hetionet"] > 0)

        if valid_mask.sum() > 0:
            hetio_p = df.loc[valid_mask, "hetio_pvalue"]
            our_p = df.loc[valid_mask, "our_pvalue_hetionet"]

            # Log-scale correlation (more appropriate for p-values)
            log_hetio = np.log10(hetio_p)
            log_our = np.log10(our_p)
            log_corr = np.corrcoef(log_hetio, log_our)[0, 1]

            abs_diff = np.abs(hetio_p - our_p)
            rel_diff = abs_diff / hetio_p

            print(f"\nOur p-value (Hetionet) vs het.io p-value:")
            print(f"  N valid pairs: {valid_mask.sum()}")
            print(f"  Log10 correlation: {log_corr:.6f}")
            print(f"  Mean absolute diff: {abs_diff.mean():.6f}")
            print(f"  Median absolute diff: {abs_diff.median():.6f}")
            print(f"  Max absolute diff: {abs_diff.max():.6f}")
            print(f"  Mean relative diff: {rel_diff.mean():.4f}")

            # Distribution comparison
            print(f"\n  Het.io p-value stats:")
            print(f"    Mean: {hetio_p.mean():.4f}")
            print(f"    Median: {hetio_p.median():.4f}")
            print(f"  Our p-value stats:")
            print(f"    Mean: {our_p.mean():.4f}")
            print(f"    Median: {our_p.median():.4f}")
    else:
        print("  (het.io p-value not available in input file)")

    # =========================================================================
    # 3. Perm0 vs Hetionet P-Values
    # =========================================================================
    print("\n--- Perm0 vs Hetionet P-Values ---")

    perm0_p = df["our_pvalue_perm0"]
    hetionet_p = df["our_pvalue_hetionet"]

    print(f"\nPerm0 p-value distribution:")
    print(f"  Mean: {perm0_p.mean():.4f}")
    print(f"  Median: {perm0_p.median():.4f}")
    print(f"  Std: {perm0_p.std():.4f}")
    print(f"  Proportion < 0.05: {(perm0_p < 0.05).mean():.3f}")
    print(f"  Proportion < 0.50: {(perm0_p < 0.50).mean():.3f}")

    print(f"\nHetionet p-value distribution (using our null):")
    print(f"  Mean: {hetionet_p.mean():.4f}")
    print(f"  Median: {hetionet_p.median():.4f}")
    print(f"  Std: {hetionet_p.std():.4f}")
    print(f"  Proportion < 0.05: {(hetionet_p < 0.05).mean():.3f}")
    print(f"  Proportion < 0.50: {(hetionet_p < 0.50).mean():.3f}")

    # =========================================================================
    # 4. Null Distribution Comparison
    # =========================================================================
    print("\n--- Null Distribution Comparison ---")

    if "hetio_dgp_n_dwpcs" in df.columns:
        print(f"\nHet.io null stats:")
        print(f"  Mean N total: {df['hetio_dgp_n_dwpcs'].mean():.0f}")
        if "hetio_dgp_n_nonzero" in df.columns:
            print(f"  Mean N nonzero: {df['hetio_dgp_n_nonzero'].mean():.0f}")
        if "hetio_dgp_mean" in df.columns:
            print(f"  Mean nonzero_mean: {df['hetio_dgp_mean'].mean():.6f}")

    print(f"\nOur null stats (from HPC):")
    print(f"  Mean N total: {df['null_n_total'].mean():.0f}")
    print(f"  Mean N nonzero: {df['null_n_nonzero'].mean():.0f}")
    print(f"  Mean lambda: {df['null_lambda'].mean():.4f}")
    print(f"  Mean nonzero_mean: {df['null_nonzero_mean'].mean():.6f}")

    # =========================================================================
    # 5. Sample Output
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAMPLE COMPARISONS (first 15 pairs)")
    print("=" * 80)

    display_cols = ["go_id", "entrez_gene_id", "dwpc_hetionet", "dwpc_perm0"]
    if "hetio_pvalue" in df.columns:
        display_cols.append("hetio_pvalue")
    display_cols.extend(["our_pvalue_hetionet", "our_pvalue_perm0"])

    print(df[display_cols].head(15).to_string(index=False))

    # =========================================================================
    # Save enriched results
    # =========================================================================
    output_file = output_dir / f"comparison_results_{input_file.stem}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved enriched results to: {output_file}")

    # =========================================================================
    # Verdict
    # =========================================================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if "hetio_pvalue" in df.columns:
        valid_mask = df["hetio_pvalue"].notna() & df["our_pvalue_hetionet"].notna()
        valid_mask &= (df["hetio_pvalue"] > 0) & (df["our_pvalue_hetionet"] > 0)

        if valid_mask.sum() > 0:
            hetio_p = df.loc[valid_mask, "hetio_pvalue"]
            our_p = df.loc[valid_mask, "our_pvalue_hetionet"]
            log_corr = np.corrcoef(np.log10(hetio_p), np.log10(our_p))[0, 1]

            if log_corr > 0.95:
                print(f"P-values MATCH het.io (log10 correlation = {log_corr:.4f})")
            elif log_corr > 0.80:
                print(f"P-values PARTIALLY match het.io (log10 correlation = {log_corr:.4f})")
                print("Differences may be due to different number of permutations or sampling.")
            else:
                print(f"P-values DO NOT match het.io (log10 correlation = {log_corr:.4f})")
                print("Investigate null distribution differences.")
    else:
        print("Cannot compare to het.io (p-values not in input file)")
        print("Check perm0 vs Hetionet differences above.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare HPC results to het.io ground truth"
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to CSV from HPC (29b output)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: same as input)"
    )

    args = parser.parse_args()
    main(input_file=args.input_file, output_dir=args.output_dir)
