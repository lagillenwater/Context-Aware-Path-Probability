"""
Compare our computed p-values to het.io ground truth.

This script analyzes the output from 30b_compute_dwpc_pvalues.py and
compares our DWPC and p-values to het.io's values.

Key validations:
1. DWPC correlation (our vs het.io) - should be ~1.0
2. P-value correlation (our vs het.io) - expected to be correlated but not identical
3. Perm0 sanity check - should have non-zero DWPCs for length 2+ paths

Usage:
    python scripts/30c_compare_pvalues.py --input_file results/pvalue_validation/dwpc_pvalue_comparison_20perms.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def compute_degrees_from_hetmat(df):
    """
    Compute node degrees from hetmat if degree columns are missing.

    Uses bp_idx and gene_idx to look up degrees from the GpBP edge file.
    """
    try:
        import scipy.sparse as sp

        # Load the GpBP sparse matrix directly
        # GpBP: Gene (rows) -> BiologicalProcess (cols)
        edge_file = DATA_DIR / "edges" / "GpBP.sparse.npz"
        if not edge_file.exists():
            print(f"  Edge file not found: {edge_file}")
            return None, None

        adj = sp.load_npz(edge_file)
        print(f"  Loaded edge matrix: {adj.shape}")

        # Gene degrees (how many BPs each gene participates in) = row sums
        # BP degrees (how many genes participate in each BP) = column sums
        gene_degrees = np.array(adj.sum(axis=1)).flatten()
        bp_degrees = np.array(adj.sum(axis=0)).flatten()

        print(f"  Gene degrees range: {gene_degrees.min()}-{gene_degrees.max()}")
        print(f"  BP degrees range: {bp_degrees.min()}-{bp_degrees.max()}")

        source_degrees = []
        target_degrees = []

        for _, row in df.iterrows():
            bp_idx = int(row["bp_idx"])
            gene_idx = int(row["gene_idx"])
            # Source = BP, Target = Gene (since paths are BP -> Gene)
            source_degrees.append(bp_degrees[bp_idx] if bp_idx < len(bp_degrees) else np.nan)
            target_degrees.append(gene_degrees[gene_idx] if gene_idx < len(gene_degrees) else np.nan)

        return np.array(source_degrees), np.array(target_degrees)
    except Exception as e:
        print(f"  Warning: Could not compute degrees from hetmat: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main(input_file, output_dir=None):
    """
    Compare our p-values to het.io ground truth.

    Parameters
    ----------
    input_file : str
        Path to CSV from 30b (output)
    output_dir : str
        Directory for output files and plots
    """
    input_file = Path(input_file)

    if output_dir is None:
        output_dir = input_file.parent
    else:
        output_dir = Path(output_dir)

    print("=" * 70)
    print("Comparing p-values to het.io ground truth")
    print("=" * 70)
    print(f"Input file: {input_file}")

    # Load results
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} pairs")

    # Normalize column names for het.io data
    # 30a saves as "api_p_value", 30b may save as "hetio_api_pvalue"
    if "api_p_value" in df.columns and "hetio_pvalue" not in df.columns:
        df["hetio_pvalue"] = df["api_p_value"]
        print("  Renamed api_p_value -> hetio_pvalue")
    elif "hetio_api_pvalue" in df.columns and "hetio_pvalue" not in df.columns:
        df["hetio_pvalue"] = df["hetio_api_pvalue"]
        print("  Renamed hetio_api_pvalue -> hetio_pvalue")

    # 30a saves actual DWPC as "actual_pdp"
    if "actual_pdp" in df.columns and "hetio_pdp" not in df.columns:
        df["hetio_pdp"] = df["actual_pdp"]
        print("  Renamed actual_pdp -> hetio_pdp")

    # Compute degrees from hetmat if not in file
    if "source_degree" not in df.columns or "target_degree" not in df.columns:
        print("\nDegree columns not found, computing from hetmat...")
        src_deg, tgt_deg = compute_degrees_from_hetmat(df)
        if src_deg is not None:
            df["source_degree"] = src_deg
            df["target_degree"] = tgt_deg
            print(f"  Added degree columns: source_degree, target_degree")

    # =========================================================================
    # 1. DWPC Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. DWPC VALIDATION")
    print("=" * 70)

    if "hetio_pdp" in df.columns:
        valid = df["hetio_pdp"].notna() & (df["our_dwpc_true"] > 0) & (df["hetio_pdp"] > 0)
        n_valid = valid.sum()

        if n_valid > 2:
            our_dwpc = df.loc[valid, "our_dwpc_true"]
            hetio_dwpc = df.loc[valid, "hetio_pdp"]

            corr = np.corrcoef(our_dwpc, hetio_dwpc)[0, 1]
            abs_diff = np.abs(our_dwpc - hetio_dwpc)
            rel_diff = abs_diff / hetio_dwpc

            print(f"\nDWPC Comparison (our vs het.io actual PDP):")
            print(f"  Valid pairs: {n_valid}")
            print(f"  Correlation: {corr:.6f}")
            print(f"  Mean absolute diff: {abs_diff.mean():.6f}")
            print(f"  Max absolute diff: {abs_diff.max():.6f}")
            print(f"  Mean relative diff: {rel_diff.mean():.2%}")

            # Check for exact matches
            exact = np.isclose(our_dwpc, hetio_dwpc, rtol=1e-4).sum()
            print(f"  Exact matches (rtol=1e-4): {exact}/{n_valid} ({100*exact/n_valid:.1f}%)")

            if corr > 0.999:
                print("\n  VERDICT: DWPC VALIDATED (correlation > 0.999)")
            elif corr > 0.99:
                print("\n  VERDICT: DWPC mostly matches (correlation > 0.99)")
            else:
                print("\n  VERDICT: DWPC MISMATCH - investigate!")
        else:
            print("  Not enough valid pairs with het.io PDP")
    else:
        print("  No hetio_pdp column - cannot validate DWPC")
        print("  Run 30a_get_hetio_pdp.py first to get actual PDP values")

    # =========================================================================
    # 2. Perm0 Sanity Check
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. PERM0 SANITY CHECK")
    print("=" * 70)

    perm0_valid = df["our_dwpc_perm0"].notna()
    n_perm0 = perm0_valid.sum()

    if n_perm0 > 0:
        perm0_dwpc = df.loc[perm0_valid, "our_dwpc_perm0"]
        n_nonzero = (perm0_dwpc > 0).sum()

        print(f"\nPerm0 DWPC:")
        print(f"  Total pairs: {n_perm0}")
        print(f"  Non-zero DWPC: {n_nonzero} ({100*n_nonzero/n_perm0:.1f}%)")
        print(f"  Mean (non-zero): {perm0_dwpc[perm0_dwpc > 0].mean():.6f}" if n_nonzero > 0 else "  Mean: N/A")

        # Compare to true network
        true_dwpc = df.loc[perm0_valid, "our_dwpc_true"]
        true_nonzero = (true_dwpc > 0).sum()
        print(f"\nTrue network DWPC:")
        print(f"  Non-zero DWPC: {true_nonzero} ({100*true_nonzero/n_perm0:.1f}%)")
        print(f"  Mean (non-zero): {true_dwpc[true_dwpc > 0].mean():.6f}" if true_nonzero > 0 else "  Mean: N/A")

        # Check by metapath length
        if "metapath_length" in df.columns:
            print("\nBy metapath length:")
            for length in sorted(df["metapath_length"].unique()):
                mask = perm0_valid & (df["metapath_length"] == length)
                if mask.sum() > 0:
                    p0_nonzero = (df.loc[mask, "our_dwpc_perm0"] > 0).sum()
                    true_nonzero = (df.loc[mask, "our_dwpc_true"] > 0).sum()
                    print(f"  Length {length}: perm0 {p0_nonzero}/{mask.sum()} non-zero, "
                          f"true {true_nonzero}/{mask.sum()} non-zero")

        if n_nonzero == 0:
            print("\n  WARNING: All perm0 DWPC values are zero!")
            print("  This suggests a problem with permutation loading or computation.")
        else:
            print("\n  VERDICT: Perm0 computation appears working")
    else:
        print("  No perm0 DWPC values available")

    # =========================================================================
    # 3. P-Value Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. P-VALUE COMPARISON")
    print("=" * 70)

    if "hetio_pvalue" in df.columns:
        valid = (
            df["hetio_pvalue"].notna() &
            (df["hetio_pvalue"] > 0) &
            (df["hetio_pvalue"] < 1) &
            (df["our_pvalue_true"] > 0) &
            (df["our_pvalue_true"] < 1)
        )
        n_valid = valid.sum()

        if n_valid > 2:
            our_p = df.loc[valid, "our_pvalue_true"]
            hetio_p = df.loc[valid, "hetio_pvalue"]

            # Log-scale correlation
            log_our = np.log10(our_p)
            log_hetio = np.log10(hetio_p)
            log_corr = np.corrcoef(log_our, log_hetio)[0, 1]

            # Linear correlation
            lin_corr = np.corrcoef(our_p, hetio_p)[0, 1]

            print(f"\nP-Value Comparison (our vs het.io):")
            print(f"  Valid pairs: {n_valid}")
            print(f"  Log10 correlation: {log_corr:.4f}")
            print(f"  Linear correlation: {lin_corr:.4f}")

            # Distribution comparison
            print(f"\n  Het.io p-values:")
            print(f"    Mean: {hetio_p.mean():.4f}")
            print(f"    Median: {hetio_p.median():.4f}")
            print(f"    < 0.05: {(hetio_p < 0.05).mean():.1%}")

            print(f"\n  Our p-values:")
            print(f"    Mean: {our_p.mean():.4f}")
            print(f"    Median: {our_p.median():.4f}")
            print(f"    < 0.05: {(our_p < 0.05).mean():.1%}")

            # Verdict
            if log_corr > 0.9:
                print(f"\n  VERDICT: P-values STRONGLY correlated (log10 r = {log_corr:.4f})")
            elif log_corr > 0.7:
                print(f"\n  VERDICT: P-values MODERATELY correlated (log10 r = {log_corr:.4f})")
            elif log_corr > 0.5:
                print(f"\n  VERDICT: P-values WEAKLY correlated (log10 r = {log_corr:.4f})")
            else:
                print(f"\n  VERDICT: P-values NOT correlated (log10 r = {log_corr:.4f})")
                print("  This may be due to using different numbers of permutations.")
        else:
            print("  Not enough valid pairs for comparison")
    else:
        print("  No hetio_pvalue column - cannot compare p-values")

    # =========================================================================
    # 4. Perm0 P-Value Calibration
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. PERM0 P-VALUE CALIBRATION")
    print("=" * 70)

    perm0_p = df["our_pvalue_perm0"]
    valid_p = perm0_p.notna() & (perm0_p > 0)

    if valid_p.sum() > 0:
        perm0_p_valid = perm0_p[valid_p]

        print(f"\nPerm0 p-value distribution:")
        print(f"  N valid: {valid_p.sum()}")
        print(f"  Mean: {perm0_p_valid.mean():.4f} (expected ~0.5 for calibrated)")
        print(f"  Median: {perm0_p_valid.median():.4f}")
        print(f"  < 0.05: {(perm0_p_valid < 0.05).mean():.1%} (expected ~5%)")
        print(f"  < 0.50: {(perm0_p_valid < 0.50).mean():.1%} (expected ~50%)")

        # KS test for uniformity
        ks_stat, ks_pval = stats.kstest(perm0_p_valid, 'uniform')
        print(f"\n  KS test for uniformity: stat={ks_stat:.4f}, p={ks_pval:.4f}")

        if ks_pval > 0.05:
            print("  VERDICT: Perm0 p-values appear uniformly distributed (calibrated)")
        else:
            print("  VERDICT: Perm0 p-values NOT uniform (miscalibrated)")
            if perm0_p_valid.mean() < 0.3:
                print("  P-values are DEFLATED (too small)")
            elif perm0_p_valid.mean() > 0.7:
                print("  P-values are INFLATED (too large)")
    else:
        print("  No valid perm0 p-values")

    # =========================================================================
    # 5. Sample Output
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. SAMPLE COMPARISONS")
    print("=" * 70)

    display_cols = ["go_id", "metapath", "our_dwpc_true", "our_dwpc_perm0"]
    if "hetio_pdp" in df.columns:
        display_cols.append("hetio_pdp")
    display_cols.extend(["our_pvalue_true", "our_pvalue_perm0"])
    if "hetio_pvalue" in df.columns:
        display_cols.append("hetio_pvalue")

    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].head(15).to_string(index=False))

    # =========================================================================
    # 5b. P-VALUE DIAGNOSTICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("5b. P-VALUE DIAGNOSTICS")
    print("=" * 70)

    if "hetio_pvalue" in df.columns:
        valid = (
            df["our_pvalue_true"].notna() &
            df["hetio_pvalue"].notna() &
            (df["our_pvalue_true"] > 0) &
            (df["hetio_pvalue"] > 0)
        )
        if valid.sum() > 0:
            our_log = -np.log10(df.loc[valid, "our_pvalue_true"].clip(1e-300))
            hetio_log = -np.log10(df.loc[valid, "hetio_pvalue"].clip(1e-300))

            print(f"\n-log10(p-value) comparison:")
            print(f"  Hetionet: mean={our_log.mean():.1f}, max={our_log.max():.1f}")
            print(f"  Het.io:   mean={hetio_log.mean():.1f}, max={hetio_log.max():.1f}")

            # Show extreme cases
            extreme = (our_log > 50) | ((our_log - hetio_log) > 20)
            if extreme.sum() > 0:
                print(f"\n  Extreme cases ({extreme.sum()}):")
                extreme_df = df.loc[valid][extreme]
                for idx in extreme_df.index[:5]:
                    row = df.loc[idx]
                    our_val = -np.log10(max(row["our_pvalue_true"], 1e-300))
                    hetio_val = -np.log10(max(row["hetio_pvalue"], 1e-300))
                    null_n = row.get("null_n_nonzero", "N/A")
                    print(f"    {row['metapath']}: our={our_val:.1f}, hetio={hetio_val:.1f}, null_n_nonzero={null_n}")
            else:
                print("\n  No extreme cases (our -log10p > 50 or diff > 20)")
        else:
            print("  No valid p-values for comparison")
    else:
        print("  No hetio_pvalue column available")

    # =========================================================================
    # 5c. PATH COUNT CONFOUNDING ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("5c. PATH COUNT CONFOUNDING ANALYSIS")
    print("=" * 70)

    has_path_count = "path_count" in df.columns
    has_degrees = "source_degree" in df.columns and "target_degree" in df.columns

    if has_path_count and has_degrees:
        valid = df["path_count"].notna() & (df["path_count"] > 0)

        if valid.sum() > 10:
            # Correlation: path_count vs degree_product
            degree_product = df.loc[valid, "source_degree"] * df.loc[valid, "target_degree"]
            pc_deg_corr = np.corrcoef(df.loc[valid, "path_count"], degree_product)[0, 1]
            print(f"\nPath count vs degree product correlation: {pc_deg_corr:.3f}")

            # Partial correlation: path_count vs significance, controlling for degree
            for pval_col, name in [("hetio_pvalue", "Het.io"), ("our_pvalue_true", "Hetionet")]:
                if pval_col not in df.columns:
                    continue
                pval_valid = (
                    valid &
                    df[pval_col].notna() &
                    (df[pval_col] > 0) &
                    (df[pval_col] < 1)
                )
                if pval_valid.sum() > 10:
                    log_p = -np.log10(df.loc[pval_valid, pval_col])
                    pc = df.loc[pval_valid, "path_count"].values
                    dp = (df.loc[pval_valid, "source_degree"] * df.loc[pval_valid, "target_degree"]).values

                    # Simple correlation
                    raw_corr = np.corrcoef(pc, log_p)[0, 1]

                    # Partial correlation (regress out degree product)
                    pc_resid = pc - np.polyval(np.polyfit(dp, pc, 1), dp)
                    logp_resid = log_p - np.polyval(np.polyfit(dp, log_p, 1), dp)
                    partial_corr = np.corrcoef(pc_resid, logp_resid)[0, 1]

                    print(f"\n{name}:")
                    print(f"  Raw correlation (path_count vs -log10p): {raw_corr:.3f}")
                    print(f"  Partial correlation (controlling for degree): {partial_corr:.3f}")
        else:
            print("  Not enough valid path count data")
    elif not has_path_count:
        print("  No path_count column available")
    else:
        print("  No degree columns available")

    # =========================================================================
    # 6. Generate Plots (7x3 grid for comprehensive 3-way comparison)
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. GENERATING PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(9, 3, figsize=(15, 36))

    # -------------------------------------------------------------------------
    # Row 1: DWPC Comparisons
    # -------------------------------------------------------------------------
    # 1.1: Our DWPC vs Het.io DWPC
    ax = axes[0, 0]
    if "hetio_pdp" in df.columns:
        valid = df["hetio_pdp"].notna() & (df["our_dwpc_true"] > 0) & (df["hetio_pdp"] > 0)
        if valid.sum() > 0:
            ax.scatter(df.loc[valid, "hetio_pdp"], df.loc[valid, "our_dwpc_true"], alpha=0.5)
            max_val = max(df.loc[valid, "hetio_pdp"].max(), df.loc[valid, "our_dwpc_true"].max())
            ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
            ax.set_xlabel("Het.io DWPC")
            ax.set_ylabel("Hetionet DWPC")
            ax.set_title("Hetionet DWPC vs Het.io DWPC")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No het.io DWPC data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Hetionet DWPC vs Het.io DWPC (no data)")

    # 1.2: Perm0 DWPC vs Hetionet DWPC
    ax = axes[0, 1]
    valid = (df["our_dwpc_true"] > 0) & (df["our_dwpc_perm0"].notna())
    if valid.sum() > 0:
        ax.scatter(df.loc[valid, "our_dwpc_true"], df.loc[valid, "our_dwpc_perm0"], alpha=0.5)
        max_val = max(df.loc[valid, "our_dwpc_true"].max(), df.loc[valid, "our_dwpc_perm0"].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
        ax.set_xlabel("Hetionet DWPC")
        ax.set_ylabel("Perm0 DWPC")
        ax.set_title("Perm0 DWPC vs Hetionet DWPC")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Perm0 DWPC vs Hetionet DWPC (no data)")

    # 1.3: Perm0 DWPC vs Het.io DWPC
    ax = axes[0, 2]
    if "hetio_pdp" in df.columns:
        valid = df["hetio_pdp"].notna() & (df["our_dwpc_perm0"].notna()) & (df["hetio_pdp"] > 0)
        if valid.sum() > 0:
            ax.scatter(df.loc[valid, "hetio_pdp"], df.loc[valid, "our_dwpc_perm0"], alpha=0.5)
            max_val = max(df.loc[valid, "hetio_pdp"].max(), df.loc[valid, "our_dwpc_perm0"].max())
            ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
            ax.set_xlabel("Het.io DWPC")
            ax.set_ylabel("Perm0 DWPC")
            ax.set_title("Perm0 DWPC vs Het.io DWPC")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No het.io DWPC data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Perm0 DWPC vs Het.io DWPC (no data)")

    # -------------------------------------------------------------------------
    # Row 2: P-value Distributions (Histograms)
    # -------------------------------------------------------------------------
    # 2.1: Het.io P-value Distribution
    ax = axes[1, 0]
    if "hetio_pvalue" in df.columns:
        valid_p = df["hetio_pvalue"].notna() & (df["hetio_pvalue"] > 0) & (df["hetio_pvalue"] < 1)
        if valid_p.sum() > 0:
            ax.hist(df.loc[valid_p, "hetio_pvalue"], bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel("P-Value")
            ax.set_ylabel("Count")
            ax.set_title("Het.io P-value Distribution")
    else:
        ax.text(0.5, 0.5, "No het.io p-value data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Het.io P-value Distribution (no data)")

    # 2.2: Hetionet P-value Distribution (True Network)
    ax = axes[1, 1]
    valid_p = df["our_pvalue_true"].notna() & (df["our_pvalue_true"] > 0) & (df["our_pvalue_true"] < 1)
    if valid_p.sum() > 0:
        ax.hist(df.loc[valid_p, "our_pvalue_true"], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel("P-Value")
        ax.set_ylabel("Count")
        ax.set_title("Hetionet P-value Distribution")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Hetionet P-value Distribution (no data)")

    # 2.3: Perm0 P-value Distribution
    ax = axes[1, 2]
    valid_p = df["our_pvalue_perm0"].notna() & (df["our_pvalue_perm0"] > 0) & (df["our_pvalue_perm0"] < 1)
    if valid_p.sum() > 0:
        ax.hist(df.loc[valid_p, "our_pvalue_perm0"], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel("P-Value")
        ax.set_ylabel("Count")
        ax.set_title("Perm0 P-value Distribution")
    else:
        ax.text(0.5, 0.5, "No perm0 p-values", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Perm0 P-value Distribution (no data)")

    # -------------------------------------------------------------------------
    # Row 3: -log10(P-value) vs Degree Product
    # -------------------------------------------------------------------------
    has_degrees = "source_degree" in df.columns and "target_degree" in df.columns

    # 3.1: Het.io Significance vs Degree Product
    ax = axes[2, 0]
    if has_degrees and "hetio_pvalue" in df.columns:
        valid = (
            df["source_degree"].notna() & df["target_degree"].notna() &
            df["hetio_pvalue"].notna() & (df["hetio_pvalue"] > 0) & (df["hetio_pvalue"] < 1)
        )
        if valid.sum() > 0:
            degree_product = df.loc[valid, "source_degree"] * df.loc[valid, "target_degree"]
            ax.scatter(degree_product, -np.log10(df.loc[valid, "hetio_pvalue"]), alpha=0.5)
            ax.set_xlabel("Source Degree x Target Degree")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Het.io: Significance vs Degree Product")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Het.io: Significance vs Degree Product (no data)")

    # 3.2: Hetionet Significance vs Degree Product
    ax = axes[2, 1]
    if has_degrees:
        valid = (
            df["source_degree"].notna() & df["target_degree"].notna() &
            (df["our_pvalue_true"] > 0) & (df["our_pvalue_true"] < 1)
        )
        if valid.sum() > 0:
            degree_product = df.loc[valid, "source_degree"] * df.loc[valid, "target_degree"]
            ax.scatter(degree_product, -np.log10(df.loc[valid, "our_pvalue_true"]), alpha=0.5)
            ax.set_xlabel("Source Degree x Target Degree")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Hetionet: Significance vs Degree Product")
    else:
        ax.text(0.5, 0.5, "No degree data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Hetionet: Significance vs Degree Product (no data)")

    # 3.3: Perm0 Significance vs Degree Product
    ax = axes[2, 2]
    if has_degrees:
        valid = (
            df["source_degree"].notna() & df["target_degree"].notna() &
            df["our_pvalue_perm0"].notna() & (df["our_pvalue_perm0"] > 0) & (df["our_pvalue_perm0"] < 1)
        )
        if valid.sum() > 0:
            degree_product = df.loc[valid, "source_degree"] * df.loc[valid, "target_degree"]
            ax.scatter(degree_product, -np.log10(df.loc[valid, "our_pvalue_perm0"]), alpha=0.5)
            ax.set_xlabel("Source Degree x Target Degree")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Perm0: Significance vs Degree Product")
    else:
        ax.text(0.5, 0.5, "No degree data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Perm0: Significance vs Degree Product (no data)")

    # -------------------------------------------------------------------------
    # Row 4: -log10(P-value) vs Path Count
    # -------------------------------------------------------------------------
    has_path_count = "path_count" in df.columns

    # 4.1: Het.io Significance vs Path Count
    ax = axes[3, 0]
    if has_path_count and "hetio_pvalue" in df.columns:
        valid = (
            df["path_count"].notna() & (df["path_count"] > 0) &
            df["hetio_pvalue"].notna() & (df["hetio_pvalue"] > 0) & (df["hetio_pvalue"] < 1)
        )
        if valid.sum() > 0:
            ax.scatter(df.loc[valid, "path_count"], -np.log10(df.loc[valid, "hetio_pvalue"]), alpha=0.5)
            ax.set_xlabel("Path Count")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Het.io: Significance vs Path Count")
            ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Het.io: Significance vs Path Count (no data)")

    # 4.2: Hetionet Significance vs Path Count
    ax = axes[3, 1]
    if has_path_count:
        valid = (
            df["path_count"].notna() & (df["path_count"] > 0) &
            (df["our_pvalue_true"] > 0) & (df["our_pvalue_true"] < 1)
        )
        if valid.sum() > 0:
            ax.scatter(df.loc[valid, "path_count"], -np.log10(df.loc[valid, "our_pvalue_true"]), alpha=0.5)
            ax.set_xlabel("Path Count")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Hetionet: Significance vs Path Count")
            ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No path count data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Hetionet: Significance vs Path Count (no data)")

    # 4.3: Perm0 Significance vs Path Count
    ax = axes[3, 2]
    if has_path_count:
        valid = (
            df["path_count"].notna() & (df["path_count"] > 0) &
            df["our_pvalue_perm0"].notna() & (df["our_pvalue_perm0"] > 0) & (df["our_pvalue_perm0"] < 1)
        )
        if valid.sum() > 0:
            ax.scatter(df.loc[valid, "path_count"], -np.log10(df.loc[valid, "our_pvalue_perm0"]), alpha=0.5)
            ax.set_xlabel("Path Count")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Perm0: Significance vs Path Count")
            ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No path count data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Perm0: Significance vs Path Count (no data)")

    # -------------------------------------------------------------------------
    # Row 5: -log10(P-value) vs Relative Path Count (path_count / degree_product)
    # -------------------------------------------------------------------------
    has_relative_pc = has_path_count and has_degrees

    # 5.1: Het.io Significance vs Relative Path Count
    ax = axes[4, 0]
    if has_relative_pc and "hetio_pvalue" in df.columns:
        valid = (
            df["path_count"].notna() & (df["path_count"] > 0) &
            df["source_degree"].notna() & df["target_degree"].notna() &
            (df["source_degree"] > 0) & (df["target_degree"] > 0) &
            df["hetio_pvalue"].notna() & (df["hetio_pvalue"] > 0) & (df["hetio_pvalue"] < 1)
        )
        if valid.sum() > 0:
            degree_product = df.loc[valid, "source_degree"] * df.loc[valid, "target_degree"]
            relative_pc = df.loc[valid, "path_count"] / degree_product
            ax.scatter(relative_pc, -np.log10(df.loc[valid, "hetio_pvalue"]), alpha=0.5)
            ax.set_xlabel("Path Count / Degree Product")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Het.io: Significance vs Relative Path Count")
            ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Het.io: Significance vs Relative Path Count (no data)")

    # 5.2: Hetionet Significance vs Relative Path Count
    ax = axes[4, 1]
    if has_relative_pc:
        valid = (
            df["path_count"].notna() & (df["path_count"] > 0) &
            df["source_degree"].notna() & df["target_degree"].notna() &
            (df["source_degree"] > 0) & (df["target_degree"] > 0) &
            (df["our_pvalue_true"] > 0) & (df["our_pvalue_true"] < 1)
        )
        if valid.sum() > 0:
            degree_product = df.loc[valid, "source_degree"] * df.loc[valid, "target_degree"]
            relative_pc = df.loc[valid, "path_count"] / degree_product
            ax.scatter(relative_pc, -np.log10(df.loc[valid, "our_pvalue_true"]), alpha=0.5)
            ax.set_xlabel("Path Count / Degree Product")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Hetionet: Significance vs Relative Path Count")
            ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Hetionet: Significance vs Relative Path Count (no data)")

    # 5.3: Perm0 Significance vs Relative Path Count
    ax = axes[4, 2]
    if has_relative_pc:
        valid = (
            df["path_count"].notna() & (df["path_count"] > 0) &
            df["source_degree"].notna() & df["target_degree"].notna() &
            (df["source_degree"] > 0) & (df["target_degree"] > 0) &
            df["our_pvalue_perm0"].notna() & (df["our_pvalue_perm0"] > 0) & (df["our_pvalue_perm0"] < 1)
        )
        if valid.sum() > 0:
            degree_product = df.loc[valid, "source_degree"] * df.loc[valid, "target_degree"]
            relative_pc = df.loc[valid, "path_count"] / degree_product
            ax.scatter(relative_pc, -np.log10(df.loc[valid, "our_pvalue_perm0"]), alpha=0.5)
            ax.set_xlabel("Path Count / Degree Product")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Perm0: Significance vs Relative Path Count")
            ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Perm0: Significance vs Relative Path Count (no data)")

    # -------------------------------------------------------------------------
    # Row 6: Degree Heatmaps
    # -------------------------------------------------------------------------
    has_degrees = "source_degree" in df.columns and "target_degree" in df.columns

    if has_degrees:
        valid_base = df["source_degree"].notna() & df["target_degree"].notna()

        if valid_base.sum() > 0:
            df_base = df.loc[valid_base].copy()

            # Bin degrees into terciles (low, medium, high)
            src_terciles = df_base["source_degree"].quantile([0.33, 0.67])
            tgt_terciles = df_base["target_degree"].quantile([0.33, 0.67])

            def bin_degree(val, terciles):
                if val <= terciles.iloc[0]:
                    return "Low"
                elif val <= terciles.iloc[1]:
                    return "Medium"
                else:
                    return "High"

            df_base["src_bin"] = df_base["source_degree"].apply(
                lambda x: bin_degree(x, src_terciles)
            )
            df_base["tgt_bin"] = df_base["target_degree"].apply(
                lambda x: bin_degree(x, tgt_terciles)
            )

            bin_order = ["Low", "Medium", "High"]

            # Define the three p-value columns to plot
            pval_configs = [
                ("hetio_pvalue", "Het.io: Significance by Degree"),
                ("our_pvalue_true", "Hetionet: Significance by Degree"),
                ("our_pvalue_perm0", "Perm0: Significance by Degree")
            ]

            for ax_idx, (pval_col, title) in enumerate(pval_configs):
                ax = axes[5, ax_idx]

                if pval_col not in df_base.columns:
                    ax.text(0.5, 0.5, f"No data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{title} (no data)")
                    continue

                # Filter valid p-values
                valid_p = df_base[pval_col].notna() & (df_base[pval_col] > 0) & (df_base[pval_col] < 1)
                if valid_p.sum() == 0:
                    ax.text(0.5, 0.5, "No valid p-values", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{title} (no data)")
                    continue

                df_valid = df_base.loc[valid_p].copy()
                df_valid["neg_log10_pval"] = -np.log10(df_valid[pval_col])

                pivot_mean = df_valid.pivot_table(
                    values="neg_log10_pval",
                    index="src_bin",
                    columns="tgt_bin",
                    aggfunc="mean"
                ).reindex(index=bin_order, columns=bin_order)

                pivot_count = df_valid.pivot_table(
                    values="neg_log10_pval",
                    index="src_bin",
                    columns="tgt_bin",
                    aggfunc="count"
                ).reindex(index=bin_order, columns=bin_order)

                im = ax.imshow(pivot_mean.values, cmap="YlOrRd", aspect="auto")
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("-log10(P)")

                ax.set_xticks(range(len(bin_order)))
                ax.set_yticks(range(len(bin_order)))
                ax.set_xticklabels(bin_order)
                ax.set_yticklabels(bin_order)

                ax.set_xlabel("Target Degree")
                ax.set_ylabel("Source Degree")
                ax.set_title(title)

                # Add text annotations
                max_val = np.nanmax(pivot_mean.values)
                for i in range(len(bin_order)):
                    for j in range(len(bin_order)):
                        mean_val = pivot_mean.iloc[i, j]
                        count_val = pivot_count.iloc[i, j]
                        if not np.isnan(mean_val):
                            text = f"{mean_val:.1f}\n(n={int(count_val)})"
                            ax.text(j, i, text, ha="center", va="center",
                                   color="white" if mean_val > max_val * 0.6 else "black",
                                   fontsize=8)
    else:
        for col in range(3):
            axes[5, col].text(0.5, 0.5, "No degree data", ha='center', va='center', transform=axes[5, col].transAxes)
            axes[5, col].set_title("Significance by Degree (no data)")

    # -------------------------------------------------------------------------
    # Row 7: Boxplots by Metapath Length
    # -------------------------------------------------------------------------
    if "metapath_length" in df.columns:
        lengths = sorted(df["metapath_length"].unique())

        pval_configs = [
            ("hetio_pvalue", "Het.io"),
            ("our_pvalue_true", "Hetionet"),
            ("our_pvalue_perm0", "Perm0")
        ]

        for ax_idx, (pval_col, title) in enumerate(pval_configs):
            ax = axes[6, ax_idx]

            if pval_col not in df.columns:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{title}: Significance by Path Length (no data)")
                continue

            data_by_length = []
            labels = []
            for length in lengths:
                mask = (
                    (df["metapath_length"] == length) &
                    df[pval_col].notna() &
                    (df[pval_col] > 0) &
                    (df[pval_col] < 1)
                )
                if mask.sum() > 0:
                    data_by_length.append(-np.log10(df.loc[mask, pval_col]))
                    labels.append(str(length))

            if data_by_length:
                ax.boxplot(data_by_length, tick_labels=labels)
                ax.set_xlabel("Metapath Length")
                ax.set_ylabel("-log10(P-Value)")
                ax.set_title(f"{title}: Significance by Path Length")
            else:
                ax.text(0.5, 0.5, "No valid p-values", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{title}: Significance by Path Length (no data)")
    else:
        for col in range(3):
            axes[6, col].text(0.5, 0.5, "No metapath_length column", ha='center', va='center', transform=axes[6, col].transAxes)
            axes[6, col].set_title("Significance by Path Length (no data)")

    # -------------------------------------------------------------------------
    # Row 8: Path Count vs Degree Product (single plot showing relationship)
    # -------------------------------------------------------------------------
    # 8.1: Path Count vs Degree Product (all data)
    ax = axes[7, 0]
    if has_path_count and has_degrees:
        valid = (
            df["path_count"].notna() & (df["path_count"] > 0) &
            df["source_degree"].notna() & df["target_degree"].notna() &
            (df["source_degree"] > 0) & (df["target_degree"] > 0)
        )
        if valid.sum() > 0:
            degree_product = df.loc[valid, "source_degree"] * df.loc[valid, "target_degree"]
            ax.scatter(degree_product, df.loc[valid, "path_count"], alpha=0.5)
            ax.set_xlabel("Degree Product")
            ax.set_ylabel("Path Count")
            ax.set_title("Path Count vs Degree Product")
            ax.set_xscale("log")
            ax.set_yscale("log")
            # Add correlation
            corr = np.corrcoef(np.log10(degree_product), np.log10(df.loc[valid, "path_count"]))[0, 1]
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10, va='top')
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Path Count vs Degree Product (no data)")

    # 8.2: Path Count / Degree Product Distribution
    ax = axes[7, 1]
    if has_path_count and has_degrees:
        valid = (
            df["path_count"].notna() & (df["path_count"] > 0) &
            df["source_degree"].notna() & df["target_degree"].notna() &
            (df["source_degree"] > 0) & (df["target_degree"] > 0)
        )
        if valid.sum() > 0:
            degree_product = df.loc[valid, "source_degree"] * df.loc[valid, "target_degree"]
            relative_pc = df.loc[valid, "path_count"] / degree_product
            ax.hist(np.log10(relative_pc), bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel("log10(Path Count / Degree Product)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Relative Path Count")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Relative Path Count Distribution (no data)")

    # 8.3: Relative Path Count by Metapath Length
    ax = axes[7, 2]
    if has_path_count and has_degrees and "metapath_length" in df.columns:
        valid = (
            df["path_count"].notna() & (df["path_count"] > 0) &
            df["source_degree"].notna() & df["target_degree"].notna() &
            (df["source_degree"] > 0) & (df["target_degree"] > 0)
        )
        if valid.sum() > 0:
            lengths = sorted(df.loc[valid, "metapath_length"].unique())
            data_by_length = []
            labels = []
            for length in lengths:
                mask = valid & (df["metapath_length"] == length)
                if mask.sum() > 0:
                    dp = df.loc[mask, "source_degree"] * df.loc[mask, "target_degree"]
                    rel_pc = df.loc[mask, "path_count"] / dp
                    data_by_length.append(np.log10(rel_pc))
                    labels.append(str(length))
            if data_by_length:
                ax.boxplot(data_by_length, tick_labels=labels)
                ax.set_xlabel("Metapath Length")
                ax.set_ylabel("log10(Path Count / Degree Product)")
                ax.set_title("Relative Path Count by Metapath Length")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Relative Path Count by Length (no data)")

    # -------------------------------------------------------------------------
    # Row 9: -log10(P-value) vs Null DWPC Variance
    # -------------------------------------------------------------------------
    has_null_var = "null_nonzero_sd" in df.columns

    # 9.1: Het.io Significance vs Null Variance
    ax = axes[8, 0]
    if has_null_var and "hetio_pvalue" in df.columns:
        valid = (
            df["null_nonzero_sd"].notna() & (df["null_nonzero_sd"] > 0) &
            df["hetio_pvalue"].notna() & (df["hetio_pvalue"] > 0) & (df["hetio_pvalue"] < 1)
        )
        if valid.sum() > 0:
            null_var = df.loc[valid, "null_nonzero_sd"] ** 2
            ax.scatter(null_var, -np.log10(df.loc[valid, "hetio_pvalue"]), alpha=0.5)
            ax.set_xlabel("Null DWPC Variance")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Het.io: Significance vs Null Variance")
            ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Het.io: Significance vs Null Variance (no data)")

    # 9.2: Hetionet Significance vs Null Variance
    ax = axes[8, 1]
    if has_null_var:
        valid = (
            df["null_nonzero_sd"].notna() & (df["null_nonzero_sd"] > 0) &
            (df["our_pvalue_true"] > 0) & (df["our_pvalue_true"] < 1)
        )
        if valid.sum() > 0:
            null_var = df.loc[valid, "null_nonzero_sd"] ** 2
            ax.scatter(null_var, -np.log10(df.loc[valid, "our_pvalue_true"]), alpha=0.5)
            ax.set_xlabel("Null DWPC Variance")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Hetionet: Significance vs Null Variance")
            ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No null variance data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Hetionet: Significance vs Null Variance (no data)")

    # 9.3: Perm0 Significance vs Null Variance
    ax = axes[8, 2]
    if has_null_var:
        valid = (
            df["null_nonzero_sd"].notna() & (df["null_nonzero_sd"] > 0) &
            df["our_pvalue_perm0"].notna() & (df["our_pvalue_perm0"] > 0) & (df["our_pvalue_perm0"] < 1)
        )
        if valid.sum() > 0:
            null_var = df.loc[valid, "null_nonzero_sd"] ** 2
            ax.scatter(null_var, -np.log10(df.loc[valid, "our_pvalue_perm0"]), alpha=0.5)
            ax.set_xlabel("Null DWPC Variance")
            ax.set_ylabel("-log10(P-Value)")
            ax.set_title("Perm0: Significance vs Null Variance")
            ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No null variance data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Perm0: Significance vs Null Variance (no data)")

    plt.tight_layout()
    plot_file = output_dir / f"pvalue_comparison_{input_file.stem}.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Saved plot to: {plot_file}")
    plt.close()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare p-values to het.io ground truth"
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to CSV from 30b"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: same as input)"
    )

    args = parser.parse_args()
    main(input_file=args.input_file, output_dir=args.output_dir)
