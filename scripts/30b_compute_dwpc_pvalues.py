"""
Compute DWPC and p-values for het.io pairs using hetmatpy.

This script computes:
1. DWPC for Hetionet (true network) using hetmatpy.degree_weight.dwpc()
2. DWPC for permutation 0 (to verify permutation loading)
3. Null distribution from permutations 1-N
4. P-values using gamma-hurdle method

Ensures perfect alignment by grouping pairs by metapath and computing
DWPC matrices once per metapath.

Usage:
    python scripts/30b_compute_dwpc_pvalues.py --n_perms 20 --input_file data/hetio_pairs_with_actual_pdp.csv

    # Limit to first 3 metapaths (for testing)
    python scripts/30b_compute_dwpc_pvalues.py --n_perms 5 --max_metapaths 3

    # Process specific metapaths only
    python scripts/30b_compute_dwpc_pvalues.py --n_perms 20 --metapaths BPpG,BPpGpBPpG

For HPC:
    python scripts/30b_compute_dwpc_pvalues.py --n_perms 20 --hpc
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings


def get_project_root():
    """Get project root from environment or script location."""
    if "CAPP_PROJECT_ROOT" in os.environ:
        return Path(os.environ["CAPP_PROJECT_ROOT"])
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PERM_DIR = DATA_DIR / "permutations"


def load_hetmat(source="true"):
    """
    Load HetMat for true network or a permutation.

    Parameters
    ----------
    source : str
        "true" for real Hetionet, or "permN" for permutation N (0-indexed)

    Returns
    -------
    hetmatpy.hetmat.HetMat
    """
    import hetmatpy.hetmat

    if source == "true":
        return hetmatpy.hetmat.HetMat(DATA_DIR)
    elif source.startswith("perm"):
        perm_idx = int(source[4:])
        perm_path = PERM_DIR / f"{perm_idx:03d}.hetmat"
        if not perm_path.exists():
            raise FileNotFoundError(f"Permutation not found: {perm_path}")
        return hetmatpy.hetmat.HetMat(perm_path)
    else:
        raise ValueError(f"Unknown source: {source}")


def compute_dwpc_for_pairs(hetmat, metapath_str, bp_indices, gene_indices, damping=0.5):
    """
    Compute DWPC for specific pairs using hetmatpy.

    Parameters
    ----------
    hetmat : hetmatpy.hetmat.HetMat
        HetMat object
    metapath_str : str
        Metapath abbreviation (e.g., 'BPpG', 'BPpGpBPpG')
    bp_indices : array-like
        BP node indices
    gene_indices : array-like
        Gene node indices
    damping : float
        Damping exponent

    Returns
    -------
    np.ndarray : DWPC values for each pair
    """
    from hetmatpy.degree_weight import dwpc

    metapath = hetmat.metagraph.metapath_from_abbrev(metapath_str)
    rows, cols, dwpc_matrix = dwpc(hetmat, metapath, damping=damping)

    # Extract values for requested pairs
    dwpc_values = np.zeros(len(bp_indices))
    for i, (bp_idx, gene_idx) in enumerate(zip(bp_indices, gene_indices)):
        dwpc_values[i] = dwpc_matrix[bp_idx, gene_idx]

    return dwpc_values


def build_null_distribution(
    metapath_str,
    bp_indices,
    gene_indices,
    n_perms=20,
    damping=0.5,
    verbose=True
):
    """
    Build null distribution by computing DWPC on permuted networks.

    Parameters
    ----------
    metapath_str : str
        Metapath abbreviation
    bp_indices : array-like
        BP node indices
    gene_indices : array-like
        Gene node indices
    n_perms : int
        Number of permutations to use (1 to n_perms)
    damping : float
        Damping exponent
    verbose : bool
        Print progress

    Returns
    -------
    np.ndarray : Shape (n_pairs, n_perms) with DWPC values
    """
    n_pairs = len(bp_indices)
    null_dwpcs = np.zeros((n_pairs, n_perms))

    for perm_idx in range(1, n_perms + 1):
        if verbose and perm_idx % 5 == 0:
            print(f"    Processing permutation {perm_idx}/{n_perms}...")

        try:
            perm_hetmat = load_hetmat(f"perm{perm_idx}")
            perm_dwpcs = compute_dwpc_for_pairs(
                perm_hetmat, metapath_str, bp_indices, gene_indices, damping
            )
            null_dwpcs[:, perm_idx - 1] = perm_dwpcs
        except FileNotFoundError as e:
            if verbose:
                print(f"    WARNING: {e}")
            null_dwpcs[:, perm_idx - 1] = np.nan

    return null_dwpcs


def fit_gamma_hurdle(null_dwpcs):
    """
    Fit gamma-hurdle distribution to null DWPC values.

    Parameters
    ----------
    null_dwpcs : np.ndarray
        Null DWPC values (can be 1D for single pair or 2D for multiple pairs)

    Returns
    -------
    dict with lambda, alpha, beta, n_total, n_nonzero, nonzero_mean, nonzero_sd
    """
    # Handle NaN values
    valid = ~np.isnan(null_dwpcs)
    null_dwpcs_valid = null_dwpcs[valid]

    n_total = len(null_dwpcs_valid)
    if n_total == 0:
        return {
            "lambda": 0, "alpha": 1, "beta": 1,
            "n_total": 0, "n_nonzero": 0,
            "nonzero_mean": 0, "nonzero_sd": 0
        }

    nonzero = null_dwpcs_valid[null_dwpcs_valid > 0]
    n_nonzero = len(nonzero)

    lambda_param = n_nonzero / n_total if n_total > 0 else 0

    if n_nonzero < 2:
        return {
            "lambda": lambda_param,
            "alpha": 1.0,
            "beta": 1.0,
            "n_total": n_total,
            "n_nonzero": n_nonzero,
            "nonzero_mean": nonzero.mean() if n_nonzero > 0 else 0,
            "nonzero_sd": 0
        }

    mean = nonzero.mean()
    sd = nonzero.std(ddof=1)
    var = sd ** 2

    if var <= 0 or mean <= 0:
        alpha = 1.0
        beta = 1.0 / mean if mean > 0 else 1.0
    else:
        alpha = mean ** 2 / var
        beta = mean / var

    return {
        "lambda": lambda_param,
        "alpha": alpha,
        "beta": beta,
        "n_total": n_total,
        "n_nonzero": n_nonzero,
        "nonzero_mean": mean,
        "nonzero_sd": sd
    }


def calculate_pvalue(observed_dwpc, lambda_param, alpha, beta):
    """
    Calculate p-value using gamma-hurdle distribution.

    P(DWPC >= observed) = lambda * Gamma.sf(observed; alpha, scale=1/beta)
    """
    if observed_dwpc == 0:
        return 1.0

    if lambda_param == 0:
        return 0.0

    gamma_sf = stats.gamma.sf(observed_dwpc, a=alpha, scale=1.0 / beta)
    return lambda_param * gamma_sf


def main(
    n_perms=20,
    input_file=None,
    output_file=None,
    damping=0.5,
    hpc_mode=False,
    verbose=True,
    max_metapaths=None,
    metapaths=None,
    all_pairs=False
):
    """
    Compute DWPC and p-values for het.io pairs.

    Parameters
    ----------
    n_perms : int
        Number of permutations for null distribution
    input_file : str
        Input CSV with het.io pairs (should have actual_pdp column)
    output_file : str
        Output CSV path
    damping : float
        DWPC damping exponent
    hpc_mode : bool
        If True, suppress interactive output
    verbose : bool
        Print progress
    max_metapaths : int or None
        Limit to first N metapaths (for testing)
    metapaths : list or None
        Specific metapaths to process (e.g., ['BPpG', 'BPpGpBPpG'])
    all_pairs : bool
        If True, use all unique pairs for each metapath (increases sample size)
    """
    if input_file is None:
        # Try the file with actual PDP first, fall back to original
        input_file = DATA_DIR / "hetio_pairs_with_actual_pdp.csv"
        if not input_file.exists():
            input_file = DATA_DIR / "hetio_pairs_for_validation.csv"
    else:
        input_file = Path(input_file)

    if output_file is None:
        output_file = (
            PROJECT_ROOT / "results" / "pvalue_validation" /
            f"dwpc_pvalue_comparison_{n_perms}perms.csv"
        )
    else:
        output_file = Path(output_file)

    print("=" * 70)
    print("Computing DWPC and p-values for het.io pairs")
    print("=" * 70)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"N permutations: {n_perms}")
    print(f"Damping: {damping}")

    # Load input pairs
    df = pd.read_csv(input_file)
    print(f"\nLoaded {len(df)} pairs")

    # Check for required columns
    required = ["go_id", "entrez_gene_id", "bp_idx", "gene_idx", "metapath_abbreviation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Load true Hetionet
    print("\nLoading Hetionet...")
    true_hetmat = load_hetmat("true")

    # Load perm0 for verification
    print("Loading permutation 0...")
    try:
        perm0_hetmat = load_hetmat("perm0")
        have_perm0 = True
    except FileNotFoundError:
        print("  WARNING: Perm0 not found")
        have_perm0 = False

    # Group pairs by metapath for efficient computation
    all_metapaths = df["metapath_abbreviation"].unique()
    print(f"\nFound {len(all_metapaths)} unique metapaths in input file")

    # Filter metapaths if requested
    if metapaths is not None:
        # Use specific metapaths provided
        metapaths_to_process = [mp for mp in metapaths if mp in all_metapaths]
        missing = [mp for mp in metapaths if mp not in all_metapaths]
        if missing:
            print(f"  WARNING: Requested metapaths not in file: {missing}")
        print(f"  Processing {len(metapaths_to_process)} specified metapaths")
    elif max_metapaths is not None:
        metapaths_to_process = all_metapaths[:max_metapaths]
        print(f"  Limiting to first {max_metapaths} metapaths")
    else:
        metapaths_to_process = all_metapaths

    # Filter dataframe to only include selected metapaths
    df_filtered = df[df["metapath_abbreviation"].isin(metapaths_to_process)]
    print(f"  Total pairs to process: {len(df_filtered)}")

    # Get all unique pairs if using all_pairs mode
    if all_pairs:
        unique_pairs = df[["go_id", "entrez_gene_id", "bp_idx", "gene_idx"]].drop_duplicates()
        print(f"  All pairs mode: using {len(unique_pairs)} unique pairs for each metapath")
        # Keep first row's metadata for each pair
        pair_metadata = df.drop_duplicates(subset=["go_id", "entrez_gene_id"])
    else:
        df_filtered = df_filtered

    # Results storage
    results = []

    for mp_idx, metapath_str in enumerate(metapaths_to_process):
        if all_pairs:
            # Use all unique pairs for this metapath
            mp_df = unique_pairs.copy()
            n_pairs = len(mp_df)
        else:
            # Only use pairs originally assigned to this metapath
            mp_df = df_filtered[df_filtered["metapath_abbreviation"] == metapath_str].copy()
            n_pairs = len(mp_df)

        print(f"\n[{mp_idx + 1}/{len(metapaths_to_process)}] Processing {metapath_str} ({n_pairs} pairs)")

        bp_indices = mp_df["bp_idx"].values.astype(int)
        gene_indices = mp_df["gene_idx"].values.astype(int)

        # Compute DWPC for true network
        print("  Computing DWPC for Hetionet...")
        true_dwpcs = compute_dwpc_for_pairs(
            true_hetmat, metapath_str, bp_indices, gene_indices, damping
        )

        # Compute DWPC for perm0
        if have_perm0:
            print("  Computing DWPC for perm0...")
            perm0_dwpcs = compute_dwpc_for_pairs(
                perm0_hetmat, metapath_str, bp_indices, gene_indices, damping
            )
        else:
            perm0_dwpcs = np.full(n_pairs, np.nan)

        # Build null distribution from permutations 1-N
        print(f"  Building null distribution from {n_perms} permutations...")
        null_dwpcs = build_null_distribution(
            metapath_str, bp_indices, gene_indices, n_perms, damping, verbose
        )

        # Compute p-values for each pair
        print("  Computing p-values...")
        for i in range(n_pairs):
            row = mp_df.iloc[i]

            # In all_pairs mode, get metadata from pair_metadata lookup
            if all_pairs:
                go_id = row["go_id"]
                gene_id = row["entrez_gene_id"]
                meta_row = pair_metadata[
                    (pair_metadata["go_id"] == go_id) &
                    (pair_metadata["entrez_gene_id"] == gene_id)
                ]
                if len(meta_row) > 0:
                    meta_row = meta_row.iloc[0]
                else:
                    meta_row = row  # fallback
            else:
                meta_row = row

            # Fit gamma-hurdle to this pair's null distribution
            pair_null = null_dwpcs[i, :]
            gh_params = fit_gamma_hurdle(pair_null)

            # P-value for true network DWPC
            pval_true = calculate_pvalue(
                true_dwpcs[i],
                gh_params["lambda"],
                gh_params["alpha"],
                gh_params["beta"]
            )

            # P-value for perm0 DWPC
            pval_perm0 = calculate_pvalue(
                perm0_dwpcs[i],
                gh_params["lambda"],
                gh_params["alpha"],
                gh_params["beta"]
            )

            result = {
                "go_id": row["go_id"],
                "entrez_gene_id": row["entrez_gene_id"],
                "bp_idx": bp_indices[i],
                "gene_idx": gene_indices[i],
                "metapath": metapath_str,
                "metapath_length": len(metapath_str.replace("p", "").replace("<", "").replace(">", "")) // 2,
                # DWPC values
                "our_dwpc_true": true_dwpcs[i],
                "our_dwpc_perm0": perm0_dwpcs[i],
                # Null distribution stats
                "null_n_total": gh_params["n_total"],
                "null_n_nonzero": gh_params["n_nonzero"],
                "null_lambda": gh_params["lambda"],
                "null_nonzero_mean": gh_params["nonzero_mean"],
                "null_nonzero_sd": gh_params["nonzero_sd"],
                "null_alpha": gh_params["alpha"],
                "null_beta": gh_params["beta"],
                # Our p-values
                "our_pvalue_true": pval_true,
                "our_pvalue_perm0": pval_perm0,
            }

            # Add het.io values if available (only valid for original metapath assignment)
            if not all_pairs:
                if "actual_pdp" in meta_row:
                    result["hetio_pdp"] = meta_row["actual_pdp"]
                if "p_value" in meta_row:
                    result["hetio_pvalue"] = meta_row["p_value"]
                if "api_p_value" in meta_row:
                    result["hetio_api_pvalue"] = meta_row["api_p_value"]
                if "dwpc" in meta_row:
                    result["hetio_dgp_mean"] = meta_row["dwpc"]

            # Add degree and path count info (from pair metadata)
            if "dgp_source_degree" in meta_row:
                result["source_degree"] = meta_row["dgp_source_degree"]
            if "dgp_target_degree" in meta_row:
                result["target_degree"] = meta_row["dgp_target_degree"]
            if not all_pairs:
                if "actual_path_count" in meta_row:
                    result["path_count"] = meta_row["actual_path_count"]
                elif "path_count" in meta_row:
                    result["path_count"] = meta_row["path_count"]

            results.append(result)

    # Create output dataframe
    results_df = pd.DataFrame(results)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal pairs processed: {len(results_df)}")

    print(f"\nOur DWPC (true network):")
    print(f"  Non-zero: {(results_df['our_dwpc_true'] > 0).sum()}/{len(results_df)}")
    print(f"  Mean: {results_df['our_dwpc_true'].mean():.6f}")

    print(f"\nOur DWPC (perm0):")
    perm0_valid = results_df['our_dwpc_perm0'].notna()
    print(f"  Non-zero: {(results_df.loc[perm0_valid, 'our_dwpc_perm0'] > 0).sum()}/{perm0_valid.sum()}")
    print(f"  Mean: {results_df.loc[perm0_valid, 'our_dwpc_perm0'].mean():.6f}")

    print(f"\nOur p-values (true network):")
    print(f"  Mean: {results_df['our_pvalue_true'].mean():.4f}")
    print(f"  Median: {results_df['our_pvalue_true'].median():.4f}")
    print(f"  < 0.05: {(results_df['our_pvalue_true'] < 0.05).mean():.1%}")

    print(f"\nOur p-values (perm0):")
    print(f"  Mean: {results_df['our_pvalue_perm0'].mean():.4f}")
    print(f"  Median: {results_df['our_pvalue_perm0'].median():.4f}")
    print(f"  < 0.05: {(results_df['our_pvalue_perm0'] < 0.05).mean():.1%}")

    # Compare to het.io if available
    if "hetio_pdp" in results_df.columns:
        valid = results_df["hetio_pdp"].notna() & (results_df["our_dwpc_true"] > 0)
        if valid.sum() > 2:
            corr = np.corrcoef(
                results_df.loc[valid, "our_dwpc_true"],
                results_df.loc[valid, "hetio_pdp"]
            )[0, 1]
            print(f"\nDWPC correlation (our vs het.io): {corr:.6f}")

    if "hetio_pvalue" in results_df.columns:
        valid = (
            results_df["hetio_pvalue"].notna() &
            (results_df["hetio_pvalue"] > 0) &
            (results_df["our_pvalue_true"] > 0)
        )
        if valid.sum() > 2:
            log_corr = np.corrcoef(
                np.log10(results_df.loc[valid, "our_pvalue_true"]),
                np.log10(results_df.loc[valid, "hetio_pvalue"])
            )[0, 1]
            print(f"P-value log10 correlation (our vs het.io): {log_corr:.6f}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute DWPC and p-values for het.io pairs"
    )
    parser.add_argument(
        "--n_perms", type=int, default=20,
        help="Number of permutations for null distribution (default: 20)"
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Input CSV with het.io pairs"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output CSV path"
    )
    parser.add_argument(
        "--damping", type=float, default=0.5,
        help="DWPC damping exponent (default: 0.5)"
    )
    parser.add_argument(
        "--hpc", action="store_true",
        help="HPC mode (suppress interactive output)"
    )
    parser.add_argument(
        "--max_metapaths", type=int, default=None,
        help="Limit to first N metapaths (for testing)"
    )
    parser.add_argument(
        "--metapaths", type=str, default=None,
        help="Comma-separated list of metapaths to process (e.g., 'BPpG,BPpGpBPpG')"
    )
    parser.add_argument(
        "--all_pairs", action="store_true",
        help="Use all unique pairs for each metapath (increases sample size)"
    )

    args = parser.parse_args()

    # Parse metapaths list if provided
    metapaths_list = None
    if args.metapaths:
        metapaths_list = [mp.strip() for mp in args.metapaths.split(",")]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(
            n_perms=args.n_perms,
            input_file=args.input_file,
            output_file=args.output_file,
            damping=args.damping,
            hpc_mode=args.hpc,
            max_metapaths=args.max_metapaths,
            metapaths=metapaths_list,
            all_pairs=args.all_pairs
        )
