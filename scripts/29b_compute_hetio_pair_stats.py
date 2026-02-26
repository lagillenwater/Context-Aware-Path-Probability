"""
Stage 1b: Compute DWPC and null distribution stats for het.io pairs.

This script runs on HPC where permutation data is available.
It computes:
- DWPC for Hetionet (true network) using hetmatpy for any metapath length
- DWPC for permutation 0
- Null distribution statistics from permutations 1-N

Input: CSV with het.io pairs (from 29a_extract_hetio_pairs.py)
Output: CSV with computed DWPCs and null stats

Usage:
    python scripts/29b_compute_hetio_pair_stats.py --n_perms 20 --input_file data/hetio_pairs_for_validation.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.sparse as sp

import hetmatpy.hetmat
import hetmatpy.degree_weight


def get_project_root():
    """Get project root from environment or script location."""
    if "CAPP_PROJECT_ROOT" in os.environ:
        return Path(os.environ["CAPP_PROJECT_ROOT"])
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PERM_DIR = DATA_DIR / "permutations"


def load_hetmat(perm_idx=None):
    """
    Load HetMat object for true network or permutation.

    Parameters
    ----------
    perm_idx : int or None
        Permutation index (0-based). None for true Hetionet.

    Returns
    -------
    hetmatpy.hetmat.HetMat
    """
    if perm_idx is not None:
        path = PERM_DIR / f"{perm_idx:03d}.hetmat"
    else:
        path = DATA_DIR

    return hetmatpy.hetmat.HetMat(path)


def compute_dwpc_matrix(hetmat, metapath_str, damping=0.5):
    """
    Compute DWPC matrix for a metapath using hetmatpy.

    Parameters
    ----------
    hetmat : hetmatpy.hetmat.HetMat
        HetMat object
    metapath_str : str
        Metapath abbreviation (e.g., 'BPpGpPWpG')
    damping : float
        Damping exponent (default 0.5)

    Returns
    -------
    tuple : (row_ids, col_ids, dwpc_matrix)
        row_ids: list of source node identifiers
        col_ids: list of target node identifiers
        dwpc_matrix: numpy array or sparse matrix of DWPC values
    """
    metapath = hetmat.metagraph.get_metapath(metapath_str)
    row_ids, col_ids, dwpc_matrix = hetmatpy.degree_weight.dwpc(
        hetmat, metapath, damping=damping
    )
    return row_ids, col_ids, dwpc_matrix


def get_dwpc_for_pair(dwpc_matrix, row_ids, col_ids, source_id, target_id):
    """
    Look up DWPC value for a specific source-target pair.

    Parameters
    ----------
    dwpc_matrix : array-like
        DWPC matrix from compute_dwpc_matrix
    row_ids : list
        Source node identifiers
    col_ids : list
        Target node identifiers
    source_id : str or int
        Source node identifier
    target_id : str or int
        Target node identifier

    Returns
    -------
    float : DWPC value for the pair
    """
    try:
        row_idx = row_ids.index(source_id)
        col_idx = col_ids.index(target_id)
    except ValueError:
        return 0.0

    if sp.issparse(dwpc_matrix):
        return dwpc_matrix[row_idx, col_idx]
    else:
        return dwpc_matrix[row_idx, col_idx]


def fit_gamma_hurdle_stats(dwpcs):
    """
    Compute gamma-hurdle statistics from null distribution.

    Returns
    -------
    dict with keys:
        n_total, n_nonzero, lambda, nonzero_mean, nonzero_sd, alpha, beta
    """
    dwpcs = np.asarray(dwpcs)
    n_total = len(dwpcs)
    nonzero = dwpcs[dwpcs > 0]
    n_nonzero = len(nonzero)

    lambda_param = n_nonzero / n_total if n_total > 0 else 0

    if n_nonzero < 2:
        return {
            "n_total": n_total,
            "n_nonzero": n_nonzero,
            "lambda": lambda_param,
            "nonzero_mean": nonzero.mean() if n_nonzero > 0 else 0,
            "nonzero_sd": 0,
            "alpha": 1.0,
            "beta": 1.0
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
        "n_total": n_total,
        "n_nonzero": n_nonzero,
        "lambda": lambda_param,
        "nonzero_mean": mean,
        "nonzero_sd": sd,
        "alpha": alpha,
        "beta": beta
    }


def build_null_from_hetio_stats(row):
    """
    Use het.io's precomputed null statistics instead of computing our own.

    This is the correct approach for validation: use het.io's null distribution
    to compute p-values, then compare to het.io's p-values.

    Parameters
    ----------
    row : pd.Series
        Row from input dataframe with het.io dgp_* columns

    Returns
    -------
    dict : Gamma-hurdle parameters from het.io's null distribution
    """
    n_total = row.get("dgp_n_dwpcs", 0)
    n_nonzero = row.get("dgp_n_nonzero_dwpcs", 0)
    mean = row.get("dgp_nonzero_mean", 0)
    sd = row.get("dgp_nonzero_sd", 0)

    lambda_param = n_nonzero / n_total if n_total > 0 else 0
    var = sd ** 2

    if var <= 0 or mean <= 0:
        alpha = 1.0
        beta = 1.0 / mean if mean > 0 else 1.0
    else:
        alpha = mean ** 2 / var
        beta = mean / var

    return {
        "n_total": n_total,
        "n_nonzero": n_nonzero,
        "lambda": lambda_param,
        "nonzero_mean": mean,
        "nonzero_sd": sd,
        "alpha": alpha,
        "beta": beta
    }


def main(n_perms=20, input_file=None, output_file=None, use_hetio_null=True):
    """
    Compute DWPC and null stats for het.io pairs.

    Parameters
    ----------
    n_perms : int
        Number of permutations for null distribution (1 to n_perms)
        Only used if use_hetio_null=False
    input_file : str
        Path to CSV with het.io pairs
    output_file : str
        Path for output CSV
    use_hetio_null : bool
        If True, use het.io's precomputed null statistics (recommended for validation)
        If False, compute our own null from permutations
    """
    if input_file is None:
        input_file = DATA_DIR / "hetio_pairs_for_validation.csv"
    else:
        input_file = Path(input_file)

    if output_file is None:
        suffix = "hetio_null" if use_hetio_null else f"{n_perms}perms"
        output_file = (
            PROJECT_ROOT / "results" / "pvalue_validation" /
            f"hetio_pair_stats_{suffix}.csv"
        )
    else:
        output_file = Path(output_file)

    print("=" * 80)
    print("Computing DWPC and null stats for het.io pairs")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Use het.io null: {use_hetio_null}")
    if not use_hetio_null:
        print(f"N permutations: {n_perms}")

    # Load het.io pairs
    print("\nLoading het.io pairs...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} pairs")

    # Check metapath lengths
    if "metapath_length" in df.columns:
        print(f"  Metapath lengths: {df['metapath_length'].value_counts().to_dict()}")
    if "metapath_abbreviation" in df.columns:
        print(f"  Unique metapaths: {df['metapath_abbreviation'].nunique()}")

    # Load hetmat for true network
    print("\nLoading HetMat...")
    hetmat_true = load_hetmat(perm_idx=None)
    print(f"  Metagraph loaded: {hetmat_true.metagraph}")

    # Load hetmat for perm0
    hetmat_perm0 = load_hetmat(perm_idx=0)

    # Group pairs by metapath for efficient computation
    metapaths = df["metapath_abbreviation"].unique()
    print(f"\nProcessing {len(metapaths)} unique metapaths...")

    # Cache DWPC matrices per metapath
    dwpc_cache_true = {}
    dwpc_cache_perm0 = {}

    results = []

    for mp_idx, metapath_str in enumerate(metapaths):
        mp_df = df[df["metapath_abbreviation"] == metapath_str]
        print(f"\n[{mp_idx + 1}/{len(metapaths)}] Metapath: {metapath_str} "
              f"({len(mp_df)} pairs)")

        # Compute DWPC matrix for this metapath (true network)
        print(f"  Computing DWPC matrix for Hetionet...")
        row_ids_true, col_ids_true, dwpc_matrix_true = compute_dwpc_matrix(
            hetmat_true, metapath_str
        )

        # Compute DWPC matrix for perm0
        print(f"  Computing DWPC matrix for perm0...")
        row_ids_p0, col_ids_p0, dwpc_matrix_p0 = compute_dwpc_matrix(
            hetmat_perm0, metapath_str
        )

        # Process each pair with this metapath
        for idx, row in mp_df.iterrows():
            go_id = row["go_id"]
            gene_id = row["entrez_gene_id"]

            # DWPC for Hetionet
            # Source is BP (go_id), target is Gene (entrez_gene_id)
            dwpc_hetionet = get_dwpc_for_pair(
                dwpc_matrix_true, row_ids_true, col_ids_true,
                go_id, gene_id
            )

            # DWPC for perm0
            dwpc_perm0 = get_dwpc_for_pair(
                dwpc_matrix_p0, row_ids_p0, col_ids_p0,
                go_id, gene_id
            )

            # Get null distribution stats
            if use_hetio_null:
                null_stats = build_null_from_hetio_stats(row)
            else:
                # TODO: Implement our own null distribution computation
                # This would require computing DWPC matrices for each permutation
                # and sampling degree-similar pairs
                raise NotImplementedError(
                    "Own null distribution not yet implemented for multi-hop paths"
                )

            # Store results
            result = {
                "go_id": go_id,
                "entrez_gene_id": gene_id,
                "metapath": metapath_str,
                "metapath_length": row.get("metapath_length", len(metapath_str.split("p")) - 1),
                "dwpc_hetionet": float(dwpc_hetionet),
                "dwpc_perm0": float(dwpc_perm0),
                "null_n_total": null_stats["n_total"],
                "null_n_nonzero": null_stats["n_nonzero"],
                "null_lambda": null_stats["lambda"],
                "null_nonzero_mean": null_stats["nonzero_mean"],
                "null_nonzero_sd": null_stats["nonzero_sd"],
                "null_alpha": null_stats["alpha"],
                "null_beta": null_stats["beta"],
            }

            # Include original het.io values for comparison
            if "dwpc" in row:
                result["hetio_dwpc"] = row["dwpc"]
            if "p_value" in row:
                result["hetio_pvalue"] = row["p_value"]
            if "dgp_n_dwpcs" in row:
                result["hetio_dgp_n_dwpcs"] = row["dgp_n_dwpcs"]
            if "dgp_n_nonzero_dwpcs" in row:
                result["hetio_dgp_n_nonzero"] = row["dgp_n_nonzero_dwpcs"]
            if "dgp_nonzero_mean" in row:
                result["hetio_dgp_mean"] = row["dgp_nonzero_mean"]
            if "dgp_nonzero_sd" in row:
                result["hetio_dgp_sd"] = row["dgp_nonzero_sd"]

            results.append(result)

    # Create output dataframe
    results_df = pd.DataFrame(results)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Pairs processed: {len(results_df)}")
    print(f"\nHetionet DWPC:")
    print(f"  Non-zero: {(results_df['dwpc_hetionet'] > 0).sum()}/{len(results_df)}")
    print(f"  Mean: {results_df['dwpc_hetionet'].mean():.6f}")
    print(f"  Range: [{results_df['dwpc_hetionet'].min():.6f}, "
          f"{results_df['dwpc_hetionet'].max():.6f}]")
    print(f"\nPerm 0 DWPC:")
    print(f"  Non-zero: {(results_df['dwpc_perm0'] > 0).sum()}/{len(results_df)}")
    print(f"  Mean: {results_df['dwpc_perm0'].mean():.6f}")

    if "hetio_dwpc" in results_df.columns:
        # Compare our DWPC to het.io's
        valid = results_df["hetio_dwpc"].notna() & (results_df["dwpc_hetionet"] > 0)
        if valid.sum() > 0:
            corr = np.corrcoef(
                results_df.loc[valid, "dwpc_hetionet"],
                results_df.loc[valid, "hetio_dwpc"]
            )[0, 1]
            print(f"\nDWPC comparison (our vs het.io):")
            print(f"  Correlation: {corr:.6f}")
            abs_diff = np.abs(
                results_df.loc[valid, "dwpc_hetionet"] -
                results_df.loc[valid, "hetio_dwpc"]
            )
            print(f"  Mean abs diff: {abs_diff.mean():.6f}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute DWPC and null stats for het.io pairs"
    )
    parser.add_argument(
        "--n_perms", type=int, default=20,
        help="Number of permutations for null distribution (default: 20)"
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Input CSV with het.io pairs (default: data/hetio_pairs_for_validation.csv)"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output CSV path (default: results/pvalue_validation/hetio_pair_stats_*.csv)"
    )
    parser.add_argument(
        "--use_hetio_null", action="store_true", default=True,
        help="Use het.io's precomputed null statistics (default: True)"
    )
    parser.add_argument(
        "--compute_own_null", action="store_true",
        help="Compute our own null distribution from permutations"
    )

    args = parser.parse_args()

    use_hetio_null = not args.compute_own_null

    main(
        n_perms=args.n_perms,
        input_file=args.input_file,
        output_file=args.output_file,
        use_hetio_null=use_hetio_null
    )
