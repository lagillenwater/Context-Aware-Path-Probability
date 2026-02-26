"""
Get actual PDP (observed DWPC) values from het.io API.

The hetio_pairs_for_validation.csv file has the dwpc column set to
dgp_nonzero_mean (null distribution mean), NOT the actual observed DWPC.
This script queries the het.io API to get the actual PDP values.

Usage:
    python scripts/30a_get_hetio_pdp.py [--input_file FILE] [--output_file FILE]

    # Query all pairs for all metapaths in input file
    python scripts/30a_get_hetio_pdp.py --all_pairs

    # Query all pairs for specific metapaths
    python scripts/30a_get_hetio_pdp.py --all_pairs --metapaths BPpGpBPpG,BPpGiGpBP
"""

import argparse
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Multi-DWPC repository paths (for neo4j ID mappings)
MULTI_DWPC_ROOT = Path(
    "/Users/lucas/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver"
    "/Repositories/Multi-DWPC/Multi-DWPC"
)


def load_neo4j_mappings():
    """Load neo4j ID mappings from Multi-DWPC repository."""
    go_map_path = MULTI_DWPC_ROOT / "input/hetionet_neo4j_go_ids_nr.csv"
    gene_map_path = MULTI_DWPC_ROOT / "input/hetionet_neo4j_genes_ids_nr.csv"

    if not go_map_path.exists() or not gene_map_path.exists():
        print("WARNING: Neo4j mapping files not found. Cannot query API.")
        print(f"  Expected GO mapping: {go_map_path}")
        print(f"  Expected Gene mapping: {gene_map_path}")
        return None, None

    go_map = pd.read_csv(go_map_path)
    gene_map = pd.read_csv(gene_map_path)

    return go_map, gene_map


def query_hetio_api_pdp(neo4j_source_id, neo4j_target_id, metapath, max_retries=3):
    """
    Query het.io API to get actual PDP (Path Degree Product) value.

    Parameters
    ----------
    neo4j_source_id : int
        Neo4j node ID for source node
    neo4j_target_id : int
        Neo4j node ID for target node
    metapath : str
        Metapath abbreviation (e.g., 'BPpG', 'BPpGpBPpG')
    max_retries : int
        Maximum number of retry attempts

    Returns
    -------
    dict with keys: pdp, path_count, dgp_source_degree, dgp_target_degree, error
    """
    url = (
        f"https://search-api.het.io/v1/paths/"
        f"source/{int(neo4j_source_id)}/target/{int(neo4j_target_id)}/"
        f"metapath/{quote(metapath, safe='')}/?format=json"
    )

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Get actual PDP from paths array
            paths = data.get("paths", [])
            if paths:
                # Sum PDP across all paths to get total DWPC
                total_pdp = sum(p.get("PDP", 0) for p in paths)
            else:
                total_pdp = 0

            # Get path count info
            path_info = data.get("path_count_info", {})

            return {
                "pdp": total_pdp,
                "path_count": path_info.get("path_count", 0),
                "dgp_source_degree": path_info.get("dgp_source_degree", 0),
                "dgp_target_degree": path_info.get("dgp_target_degree", 0),
                "p_value_api": path_info.get("p_value", None),
                "error": None
            }

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"pdp": None, "error": "timeout"}

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"pdp": None, "error": str(e)}

    return {"pdp": None, "error": "max_retries_exceeded"}


def main(input_file=None, output_file=None, rate_limit_delay=0.1,
         all_pairs=False, metapaths=None):
    """
    Get actual PDP values from het.io API for validation pairs.

    Parameters
    ----------
    input_file : str
        Path to input CSV with het.io pairs
    output_file : str
        Path for output CSV with actual PDP values
    rate_limit_delay : float
        Delay between API calls in seconds
    all_pairs : bool
        If True, query all unique pairs for all (or specified) metapaths
    metapaths : list of str
        List of metapath abbreviations to query. If None and all_pairs=True,
        uses all metapaths from input file.
    """
    if input_file is None:
        input_file = DATA_DIR / "hetio_pairs_for_validation.csv"
    else:
        input_file = Path(input_file)

    if output_file is None:
        output_file = DATA_DIR / "hetio_pairs_with_actual_pdp.csv"
    else:
        output_file = Path(output_file)

    print("=" * 70)
    print("Getting actual PDP values from het.io API")
    print("=" * 70)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"All pairs mode: {all_pairs}")
    if metapaths:
        print(f"Metapaths to query: {metapaths}")

    # Load input pairs
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records from input file")

    # Load neo4j mappings
    print("\nLoading neo4j ID mappings...")
    go_map, gene_map = load_neo4j_mappings()

    if go_map is None:
        print("ERROR: Cannot proceed without neo4j mappings")
        return

    # Load het.io CSV to get neo4j IDs
    hetio_csv_path = (
        MULTI_DWPC_ROOT /
        "output/dwpc_com/res_hetio_bp_go_2016_filt_com_go_w_g_50_250_add_1_25_pct_w_neoj4_ids.csv"
    )

    if not hetio_csv_path.exists():
        print(f"ERROR: Het.io CSV not found: {hetio_csv_path}")
        return

    hetio_df = pd.read_csv(hetio_csv_path)
    print(f"Loaded {len(hetio_df)} records from het.io CSV")

    # Merge to get neo4j IDs for our pairs
    # First merge with GO mapping
    hetio_df = hetio_df.merge(go_map, on="neo4j_source_id", how="left")
    hetio_df = hetio_df.merge(gene_map, on="neo4j_target_id", how="left")

    # Determine metapaths to query
    if metapaths:
        target_metapaths = metapaths
    else:
        target_metapaths = df["metapath_abbreviation"].unique().tolist()
    print(f"\nMetapaths to query: {len(target_metapaths)}")
    for mp in target_metapaths:
        print(f"  - {mp}")

    # Create pairs to query
    if all_pairs:
        # Get unique pairs (by go_id and entrez_gene_id)
        unique_pairs = df[["go_id", "entrez_gene_id", "bp_idx", "gene_idx"]].drop_duplicates()
        print(f"\nAll pairs mode: {len(unique_pairs)} unique pairs x {len(target_metapaths)} metapaths")

        # Create cross-product of pairs x metapaths
        rows = []
        for _, pair_row in unique_pairs.iterrows():
            for mp in target_metapaths:
                rows.append({
                    "go_id": pair_row["go_id"],
                    "entrez_gene_id": pair_row["entrez_gene_id"],
                    "bp_idx": pair_row["bp_idx"],
                    "gene_idx": pair_row["gene_idx"],
                    "metapath_abbreviation": mp
                })
        df_to_query = pd.DataFrame(rows)
        print(f"Total combinations to query: {len(df_to_query)}")
    else:
        df_to_query = df.copy()

    # Match with het.io data to get neo4j IDs
    # We need neo4j IDs for each pair (independent of metapath)
    pair_neo4j_map = hetio_df[["go_id", "entrez_gene_id",
                               "neo4j_source_id", "neo4j_target_id"]].drop_duplicates()

    df_with_neo4j = df_to_query.merge(
        pair_neo4j_map,
        on=["go_id", "entrez_gene_id"],
        how="left"
    )

    matched = df_with_neo4j["neo4j_source_id"].notna().sum()
    print(f"Matched {matched}/{len(df_with_neo4j)} pairs to neo4j IDs")

    # Query API for actual PDP values
    print(f"\nQuerying het.io API for {matched} pairs...")
    print("(This may take a few minutes)")

    actual_pdp = []
    actual_path_count = []
    api_p_values = []
    dgp_source_degrees = []
    dgp_target_degrees = []
    errors = []

    for idx, row in df_with_neo4j.iterrows():
        if pd.isna(row.get("neo4j_source_id")):
            actual_pdp.append(np.nan)
            actual_path_count.append(np.nan)
            api_p_values.append(np.nan)
            dgp_source_degrees.append(np.nan)
            dgp_target_degrees.append(np.nan)
            errors.append("no_neo4j_mapping")
            continue

        result = query_hetio_api_pdp(
            row["neo4j_source_id"],
            row["neo4j_target_id"],
            row["metapath_abbreviation"]
        )

        actual_pdp.append(result.get("pdp"))
        actual_path_count.append(result.get("path_count"))
        api_p_values.append(result.get("p_value_api"))
        dgp_source_degrees.append(result.get("dgp_source_degree"))
        dgp_target_degrees.append(result.get("dgp_target_degree"))
        errors.append(result.get("error"))

        if (idx + 1) % 10 == 0:
            success = sum(1 for e in errors if e is None)
            print(f"  Processed {idx + 1}/{len(df_with_neo4j)} pairs ({success} successful)")

        time.sleep(rate_limit_delay)

    # Add results to dataframe
    df_with_neo4j["actual_pdp"] = actual_pdp
    df_with_neo4j["actual_path_count"] = actual_path_count
    df_with_neo4j["api_p_value"] = api_p_values
    df_with_neo4j["dgp_source_degree"] = dgp_source_degrees
    df_with_neo4j["dgp_target_degree"] = dgp_target_degrees
    df_with_neo4j["api_error"] = errors

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_with_neo4j.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = df_with_neo4j["actual_pdp"].notna().sum()
    print(f"Successfully queried: {successful}/{len(df_with_neo4j)}")

    if successful > 0:
        print(f"\nActual PDP (DWPC) statistics:")
        print(f"  Mean: {df_with_neo4j['actual_pdp'].mean():.6f}")
        print(f"  Std: {df_with_neo4j['actual_pdp'].std():.6f}")
        print(f"  Min: {df_with_neo4j['actual_pdp'].min():.6f}")
        print(f"  Max: {df_with_neo4j['actual_pdp'].max():.6f}")

        # Compare to original dwpc column if it exists (only in non-all_pairs mode)
        if "dwpc" in df_with_neo4j.columns:
            print(f"\nOriginal 'dwpc' column (dgp_nonzero_mean) statistics:")
            print(f"  Mean: {df_with_neo4j['dwpc'].mean():.6f}")
            print(f"  Std: {df_with_neo4j['dwpc'].std():.6f}")

            # Correlation check
            valid_mask = df_with_neo4j["actual_pdp"].notna()
            if valid_mask.sum() > 2:
                corr = np.corrcoef(
                    df_with_neo4j.loc[valid_mask, "actual_pdp"],
                    df_with_neo4j.loc[valid_mask, "dwpc"]
                )[0, 1]
                print(f"\nCorrelation (actual_pdp vs dgp_nonzero_mean): {corr:.4f}")

    return df_with_neo4j


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get actual PDP values from het.io API"
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Input CSV with het.io pairs"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output CSV with actual PDP values"
    )
    parser.add_argument(
        "--rate_limit", type=float, default=0.1,
        help="Delay between API calls in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--all_pairs", action="store_true",
        help="Query all unique pairs for all metapaths (cross-product)"
    )
    parser.add_argument(
        "--metapaths", type=str, default=None,
        help="Comma-separated list of metapath abbreviations to query"
    )

    args = parser.parse_args()

    # Parse metapaths if provided
    metapaths_list = None
    if args.metapaths:
        metapaths_list = [mp.strip() for mp in args.metapaths.split(",")]

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        rate_limit_delay=args.rate_limit,
        all_pairs=args.all_pairs,
        metapaths=metapaths_list
    )
