"""
Stage 1a: Extract het.io pair list for HPC processing.

This script extracts the list of Gene-BP pairs from the Multi-DWPC het.io
output and saves them to a CSV that can be uploaded to HPC.

Run locally where Multi-DWPC data is available.

Usage:
    python scripts/29a_extract_hetio_pairs.py [--n_pairs N] [--output_file FILE]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Local paths (Multi-DWPC data)
MULTI_DWPC_ROOT = Path(
    "/Users/lucas/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver"
    "/Repositories/Multi-DWPC/Multi-DWPC"
)
HETIO_OUTPUT = (
    MULTI_DWPC_ROOT /
    "output/dwpc_com/res_hetio_bp_go_2016_filt_com_go_w_g_50_250_add_1_25_pct_w_neoj4_ids.csv"
)
GO_MAPPING = MULTI_DWPC_ROOT / "input/hetionet_neo4j_go_ids_nr.csv"
GENE_MAPPING = MULTI_DWPC_ROOT / "input/hetionet_neo4j_genes_ids_nr.csv"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_node_mappings():
    """Load node identifier to index mappings from hetmat."""
    bp_df = pd.read_csv(DATA_DIR / "nodes" / "Biological Process.tsv", sep="\t")
    gene_df = pd.read_csv(DATA_DIR / "nodes" / "Gene.tsv", sep="\t")

    bp_to_idx = dict(zip(bp_df["identifier"], bp_df["position"]))
    gene_to_idx = dict(zip(gene_df["identifier"], gene_df["position"]))

    return bp_to_idx, gene_to_idx


def get_metapath_length(metapath):
    """Get the length of a metapath (number of edges)."""
    length = 0
    in_edge = False
    for char in metapath:
        if char.islower() or char in "<>":
            if not in_edge:
                length += 1
                in_edge = True
        else:
            in_edge = False
    return length


def main(n_pairs=100, output_file=None, min_path_length=1):
    """
    Extract het.io pairs and save for HPC processing.

    Parameters
    ----------
    n_pairs : int
        Number of pairs to extract
    output_file : str
        Output CSV path
    min_path_length : int
        Minimum metapath length (1 for BPpG, 2+ for longer paths)
    """
    if output_file is None:
        output_file = PROJECT_ROOT / "data" / "hetio_pairs_for_validation.csv"
    else:
        output_file = Path(output_file)

    print("=" * 70)
    print("Extracting het.io pairs for HPC validation")
    print("=" * 70)

    # Load het.io data
    print(f"\nLoading het.io data from: {HETIO_OUTPUT}")
    df = pd.read_csv(HETIO_OUTPUT)
    print(f"  Total records: {len(df)}")

    # Load neo4j ID mappings
    print("\nLoading neo4j ID mappings...")
    go_map = pd.read_csv(GO_MAPPING)
    gene_map = pd.read_csv(GENE_MAPPING)

    # Merge to get identifiers
    df = df.merge(go_map, on="neo4j_source_id", how="left")
    df = df.merge(gene_map, on="neo4j_target_id", how="left")

    # Filter by metapath length
    df["metapath_length"] = df["metapath_abbreviation"].apply(get_metapath_length)
    df = df[df["metapath_length"] >= min_path_length]
    print(f"  After filtering length >= {min_path_length}: {len(df)}")

    # Load hetmat node mappings
    print("\nLoading hetmat node mappings...")
    bp_to_idx, gene_to_idx = load_node_mappings()

    # Map to hetmat indices
    df["bp_idx"] = df["go_id"].map(bp_to_idx)
    df["gene_idx"] = df["entrez_gene_id"].map(gene_to_idx)

    # Drop rows where mapping failed
    df_valid = df.dropna(subset=["bp_idx", "gene_idx"])
    df_valid = df_valid.copy()
    df_valid["bp_idx"] = df_valid["bp_idx"].astype(int)
    df_valid["gene_idx"] = df_valid["gene_idx"].astype(int)
    print(f"  After mapping to hetmat indices: {len(df_valid)}")

    # Filter to valid p-values (not NaN, not 0, not 1)
    df_valid = df_valid[
        (df_valid["p_value"].notna()) &
        (df_valid["p_value"] > 0) &
        (df_valid["p_value"] < 1)
    ]
    print(f"  After filtering valid p-values: {len(df_valid)}")

    # Sample if needed
    if len(df_valid) > n_pairs:
        df_sample = df_valid.sample(n=n_pairs, random_state=42)
    else:
        df_sample = df_valid
    print(f"  Final sample size: {len(df_sample)}")

    # Select columns to save
    columns_to_save = [
        "go_id",
        "entrez_gene_id",
        "bp_idx",
        "gene_idx",
        "metapath_abbreviation",
        "metapath_length",
        "path_count",
        "dwpc",
        "p_value",
        "dgp_n_dwpcs",
        "dgp_n_nonzero_dwpcs",
        "dgp_nonzero_mean",
        "dgp_nonzero_sd",
        "dgp_source_degree",
        "dgp_target_degree"
    ]

    # Only include columns that exist
    columns_to_save = [c for c in columns_to_save if c in df_sample.columns]
    df_output = df_sample[columns_to_save].copy()

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_file, index=False)
    print(f"\nSaved {len(df_output)} pairs to: {output_file}")

    # Summary statistics
    print("\nSummary of extracted pairs:")
    print(f"  Metapaths: {df_output['metapath_abbreviation'].nunique()}")
    print(f"  Metapath distribution:")
    print(df_output["metapath_abbreviation"].value_counts().head(10).to_string())
    print(f"\n  P-value range: [{df_output['p_value'].min():.4f}, {df_output['p_value'].max():.4f}]")
    print(f"  Mean p-value: {df_output['p_value'].mean():.4f}")

    return df_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract het.io pairs for HPC validation"
    )
    parser.add_argument(
        "--n_pairs", type=int, default=100,
        help="Number of pairs to extract (default: 100)"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output CSV path (default: data/hetio_pairs_for_validation.csv)"
    )
    parser.add_argument(
        "--min_path_length", type=int, default=1,
        help="Minimum metapath length (default: 1)"
    )

    args = parser.parse_args()
    main(
        n_pairs=args.n_pairs,
        output_file=args.output_file,
        min_path_length=args.min_path_length
    )
