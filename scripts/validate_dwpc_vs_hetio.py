"""
DWPC Validation Script: Compare our calculations to het.io ground truth.

This script validates our DWPC calculation by comparing with known het.io
values from the Multi-DWPC repository output.

Approach:
1. Load neo4j ID mappings from Multi-DWPC
2. Load het.io DWPC results for BP-Gene pairs
3. Calculate our DWPC for the same pairs
4. Compare and report differences
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dwpc_pvalue_validation.data_loading import get_loader
from src.dwpc_pvalue_validation.dwpc_calculation import (
    calculate_dwpc_pairs,
    get_total_node_degrees
)

# Paths
MULTI_DWPC_ROOT = Path("/Users/lucas/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Repositories/Multi-DWPC/Multi-DWPC")
HETIO_OUTPUT = MULTI_DWPC_ROOT / "output/dwpc_com/res_hetio_bp_go_2016_filt_com_go_w_g_50_250_add_1_25_pct_w_neoj4_ids.csv"
NEO4J_GO_MAPPING = MULTI_DWPC_ROOT / "input/hetionet_neo4j_go_ids_nr.csv"
NEO4J_GENE_MAPPING = MULTI_DWPC_ROOT / "input/hetionet_neo4j_genes_ids_nr.csv"


def load_neo4j_mappings():
    """Load neo4j ID mappings for GO terms and genes."""
    go_mapping = pd.read_csv(NEO4J_GO_MAPPING)
    gene_mapping = pd.read_csv(NEO4J_GENE_MAPPING)

    print(f"Loaded {len(go_mapping)} GO term mappings")
    print(f"Loaded {len(gene_mapping)} gene mappings")

    return go_mapping, gene_mapping


def load_our_node_mappings():
    """Load our node index to identifier mappings."""
    data_dir = project_root / "data" / "nodes"

    # Load BP nodes
    bp_df = pd.read_csv(data_dir / "Biological Process.tsv", sep="\t")
    bp_df = bp_df.rename(columns={"position": "our_idx", "identifier": "go_id"})

    # Load Gene nodes
    gene_df = pd.read_csv(data_dir / "Gene.tsv", sep="\t")
    gene_df = gene_df.rename(columns={"position": "our_idx", "identifier": "entrez_gene_id"})
    gene_df["entrez_gene_id"] = gene_df["entrez_gene_id"].astype(int)

    print(f"Loaded {len(bp_df)} BP nodes from our data")
    print(f"Loaded {len(gene_df)} Gene nodes from our data")

    return bp_df, gene_df


def load_hetio_dwpc_results(metapath="GpBP", n_samples=100):
    """
    Load het.io DWPC results for specific metapath.

    Note: het.io uses BPpG (BP to Gene), but our matrix is GpBP (Gene to BP).
    For length-1 undirected paths, DWPC should be symmetric.
    """
    df = pd.read_csv(HETIO_OUTPUT)

    # Filter to specific metapath
    df_metapath = df[df["metapath_abbreviation"] == "BPpG"].copy()

    print(f"Found {len(df_metapath)} BP-Gene pairs with BPpG metapath")

    # Sample for efficiency
    if len(df_metapath) > n_samples:
        df_metapath = df_metapath.sample(n=n_samples, random_state=42)

    return df_metapath


def create_merged_mapping(hetio_df, go_mapping, gene_mapping, our_bp_df, our_gene_df):
    """Create mapping from het.io neo4j IDs to our indices."""

    # Merge het.io data with GO mapping (source is GO term)
    merged = hetio_df.merge(
        go_mapping,
        left_on="neo4j_source_id",
        right_on="neo4j_source_id",
        how="inner"
    )

    # Merge with gene mapping (target is Gene)
    merged = merged.merge(
        gene_mapping,
        left_on="neo4j_target_id",
        right_on="neo4j_target_id",
        how="inner"
    )

    # Merge with our BP mapping to get our indices
    merged = merged.merge(
        our_bp_df[["our_idx", "go_id"]],
        on="go_id",
        how="inner"
    )
    merged = merged.rename(columns={"our_idx": "our_bp_idx"})

    # Merge with our Gene mapping to get our indices
    merged = merged.merge(
        our_gene_df[["our_idx", "entrez_gene_id"]],
        on="entrez_gene_id",
        how="inner"
    )
    merged = merged.rename(columns={"our_idx": "our_gene_idx"})

    print(f"Successfully mapped {len(merged)} pairs to our indices")

    return merged


def calculate_our_dwpc(merged_df, damping_exponent=0.5, use_edge_degrees=True):
    """
    Calculate DWPC using our implementation for the same pairs.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged dataframe with our indices and het.io values.
    damping_exponent : float
        Damping exponent (default 0.5).
    use_edge_degrees : bool
        If True, use edge-specific degrees (matching het.io).
        If False, use total node degrees.
    """

    loader = get_loader()
    hetmat = loader.load_hetmat("true")

    # GpBP: Gene (rows) to Biological Process (cols)
    # For BPpG metapath, we need BP -> Gene direction
    # Our matrix is stored as Gene (source) -> BP (target)
    # So for BP -> Gene, we need the transpose or query [bp, gene]

    # Load GpBP matrix
    adj_matrix = loader.load_edge_matrix("GpBP", "true")

    if use_edge_degrees:
        # IMPORTANT: Het.io uses EDGE-SPECIFIC degrees, not total degrees
        # For GpBP: gene_degree = sum over BP columns, bp_degree = sum over Gene rows
        gene_degrees = np.asarray(adj_matrix.sum(axis=1)).flatten()  # row sums
        bp_degrees = np.asarray(adj_matrix.sum(axis=0)).flatten()    # col sums
        print(f"Using EDGE-SPECIFIC degrees (matching het.io)")
        print(f"  Gene degrees: mean={gene_degrees.mean():.1f}, max={gene_degrees.max()}")
        print(f"  BP degrees: mean={bp_degrees.mean():.1f}, max={bp_degrees.max()}")
    else:
        # Total degrees across all edge types
        gene_degrees = get_total_node_degrees("Gene", "true")
        bp_degrees = get_total_node_degrees("Biological Process", "true")
        print(f"Using TOTAL degrees across all edge types")

    # Calculate DWPC for each pair
    # BPpG means: BP source, Gene target
    # Our matrix GpBP has Gene as source (rows), BP as target (cols)
    # So BPpG traverses GpBP in reverse: A[gene, bp] or A.T[bp, gene]

    dwpcs = []
    for _, row in merged_df.iterrows():
        bp_idx = int(row["our_bp_idx"])
        gene_idx = int(row["our_gene_idx"])

        # For undirected edge GpBP, BPpG is the same edge traversed in reverse
        # A[gene, bp] = A.T[bp, gene]
        # The adjacency value is the same due to symmetry for undirected edges
        edge_val = adj_matrix[gene_idx, bp_idx]

        # Apply damping: deg(bp)^-w * deg(gene)^-w
        # For BPpG direction, BP is source (col sums) and Gene is target (row sums)
        if bp_degrees[bp_idx] > 0 and gene_degrees[gene_idx] > 0:
            damping = (bp_degrees[bp_idx] ** -damping_exponent) * \
                      (gene_degrees[gene_idx] ** -damping_exponent)
            dwpc = edge_val * damping
        else:
            dwpc = 0.0

        dwpcs.append(dwpc)

    return np.array(dwpcs)


def query_hetio_api_pdp(neo4j_source_id, neo4j_target_id, metapath="BPpG"):
    """Query het.io API to get actual PDP (Path Degree Product) value."""
    import requests
    from urllib.parse import quote

    url = (
        f"https://search-api.het.io/v1/paths/"
        f"source/{neo4j_source_id}/target/{neo4j_target_id}/"
        f"metapath/{quote(metapath, safe='')}/?format=json"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Get actual PDP from paths array
        paths = data.get("paths", [])
        if paths:
            pdp = paths[0].get("PDP", 0)
        else:
            pdp = 0

        # Also get dgp degrees for verification
        path_info = data.get("path_count_info", {})
        dgp_source_deg = path_info.get("dgp_source_degree", 0)
        dgp_target_deg = path_info.get("dgp_target_degree", 0)

        return {
            "pdp": pdp,
            "dgp_source_degree": dgp_source_deg,
            "dgp_target_degree": dgp_target_deg
        }
    except Exception as e:
        return {"pdp": None, "error": str(e)}


def main():
    print("=" * 70)
    print("DWPC Validation: Comparing our calculations to het.io")
    print("=" * 70)

    # Load mappings
    print("\nLoading neo4j ID mappings...")
    go_mapping, gene_mapping = load_neo4j_mappings()

    print("\nLoading our node mappings...")
    our_bp_df, our_gene_df = load_our_node_mappings()

    # Load het.io results
    print("\nLoading het.io DWPC results...")
    hetio_df = load_hetio_dwpc_results(n_samples=20)  # Smaller sample for API queries

    # Create merged mapping
    print("\nMerging mappings...")
    merged_df = create_merged_mapping(
        hetio_df, go_mapping, gene_mapping, our_bp_df, our_gene_df
    )

    if len(merged_df) == 0:
        print("ERROR: No pairs could be mapped. Check data compatibility.")
        return

    # Calculate our DWPC
    print("\nCalculating our DWPC values...")
    our_dwpcs = calculate_our_dwpc(merged_df)

    # Query het.io API for actual PDP values
    print("\nQuerying het.io API for true PDP values...")
    print("(Note: 'dwpc' column in CSV is dgp_nonzero_mean, NOT actual DWPC)")

    hetio_pdps = []
    for idx, row in merged_df.iterrows():
        result = query_hetio_api_pdp(
            int(row["neo4j_source_id"]),
            int(row["neo4j_target_id"])
        )
        hetio_pdps.append(result.get("pdp", 0) or 0)
        if (idx + 1) % 5 == 0:
            print(f"  Queried {idx + 1}/{len(merged_df)} pairs...")

    hetio_pdps = np.array(hetio_pdps)

    # Compare results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS (comparing to actual PDP from API)")
    print("=" * 70)

    # Calculate differences
    abs_diff = np.abs(our_dwpcs - hetio_pdps)
    rel_diff = np.where(hetio_pdps > 0, abs_diff / hetio_pdps, 0)

    # Summary statistics
    print(f"\nNumber of pairs compared: {len(merged_df)}")

    print(f"\nHet.io PDP (actual DWPC):")
    print(f"  Mean: {np.mean(hetio_pdps):.6f}")
    print(f"  Std:  {np.std(hetio_pdps):.6f}")
    print(f"  Min:  {np.min(hetio_pdps):.6f}")
    print(f"  Max:  {np.max(hetio_pdps):.6f}")

    print(f"\nOur DWPC:")
    print(f"  Mean: {np.mean(our_dwpcs):.6f}")
    print(f"  Std:  {np.std(our_dwpcs):.6f}")
    print(f"  Min:  {np.min(our_dwpcs):.6f}")
    print(f"  Max:  {np.max(our_dwpcs):.6f}")

    print(f"\nAbsolute Difference:")
    print(f"  Mean: {np.mean(abs_diff):.6f}")
    print(f"  Max:  {np.max(abs_diff):.6f}")

    print(f"\nRelative Difference (for non-zero het.io PDP):")
    nonzero_mask = hetio_pdps > 0
    if nonzero_mask.sum() > 0:
        print(f"  Mean: {np.mean(rel_diff[nonzero_mask]):.4%}")
        print(f"  Max:  {np.max(rel_diff[nonzero_mask]):.4%}")

    # Correlation
    if np.std(our_dwpcs) > 0 and np.std(hetio_pdps) > 0:
        correlation = np.corrcoef(our_dwpcs, hetio_pdps)[0, 1]
        print(f"\nPearson correlation: {correlation:.6f}")

    # Count exact matches (within floating point tolerance)
    exact_matches = np.sum(np.isclose(our_dwpcs, hetio_pdps, rtol=1e-4))
    print(f"\nExact matches (rtol=1e-4): {exact_matches}/{len(merged_df)} ({100*exact_matches/len(merged_df):.1f}%)")

    # Show some examples
    print("\n" + "-" * 70)
    print("Sample comparisons:")
    print("-" * 70)
    sample_df = merged_df.head(10).copy()
    sample_df["hetio_pdp"] = hetio_pdps[:10]
    sample_df["our_dwpc"] = our_dwpcs[:10]
    sample_df["abs_diff"] = abs_diff[:10]

    display_cols = ["go_id", "entrez_gene_id", "hetio_pdp", "our_dwpc", "abs_diff"]
    print(sample_df[display_cols].to_string(index=False))

    # Verdict
    print("\n" + "=" * 70)
    if exact_matches >= len(merged_df) * 0.95:
        print("VALIDATION PASSED: DWPC calculations match het.io PDP values")
        print("Our DWPC formula is correct!")
    elif correlation > 0.99 and np.mean(rel_diff[nonzero_mask]) < 0.01:
        print("VALIDATION MOSTLY PASSED: High correlation with small differences")
        print("Likely due to floating point precision or minor degree calculation differences")
    else:
        print("VALIDATION FAILED: Significant differences from het.io")
        print("Check DWPC calculation implementation")
    print("=" * 70)

    return merged_df, our_dwpcs, hetio_pdps


if __name__ == "__main__":
    main()
