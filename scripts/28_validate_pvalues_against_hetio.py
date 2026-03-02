#!/usr/bin/env python
"""
Validate P-Values Against Het.io Reported Values

This script validates that our p-value calculations match het.io's reported
p-values from https://het.io/search/. This tests the entire pipeline including
degree-stratified null distributions.

Usage:
    python scripts/28_validate_pvalues_against_hetio.py --metapath CbGpPW --n-samples 10
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import requests
import time
from urllib.parse import quote
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dwpc_pvalue_validation import (
    config, utils, data_loading, dwpc_calculation,
    null_distribution, pvalue_calculation
)

logger = utils.setup_logging(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate p-values against het.io"
    )

    parser.add_argument(
        '--metapath',
        type=str,
        required=True,
        help='Metapath to validate (e.g., CbGpPW, CtDaG)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=20,
        help='Number of node pairs to validate (default: 20)'
    )
    parser.add_argument(
        '--null-perms',
        nargs='+',
        type=int,
        default=list(range(1, 21)),
        help='Null permutation indices (default: 1-20)'
    )
    parser.add_argument(
        '--method',
        choices=['gamma_hurdle', 'empirical'],
        default='gamma_hurdle',
        help='P-value calculation method (default: gamma_hurdle)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    return parser.parse_args()


def get_node_identifiers(hetmat, node_type):
    """
    Get node identifiers for a given node type.

    Args:
        hetmat: HetMat object
        node_type: Node type string (e.g., 'Compound', 'Disease')

    Returns:
        list: Node identifiers
    """
    metanode = hetmat.metagraph.get_metanode(node_type)
    identifiers = hetmat.get_node_identifiers(metanode)
    return identifiers


def build_identifier_to_id_mapping(api_url="http://localhost:8015", max_attempts=1000):
    """
    Build a mapping from (identifier, node_type) to integer node IDs from het.io API.

    Args:
        api_url: Base URL for API
        max_attempts: Maximum number of pages to fetch (failsafe, default 1000)

    Returns:
        dict: (identifier, node_type) -> integer ID mapping
    """
    mapping = {}
    url = f"{api_url}/v1/nodes/?limit=100"
    attempts = 0

    logger.info("Building identifier to ID mapping from het.io API...")

    while url and attempts < max_attempts:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            for node in data.get('results', []):
                identifier = node.get('identifier')
                node_id = node.get('id')
                node_type = node.get('metanode')
                if identifier and node_id is not None and node_type:
                    mapping[(str(identifier), node_type)] = node_id

            url = data.get('next')
            attempts += 1

            if attempts % 100 == 0:
                logger.info(f"  Fetched {len(mapping)} node mappings...")

        except Exception as e:
            logger.error(f"Error fetching nodes: {e}")
            break

    logger.info(f"Built mapping with {len(mapping)} node mappings")
    return mapping


def query_hetio_pvalue(source_identifier, target_identifier,
                       source_node_type, target_node_type,
                       metapath, identifier_mapping, max_retries=3,
                       api_url="http://localhost:8015"):
    """
    Query het.io search API for p-value.

    Args:
        source_identifier: Source node identifier (e.g., 'DB00001')
        target_identifier: Target node identifier (e.g., '1234')
        source_node_type: Source node type (e.g., 'Compound')
        target_node_type: Target node type (e.g., 'Gene')
        metapath: Metapath abbreviation (e.g., 'CbG')
        identifier_mapping: Dictionary mapping (identifier, node_type) to integer node IDs
        max_retries: Maximum number of retries for API calls
        api_url: Base URL for API (default: http://localhost:8015 for docker)

    Returns:
        dict with 'dwpc', 'pvalue', 'path_count', or None if not found
    """
    # Lookup integer node IDs from (identifier, node_type) tuples
    source_id = identifier_mapping.get((source_identifier, source_node_type))
    target_id = identifier_mapping.get((target_identifier, target_node_type))

    if source_id is None:
        logger.warning(f"Source ({source_identifier}, {source_node_type}) not found in mapping")
        return None
    if target_id is None:
        logger.warning(f"Target ({target_identifier}, {target_node_type}) not found in mapping")
        return None

    # Het.io search API endpoint
    # Format: {api_url}/v1/paths/source/{source_id}/target/{target_id}/metapath/{metapath}/
    # Note: Uses integer node IDs, not identifiers
    # Default uses local docker (port 8015), can override with https://search-api.het.io

    url = (
        f"{api_url}/v1/paths/"
        f"source/{source_id}/target/{target_id}/metapath/{quote(metapath, safe='')}/"
        "?format=json"
    )

    logger.info(f"  Querying het.io: source_id={source_id}, target_id={target_id}")
    logger.debug(f"  URL: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Het.io API returns path_count_info with p-value and null distribution stats
                # AND paths array with individual path PDPs (observed DWPCs)
                if 'path_count_info' in data:
                    info = data['path_count_info']

                    # Extract observed DWPC from paths array (sum of PDPs)
                    # For single-edge metapaths, usually path_count=0 or 1
                    paths = data.get('paths', [])
                    observed_dwpc = sum(p.get('PDP', 0.0) for p in paths)
                    path_count = len(paths)

                    return {
                        'dwpc': observed_dwpc,  # Use PDP sum, not info['dwpc']
                        'pvalue': info.get('p_value'),
                        'adjusted_pvalue': info.get('adjusted_p_value'),
                        'path_count': path_count,
                        'dgp_source_degree': info.get('dgp_source_degree'),
                        'dgp_target_degree': info.get('dgp_target_degree'),
                        'dgp_n_dwpcs': info.get('dgp_n_dwpcs'),
                        'dgp_nonzero_mean': info.get('dgp_nonzero_mean'),
                        'dgp_nonzero_sd': info.get('dgp_nonzero_sd')
                    }
                else:
                    logger.debug(f"No path_count_info in response for {source_id} -> {target_id}")
                    return None

            elif response.status_code == 404:
                logger.debug(f"No path found: {source_id} -> {target_id}")
                return None

            else:
                logger.warning(f"API returned status {response.status_code}: {response.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None

    return None


def calculate_our_pvalue(metapath, source_idx, target_idx, source_degree, target_degree,
                         null_perms, method='gamma_hurdle'):
    """
    Calculate p-value using our implementation with exact degree grouping.

    Args:
        metapath: Metapath abbreviation
        source_idx: Source node index
        target_idx: Target node index
        source_degree: Source node degree
        target_degree: Target node degree
        null_perms: List of permutation indices for null
        method: 'gamma_hurdle' or 'empirical'

    Returns:
        dict with 'dwpc', 'pvalue', 'null_size'
    """
    # Calculate observed DWPC
    observed_dwpc = dwpc_calculation.calculate_dwpc_metapath(
        metapath,
        [source_idx],
        [target_idx],
        source='true',
        damping_exponent=config.DAMPING_EXPONENT
    )[0]

    # Build sample structure
    sample = {
        'source_idx': source_idx,
        'target_idx': target_idx,
        'source_degree': source_degree,
        'target_degree': target_degree,
        'metapath': metapath
    }

    # Build null distribution using exact degree grouping
    null_by_degree = null_distribution.build_null_distributions_for_samples(
        samples=[sample],
        metapath=metapath,
        perm_indices=null_perms,
        damping_exponent=config.DAMPING_EXPONENT
    )

    degree_key = (source_degree, target_degree)

    if degree_key not in null_by_degree:
        logger.warning(f"No null distribution for degree {degree_key}")
        return None

    null_dwpcs = null_by_degree[degree_key]

    # Calculate p-value
    if method == 'empirical':
        pvalue = pvalue_calculation.calculate_empirical_pvalue(
            observed_dwpc, null_dwpcs
        )
    elif method == 'gamma_hurdle':
        params = pvalue_calculation.fit_gamma_hurdle(null_dwpcs)
        pvalue = pvalue_calculation.calculate_pvalues(
            np.array([observed_dwpc]), params
        )[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        'dwpc': observed_dwpc,
        'pvalue': pvalue,
        'null_size': len(null_dwpcs),
        'null_mean': float(np.mean(null_dwpcs)),
        'null_std': float(np.std(null_dwpcs))
    }


def get_node_degrees(hetmat, metapath, source_idx, target_idx):
    """
    Get degrees for source and target nodes.

    Args:
        hetmat: HetMat object
        metapath: Metapath abbreviation
        source_idx: Source node index
        target_idx: Target node index

    Returns:
        tuple: (source_degree, target_degree)
    """
    from dwpc_pvalue_validation.data_loading import get_metaedges_for_metapath

    metaedges = get_metaedges_for_metapath(metapath)
    first_metaedge = metaedges[0]
    last_metaedge = metaedges[-1]

    loader = data_loading.get_loader()

    source_degrees = loader.get_node_degrees(first_metaedge, 'true', 'source')
    target_degrees = loader.get_node_degrees(last_metaedge, 'true', 'target')

    return int(source_degrees[source_idx]), int(target_degrees[target_idx])


def sample_connected_pairs(metapath, n_samples, random_seed=42):
    """
    Sample node pairs with non-zero DWPC.

    For single-edge metapaths: samples directly from existing edges.
    For multi-edge metapaths: samples random pairs and tests for connectivity.

    Args:
        metapath: Metapath abbreviation
        n_samples: Number of pairs to sample
        random_seed: Random seed

    Returns:
        list: List of dicts with source_idx, target_idx, source_id, target_id,
              source_degree, target_degree, source_node_type, target_node_type, dwpc
    """
    np.random.seed(random_seed)

    from dwpc_pvalue_validation.data_loading import get_metaedges_for_metapath

    metaedges = get_metaedges_for_metapath(metapath)
    first_metaedge = metaedges[0]
    last_metaedge = metaedges[-1]

    loader = data_loading.get_loader()
    hetmat = loader.load_hetmat(source='true')

    # Get metanodes from metaedges
    first_metaedge_obj = hetmat.metagraph.get_metaedge(first_metaedge)
    last_metaedge_obj = hetmat.metagraph.get_metaedge(last_metaedge)

    source_metanode = first_metaedge_obj.source
    target_metanode = last_metaedge_obj.target

    # Get node type names
    source_node_type = source_metanode.identifier
    target_node_type = target_metanode.identifier

    # Get identifiers
    source_ids = hetmat.get_node_identifiers(source_metanode)
    target_ids = hetmat.get_node_identifiers(target_metanode)

    # Get degrees
    source_degrees = loader.get_node_degrees(first_metaedge, 'true', 'source')
    target_degrees = loader.get_node_degrees(last_metaedge, 'true', 'target')

    logger.info(f"Sampling {n_samples} connected pairs from {len(source_ids)} {source_metanode.identifier} x {len(target_ids)} {target_metanode.identifier}")

    pairs = []

    # For single-edge metapaths, sample directly from edges
    if len(metaedges) == 1:
        logger.info(f"  Single-edge metapath - sampling from adjacency matrix")

        # Load adjacency matrix for the single edge
        # metaedge_to_adjacency_matrix returns (source_nodes, target_nodes, matrix)
        _, _, adj_matrix = hetmat.metaedge_to_adjacency_matrix(first_metaedge_obj)

        # Get edges (non-zero entries)
        edges = adj_matrix.nonzero()
        n_edges = len(edges[0])

        logger.info(f"  Found {n_edges} edges")

        if n_edges == 0:
            logger.warning("  No edges found in adjacency matrix!")
            return []

        # Sample n_samples edges
        sample_indices = np.random.choice(n_edges, size=min(n_samples, n_edges), replace=False)

        for idx in sample_indices:
            source_idx = int(edges[0][idx])
            target_idx = int(edges[1][idx])

            # Calculate DWPC for this edge
            dwpc = dwpc_calculation.calculate_dwpc_metapath(
                metapath,
                [source_idx],
                [target_idx],
                source='true',
                damping_exponent=config.DAMPING_EXPONENT
            )[0]

            pairs.append({
                'source_idx': source_idx,
                'target_idx': target_idx,
                'source_id': str(source_ids[source_idx]),
                'target_id': str(target_ids[target_idx]),
                'source_node_type': source_node_type,
                'target_node_type': target_node_type,
                'source_degree': int(source_degrees[source_idx]),
                'target_degree': int(target_degrees[target_idx]),
                'dwpc': dwpc
            })

        logger.info(f"  Sampled {len(pairs)} connected pairs from edges")

    else:
        # Multi-edge metapath: use random sampling with connectivity check
        logger.info(f"  Multi-edge metapath ({len(metaedges)} edges) - random sampling with connectivity check")

        attempts = 0
        max_attempts = n_samples * 1000

        while len(pairs) < n_samples and attempts < max_attempts:
            attempts += 1

            source_idx = np.random.randint(0, len(source_ids))
            target_idx = np.random.randint(0, len(target_ids))

            # Calculate DWPC
            dwpc = dwpc_calculation.calculate_dwpc_metapath(
                metapath,
                [source_idx],
                [target_idx],
                source='true',
                damping_exponent=config.DAMPING_EXPONENT
            )[0]

            if dwpc > 0:
                pairs.append({
                    'source_idx': source_idx,
                    'target_idx': target_idx,
                    'source_id': str(source_ids[source_idx]),
                    'target_id': str(target_ids[target_idx]),
                    'source_node_type': source_node_type,
                    'target_node_type': target_node_type,
                    'source_degree': int(source_degrees[source_idx]),
                    'target_degree': int(target_degrees[target_idx]),
                    'dwpc': dwpc
                })

                if len(pairs) % 5 == 0:
                    logger.info(f"    Found {len(pairs)}/{n_samples} connected pairs (attempt {attempts})")

        logger.info(f"  Sampled {len(pairs)} connected pairs in {attempts} attempts")

    return pairs


def main():
    """Main execution function."""
    args = parse_args()

    logger.info(f"Validating p-values for metapath: {args.metapath}")
    logger.info(f"Using null permutations: {args.null_perms[0]}-{args.null_perms[-1]}")
    logger.info(f"Method: {args.method}")

    # Build identifier to ID mapping from het.io API
    api_url = "http://localhost:8015"
    logger.info(f"\nBuilding node identifier mapping from {api_url}...")
    identifier_mapping = build_identifier_to_id_mapping(api_url=api_url)

    if len(identifier_mapping) == 0:
        logger.error("Failed to build identifier mapping. Is docker running?")
        sys.exit(1)

    # Sample connected pairs
    logger.info(f"\nSampling {args.n_samples} connected node pairs...")
    pairs = sample_connected_pairs(args.metapath, args.n_samples, args.random_seed)

    if len(pairs) == 0:
        logger.error("No connected pairs found. Try increasing --n-samples")
        sys.exit(1)

    # Validate each pair
    results = []

    logger.info(f"\nValidating {len(pairs)} pairs against het.io...")

    for idx, pair in enumerate(pairs):
        logger.info(f"\n[{idx+1}/{len(pairs)}] Validating pair:")
        logger.info(f"  {pair['source_id']} (idx={pair['source_idx']}, deg={pair['source_degree']}) -> "
                   f"{pair['target_id']} (idx={pair['target_idx']}, deg={pair['target_degree']})")
        logger.info(f"  Our DWPC: {pair['dwpc']:.6f}")

        # Query het.io
        hetio_result = query_hetio_pvalue(
            pair['source_id'],
            pair['target_id'],
            pair['source_node_type'],
            pair['target_node_type'],
            args.metapath,
            identifier_mapping,
            api_url=api_url
        )

        if hetio_result is None:
            logger.warning("  Het.io query failed - skipping")
            continue

        logger.info(f"  Het.io DWPC: {hetio_result['dwpc']:.6f}")
        logger.info(f"  Het.io p-value: {hetio_result['pvalue']:.6f}")

        # Calculate our p-value
        our_result = calculate_our_pvalue(
            args.metapath,
            pair['source_idx'],
            pair['target_idx'],
            pair['source_degree'],
            pair['target_degree'],
            args.null_perms,
            args.method
        )

        if our_result is None:
            logger.warning("  Our calculation failed - skipping")
            continue

        logger.info(f"  Our p-value: {our_result['pvalue']:.6f}")
        logger.info(f"  Null distribution: n={our_result['null_size']}, "
                   f"mean={our_result['null_mean']:.6f}, std={our_result['null_std']:.6f}")

        # Compare
        dwpc_diff = abs(pair['dwpc'] - hetio_result['dwpc'])
        pvalue_diff = abs(our_result['pvalue'] - hetio_result['pvalue'])

        logger.info(f"  DWPC difference: {dwpc_diff:.6e}")
        logger.info(f"  P-value difference: {pvalue_diff:.6f}")

        results.append({
            'source_id': pair['source_id'],
            'target_id': pair['target_id'],
            'source_degree': pair['source_degree'],
            'target_degree': pair['target_degree'],
            'our_dwpc': pair['dwpc'],
            'hetio_dwpc': hetio_result['dwpc'],
            'our_pvalue': our_result['pvalue'],
            'hetio_pvalue': hetio_result['pvalue'],
            'dwpc_diff': dwpc_diff,
            'pvalue_diff': pvalue_diff,
            'null_size': our_result['null_size'],
            'dwpc_match': dwpc_diff < 1e-6,
            'pvalue_close': pvalue_diff < 0.1
        })

        # Rate limit
        time.sleep(0.5)

    if not results:
        logger.error("No successful validations")
        sys.exit(1)

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Metapath: {args.metapath}")
    print(f"Method: {args.method}")
    print(f"Null permutations: {args.null_perms[0]}-{args.null_perms[-1]} ({len(args.null_perms)} total)")
    print(f"Samples validated: {len(df)}")
    print(f"\nDWPC Validation:")
    print(f"  Exact matches (diff < 1e-6): {df['dwpc_match'].sum()}/{len(df)}")
    print(f"  Mean absolute difference: {df['dwpc_diff'].mean():.6e}")
    print(f"  Max absolute difference: {df['dwpc_diff'].max():.6e}")
    print(f"\nP-Value Validation:")
    print(f"  Close matches (diff < 0.1): {df['pvalue_close'].sum()}/{len(df)}")
    print(f"  Mean absolute difference: {df['pvalue_diff'].mean():.4f}")
    print(f"  Max absolute difference: {df['pvalue_diff'].max():.4f}")
    print(f"  Correlation: {df[['our_pvalue', 'hetio_pvalue']].corr().iloc[0,1]:.4f}")

    print("\nDetailed Results:")
    print("-"*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df[['source_id', 'target_id', 'source_degree', 'target_degree',
              'our_pvalue', 'hetio_pvalue', 'pvalue_diff', 'pvalue_close']])

    output_file = config.RESULTS_DIR / f"pvalue_validation_{args.metapath}_{args.method}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"\nSaved results to: {output_file}")

    if df['pvalue_close'].mean() > 0.8:
        print("\n" + "="*80)
        print("SUCCESS: P-values closely match het.io!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("WARNING: P-values show significant differences from het.io")
        print("This may be due to using fewer permutations or methodology differences")
        print("="*80)


if __name__ == "__main__":
    main()
