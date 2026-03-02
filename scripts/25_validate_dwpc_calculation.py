#!/usr/bin/env python
"""
Validate DWPC Calculation Against Het.io

This script validates that our DWPC calculations match het.io's Neo4j database
by comparing a small sample of node pairs. This is a sanity check to ensure
the calculation methodology is correct before interpreting p-value calibration.

Usage:
    python scripts/25_validate_dwpc_calculation.py --metapath CbGpPW --n-samples 10
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dwpc_pvalue_validation import config, utils, data_loading, dwpc_calculation

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("WARNING: neo4j package not installed")
    print("Install with: conda run -n CAPP pip install neo4j")
    sys.exit(1)

logger = utils.setup_logging(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate DWPC calculations against het.io"
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
        default=10,
        help='Number of node pairs to validate (default: 10)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Data directory (default: data/)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    return parser.parse_args()


def get_metapath_info(metapath_abbrev):
    """
    Get metapath information from abbreviation.

    Args:
        metapath_abbrev: String like 'CbGpPW'

    Returns:
        dict with keys: 'abbreviation', 'node_types', 'edge_abbrevs',
                       'cypher_pattern', 'full_name'
    """
    metapath_mapping = {
        'CbGpPW': {
            'node_types': ['Compound', 'Gene', 'Pathway'],
            'edge_abbrevs': ['CbG', 'GpPW'],
            'relationships': ['BINDS_CbG', 'PARTICIPATES_GpPW'],
            'full_name': 'Compound-binds-Gene-participates-Pathway',
            'cypher_pattern': '(c:Compound)-[:BINDS_CbG]-(g:Gene)-[:PARTICIPATES_GpPW]-(p:Pathway)'
        },
        'CtDaG': {
            'node_types': ['Compound', 'Disease', 'Gene'],
            'edge_abbrevs': ['CtD', 'DaG'],
            'relationships': ['TREATS_CtD', 'ASSOCIATES_DaG'],
            'full_name': 'Compound-treats-Disease-associates-Gene',
            'cypher_pattern': '(c:Compound)-[:TREATS_CtD]-(d:Disease)-[:ASSOCIATES_DaG]-(g:Gene)'
        },
        'GiGaD': {
            'node_types': ['Gene', 'Gene', 'Disease'],
            'edge_abbrevs': ['GiG', 'GaD'],
            'relationships': ['INTERACTS_GiG', 'ASSOCIATES_GaD'],
            'full_name': 'Gene-interacts-Gene-associates-Disease',
            'cypher_pattern': '(g1:Gene)-[:INTERACTS_GiG]-(g2:Gene)-[:ASSOCIATES_GaD]-(d:Disease)'
        },
        'CbGpPWpG': {
            'node_types': ['Compound', 'Gene', 'Pathway', 'Gene'],
            'edge_abbrevs': ['CbG', 'GpPW', 'PWpG'],
            'relationships': ['BINDS_CbG', 'PARTICIPATES_GpPW', 'PARTICIPATES_PWpG'],
            'full_name': 'Compound-binds-Gene-participates-Pathway-participates-Gene',
            'cypher_pattern': '(c:Compound)-[:BINDS_CbG]-(g1:Gene)-[:PARTICIPATES_GpPW]-(pw:Pathway)-[:PARTICIPATES_GpPW]-(g2:Gene)'
        },
        'CtDaGiG': {
            'node_types': ['Compound', 'Disease', 'Gene', 'Gene'],
            'edge_abbrevs': ['CtD', 'DaG', 'GiG'],
            'relationships': ['TREATS_CtD', 'ASSOCIATES_DaG', 'INTERACTS_GiG'],
            'full_name': 'Compound-treats-Disease-associates-Gene-interacts-Gene',
            'cypher_pattern': '(c:Compound)-[:TREATS_CtD]-(d:Disease)-[:ASSOCIATES_DaG]-(g1:Gene)-[:INTERACTS_GiG]-(g2:Gene)'
        },
        'CbGpPWpGaD': {
            'node_types': ['Compound', 'Gene', 'Pathway', 'Gene', 'Disease'],
            'edge_abbrevs': ['CbG', 'GpPW', 'PWpG', 'GaD'],
            'relationships': ['BINDS_CbG', 'PARTICIPATES_GpPW', 'PARTICIPATES_PWpG', 'ASSOCIATES_GaD'],
            'full_name': 'Compound-binds-Gene-participates-Pathway-participates-Gene-associates-Disease',
            'cypher_pattern': '(c:Compound)-[:BINDS_CbG]-(g1:Gene)-[:PARTICIPATES_GpPW]-(pw:Pathway)-[:PARTICIPATES_GpPW]-(g2:Gene)-[:ASSOCIATES_GaD]-(d:Disease)'
        }
    }

    if metapath_abbrev not in metapath_mapping:
        logger.error(f"Unknown metapath: {metapath_abbrev}")
        logger.error(f"Supported metapaths: {list(metapath_mapping.keys())}")
        return None

    info = metapath_mapping[metapath_abbrev]
    info['abbreviation'] = metapath_abbrev
    return info


def query_hetio_dwpc(driver, source_name, target_name, metapath_info, damping=0.5):
    """
    Query het.io Neo4j database for DWPC between two nodes.

    Args:
        driver: Neo4j driver instance
        source_name: Source node identifier (e.g., DB00001)
        target_name: Target node identifier (e.g., PC7_1234)
        metapath_info: Dictionary from get_metapath_info
        damping: Damping exponent (default 0.5)

    Returns:
        tuple: (dwpc_value, path_count) or (None, None) if query fails
    """
    pattern = metapath_info['cypher_pattern']

    parts = pattern.replace('(', '|').replace(')', '|').split('|')
    parts = [p.strip() for p in parts if ':' in p]

    source_var = parts[0].split(':')[0]
    target_var = parts[-1].split(':')[0]

    query = f"""
    MATCH path = {pattern}
    WHERE {source_var}.identifier = $source_name
      AND {target_var}.identifier = $target_name
    WITH path,
         [node in nodes(path) | size((node)--())] as degrees
    WITH collect(degrees) as all_degrees
    RETURN
        size(all_degrees) as path_count,
        reduce(total = 0.0, deg_list in all_degrees |
            total + reduce(pdp = 1.0, d in deg_list | pdp * d ^ $damping)
        ) as dwpc
    """

    try:
        with driver.session() as session:
            result = session.run(
                query,
                source_name=source_name,
                target_name=target_name,
                damping=-damping
            )
            record = result.single()

            if record:
                return record['dwpc'], record['path_count']
            else:
                return 0.0, 0

    except Exception as e:
        logger.error(f"Neo4j query failed: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Parameters: source={source_name}, target={target_name}")
        return None, None


def calculate_our_dwpc(metapath, source_idx, target_idx):
    """
    Calculate DWPC using our implementation.

    Args:
        metapath: Metapath abbreviation string (e.g., 'CbGpPW')
        source_idx: Source node index
        target_idx: Target node index

    Returns:
        float: DWPC value
    """
    dwpcs = dwpc_calculation.calculate_dwpc_metapath(
        metapath,
        [source_idx],
        [target_idx],
        source='true',
        damping_exponent=config.DAMPING_EXPONENT
    )

    return dwpcs[0]


def get_node_names_from_hetmat(hetmat, node_type):
    """
    Get node names from hetmat for a given node type.

    Args:
        hetmat: HetMat object
        node_type: Node type string (e.g., 'Compound', 'Gene')

    Returns:
        list: Node identifiers
    """
    try:
        metanode = hetmat.metagraph.get_metanode(node_type)
        identifiers = hetmat.get_node_identifiers(metanode)
        return identifiers
    except Exception as e:
        logger.error(f"Error getting nodes for {node_type}: {e}")
        raise


def sample_node_pairs(data_loader, metapath_info, n_samples, random_seed=42):
    """
    Sample random node pairs for validation.

    Args:
        data_loader: HetionetLoader instance
        metapath_info: Dictionary from get_metapath_info
        n_samples: Number of pairs to sample
        random_seed: Random seed

    Returns:
        list: List of (source_idx, target_idx, source_name, target_name) tuples
    """
    np.random.seed(random_seed)

    source_type = metapath_info['node_types'][0]
    target_type = metapath_info['node_types'][-1]

    hetmat = data_loader.load_hetmat(source='true')

    source_nodes = get_node_names_from_hetmat(hetmat, source_type)
    target_nodes = get_node_names_from_hetmat(hetmat, target_type)

    n_source = len(source_nodes)
    n_target = len(target_nodes)

    logger.info(f"Sampling {n_samples} pairs from {n_source} {source_type} x {n_target} {target_type}")

    pairs = []
    for _ in range(n_samples):
        source_idx = np.random.randint(0, n_source)
        target_idx = np.random.randint(0, n_target)

        source_name = source_nodes[source_idx]
        target_name = target_nodes[target_idx]

        pairs.append((source_idx, target_idx, source_name, target_name))

    return pairs


def main():
    """Main execution function."""
    args = parse_args()

    logger.info(f"Validating DWPC calculation for metapath: {args.metapath}")

    metapath_info = get_metapath_info(args.metapath)
    if metapath_info is None:
        sys.exit(1)

    logger.info(f"Full metapath: {metapath_info['full_name']}")

    logger.info("Loading data...")
    data_loader = data_loading.get_loader()

    logger.info(f"Sampling {args.n_samples} node pairs...")
    pairs = sample_node_pairs(
        data_loader,
        metapath_info,
        args.n_samples,
        args.random_seed
    )

    logger.info("Connecting to het.io Neo4j database...")
    try:
        driver = GraphDatabase.driver("bolt://neo4j.het.io:7687")
        logger.info("Connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        sys.exit(1)

    results = []

    logger.info(f"\nFirst pass: Finding pairs with non-zero DWPC...")
    candidate_pairs = []

    for idx, (source_idx, target_idx, source_name, target_name) in enumerate(pairs):
        our_dwpc = calculate_our_dwpc(
            args.metapath,
            source_idx,
            target_idx
        )

        if our_dwpc > 0:
            candidate_pairs.append((source_idx, target_idx, source_name, target_name, our_dwpc))
            logger.debug(f"  [{idx+1}/{args.n_samples}] {source_name} → {target_name}: DWPC={our_dwpc:.6f}")

    logger.info(f"Found {len(candidate_pairs)} pairs with non-zero DWPC out of {args.n_samples} sampled")

    if len(candidate_pairs) == 0:
        logger.error("No non-zero DWPC pairs found. Try increasing --n-samples")
        driver.close()
        sys.exit(1)

    logger.info(f"\nSecond pass: Validating {len(candidate_pairs)} non-zero pairs against het.io...")

    for idx, (source_idx, target_idx, source_name, target_name, our_dwpc) in enumerate(candidate_pairs):
        logger.info(f"\n[{idx+1}/{len(candidate_pairs)}] Validating pair:")
        logger.info(f"  Source: {source_name} (idx={source_idx})")
        logger.info(f"  Target: {target_name} (idx={target_idx})")
        logger.info(f"  Our DWPC: {our_dwpc:.6f}")

        hetio_dwpc, path_count = query_hetio_dwpc(
            driver,
            source_name,
            target_name,
            metapath_info,
            damping=config.DAMPING_EXPONENT
        )

        if hetio_dwpc is not None:
            logger.info(f"  Het.io DWPC: {hetio_dwpc:.6f}")
            logger.info(f"  Path count: {path_count}")

            abs_diff = abs(our_dwpc - hetio_dwpc)
            rel_diff = abs_diff / max(hetio_dwpc, 1e-10) if hetio_dwpc > 0 else np.nan

            logger.info(f"  Absolute difference: {abs_diff:.6f}")
            if not np.isnan(rel_diff):
                logger.info(f"  Relative difference: {rel_diff:.2%}")

            results.append({
                'source_name': source_name,
                'target_name': target_name,
                'source_idx': source_idx,
                'target_idx': target_idx,
                'our_dwpc': our_dwpc,
                'hetio_dwpc': hetio_dwpc,
                'path_count': path_count,
                'abs_difference': abs_diff,
                'rel_difference': rel_diff,
                'match': abs_diff < 1e-6
            })
        else:
            logger.warning("  Het.io query failed - skipping")

    driver.close()

    if not results:
        logger.error("No successful validations")
        sys.exit(1)

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Metapath: {metapath_info['full_name']}")
    print(f"Samples validated: {len(df)}")
    print(f"\nMatches (abs diff < 1e-6): {df['match'].sum()}/{len(df)}")
    print(f"Mean absolute difference: {df['abs_difference'].mean():.6e}")
    print(f"Max absolute difference: {df['abs_difference'].max():.6e}")
    print(f"Mean relative difference: {df['rel_difference'].mean():.2%}")

    print("\nDetailed Results:")
    print("-"*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df[['source_name', 'target_name', 'our_dwpc', 'hetio_dwpc',
              'abs_difference', 'rel_difference', 'match']])

    output_file = config.RESULTS_DIR / f"dwpc_validation_{args.metapath}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"\nSaved results to: {output_file}")

    if df['match'].all():
        print("\n" + "="*80)
        print("SUCCESS: All DWPC calculations match het.io!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("WARNING: Some DWPC calculations do not match!")
        print("Check the detailed results above.")
        print("="*80)


if __name__ == "__main__":
    main()
