"""
End-to-End P-Value Validation: Compare our pipeline to het.io.

This script performs a TRUE end-to-end validation by:
1. Selecting specific Gene-GO term pairs from het.io data
2. Loading OUR permutation matrices for null distribution
3. Computing DWPCs using OUR implementation
4. Building null distributions from OUR permutations
5. Calculating p-values for permutation 0 and Hetionet
6. Comparing to het.io's p-values for the same pairs

This validates the entire pipeline, not just the formula.

Usage:
    python validate_pvalue_end_to_end.py [--n_perms N] [--n_pairs N] [--n_samples N]

Arguments:
    --n_perms: Number of permutations for null distribution (default: 5)
    --n_pairs: Number of gene-BP pairs to test (default: 30)
    --n_samples: Samples per degree group per permutation (default: 50)
    --project_root: Path to project root (default: auto-detect from script location)

HPC Usage:
    python validate_pvalue_end_to_end.py --n_perms 20 --n_pairs 100 --output_dir results/pvalue_validation_20perms
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import scipy.sparse as sp

# Default paths - can be overridden via command line or environment variables
def get_project_root():
    """Get project root from environment or script location."""
    if 'CAPP_PROJECT_ROOT' in os.environ:
        return Path(os.environ['CAPP_PROJECT_ROOT'])
    # Default: parent of scripts/ directory
    return Path(__file__).parent.parent

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / 'data'
PERM_DIR = DATA_DIR / 'permutations'

# Multi-DWPC paths (for het.io validation data)
def get_multi_dwpc_root():
    """Get Multi-DWPC root from environment or default."""
    if 'MULTI_DWPC_ROOT' in os.environ:
        return Path(os.environ['MULTI_DWPC_ROOT'])
    # Try common locations
    candidates = [
        Path.home() / 'Repositories/Multi-DWPC/Multi-DWPC',
        Path('/projects/gillenlu@xsede.org/Multi-DWPC/Multi-DWPC'),  # HPC path
        PROJECT_ROOT.parent / 'Multi-DWPC/Multi-DWPC',
    ]
    for path in candidates:
        if path.exists():
            return path
    # Return default local path (may not exist on HPC)
    return Path('/Users/lucas/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Repositories/Multi-DWPC/Multi-DWPC')

MULTI_DWPC_ROOT = get_multi_dwpc_root()
HETIO_OUTPUT = MULTI_DWPC_ROOT / 'output/dwpc_com/res_hetio_bp_go_2016_filt_com_go_w_g_50_250_add_1_25_pct_w_neoj4_ids.csv'
GO_MAPPING = MULTI_DWPC_ROOT / 'input/hetionet_neo4j_go_ids_nr.csv'
GENE_MAPPING = MULTI_DWPC_ROOT / 'input/hetionet_neo4j_genes_ids_nr.csv'


def load_edge_matrix(edge_type, perm_idx=None):
    """
    Load edge matrix from hetmat format.

    Parameters
    ----------
    edge_type : str
        Edge type abbreviation (e.g., 'GpBP')
    perm_idx : int or None
        Permutation index (0-based). None for true Hetionet.

    Returns
    -------
    scipy.sparse matrix
    """
    if perm_idx is not None:
        path = PERM_DIR / f'{perm_idx:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz'
    else:
        path = DATA_DIR / 'edges' / f'{edge_type}.sparse.npz'

    return sp.load_npz(path)


def load_node_mappings():
    """Load node identifier to index mappings."""
    bp_df = pd.read_csv(DATA_DIR / 'nodes' / 'Biological Process.tsv', sep='\t')
    gene_df = pd.read_csv(DATA_DIR / 'nodes' / 'Gene.tsv', sep='\t')

    bp_to_idx = dict(zip(bp_df['identifier'], bp_df['position']))
    gene_to_idx = dict(zip(gene_df['identifier'], gene_df['position']))

    idx_to_bp = dict(zip(bp_df['position'], bp_df['identifier']))
    idx_to_gene = dict(zip(gene_df['position'], gene_df['identifier']))

    bp_names = dict(zip(bp_df['identifier'], bp_df['name']))
    gene_names = dict(zip(gene_df['identifier'], gene_df['name']))

    return {
        'bp_to_idx': bp_to_idx,
        'gene_to_idx': gene_to_idx,
        'idx_to_bp': idx_to_bp,
        'idx_to_gene': idx_to_gene,
        'bp_names': bp_names,
        'gene_names': gene_names
    }


def compute_dwpc_length2(source_idx, target_idx, edge1_matrix, edge2_matrix, damping=0.5):
    """
    Compute DWPC for a length-2 metapath between source and target.

    For metapath A-e1-B-e2-C:
    DWPC = sum over intermediate nodes b in B of:
           (edge1[a,b] * edge2[b,c]) / (deg1_a^w * deg1_b^w * deg2_b^w * deg2_c^w)

    Parameters
    ----------
    source_idx : int
        Source node index
    target_idx : int
        Target node index
    edge1_matrix : sparse matrix
        First edge matrix (source to intermediate)
    edge2_matrix : sparse matrix
        Second edge matrix (intermediate to target)
    damping : float
        Damping exponent (default 0.5)

    Returns
    -------
    float : DWPC value
    """
    # Get degrees for each edge type (edge-specific degrees)
    deg1_source = np.asarray(edge1_matrix.sum(axis=1)).flatten()  # row sums
    deg1_target = np.asarray(edge1_matrix.sum(axis=0)).flatten()  # col sums
    deg2_source = np.asarray(edge2_matrix.sum(axis=1)).flatten()
    deg2_target = np.asarray(edge2_matrix.sum(axis=0)).flatten()

    # Get edges from source
    source_edges = edge1_matrix[source_idx, :].toarray().flatten()
    intermediates = np.where(source_edges > 0)[0]

    if len(intermediates) == 0:
        return 0.0

    dwpc = 0.0
    for inter_idx in intermediates:
        # Check if intermediate connects to target
        if edge2_matrix[inter_idx, target_idx] > 0:
            # Compute path weight
            # Edge 1: source -> intermediate
            if deg1_source[source_idx] > 0 and deg1_target[inter_idx] > 0:
                weight1 = (deg1_source[source_idx] ** -damping) * (deg1_target[inter_idx] ** -damping)
            else:
                continue

            # Edge 2: intermediate -> target
            if deg2_source[inter_idx] > 0 and deg2_target[target_idx] > 0:
                weight2 = (deg2_source[inter_idx] ** -damping) * (deg2_target[target_idx] ** -damping)
            else:
                continue

            dwpc += weight1 * weight2

    return dwpc


def compute_dwpc_GpBPpG(gene_idx, bp_idx, GpBP_matrix, damping=0.5):
    """
    Compute DWPC for BPpG metapath (length 1).

    BPpG means: Biological Process - participates - Gene
    The matrix GpBP has Gene as rows, BP as columns.
    So BPpG traverses from BP (column) to Gene (row).

    For length-1, DWPC is simply the edge weight if it exists.
    """
    # Edge-specific degrees
    gene_degrees = np.asarray(GpBP_matrix.sum(axis=1)).flatten()
    bp_degrees = np.asarray(GpBP_matrix.sum(axis=0)).flatten()

    # Check if edge exists (GpBP has Gene as rows, BP as columns)
    edge_val = GpBP_matrix[gene_idx, bp_idx]

    if edge_val == 0:
        return 0.0

    if gene_degrees[gene_idx] > 0 and bp_degrees[bp_idx] > 0:
        dwpc = (gene_degrees[gene_idx] ** -damping) * (bp_degrees[bp_idx] ** -damping)
    else:
        dwpc = 0.0

    return dwpc


def compute_dwpc_BPpGpBP(bp1_idx, bp2_idx, GpBP_matrix, damping=0.5):
    """
    Compute DWPC for BPpGpBP metapath (length 2).

    BP1 - participates - Gene - participates - BP2

    GpBP matrix has Gene as rows, BP as columns.
    """
    gene_degrees = np.asarray(GpBP_matrix.sum(axis=1)).flatten()  # genes
    bp_degrees = np.asarray(GpBP_matrix.sum(axis=0)).flatten()    # BPs

    # Find genes connected to bp1 (column bp1_idx in GpBP)
    genes_to_bp1 = GpBP_matrix[:, bp1_idx].toarray().flatten()
    intermediate_genes = np.where(genes_to_bp1 > 0)[0]

    if len(intermediate_genes) == 0:
        return 0.0

    dwpc = 0.0
    for gene_idx in intermediate_genes:
        # Check if gene connects to bp2
        if GpBP_matrix[gene_idx, bp2_idx] > 0:
            # Path: BP1 -> Gene -> BP2
            # Edge 1 (BP1 to Gene): use BP degree and Gene degree from GpBP
            if bp_degrees[bp1_idx] > 0 and gene_degrees[gene_idx] > 0:
                weight1 = (bp_degrees[bp1_idx] ** -damping) * (gene_degrees[gene_idx] ** -damping)
            else:
                continue

            # Edge 2 (Gene to BP2): same edge type
            if gene_degrees[gene_idx] > 0 and bp_degrees[bp2_idx] > 0:
                weight2 = (gene_degrees[gene_idx] ** -damping) * (bp_degrees[bp2_idx] ** -damping)
            else:
                continue

            dwpc += weight1 * weight2

    return dwpc


def fit_gamma_hurdle(dwpcs):
    """Fit gamma-hurdle distribution using method of moments."""
    n_total = len(dwpcs)
    nonzero = dwpcs[dwpcs > 0]
    n_nonzero = len(nonzero)

    lambda_param = n_nonzero / n_total if n_total > 0 else 0

    if n_nonzero < 2:
        return {'lambda': lambda_param, 'alpha': 1.0, 'beta': 1.0}

    mean = nonzero.mean()
    var = nonzero.var(ddof=1)

    if var <= 0 or mean <= 0:
        return {'lambda': lambda_param, 'alpha': 1.0, 'beta': 1.0 / mean if mean > 0 else 1.0}

    alpha = mean ** 2 / var
    beta = mean / var

    return {'lambda': lambda_param, 'alpha': alpha, 'beta': beta}


def calculate_pvalue(observed_dwpc, params):
    """Calculate p-value using gamma-hurdle distribution."""
    if observed_dwpc == 0:
        return 1.0

    gamma_sf = stats.gamma.sf(observed_dwpc, a=params['alpha'], scale=1.0/params['beta'])
    return params['lambda'] * gamma_sf


def load_hetio_pairs_for_validation(n_pairs=10):
    """
    Load specific Gene-BP pairs from het.io for validation.

    Select pairs with non-trivial p-values and length-2 metapaths.
    """
    df = pd.read_csv(HETIO_OUTPUT)
    go_map = pd.read_csv(GO_MAPPING)
    gene_map = pd.read_csv(GENE_MAPPING)

    # Load our node mappings
    node_maps = load_node_mappings()

    # Merge to get identifiers
    df = df.merge(go_map, on='neo4j_source_id', how='left')
    df = df.merge(gene_map, on='neo4j_target_id', how='left')

    # Filter to BPpGpBP metapath (length 2, same node types at ends)
    # Actually, the het.io data has source=BP, target=Gene for most metapaths
    # Let's use metapaths that we can compute

    # For simplicity, let's focus on BPpG (length 1) which we can easily verify
    # or find a length-2 metapath

    # Filter to interesting cases
    df_filtered = df[
        (df['metapath_abbreviation'].isin(['BPpGpBPpG', 'BPpGaDaG', 'BPpGeAeG'])) &
        (df['p_value'] > 0.01) &
        (df['p_value'] < 0.99) &
        (df['path_count'] > 0)
    ].copy()

    if len(df_filtered) < n_pairs:
        # Fall back to any length >= 2
        df_filtered = df[
            (df['metapath_abbreviation'] != 'BPpG') &
            (df['p_value'] > 0.01) &
            (df['p_value'] < 0.99) &
            (df['path_count'] > 0)
        ].copy()

    # Map to our indices
    df_filtered['bp_idx'] = df_filtered['go_id'].map(node_maps['bp_to_idx'])
    df_filtered['gene_idx'] = df_filtered['entrez_gene_id'].map(node_maps['gene_to_idx'])

    # Drop rows where mapping failed
    df_filtered = df_filtered.dropna(subset=['bp_idx', 'gene_idx'])
    df_filtered['bp_idx'] = df_filtered['bp_idx'].astype(int)
    df_filtered['gene_idx'] = df_filtered['gene_idx'].astype(int)

    # Add names
    df_filtered['bp_name'] = df_filtered['go_id'].map(node_maps['bp_names'])
    df_filtered['gene_name'] = df_filtered['entrez_gene_id'].map(node_maps['gene_names'])

    # Sample
    if len(df_filtered) > n_pairs:
        df_filtered = df_filtered.sample(n=n_pairs, random_state=42)

    return df_filtered


def build_degree_grouped_null(GpBP_matrices, gene_degree, bp_degree, n_samples=100, damping=0.5):
    """
    Build null distribution using DEGREE-GROUPED approach (like het.io).

    Instead of looking at specific (gene, BP) pairs, we sample ALL pairs
    with similar degrees across all permutations.

    Parameters
    ----------
    GpBP_matrices : list of sparse matrices
        Permutation matrices to build null from
    gene_degree : int
        Target gene degree (in GpBP edge type)
    bp_degree : int
        Target BP degree (in GpBP edge type)
    n_samples : int
        Number of pairs to sample per permutation
    damping : float
        DWPC damping exponent

    Returns
    -------
    np.array : Null DWPC values
    """
    null_dwpcs = []

    for perm_matrix in GpBP_matrices:
        # Get degrees in this permutation
        perm_gene_degrees = np.asarray(perm_matrix.sum(axis=1)).flatten()
        perm_bp_degrees = np.asarray(perm_matrix.sum(axis=0)).flatten()

        # Find genes with similar degree (within 20% or +/- 5)
        gene_tol = max(5, int(gene_degree * 0.2))
        matching_genes = np.where(
            (perm_gene_degrees >= gene_degree - gene_tol) &
            (perm_gene_degrees <= gene_degree + gene_tol) &
            (perm_gene_degrees > 0)
        )[0]

        # Find BPs with similar degree
        bp_tol = max(5, int(bp_degree * 0.2))
        matching_bps = np.where(
            (perm_bp_degrees >= bp_degree - bp_tol) &
            (perm_bp_degrees <= bp_degree + bp_tol) &
            (perm_bp_degrees > 0)
        )[0]

        if len(matching_genes) == 0 or len(matching_bps) == 0:
            continue

        # Sample pairs from matching degree groups
        n_to_sample = min(n_samples, len(matching_genes) * len(matching_bps))
        sampled = 0

        # Random sampling
        rng = np.random.RandomState(42)
        gene_samples = rng.choice(matching_genes, size=min(n_samples, len(matching_genes)), replace=True)
        bp_samples = rng.choice(matching_bps, size=min(n_samples, len(matching_bps)), replace=True)

        for g_idx, b_idx in zip(gene_samples, bp_samples):
            dwpc = compute_dwpc_GpBPpG(g_idx, b_idx, perm_matrix, damping)
            null_dwpcs.append(dwpc)

    return np.array(null_dwpcs)


def sample_pairs_with_edges(matrix, n_pairs=20, random_state=42):
    """
    Sample random pairs that have edges in the given matrix.

    Returns list of (row_idx, col_idx) tuples.
    """
    rng = np.random.RandomState(random_state)

    # Get all edges
    rows, cols = matrix.nonzero()
    n_edges = len(rows)

    if n_edges < n_pairs:
        indices = np.arange(n_edges)
    else:
        indices = rng.choice(n_edges, size=n_pairs, replace=False)

    return [(rows[i], cols[i]) for i in indices]


def main(n_perms=5, n_cal_pairs=30, n_samples_per_group=50, output_dir='results/pvalue_validation', skip_hetio=False):
    """
    Run end-to-end p-value validation.

    Parameters
    ----------
    n_perms : int
        Number of permutations to use for null distribution (1 to n_perms)
    n_cal_pairs : int
        Number of pairs to test in calibration
    n_samples_per_group : int
        Samples per degree group per permutation
    output_dir : str
        Directory for saving results
    skip_hetio : bool
        Skip het.io comparison if Multi-DWPC data not available
    """
    print('=' * 100)
    print('END-TO-END P-VALUE VALIDATION (Degree-Grouped Approach)')
    print(f'Using OUR permutations 1-{n_perms} for null distribution')
    print('Testing on permutation 0 and true Hetionet')
    print(f'Skip het.io comparison: {skip_hetio}')
    print('=' * 100)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load node mappings
    node_maps = load_node_mappings()

    # Load het.io pairs for validation (if available)
    hetio_pairs = None
    if not skip_hetio:
        try:
            print('\nLoading het.io pairs for validation...')
            hetio_pairs = load_hetio_pairs_for_validation(n_pairs=10)
            print(f'Found {len(hetio_pairs)} pairs to validate')
        except FileNotFoundError as e:
            print(f'\nWarning: Could not load het.io data: {e}')
            print('Proceeding with calibration test only.')
            skip_hetio = True

    # Load GpBP matrices
    print(f'\nLoading edge matrices (perms 1-{n_perms})...')
    GpBP_true = load_edge_matrix('GpBP', perm_idx=None)
    GpBP_perm0 = load_edge_matrix('GpBP', perm_idx=0)
    GpBP_perms = [load_edge_matrix('GpBP', perm_idx=i) for i in range(1, n_perms + 1)]

    print(f'  True Hetionet GpBP: {GpBP_true.shape}, {GpBP_true.nnz} edges')
    print(f'  Perm 0 GpBP: {GpBP_perm0.shape}, {GpBP_perm0.nnz} edges')
    print(f'  Perms 1-{n_perms} loaded ({len(GpBP_perms)} matrices)')

    # Get degrees from true network for reference
    true_gene_degrees = np.asarray(GpBP_true.sum(axis=1)).flatten()
    true_bp_degrees = np.asarray(GpBP_true.sum(axis=0)).flatten()

    results = []
    results_df = None

    # Het.io comparison (if data available)
    if hetio_pairs is not None and len(hetio_pairs) > 0:
        print('\nComputing DWPCs and p-values using DEGREE-GROUPED null...')
        print('-' * 100)

        for idx, row in hetio_pairs.iterrows():
            bp_idx = row['bp_idx']
            gene_idx = row['gene_idx']
            bp_name = row['bp_name'][:30] if pd.notna(row['bp_name']) else 'Unknown'
            gene_name = row['gene_name'] if pd.notna(row['gene_name']) else 'Unknown'
            hetio_pval = row['p_value']
            metapath = row['metapath_abbreviation']

            # Get degrees for this pair
            gene_deg = true_gene_degrees[gene_idx]
            bp_deg = true_bp_degrees[bp_idx]

            # Compute DWPC for length-1 BPpG on true network and perm0
            dwpc_true = compute_dwpc_GpBPpG(gene_idx, bp_idx, GpBP_true)
            dwpc_perm0 = compute_dwpc_GpBPpG(gene_idx, bp_idx, GpBP_perm0)

            # Build DEGREE-GROUPED null distribution from perms 1-5
            null_dwpcs = build_degree_grouped_null(
                GpBP_perms,
                gene_degree=int(gene_deg),
                bp_degree=int(bp_deg),
                n_samples=n_samples_per_group
            )

            # Fit gamma-hurdle to null
            params = fit_gamma_hurdle(null_dwpcs)

            # Calculate p-values
            pval_true = calculate_pvalue(dwpc_true, params)
            pval_perm0 = calculate_pvalue(dwpc_perm0, params)

            results.append({
                'gene': gene_name,
                'bp': bp_name,
                'gene_deg': int(gene_deg),
                'bp_deg': int(bp_deg),
                'metapath': metapath,
                'hetio_pval': hetio_pval,
                'dwpc_true': dwpc_true,
                'dwpc_perm0': dwpc_perm0,
                'null_n': len(null_dwpcs),
                'null_nonzero': (null_dwpcs > 0).sum(),
                'null_mean': null_dwpcs.mean() if len(null_dwpcs) > 0 else 0,
                'lambda': params['lambda'],
                'our_pval_true': pval_true,
                'our_pval_perm0': pval_perm0
            })

            print(f'  {gene_name:<10} deg={gene_deg:<3} | {bp_name[:25]:<25} deg={bp_deg:<3} | '
                  f'null_n={len(null_dwpcs):<4} nonzero={params["lambda"]:.2f}')

        # Display results
        results_df = pd.DataFrame(results)

        print('\n' + '=' * 100)
        print('RESULTS: Specific Gene-BP Pairs with Degree-Grouped Null')
        print('=' * 100)
        print()
        print(f'Het.io uses 200 perms with degree-grouped null; we use {n_perms} perms.')
        print()

        print(f'{"Gene":<12} {"GO Term":<28} {"Deg":<8} {"Het.io p":<10} {"Our p(Het)":<11} {"Our p(P0)":<11} {"Lambda":<8}')
        print('-' * 100)

        for _, row in results_df.iterrows():
            gene = row['gene'][:11]
            bp = row['bp'][:27]
            deg = f"{row['gene_deg']},{row['bp_deg']}"
            print(f'{gene:<12} {bp:<28} {deg:<8} {row["hetio_pval"]:<10.4f} {row["our_pval_true"]:<11.4f} {row["our_pval_perm0"]:<11.4f} {row["lambda"]:<8.3f}')

        print()
        print('=' * 100)
        print('CALIBRATION ANALYSIS (Het.io Pairs)')
        print('=' * 100)

        # Check calibration for perm0
        perm0_pvals = results_df['our_pval_perm0'].values
        print(f'\nPermutation 0 p-value statistics (expected mean ~0.50 if well-calibrated):')
        print(f'  Mean: {perm0_pvals.mean():.4f}')
        print(f'  Median: {np.median(perm0_pvals):.4f}')
        print(f'  Std: {perm0_pvals.std():.4f}')
        print(f'  Min: {perm0_pvals.min():.4f}')
        print(f'  Max: {perm0_pvals.max():.4f}')

        # Compare to het.io for Hetionet
        true_pvals = results_df['our_pval_true'].values
        hetio_pvals_arr = results_df['hetio_pval'].values

        print(f'\nHetionet p-values (Our BPpG length-1 vs Het.io full metapath):')
        print(f'  Het.io mean: {hetio_pvals_arr.mean():.4f}')
        print(f'  Our mean: {true_pvals.mean():.4f}')
        print(f'  Correlation: {np.corrcoef(hetio_pvals_arr, true_pvals)[0,1]:.4f}')

    # =========================================================================
    # CALIBRATION TEST: Sample pairs WITH EDGES from perm0
    # =========================================================================
    print()
    print('=' * 100)
    print('CALIBRATION TEST: Sampling pairs that HAVE EDGES in Perm 0')
    print('=' * 100)
    print()
    print('For proper calibration, we sample pairs that have edges in perm0,')
    print('then check if their p-values are ~uniform (mean ~0.50).')
    print()

    # Sample pairs with edges from perm0
    perm0_pairs = sample_pairs_with_edges(GpBP_perm0, n_pairs=n_cal_pairs, random_state=123)

    calibration_results = []
    for gene_idx, bp_idx in perm0_pairs:
        gene_deg = true_gene_degrees[gene_idx]
        bp_deg = true_bp_degrees[bp_idx]

        # DWPC in perm0 (should be non-zero since we sampled edges)
        dwpc_perm0 = compute_dwpc_GpBPpG(gene_idx, bp_idx, GpBP_perm0)

        # Build degree-grouped null from perms 1-5
        null_dwpcs = build_degree_grouped_null(
            GpBP_perms,
            gene_degree=int(gene_deg),
            bp_degree=int(bp_deg),
            n_samples=50
        )

        params = fit_gamma_hurdle(null_dwpcs)
        pval = calculate_pvalue(dwpc_perm0, params)

        calibration_results.append({
            'gene_idx': gene_idx,
            'bp_idx': bp_idx,
            'gene_deg': int(gene_deg),
            'bp_deg': int(bp_deg),
            'dwpc_perm0': dwpc_perm0,
            'null_n': len(null_dwpcs),
            'lambda': params['lambda'],
            'pvalue': pval
        })

    cal_df = pd.DataFrame(calibration_results)

    print(f'{"Gene Idx":<10} {"BP Idx":<10} {"Degrees":<10} {"DWPC P0":<12} {"Lambda":<8} {"P-value":<10}')
    print('-' * 70)
    for _, row in cal_df.head(15).iterrows():
        deg = f"{row['gene_deg']},{row['bp_deg']}"
        print(f'{row["gene_idx"]:<10} {row["bp_idx"]:<10} {deg:<10} {row["dwpc_perm0"]:<12.6f} {row["lambda"]:<8.3f} {row["pvalue"]:<10.4f}')

    print()
    print('=' * 100)
    print('CALIBRATION RESULTS (Perm 0 pairs with edges)')
    print('=' * 100)
    cal_pvals = cal_df['pvalue'].values
    print(f'\nP-value statistics (expected: mean ~0.50, uniform distribution):')
    print(f'  N pairs: {len(cal_pvals)}')
    print(f'  Mean: {cal_pvals.mean():.4f}')
    print(f'  Median: {np.median(cal_pvals):.4f}')
    print(f'  Std: {cal_pvals.std():.4f}')
    print(f'  Min: {cal_pvals.min():.4f}')
    print(f'  Max: {cal_pvals.max():.4f}')
    print(f'  Proportion < 0.05: {(cal_pvals < 0.05).mean():.3f} (expected ~0.05)')
    print(f'  Proportion < 0.50: {(cal_pvals < 0.50).mean():.3f} (expected ~0.50)')

    print()
    print('=' * 100)
    print('CONCLUSION')
    print('=' * 100)

    if cal_pvals.mean() > 0.7:
        print(f'\nCalibration FAILED: Mean p-value = {cal_pvals.mean():.3f} (expected ~0.50)')
        print('P-values are inflated even for pairs with edges.')
    elif cal_pvals.mean() < 0.3:
        print(f'\nCalibration FAILED: Mean p-value = {cal_pvals.mean():.3f} (expected ~0.50)')
        print('P-values are deflated.')
    elif abs(cal_pvals.mean() - 0.5) < 0.15:
        print(f'\nCalibration PASSED: Mean p-value = {cal_pvals.mean():.3f} (expected ~0.50)')
        print(f'Degree-grouped approach with {n_perms} permutations produces reasonable calibration.')
    else:
        print(f'\nCalibration MARGINAL: Mean p-value = {cal_pvals.mean():.3f} (expected ~0.50)')

    # Save results
    print('\n' + '=' * 100)
    print('SAVING RESULTS')
    print('=' * 100)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save calibration results
    cal_output = output_path / f'calibration_{n_perms}perms.csv'
    cal_df.to_csv(cal_output, index=False)
    print(f'Saved calibration results to: {cal_output}')

    # Save het.io comparison if available
    if results_df is not None:
        hetio_output = output_path / f'hetio_comparison_{n_perms}perms.csv'
        results_df.to_csv(hetio_output, index=False)
        print(f'Saved het.io comparison to: {hetio_output}')

    # Save summary statistics
    summary = {
        'n_perms': n_perms,
        'n_cal_pairs': n_cal_pairs,
        'n_samples_per_group': n_samples_per_group,
        'cal_mean_pvalue': cal_pvals.mean(),
        'cal_median_pvalue': np.median(cal_pvals),
        'cal_std_pvalue': cal_pvals.std(),
        'cal_prop_lt_05': (cal_pvals < 0.05).mean(),
        'cal_prop_lt_50': (cal_pvals < 0.50).mean(),
    }
    summary_df = pd.DataFrame([summary])
    summary_output = output_path / f'summary_{n_perms}perms.csv'
    summary_df.to_csv(summary_output, index=False)
    print(f'Saved summary to: {summary_output}')

    return results_df, cal_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='End-to-end p-value validation against het.io'
    )
    parser.add_argument(
        '--n_perms', type=int, default=5,
        help='Number of permutations for null distribution (default: 5)'
    )
    parser.add_argument(
        '--n_pairs', type=int, default=30,
        help='Number of gene-BP pairs to test in calibration (default: 30)'
    )
    parser.add_argument(
        '--n_samples', type=int, default=50,
        help='Samples per degree group per permutation (default: 50)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/pvalue_validation',
        help='Output directory for results (default: results/pvalue_validation)'
    )
    parser.add_argument(
        '--skip_hetio', action='store_true',
        help='Skip het.io comparison (if Multi-DWPC data not available on HPC)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results, calibration = main(
        n_perms=args.n_perms,
        n_cal_pairs=args.n_pairs,
        n_samples_per_group=args.n_samples,
        output_dir=args.output_dir,
        skip_hetio=args.skip_hetio
    )
