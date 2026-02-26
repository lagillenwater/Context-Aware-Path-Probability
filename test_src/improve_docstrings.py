#!/usr/bin/env python
"""
Improve docstrings in notebooks to include mathematical explanations and references.

This script enhances function docstrings with:
1. Mathematical formula explanations
2. References to the analytical prior derivation
3. Clear parameter and return value descriptions
"""

import json
import re
from pathlib import Path


IMPROVED_COMPUTE_FUNCTION_DOCSTRING = '''def compute_metapath_compositionality(edge1_matrix, edge2_matrix, perm_id):
    """
    Compute compositionality analysis for a metapath using Option A probabilistic combination.

    This function calculates compositional probabilities for metapaths (e.g., Compound→Gene→Pathway)
    using the Option A formula that ensures all probabilities remain ≤ 1.0:

        P(compound→pathway) = 1 - ∏_gene (1 - P(compound→gene) × P(gene→pathway))

    This approach treats multiple gene pathways as independent alternative routes with redundancy,
    which is biologically realistic and mathematically valid.

    The function computes:
    1. Observed frequencies: Fraction of compound-pathway pairs connected through each gene
    2. Compositional probabilities: Using analytical priors for individual edges
    3. PMI (Pointwise Mutual Information): log2(observed / compositional)
    4. Residuals: observed - compositional

    Analytical Prior Formula
    ------------------------
    Edge probabilities are estimated using the analytical prior:

        P(u,v) = (deg_u × deg_v) / sqrt((deg_u × deg_v)² + (m - deg_u - deg_v + 1)²)

    where:
    - deg_u, deg_v are the degrees of nodes u and v
    - m is the total number of edges in the network

    This formula approximates the probability that an edge exists between nodes u and v
    in a random graph with the observed degree sequence.

    Reference: Himmelstein et al. (2017) systematic integration of biomedical knowledge
    prioritizes drugs for repurposing. eLife. https://doi.org/10.7554/eLife.26726

    Parameters
    ----------
    edge1_matrix : scipy.sparse.csr_matrix
        First edge matrix (e.g., CbG: Compounds × Genes)
        Rows represent source nodes (e.g., compounds)
        Columns represent target nodes (e.g., genes)
    edge2_matrix : scipy.sparse.csr_matrix
        Second edge matrix (e.g., GpPW: Genes × Pathways)
        Rows must correspond to edge1_matrix columns (genes)
        Columns represent final target nodes (e.g., pathways)
    perm_id : int
        Permutation ID for tracking (0 for Hetionet, 1-200 for permutations)

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame with one row per metapath pair containing:
        - perm_id : int - Permutation identifier
        - compound_idx : int - Index of source node (compound)
        - pathway_idx : int - Index of target node (pathway)
        - compound_degree : int - Degree of source node
        - pathway_degree : int - Degree of target node
        - observed_freq : float - Empirical frequency of connection
        - compositional_prob : float - Predicted probability using Option A
        - pmi : float - Pointwise mutual information
        - residual : float - observed_freq - compositional_prob

    Notes
    -----
    This implementation uses Option A probabilistic combination. For comparison with the
    broken summation method, see notebook 11.1.

    The Option A formula automatically enforces an upper bound of 1.0 on probabilities,
    unlike simple summation which can produce invalid probabilities > 1.0.

    See Also
    --------
    Notebook 11.1 : Demonstrates the broken OLD summation method
    Notebook 11.2 : Full implementation and validation of this corrected method
    """'''


IMPROVED_ANALYTICAL_PRIOR_DOCSTRING = '''    def analytical_prior(u, v, m):
        """
        Compute analytical edge probability prior based on node degrees.

        This formula estimates P(edge exists | degrees, network size) using a
        geometric mean normalization that accounts for the total network size.

        Formula:
            P(u,v) = (u × v) / sqrt((u × v)² + (m - u - v + 1)²)

        Intuition:
        - Numerator (u × v): Higher-degree nodes are more likely to connect
        - Denominator: Normalizes by network size and degree constraints
        - Result: Probability ∈ [0, 1] that increases with node degrees

        Parameters
        ----------
        u : int
            Degree of source node
        v : int
            Degree of target node
        m : int
            Total number of edges in the network

        Returns
        -------
        float
            Edge probability in range [0, 1]

        References
        ----------
        Himmelstein et al. (2017) Systematic integration of biomedical knowledge
        prioritizes drugs for repurposing. eLife. https://doi.org/10.7554/eLife.26726
        """'''


IMPROVED_LOAD_EDGE_MATRIX_DOCSTRING = '''def load_edge_matrix(edge_type: str, perm_id: int = 0) -> sp.csr_matrix:
    """
    Load edge matrix for given edge type and permutation.

    Loads sparse adjacency matrices in hetmat format from the permutations directory.
    Permutation 000 corresponds to the real Hetionet network, while permutations
    001-200 are degree-preserving randomized networks generated using XSwap.

    Parameters
    ----------
    edge_type : str
        Edge type code (e.g., 'CbG' for Compound-binds-Gene)
        Format: {source_abbrev}{relationship_abbrev}{target_abbrev}
    perm_id : int, default=0
        Permutation identifier (0 for Hetionet, 1-200 for degree-preserving permutations)

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix in CSR format
        Rows: source nodes
        Columns: target nodes
        Non-zero entries: edges

    Notes
    -----
    File structure: data/permutations/{perm_id:03d}.hetmat/edges/{edge_type}.sparse.npz
    """'''


IMPROVED_BIN_DEGREES_DOCSTRING = '''def bin_degrees(df: pd.DataFrame, bins=DEGREE_BINS, labels=DEGREE_LABELS):
    """
    Add degree bin columns to DataFrame for stratified analysis.

    Creates categorical degree bins for both source and target nodes, enabling
    degree-stratified correlation analysis. Uses ordered categories to ensure
    proper heatmap orientation (low values in lower-left corner).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'compound_degree' and 'pathway_degree' columns
    bins : list of float, default=DEGREE_BINS
        Bin edges for pd.cut (e.g., [0, 5, 20, 100, np.inf])
    labels : list of str, default=DEGREE_LABELS
        Bin labels (e.g., ['Very Low (0-5)', 'Low (5-20)', ...])

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added columns:
        - compound_degree_bin : ordered categorical
        - pathway_degree_bin : ordered categorical

    Notes
    -----
    Ordered categories ensure proper heatmap rendering with low degree bins
    appearing in the lower-left corner of degree-stratified heatmaps.
    """'''


def improve_function_docstrings(source_code):
    """
    Replace basic docstrings with improved versions containing mathematical explanations.

    Parameters
    ----------
    source_code : str
        Python source code

    Returns
    -------
    str
        Source code with improved docstrings
    """
    # Replace compute_metapath_compositionality docstring
    pattern = r'def compute_metapath_compositionality\(edge1_matrix, edge2_matrix, perm_id\):\s*"""[^"]*"""'
    if re.search(pattern, source_code, re.DOTALL):
        source_code = re.sub(pattern, IMPROVED_COMPUTE_FUNCTION_DOCSTRING, source_code, flags=re.DOTALL)

    # Replace analytical_prior docstring
    pattern = r'def analytical_prior\(u, v, m\):\s*"""[^"]*"""'
    if re.search(pattern, source_code, re.DOTALL):
        source_code = re.sub(pattern, IMPROVED_ANALYTICAL_PRIOR_DOCSTRING, source_code, flags=re.DOTALL)

    # Replace load_edge_matrix docstring
    pattern = r'def load_edge_matrix\(edge_type: str, perm_id: int = 0\) -> sp\.csr_matrix:\s*"""[^"]*"""'
    if re.search(pattern, source_code, re.DOTALL):
        source_code = re.sub(pattern, IMPROVED_LOAD_EDGE_MATRIX_DOCSTRING, source_code, flags=re.DOTALL)

    # Replace bin_degrees docstring
    pattern = r'def bin_degrees\(df: pd\.DataFrame, bins=DEGREE_BINS, labels=DEGREE_LABELS\):\s*"""[^"]*"""'
    if re.search(pattern, source_code, re.DOTALL):
        source_code = re.sub(pattern, IMPROVED_BIN_DEGREES_DOCSTRING, source_code, flags=re.DOTALL)

    return source_code


def process_notebook(notebook_path):
    """
    Process a Jupyter notebook to improve docstrings.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook file

    Returns
    -------
    int
        Number of cells modified
    """
    print(f"\nProcessing: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified_count = 0

    # Process each code cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            original_source = ''.join(cell.get('source', []))
            new_source = improve_function_docstrings(original_source)

            if new_source != original_source:
                cell['source'] = new_source
                modified_count += 1

    # Write back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"  Cells modified: {modified_count}")

    return modified_count


def main():
    """Improve docstrings in all notebooks in the series."""
    repo_dir = Path(__file__).parent
    notebooks_dir = repo_dir / 'notebooks'

    notebook_files = [
        '11_degree_conditioned_compositionality.ipynb',
        '11.1_empirical_vs_analytical_compositional.ipynb',
        '11.2_empirical_vs_analytical_compositional.ipynb',
        '11.3_empirical_vs_analytical_compositional.ipynb',
    ]

    print("="*70)
    print("IMPROVING FUNCTION DOCSTRINGS")
    print("="*70)

    total_modified = 0

    for notebook_file in notebook_files:
        notebook_path = notebooks_dir / notebook_file

        if notebook_path.exists():
            modified = process_notebook(notebook_path)
            total_modified += modified
        else:
            print(f"\nSkipping: {notebook_file} (not found)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total cells modified: {total_modified}")
    print("\nFunction docstrings have been enhanced with:")
    print("  - Mathematical formula explanations")
    print("  - References to Himmelstein et al. (2017)")
    print("  - Clear parameter descriptions")
    print("  - Biological and mathematical intuition")
    print("="*70)


if __name__ == '__main__':
    main()
