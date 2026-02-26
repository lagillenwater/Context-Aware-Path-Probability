#!/usr/bin/env python
"""
Add the missing compute_metapath_compositionality_old() function to notebook 11.1.

This function implements the OLD broken summation method that produces probabilities > 1.0.
"""

import json
from pathlib import Path


OLD_FUNCTION_CODE = '''def compute_metapath_compositionality_old(edge1_matrix, edge2_matrix, perm_id):
    """
    Compute compositionality analysis using OLD SUMMATION METHOD (BROKEN).

    WARNING: This function implements a BROKEN method that produces probabilities > 1.0.
    It is included for educational purposes to demonstrate the mathematical error.

    The OLD formula uses simple summation:
        P(compound→pathway) = Σ_gene P(compound→gene) × P(gene→pathway)

    This treats multiple gene pathways as additive, which violates probability axioms
    and produces invalid probabilities > 1.0.

    DO NOT USE THIS METHOD FOR PRODUCTION ANALYSIS.
    See notebook 11.2 for the corrected Option A method.

    Parameters
    ----------
    edge1_matrix : scipy.sparse.csr_matrix
        First edge matrix (e.g., CbG: Compounds × Genes)
    edge2_matrix : scipy.sparse.csr_matrix
        Second edge matrix (e.g., GpPW: Genes × Pathways)
    perm_id : int
        Permutation ID for tracking

    Returns
    -------
    results_df : pandas.DataFrame
        Results with compositional probabilities (may exceed 1.0!)

    Notes
    -----
    This implementation is intentionally flawed to demonstrate the bug.
    Expected result: Some probabilities will exceed 1.0, proving the method is invalid.
    """
    print(f"DEBUG: OLD SUMMATION METHOD CALLED (Perm {perm_id})")

    # Align gene dimensions
    assert edge1_matrix.shape[1] == edge2_matrix.shape[0], "Gene dimension mismatch!"

    # Filter zero-degree compounds and pathways
    compound_degrees = np.array(edge1_matrix.sum(axis=1)).flatten()
    pathway_degrees = np.array(edge2_matrix.sum(axis=0)).flatten()

    compound_nonzero = np.where(compound_degrees > 0)[0]
    pathway_nonzero = np.where(pathway_degrees > 0)[0]

    edge1_aligned = edge1_matrix[compound_nonzero, :]
    edge2_aligned = edge2_matrix[:, pathway_nonzero]

    n_compounds = edge1_aligned.shape[0]
    n_pathways = edge2_aligned.shape[1]

    # Compute metapath matrix
    metapath_matrix = edge1_aligned @ edge2_aligned

    # 1. Compute observed frequencies
    observed_freq = {}
    for i, j in zip(*metapath_matrix.nonzero()):
        compound_genes = edge1_aligned.getrow(i).nonzero()[1]
        pathway_genes = edge2_aligned.getcol(j).nonzero()[0]
        shared_genes = set(compound_genes) & set(pathway_genes)

        n_paths = len(shared_genes)
        n_possible = len(compound_genes)

        if n_possible > 0:
            observed_freq[(i, j)] = n_paths / n_possible

    # 2. Compute compositional probabilities using analytical prior
    def analytical_prior(u, v, m):
        """Analytical formula for edge probability."""
        uv = u * v
        denominator = np.sqrt(uv**2 + (m - u - v + 1)**2)
        return uv / denominator if denominator > 0 else 0.0

    # Compute edge priors
    edge1_priors = {}
    edge2_priors = {}

    # Edge1 priors (CbG)
    n_edges_edge1 = edge1_aligned.nnz
    source_degrees = np.array(edge1_aligned.sum(axis=1)).flatten()
    target_degrees = np.array(edge1_aligned.sum(axis=0)).flatten()

    rows, cols = edge1_aligned.nonzero()
    for i, j in zip(rows, cols):
        u, v = source_degrees[i], target_degrees[j]
        if u > 0 and v > 0:
            edge1_priors[(i, j)] = analytical_prior(u, v, n_edges_edge1)

    # Edge2 priors (GpPW)
    n_edges_edge2 = edge2_aligned.nnz
    source_degrees = np.array(edge2_aligned.sum(axis=1)).flatten()
    target_degrees = np.array(edge2_aligned.sum(axis=0)).flatten()

    rows, cols = edge2_aligned.nonzero()
    for i, j in zip(rows, cols):
        u, v = source_degrees[i], target_degrees[j]
        if u > 0 and v > 0:
            edge2_priors[(i, j)] = analytical_prior(u, v, n_edges_edge2)

    # 3. Compute compositional probabilities using OLD SUMMATION (BROKEN!)
    compositional_prob = {}
    prob_over_1_count = 0

    for i in range(n_compounds):
        compound_genes = edge1_aligned.getrow(i).nonzero()[1]

        for j in range(n_pathways):
            pathway_genes = edge2_aligned.getcol(j).nonzero()[0]
            shared_genes = set(compound_genes) & set(pathway_genes)

            if shared_genes:
                # OLD METHOD: Simple summation (BROKEN - can exceed 1.0!)
                # P(compound→pathway) = Σ_gene P(compound→gene) × P(gene→pathway)
                total_prob = 0.0

                for gene in shared_genes:
                    p_edge1 = edge1_priors.get((i, gene), 0.0)
                    p_edge2 = edge2_priors.get((gene, j), 0.0)
                    individual_prob = p_edge1 * p_edge2
                    total_prob += individual_prob  # BUG: Simple addition!

                if total_prob > 1.0:
                    prob_over_1_count += 1

                if total_prob > 0:
                    compositional_prob[(i, j)] = total_prob

    print(f"OLD METHOD RESULT: {prob_over_1_count} probabilities > 1.0 (INVALID!)")

    # 4. Compute PMI and create results
    results_data = []
    common_pairs = set(observed_freq.keys()) & set(compositional_prob.keys())

    for pair in common_pairs:
        i, j = pair
        p_observed = observed_freq[pair]
        p_compositional = compositional_prob[pair]

        # PMI calculation
        if p_observed > 0 and p_compositional > 0:
            pmi = np.log2(p_observed / p_compositional)
        else:
            pmi = np.nan

        # Get degrees (map back to original indices)
        orig_compound_idx = compound_nonzero[i]
        orig_pathway_idx = pathway_nonzero[j]

        compound_degree = compound_degrees[orig_compound_idx]
        pathway_degree = pathway_degrees[orig_pathway_idx]

        results_data.append({
            'perm_id': perm_id,
            'compound_idx': orig_compound_idx,
            'pathway_idx': orig_pathway_idx,
            'compound_degree': int(compound_degree),
            'pathway_degree': int(pathway_degree),
            'observed_freq': p_observed,
            'compositional_prob': p_compositional,
            'pmi': pmi,
            'residual': p_observed - p_compositional
        })

    print(f"Returning {len(results_data)} results (OLD SUMMATION METHOD)")
    return pd.DataFrame(results_data)
'''


def add_function_to_notebook():
    """Add the missing OLD function to notebook 11.1."""
    notebook_path = Path('notebooks/11.1_empirical_vs_analytical_compositional.ipynb')

    print("="*70)
    print("ADDING MISSING FUNCTION TO NOTEBOOK 11.1")
    print("="*70)

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Find cell 7 (should be empty)
    cells = notebook.get('cells', [])
    if len(cells) > 7:
        cell7 = cells[7]

        # Add the function code to cell 7
        cell7['source'] = OLD_FUNCTION_CODE
        cell7['cell_type'] = 'code'
        cell7['execution_count'] = None
        cell7['outputs'] = []

        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        print("\nSuccess: Added compute_metapath_compositionality_old() to cell 7")
        print("This function implements the BROKEN summation method for demonstration.")
        print("\nThe notebook can now be executed to show probabilities > 1.0")
    else:
        print("\nError: Notebook doesn't have enough cells")

    print("="*70)


if __name__ == '__main__':
    add_function_to_notebook()
