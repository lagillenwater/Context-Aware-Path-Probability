#!/usr/bin/env python
"""
Replace verbose headers with concise, professional summaries.

The previous headers were too long (50+ lines). Greene Lab standards require
concise, professional documentation - not encyclopedia entries.
"""

import json
from pathlib import Path


# Concise headers following Greene Lab "concise and professional" standard
CONCISE_HEADERS = {
    '11_degree_conditioned_compositionality.ipynb': """# Degree-Conditioned Compositionality Analysis

Production analysis testing whether PMI depends on node degrees and comparing Hetionet with degree-preserving permutations. Uses Option A probabilistic combination method. See notebooks 11.1 (old broken method) and 11.2 (corrected method) for methodological comparison.

## Research Questions

1. Is compositionality degree-dependent? Does PMI vary systematically with node degrees?
2. Are null (permuted) metapaths compositional? Do degree-preserving permutations preserve or break compositionality?
""",

    '11.1_empirical_vs_analytical_compositional.ipynb': """# Empirical vs Analytical Compositional Analysis (OLD METHOD)

**Purpose:** Demonstrates the broken OLD summation method that produces probabilities > 1.0

**Status:** DEPRECATED - For comparison purposes only. See notebook 11.2 for corrected method.

**Formula (BROKEN):** P(compound→pathway) = Σ_gene P(compound→gene) × P(gene→pathway)

This simple summation violates probability axioms by treating multiple pathways as additive, producing invalid probabilities exceeding 1.0 (up to ~3.2 in analysis).
""",

    '11.2_empirical_vs_analytical_compositional.ipynb': """# Empirical vs Analytical Compositional Analysis (OPTION A - CORRECTED)

**Purpose:** Demonstrates the corrected Option A probabilistic combination method

**Formula (CORRECTED):** P(compound→pathway) = 1 - Π_gene (1 - P(compound→gene) × P(gene→pathway))

Treats multiple pathways as independent alternative routes, ensuring all probabilities ≤ 1.0. Compare with notebook 11.1 (old broken method).

Reference: Himmelstein et al. (2017) eLife. https://doi.org/10.7554/eLife.26726
""",

    '11.3_empirical_vs_analytical_compositional.ipynb': """# Empirical vs Analytical Compositional Analysis (DEGREE-STRATIFIED)

**Purpose:** Degree-stratified correlation analysis using pre-computed data from notebook 11

Loads results from notebook 11's degree-conditioned analysis and computes correlations within degree bins to reveal where the independence assumption (compositionality) works vs fails.

Validates expected PMI-correlation relationship: High PMI bins → Low correlations
"""
}


def restore_concise_header(notebook_path, new_header):
    """
    Replace verbose header with concise version.

    Parameters
    ----------
    notebook_path : Path
        Path to notebook file
    new_header : str
        Concise header text

    Returns
    -------
    bool
        True if updated
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook.get('cells', [])
    if not cells or cells[0].get('cell_type') != 'markdown':
        return False

    # Replace first cell
    cells[0]['source'] = new_header

    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    return True


def main():
    """Restore concise headers to all notebooks."""
    repo_dir = Path(__file__).parent
    notebooks_dir = repo_dir / 'notebooks'

    print("="*70)
    print("RESTORING CONCISE HEADERS")
    print("="*70)
    print("\nReplacing verbose documentation with professional summaries")
    print("Greene Lab standard: Concise and readable\n")

    for notebook_file, header_text in CONCISE_HEADERS.items():
        notebook_path = notebooks_dir / notebook_file

        if notebook_path.exists():
            print(f"Updating: {notebook_file}")
            restored = restore_concise_header(notebook_path, header_text)
            if restored:
                lines = len(header_text.split('\n'))
                print(f"  New header: {lines} lines (concise)")
            else:
                print(f"  Warning: Could not update")
        else:
            print(f"Skipping: {notebook_file} (not found)")

    print("\n" + "="*70)
    print("CONCISE HEADERS RESTORED")
    print("="*70)
    print("Notebooks now have professional 5-10 line headers")
    print("Verbose documentation moved to NOTEBOOK_11_SERIES_IMPROVEMENTS.md")
    print("="*70)


if __name__ == '__main__':
    main()
