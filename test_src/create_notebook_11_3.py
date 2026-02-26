#!/usr/bin/env python3
"""
Create notebook 11.3: Full permutation analysis with Option A method.

Uses notebook 11 structure with 20 permutations + Option A function from 11.2.
"""

import json
from pathlib import Path


def load_notebook(path):
    """Load a notebook as JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(notebook, path):
    """Save notebook as JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


def create_notebook_11_3():
    """Create notebook 11.3 from templates."""

    print("Loading template notebooks...")

    # Load notebook 11 as structure template
    nb_11 = load_notebook('notebooks/11_degree_conditioned_compositionality.ipynb')

    # Load notebook 11.2 to get Option A function
    nb_11_2 = load_notebook('notebooks/11.2_empirical_vs_analytical_compositional.ipynb')

    # Create new notebook based on nb_11
    nb_11_3 = {
        'cells': [],
        'metadata': nb_11['metadata'],
        'nbformat': nb_11['nbformat'],
        'nbformat_minor': nb_11['nbformat_minor']
    }

    print("Building notebook 11.3...")

    # Copy cells from nb_11 but modify key cells
    for i, cell in enumerate(nb_11['cells']):

        # Cell 0: Replace header
        if i == 0:
            new_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '# Full Permutation Analysis with Option A Method\n',
                    '\n',
                    '**Purpose:** Statistical comparison of Hetionet vs 20 degree-preserving permutations using corrected Option A probabilistic combination.\n',
                    '\n',
                    '**Methodological Context:**\n',
                    '- **[Notebook 11.1](11.1_empirical_vs_analytical_compositional.ipynb):** OLD broken summation method (probabilities > 1.0)\n',
                    '- **[Notebook 11.2](11.2_empirical_vs_analytical_compositional.ipynb):** CORRECTED Option A method (single network demo)\n',
                    '- **This notebook (11.3):** Full 20-permutation analysis with Option A\n',
                    '- **[Notebook 11](11_degree_conditioned_compositionality.ipynb):** Production version (same analysis)\n',
                    '\n',
                    '**Formula (CORRECTED):**\n',
                    '```\n',
                    'P(compound→pathway) = 1 - ∏_gene (1 - P(compound→gene) × P(gene→pathway))\n',
                    '```\n',
                    '\n',
                    'All probabilities guaranteed ≤ 1.0.\n',
                    '\n',
                    '## Research Questions\n',
                    '\n',
                    '1. **Is compositionality degree-dependent?** Does PMI vary systematically with node degrees?\n',
                    '2. **Are null (permuted) metapaths compositional?** Do degree-preserving permutations preserve or break compositionality?\n',
                    '3. **Does Option A method produce valid, reproducible results?**'
                ]
            }
            nb_11_3['cells'].append(new_cell)

        # Cell 5: Replace load_edge_matrix to include from notebook 11
        elif i == 5:
            # Get both functions from nb_11 cell 5
            nb_11_3['cells'].append(cell)

        # Cell 6: Update header
        elif i == 6:
            new_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## Option A Probabilistic Combination Function\n',
                    '\n',
                    'This function uses the corrected Option A formula (from notebook 11.2) that guarantees all probabilities ≤ 1.0.'
                ]
            }
            nb_11_3['cells'].append(new_cell)

        # Cell 7: Replace with Option A function from 11.2
        elif i == 7:
            # Get the compute function from nb_11_2 (cell 7)
            option_a_cell = nb_11_2['cells'][7]
            nb_11_3['cells'].append(option_a_cell)

        # All other cells: copy as-is
        else:
            nb_11_3['cells'].append(cell)

    # Now add degree-colored scatter plots at the end (before conclusions)
    # Find the "Save Results" markdown cell
    save_results_idx = None
    for i, cell in enumerate(nb_11_3['cells']):
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            if '## Save Results' in source:
                save_results_idx = i
                break

    if save_results_idx:
        # Insert degree scatter plots before "Save Results"
        scatter_header = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Degree-Stratified Scatter Plots\n',
                '\n',
                'Visualize how node degrees affect compositional predictions using Option A method.\n',
                'All probabilities remain ≤ 1.0 (mathematically valid).'
            ]
        }

        # Get scatter plot code from nb_11_2 (cell 13)
        scatter_code = nb_11_2['cells'][13]

        nb_11_3['cells'].insert(save_results_idx, scatter_header)
        nb_11_3['cells'].insert(save_results_idx + 1, scatter_code)

        print(f"  Inserted degree scatter plots at cells {save_results_idx}-{save_results_idx+1}")

    # Update save results cell to use different filenames
    for cell in nb_11_3['cells']:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                source_str = ''.join(source)
            else:
                source_str = source

            # Update filenames in save commands
            if 'hetionet_degree_results.csv' in source_str:
                source_str = source_str.replace(
                    'hetionet_degree_results.csv',
                    'hetionet_degree_results_OPTION_A_FULL.csv'
                )
                source_str = source_str.replace(
                    'null_degree_results.csv',
                    'null_degree_results_OPTION_A_FULL.csv'
                )
                source_str = source_str.replace(
                    'degree_conditioned_summary.csv',
                    'degree_conditioned_summary_OPTION_A_FULL.csv'
                )
                source_str = source_str.replace(
                    'degree_conditioned_analysis.png',
                    'option_a_full_degree_conditioned_analysis.png'
                )
                source_str = source_str.replace(
                    'pmi_heatmap_FIXED.png',
                    'option_a_full_pmi_heatmap.png'
                )
                # Properly split while preserving newlines
                lines = source_str.split('\n')
                # Add back newlines to all but last line
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

    # Save notebook
    output_path = Path('notebooks/11.3_empirical_vs_analytical_compositional.ipynb')
    save_notebook(nb_11_3, output_path)

    print(f"\nCreated: {output_path}")
    print(f"Total cells: {len(nb_11_3['cells'])}")

    return output_path


if __name__ == '__main__':
    print("="*70)
    print("CREATING NOTEBOOK 11.3 (FULL PERMUTATION ANALYSIS - OPTION A)")
    print("="*70)

    try:
        notebook_path = create_notebook_11_3()

        print("\n" + "="*70)
        print("SUCCESS: Notebook 11.3 created")
        print("="*70)
        print("\nNotebook includes:")
        print("  - Option A probabilistic combination (corrected method)")
        print("  - Analysis of 20 permutations + Hetionet")
        print("  - Statistical comparisons (Mann-Whitney, t-test)")
        print("  - Degree-stratified analysis")
        print("  - PMI histograms and heatmaps")
        print("  - Degree-colored scatter plots")
        print("\nExpected runtime: ~26 minutes")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise
