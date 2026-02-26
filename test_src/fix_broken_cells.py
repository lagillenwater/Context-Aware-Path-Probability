#!/usr/bin/env python
"""
Fix cells broken by the docstring improvement script.

The regex replacement removed newlines after closing triple quotes,
causing syntax errors. This script fixes the formatting.
"""

import json
import re
from pathlib import Path


def fix_cell_formatting(source_code):
    """
    Fix broken formatting where closing triple quotes are
    followed immediately by code.

    Parameters
    ----------
    source_code : str
        Python source code

    Returns
    -------
    str
        Fixed source code
    """
    # Fix pattern: triple-quotes + word (no newline)
    # Should be: triple-quotes + newline + indent + word
    source_code = re.sub(r'"""([a-zA-Z_])', r'"""\n    \1', source_code)

    # Fix pattern: triple-quotes + comment (no newline)
    source_code = re.sub(r'"""(#)', r'"""\n    \1', source_code)

    return source_code


def fix_notebook(notebook_path):
    """
    Fix broken cells in a notebook.

    Parameters
    ----------
    notebook_path : Path
        Path to notebook file

    Returns
    -------
    int
        Number of cells fixed
    """
    print(f"\nFixing: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    fixed_count = 0

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            original_source = ''.join(cell.get('source', []))

            # Check if there's a syntax issue
            if '"""' in original_source and re.search(r'"""[a-zA-Z_#]', original_source):
                fixed_source = fix_cell_formatting(original_source)

                if fixed_source != original_source:
                    cell['source'] = fixed_source
                    fixed_count += 1

    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"  Cells fixed: {fixed_count}")

    return fixed_count


def main():
    """Fix all broken notebooks."""
    repo_dir = Path(__file__).parent
    notebooks_dir = repo_dir / 'notebooks'

    notebook_files = [
        '11_degree_conditioned_compositionality.ipynb',
        '11.1_empirical_vs_analytical_compositional.ipynb',
        '11.2_empirical_vs_analytical_compositional.ipynb',
        '11.3_empirical_vs_analytical_compositional.ipynb',
    ]

    print("="*70)
    print("FIXING BROKEN CELL FORMATTING")
    print("="*70)
    print("\nIssue: Docstring script removed newlines after closing triple quotes")
    print("Fix: Restoring proper formatting\n")

    total_fixed = 0

    for notebook_file in notebook_files:
        notebook_path = notebooks_dir / notebook_file

        if notebook_path.exists():
            fixed = fix_notebook(notebook_path)
            total_fixed += fixed
        else:
            print(f"\nSkipping: {notebook_file} (not found)")

    print("\n" + "="*70)
    print("CELLS FIXED")
    print("="*70)
    print(f"Total cells repaired: {total_fixed}")
    print("Notebooks should now have valid Python syntax")
    print("="*70)


if __name__ == '__main__':
    main()
