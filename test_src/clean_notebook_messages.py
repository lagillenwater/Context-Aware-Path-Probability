#!/usr/bin/env python
"""
Clean up debug and validation messages in notebooks to meet Greene Lab standards.

This script:
1. Removes excessive DEBUG print statements
2. Consolidates redundant SUCCESS/VALIDATION messages
3. Makes output messages professional and concise
"""

import json
import re
from pathlib import Path


def clean_print_statements(code):
    """
    Clean excessive debug and validation print statements.

    Parameters
    ----------
    code : str
        Source code

    Returns
    -------
    str
        Cleaned code
    """
    lines = code.split('\n')
    cleaned_lines = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        # Remove lines with DEBUG markers
        if 'DEBUG:' in line and 'print(' in line:
            continue

        # Remove excessive surrounding markers like " OPTION A RESULT:"
        if re.search(r'print\(f["\'].*OPTION A (RESULT|FUNCTION CALLED):.*["\']', line):
            continue

        # Remove "RETURNING ... results with Option A" debug lines
        if 'RETURNING' in line and 'results with Option A' in line:
            continue

        # Clean up multiple SUCCESS markers
        line = re.sub(r'Success:\s+Success:', 'Success:', line)
        line = re.sub(r'SUCCESS:\s+SUCCESS:', 'Success:', line)

        # Simplify validation success messages
        if 'Success: All compositional probabilities are' in line:
            line = line.replace('Success: All compositional probabilities are',
                              'All compositional probabilities are')

        # Remove excessive celebration messages
        if re.search(r'(Option A fix worked correctly|No more impossible probabilities)', line):
            if i + 1 < len(lines) and 'print(' in lines[i+1]:
                # Keep only one success message
                continue

        # Clean up "HEATMAP ORIENTATION FIXED" excessive message
        if 'HEATMAP ORIENTATION FIXED:' in line:
            line = line.replace('HEATMAP ORIENTATION FIXED:', 'Heatmap orientation fixed:')

        # Clean up multiple validation markers
        line = re.sub(r'VALIDATION:\s+VALIDATION:', 'Validation:', line)
        line = re.sub(r'Validation:\s+Validation:', 'Validation:', line)

        # Remove "mathematical impossibility" dramatic language
        line = re.sub(r'Mathematical Impossibility Demonstrated',
                     'Compositional Probability Distribution', line)
        line = re.sub(r'mathematically impossible', 'invalid', line)
        line = re.sub(r'MATHEMATICAL ERROR', 'Invalid probability', line)

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def process_notebook(notebook_path):
    """
    Process a Jupyter notebook to clean messages.

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

    # Process each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            original_source = ''.join(cell.get('source', []))
            new_source = clean_print_statements(original_source)

            if new_source != original_source:
                cell['source'] = new_source
                modified_count += 1

    # Write back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"  Cells modified: {modified_count}")

    return modified_count


def main():
    """Process all notebooks in the series."""
    repo_dir = Path(__file__).parent
    notebooks_dir = repo_dir / 'notebooks'

    notebook_files = [
        '11_degree_conditioned_compositionality.ipynb',
        '11.1_empirical_vs_analytical_compositional.ipynb',
        '11.2_empirical_vs_analytical_compositional.ipynb',
        '11.3_empirical_vs_analytical_compositional.ipynb',
    ]

    print("="*70)
    print("CLEANING DEBUG AND VALIDATION MESSAGES")
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
    print("\nDebug and validation messages have been cleaned.")
    print("="*70)


if __name__ == '__main__':
    main()
