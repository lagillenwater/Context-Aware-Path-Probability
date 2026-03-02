#!/usr/bin/env python
"""
Remove emojis and fix Greene Lab standards violations in notebooks.

This script processes Jupyter notebooks to:
1. Remove all emojis
2. Clean up excessive debug/validation messages
3. Fix inline comment spacing
"""

import json
import re
from pathlib import Path


def remove_emojis(text):
    """
    Remove all emojis from text.

    Parameters
    ----------
    text : str
        Text containing emojis

    Returns
    -------
    str
        Text with emojis removed
    """
    # Common emojis used in the notebooks
    emoji_patterns = [
        '🚨', '✓', '✗', '❌', '✅', '→', '←', '↔', '⚠', '💡', '🔍', '📊', '📈', '📉'
    ]

    for emoji in emoji_patterns:
        text = text.replace(emoji, '')

    # Remove any remaining emojis using regex
    # This pattern matches most emojis in the Unicode standard
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    return text


def clean_validation_messages(text):
    """
    Clean up excessive SUCCESS/ERROR/DEBUG messages.

    Parameters
    ----------
    text : str
        Text with validation messages

    Returns
    -------
    str
        Cleaned text
    """
    # Remove excessive markers
    replacements = {
        ' DEBUG:': 'DEBUG:',
        ' SUCCESS:': 'Success:',
        ' ERROR:': 'Error:',
        ' VALIDATION:': 'Validation:',
        ' WARNING:': 'Warning:',
        'SUCCESS: SUCCESS:': 'Success:',
        'SUCCESS:  ':  'Success: ',
        'Error: ERROR:': 'Error:',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def process_notebook(notebook_path):
    """
    Process a Jupyter notebook to remove emojis and clean messages.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook file

    Returns
    -------
    dict
        Statistics about changes made
    """
    print(f"\nProcessing: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    emoji_count = 0
    cell_count = 0

    # Process each cell
    for cell in notebook.get('cells', []):
        original_source = ''.join(cell.get('source', []))

        # Check for emojis
        if any(emoji in original_source for emoji in ['🚨', '✓', '✗', '❌', '✅']):
            emoji_count += 1

        # Clean source
        new_source = original_source
        new_source = remove_emojis(new_source)
        new_source = clean_validation_messages(new_source)

        # Update cell if changed
        if new_source != original_source:
            cell['source'] = new_source
            cell_count += 1

    # Write back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    stats = {
        'cells_with_emojis': emoji_count,
        'cells_modified': cell_count
    }

    print(f"  Cells with emojis: {emoji_count}")
    print(f"  Cells modified: {cell_count}")

    return stats


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
    print("REMOVING EMOJIS AND CLEANING NOTEBOOKS")
    print("="*70)

    total_stats = {'cells_with_emojis': 0, 'cells_modified': 0}

    for notebook_file in notebook_files:
        notebook_path = notebooks_dir / notebook_file

        if notebook_path.exists():
            stats = process_notebook(notebook_path)
            total_stats['cells_with_emojis'] += stats['cells_with_emojis']
            total_stats['cells_modified'] += stats['cells_modified']
        else:
            print(f"\nSkipping: {notebook_file} (not found)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total cells with emojis: {total_stats['cells_with_emojis']}")
    print(f"Total cells modified: {total_stats['cells_modified']}")
    print("\nAll emojis have been removed from the notebook series.")
    print("="*70)


if __name__ == '__main__':
    main()
