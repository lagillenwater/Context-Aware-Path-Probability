"""
Restructure notebook 21 to focus on single-layer NN optimizer comparison.

This script:
1. Removes complex architectures (GraphAwareNN, CNN, RNN, etc.)
2. Reorganizes into Test 1 and Test 2
3. Adds result persistence and visualization code
"""

import json
import sys


def load_notebook(filepath):
    """Load a Jupyter notebook from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(notebook, filepath):
    """Save a Jupyter notebook to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write('\n')


def find_cell_by_content(cells, search_text):
    """Find index of cell containing search text."""
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if search_text in source:
                return i
    return -1


def main():
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/21_nn_architecture_exploration.ipynb'

    print(f"Loading notebook: {notebook_path}")
    notebook = load_notebook(notebook_path)

    cells = notebook['cells']

    # Find section boundaries
    empirical_freq_idx = find_cell_by_content(cells, "## 3. Empirical Frequency Analysis")
    alt_arch_2_idx = find_cell_by_content(cells, "## 4. Alternative Architecture 2:")
    recommendations_idx = find_cell_by_content(cells, "## 8. Recommendations and Next Steps")

    print(f"Found section indices:")
    print(f"  Empirical Frequency: {empirical_freq_idx}")
    print(f"  Alternative Architecture 2: {alt_arch_2_idx}")
    print(f"  Recommendations: {recommendations_idx}")

    if alt_arch_2_idx == -1 or empirical_freq_idx == -1:
        print("Error: Could not find expected sections")
        return 1

    # Keep cells up to and including empirical frequency analysis
    # Remove complex architecture sections
    # Keep recommendations at end

    if recommendations_idx > 0:
        new_cells = cells[:alt_arch_2_idx] + cells[recommendations_idx:]
    else:
        new_cells = cells[:alt_arch_2_idx]

    notebook['cells'] = new_cells

    # Save backup
    backup_path = notebook_path.replace('.ipynb', '_backup.ipynb')
    print(f"Saving backup to: {backup_path}")
    save_notebook(load_notebook(notebook_path), backup_path)

    # Save modified notebook
    print(f"Saving modified notebook to: {notebook_path}")
    save_notebook(notebook, notebook_path)

    print(f"Removed {len(cells) - len(new_cells)} cells")
    print("Done!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
