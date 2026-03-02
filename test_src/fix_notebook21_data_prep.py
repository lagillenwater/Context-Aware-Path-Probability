"""
Fix notebook 21 data preparation.

This script:
1. Removes old sanity check cells (4-7)
2. Adds tensor creation after data loading
3. Ensures clean transition to Test 1
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


def create_code_cell(content):
    """Create a code cell with given content."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content if isinstance(content, list) else [content]
    }


def main():
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/21_nn_architecture_exploration.ipynb'

    print(f"Loading notebook: {notebook_path}")
    notebook = load_notebook(notebook_path)

    cells = notebook['cells']

    # Keep cells 0-3 (intro, data loading)
    # Remove cells 4-7 (old sanity check)
    # Keep rest (Test 1, Test 2, etc.)

    print(f"Original cell count: {len(cells)}")
    print(f"Removing cells 4-7 (old sanity check code)")

    # Create tensor conversion cell
    tensor_cell = create_code_cell([
        "# Create PyTorch tensors for training\n",
        "\n",
        "X_train_tensor = torch.FloatTensor(X_train)\n",
        "X_test_tensor = torch.FloatTensor(X_test)\n",
        "y_train_tensor = torch.FloatTensor(y_train)\n",
        "y_test_tensor = torch.FloatTensor(y_test)\n",
        "\n",
        "print(\"\\nCreated PyTorch tensors:\")\n",
        "print(f\"  X_train_tensor: {X_train_tensor.shape}\")\n",
        "print(f\"  X_test_tensor: {X_test_tensor.shape}\")\n",
        "print(f\"  y_train_tensor: {y_train_tensor.shape}\")\n",
        "print(f\"  y_test_tensor: {y_test_tensor.shape}\")\n"
    ])

    # Build new cell list: cells 0-3, tensor cell, then cells 8 onward (skip 4-7)
    new_cells = cells[:4] + [tensor_cell] + cells[8:]

    notebook['cells'] = new_cells

    print(f"New cell count: {len(new_cells)}")
    print(f"Removed {len(cells) - len(new_cells)} cells")

    # Save modified notebook
    print(f"\nSaving modified notebook to: {notebook_path}")
    save_notebook(notebook, notebook_path)

    print("Notebook fixed successfully!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
