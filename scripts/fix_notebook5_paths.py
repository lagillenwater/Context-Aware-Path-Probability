"""
Make notebook 5 path setup more robust to handle running from different directories.
"""

import json
from pathlib import Path

nb_path = Path('notebooks/5_model_testing_summary.ipynb')
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Find cell 2 (imports and setup)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_lines = cell.get('source', [])
        source = '\n'.join(source_lines) if isinstance(source_lines, list) else source_lines

        if '# Setup paths' in source and 'repo_dir = Path.cwd().parent' in source:
            print(f'Updating cell {i}: path setup to be more robust')

            # Replace the path setup to handle both running from notebooks/ and repo root
            old_setup = '''# Setup paths
repo_dir = Path.cwd().parent'''

            new_setup = '''# Setup paths
# Handle running from both notebooks/ directory and repo root
if Path.cwd().name == 'notebooks':
    repo_dir = Path.cwd().parent
else:
    repo_dir = Path.cwd()'''

            source = source.replace(old_setup, new_setup)
            cell['source'] = source.split('\n')
            print('  ✓ Updated to handle both directories')
            break

# Save
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'\n✓ Updated {nb_path}')
print('\nNow the notebook will work whether you run it from:')
print('  - notebooks/ directory')
print('  - repo root directory')
