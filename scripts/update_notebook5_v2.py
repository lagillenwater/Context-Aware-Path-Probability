"""
Update notebook 5 to properly load and display analytical vs empirical comparison - V2
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path('notebooks/5_model_testing_summary.ipynb')
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Update cell 4 - add analytical_vs_empirical loading
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_lines = cell.get('source', [])
        source = '\n'.join(source_lines) if isinstance(source_lines, list) else source_lines

        if 'def load_edge_type_results' in source and 'return results' in source:
            print(f"Updating cell {i}: load_edge_type_results")

            # Insert new lines before 'return results'
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                if line.strip() == 'return results':
                    # Add analytical_vs_empirical loading before return
                    new_lines.append('')
                    new_lines.append('    # Load analytical vs empirical comparison')
                    new_lines.append("    analytical_empirical_file = edge_results_dir / 'analytical_vs_empirical_comparison.csv'")
                    new_lines.append('    if analytical_empirical_file.exists():')
                    new_lines.append("        results['analytical_vs_empirical'] = pd.read_csv(analytical_empirical_file)")
                new_lines.append(line)

            cell['source'] = new_lines
            print("  ✓ Updated")
            break

# Update cell 20 - add analytical_vs_empirical aggregation
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_lines = cell.get('source', [])
        source = '\n'.join(source_lines) if isinstance(source_lines, list) else source_lines

        if '# Aggregate analytical and empirical comparisons' in source and 'empirical_df = pd.DataFrame()' in source:
            print(f"Updating cell {i}: aggregate analytical_vs_empirical data")

            lines = source.split('\n')
            new_lines = []
            for j, line in enumerate(lines):
                new_lines.append(line)
                # Add after 'empirical_df = pd.DataFrame()'
                if 'empirical_df = pd.DataFrame()' in line:
                    new_lines.append('')
                    new_lines.append('# Load analytical vs empirical comparison data')
                    new_lines.append('analytical_vs_empirical_data = []')
                    new_lines.append('for edge_type, results in all_results.items():')
                    new_lines.append("    if 'analytical_vs_empirical' in results:")
                    new_lines.append("        df = results['analytical_vs_empirical'].copy()")
                    new_lines.append("        df['edge_type'] = edge_type")
                    new_lines.append('        analytical_vs_empirical_data.append(df)')
                    new_lines.append('')
                    new_lines.append('if analytical_vs_empirical_data:')
                    new_lines.append('    analytical_vs_empirical_df = pd.concat(analytical_vs_empirical_data, ignore_index=True)')
                    new_lines.append('    print(f"Analytical vs Empirical comparison data: {len(analytical_vs_empirical_df)} records")')
                    new_lines.append('else:')
                    new_lines.append('    analytical_vs_empirical_df = pd.DataFrame()')

            cell['source'] = new_lines
            print("  ✓ Updated")
            break

# Update cell 21 - replace hack code with proper loading
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_lines = cell.get('source', [])
        source = '\n'.join(source_lines) if isinstance(source_lines, list) else source_lines

        if "if model == 'Current Analytical':" in source and 'For now, get analytical vs empirical correlation from existing data' in source:
            print(f"Updating cell {i}: replace hack code in boxplot")

            # Find and replace the hack section
            lines = source.split('\n')
            new_lines = []
            skip_until_else = False
            indent_level = 0

            j = 0
            while j < len(lines):
                line = lines[j]

                # Detect start of hack code
                if "if model == 'Current Analytical':" in line:
                    # Add this line
                    new_lines.append(line)
                    indent_level = len(line) - len(line.lstrip())
                    base_indent = ' ' * indent_level
                    next_indent = ' ' * (indent_level + 4)

                    # Skip all the hack code until we find the 'else:' at the same indent level
                    j += 1
                    while j < len(lines):
                        current_line = lines[j]
                        current_indent = len(current_line) - len(current_line.lstrip())

                        # Check if we reached the else at the same level
                        if current_indent == indent_level and 'else:' in current_line:
                            # Insert our new clean code before the else
                            new_lines.append(next_indent + '# Load analytical vs empirical from the dedicated CSV files')
                            new_lines.append(next_indent + 'if not analytical_vs_empirical_df.empty:')
                            new_lines.append(next_indent + '    # Merge with density categories if not already done')
                            new_lines.append(next_indent + "    if 'density_category' not in analytical_vs_empirical_df.columns:")
                            new_lines.append(next_indent + '        analytical_vs_empirical_df = analytical_vs_empirical_df.merge(')
                            new_lines.append(next_indent + "            graph_chars_df[['edge_type', 'density']],")
                            new_lines.append(next_indent + "            on='edge_type',")
                            new_lines.append(next_indent + "            how='left'")
                            new_lines.append(next_indent + '        )')
                            new_lines.append(next_indent + '        analytical_vs_empirical_df[\'density_category\'] = pd.cut(')
                            new_lines.append(next_indent + '            analytical_vs_empirical_df[\'density\'],')
                            new_lines.append(next_indent + '            bins=[0, 0.01, 0.03, 0.05, 1.0],')
                            new_lines.append(next_indent + "            labels=['Very Sparse (<1%)', 'Sparse (1-3%)', 'Medium (3-5%)', 'Dense (>5%)']")
                            new_lines.append(next_indent + '        )')
                            new_lines.append('')
                            new_lines.append(next_indent + '    # Get data for this density category')
                            new_lines.append(next_indent + '    analytical_cat_data = analytical_vs_empirical_df[')
                            new_lines.append(next_indent + '        analytical_vs_empirical_df[\'density_category\'] == cat')
                            new_lines.append(next_indent + '    ]')
                            new_lines.append(next_indent + '    if len(analytical_cat_data) > 0:')
                            new_lines.append(next_indent + '        box_data_empirical.append(analytical_cat_data[\'pearson_r\'].values)')
                            new_lines.append(next_indent + '        box_positions_empirical.append(pos)')
                            # Now add the else line we found
                            new_lines.append(current_line)
                            break
                        j += 1
                    j += 1
                    continue

                new_lines.append(line)
                j += 1

            cell['source'] = new_lines
            print("  ✓ Updated")
            break

# Save
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✓ Successfully updated {notebook_path}")
