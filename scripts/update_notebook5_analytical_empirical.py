"""
Update notebook 5 to properly load and display analytical vs empirical comparison.

This script:
1. Updates load_edge_type_results to load analytical_vs_empirical_comparison.csv
2. Simplifies cell 21 to load the new data instead of the hack code
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path('notebooks/5_model_testing_summary.ipynb')
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find and update cell 4 (load_edge_type_results function)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def load_edge_type_results' in source and 'test_vs_empirical_comparison.csv' in source:
            print(f"Updating cell {i}: load_edge_type_results function")

            # Add loading of analytical_vs_empirical_comparison.csv
            old_code = """    # Load empirical comparison
    empirical_file = edge_results_dir / 'test_vs_empirical_comparison.csv'
    if empirical_file.exists():
        results['empirical_comparison'] = pd.read_csv(empirical_file)

    return results"""

            new_code = """    # Load empirical comparison
    empirical_file = edge_results_dir / 'test_vs_empirical_comparison.csv'
    if empirical_file.exists():
        results['empirical_comparison'] = pd.read_csv(empirical_file)

    # Load analytical vs empirical comparison
    analytical_empirical_file = edge_results_dir / 'analytical_vs_empirical_comparison.csv'
    if analytical_empirical_file.exists():
        results['analytical_vs_empirical'] = pd.read_csv(analytical_empirical_file)

    return results"""

            source = source.replace(old_code, new_code)
            cell['source'] = source.split('\n')
            if not cell['source'][-1]:  # Remove empty last element if present
                cell['source'] = cell['source'][:-1]
            print("  ✓ Updated")
            break

# Find and update cell 20 (aggregate analytical_vs_empirical data)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "# Aggregate analytical and empirical comparisons" in source and "'empirical_comparison' in results" in source:
            print(f"Updating cell {i}: aggregate data loading")

            # Add aggregation of analytical_vs_empirical data
            old_code = """if empirical_data:
    empirical_df = pd.concat(empirical_data, ignore_index=True)
    print(f"Empirical comparison data: {len(empirical_df)} records")
else:
    empirical_df = pd.DataFrame()"""

            new_code = """if empirical_data:
    empirical_df = pd.concat(empirical_data, ignore_index=True)
    print(f"Empirical comparison data: {len(empirical_df)} records")
else:
    empirical_df = pd.DataFrame()

# Load analytical vs empirical comparison data
analytical_vs_empirical_data = []
for edge_type, results in all_results.items():
    if 'analytical_vs_empirical' in results:
        df = results['analytical_vs_empirical'].copy()
        df['edge_type'] = edge_type
        analytical_vs_empirical_data.append(df)

if analytical_vs_empirical_data:
    analytical_vs_empirical_df = pd.concat(analytical_vs_empirical_data, ignore_index=True)
    print(f"Analytical vs Empirical comparison data: {len(analytical_vs_empirical_df)} records")
else:
    analytical_vs_empirical_df = pd.DataFrame()"""

            source = source.replace(old_code, new_code)
            cell['source'] = source.split('\n')
            if not cell['source'][-1]:
                cell['source'] = cell['source'][:-1]
            print("  ✓ Updated")
            break

# Find and update cell 21 (plotting code with hack)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'For analytical, get its correlation with empirical' in source and '# For now, get analytical vs empirical correlation from existing data' in source:
            print(f"Updating cell {i}: boxplot plotting code")

            # Replace the entire hack section with simple load from analytical_vs_empirical_df
            old_hack = """                if model == 'Current Analytical':
                    # For analytical, get its correlation with empirical from analytical_df
                    # The analytical formula's correlation with empirical is computed in notebook 4
                    # We need to load it from the analytical comparison data
                    # For now, get analytical vs empirical correlation from existing data
                    if not analytical_df.empty:
                        analytical_cat_data = analytical_df[analytical_df['density_category'] == cat]
                        if len(analytical_cat_data) > 0:
                            # Get unique edge types in this category
                            edge_types_in_cat = empirical_df[empirical_df['density_category'] == cat]['edge_type'].unique()
                            # For each edge type, get analytical vs empirical correlation
                            analytical_empirical_corrs = []
                            for et in edge_types_in_cat:
                                # Load analytical vs empirical from results
                                et_results = all_results.get(et, {})
                                if 'empirical_comparison' in et_results:
                                    emp_comp = et_results['empirical_comparison']
                                    # Look for "Current Analytical" or "Analytical" in Model column
                                    analytical_rows = emp_comp[emp_comp['Model'].str.contains('Analytical', case=False, na=False)]
                                    if len(analytical_rows) > 0:
                                        analytical_empirical_corrs.append(analytical_rows['Correlation vs Empirical'].values[0])

                            if len(analytical_empirical_corrs) > 0:
                                box_data_empirical.append(analytical_empirical_corrs)
                                box_positions_empirical.append(pos)"""

            new_simple_code = """                if model == 'Current Analytical':
                    # Load analytical vs empirical from the dedicated CSV files
                    if not analytical_vs_empirical_df.empty:
                        # Merge with density categories
                        if 'density_category' not in analytical_vs_empirical_df.columns:
                            analytical_vs_empirical_df = analytical_vs_empirical_df.merge(
                                graph_chars_df[['edge_type', 'density']],
                                on='edge_type',
                                how='left'
                            )
                            analytical_vs_empirical_df['density_category'] = pd.cut(
                                analytical_vs_empirical_df['density'],
                                bins=[0, 0.01, 0.03, 0.05, 1.0],
                                labels=['Very Sparse (<1%)', 'Sparse (1-3%)', 'Medium (3-5%)', 'Dense (>5%)']
                            )

                        # Get data for this density category
                        analytical_cat_data = analytical_vs_empirical_df[
                            analytical_vs_empirical_df['density_category'] == cat
                        ]
                        if len(analytical_cat_data) > 0:
                            box_data_empirical.append(analytical_cat_data['pearson_r'].values)
                            box_positions_empirical.append(pos)"""

            source = source.replace(old_hack, new_simple_code)
            cell['source'] = source.split('\n')
            if not cell['source'][-1]:
                cell['source'] = cell['source'][:-1]
            print("  ✓ Updated")
            break

# Save updated notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✓ Successfully updated {notebook_path}")
print("\nChanges made:")
print("1. Added loading of analytical_vs_empirical_comparison.csv in load_edge_type_results")
print("2. Added aggregation of analytical_vs_empirical data in cell 20")
print("3. Replaced hack code with simple load from analytical_vs_empirical_df in cell 21")
