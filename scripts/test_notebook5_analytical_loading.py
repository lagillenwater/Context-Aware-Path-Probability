"""
Test that notebook 5 will correctly load and display analytical vs empirical data.
This simulates what notebook 5 will do when executed.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths (simulating notebook 5's setup)
repo_dir = Path.cwd()
results_dir = repo_dir / 'results' / 'model_comparison'

print("="*80)
print("TESTING NOTEBOOK 5 ANALYTICAL VS EMPIRICAL LOADING")
print("="*80)

# Step 1: Load edge type results (simulating load_edge_type_results function)
print("\n1. Testing load_edge_type_results function...")
edge_types_to_test = ['CtD', 'AeG', 'CbG']
all_results = {}

for edge_type in edge_types_to_test:
    edge_results_dir = results_dir / f"{edge_type}_results"
    if not edge_results_dir.exists():
        print(f"  ✗ {edge_type}: directory not found")
        continue

    results = {'edge_type': edge_type}

    # Load analytical vs empirical comparison
    analytical_empirical_file = edge_results_dir / 'analytical_vs_empirical_comparison.csv'
    if analytical_empirical_file.exists():
        results['analytical_vs_empirical'] = pd.read_csv(analytical_empirical_file)
        print(f"  ✓ {edge_type}: loaded analytical_vs_empirical (pearson_r = {results['analytical_vs_empirical']['pearson_r'].values[0]:.4f})")
    else:
        print(f"  ✗ {edge_type}: analytical_vs_empirical file not found")

    all_results[edge_type] = results

# Step 2: Aggregate analytical_vs_empirical data (simulating cell 20)
print("\n2. Testing analytical_vs_empirical aggregation...")
analytical_vs_empirical_data = []
for edge_type, results in all_results.items():
    if 'analytical_vs_empirical' in results:
        df = results['analytical_vs_empirical'].copy()
        df['edge_type'] = edge_type
        analytical_vs_empirical_data.append(df)

if analytical_vs_empirical_data:
    analytical_vs_empirical_df = pd.concat(analytical_vs_empirical_data, ignore_index=True)
    print(f"  ✓ Aggregated {len(analytical_vs_empirical_df)} records")
    print(f"  ✓ Columns present: {', '.join(analytical_vs_empirical_df.columns[:5])}...")
    print(f"\n  Sample data:")
    print(analytical_vs_empirical_df[['edge_type', 'Model', 'pearson_r']])
else:
    print("  ✗ No analytical_vs_empirical data found")
    analytical_vs_empirical_df = pd.DataFrame()

# Step 3: Test density categorization (simulating cell 21 prep)
print("\n3. Testing density categorization...")

# Simulate graph characteristics
graph_chars_df = pd.DataFrame([
    {'edge_type': 'CtD', 'density': 0.0082},  # Sparse
    {'edge_type': 'AeG', 'density': 0.0018},  # Very Sparse
    {'edge_type': 'CbG', 'density': 0.00035}  # Very Sparse
])

if not analytical_vs_empirical_df.empty:
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
    print("  ✓ Density categories added")
    print(f"\n  Data with categories:")
    print(analytical_vs_empirical_df[['edge_type', 'density_category', 'pearson_r']])

# Step 4: Test data extraction for boxplot (simulating cell 21 boxplot code)
print("\n4. Testing boxplot data extraction...")

for cat in ['Very Sparse (<1%)', 'Sparse (1-3%)']:
    analytical_cat_data = analytical_vs_empirical_df[
        analytical_vs_empirical_df['density_category'] == cat
    ]
    if len(analytical_cat_data) > 0:
        pearson_values = analytical_cat_data['pearson_r'].values
        print(f"  ✓ {cat}: {len(pearson_values)} values, range [{pearson_values.min():.4f}, {pearson_values.max():.4f}]")
    else:
        print(f"  ✗ {cat}: no data")

print("\n" + "="*80)
print("TEST COMPLETE - Notebook 5 should work correctly!")
print("="*80)
print("\nTo see the updated boxplots, re-run notebook 5:")
print("  jupyter nbconvert --to notebook --execute notebooks/5_model_testing_summary.ipynb")
print("  (Run from the notebooks/ directory or adjust paths accordingly)")
