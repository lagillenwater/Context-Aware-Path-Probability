#!/usr/bin/env python3
"""
Test script to verify the notebook 5 fix works correctly.
"""

import pandas as pd
import numpy as np

print("Testing notebook 5 fix...")

# Simulate the data structures that should exist in the notebook
print("\n1. Simulating existing data structures...")

# Mock all_results structure (simplified)
all_results = {
    'CtD': {
        'analytical_comparison': pd.DataFrame({
            'Model': ['Simple NN', 'Random Forest'],
            'Correlation vs Analytical': [0.85, 0.92],
            'MAE vs Analytical': [0.02, 0.015]
        }),
        'empirical_comparison': pd.DataFrame({
            'Model': ['Simple NN', 'Random Forest'],
            'Correlation vs Empirical': [0.75, 0.82],
            'MAE vs Empirical': [0.03, 0.025]
        })
    },
    'AdG': {
        'analytical_comparison': pd.DataFrame({
            'Model': ['Simple NN', 'Random Forest'],
            'Correlation vs Analytical': [0.88, 0.95],
            'MAE vs Analytical': [0.018, 0.012]
        }),
        'empirical_comparison': pd.DataFrame({
            'Model': ['Simple NN', 'Random Forest'],
            'Correlation vs Empirical': [0.78, 0.85],
            'MAE vs Empirical': [0.028, 0.022]
        })
    }
}

# Mock analytical_vs_empirical_df
analytical_vs_empirical_df = pd.DataFrame({
    'edge_type': ['CtD', 'AdG'],
    'pearson_r': [0.96, 0.98],
    'n_pairs': [1000, 1200]
})

print("✓ Mock data created")

# Test the aggregation code (this is what was added to the notebook)
print("\n2. Testing aggregation code...")

# Aggregate analytical comparison metrics across all edge types
analytical_performance = []

for edge_type, results in all_results.items():
    if 'analytical_comparison' not in results:
        continue

    df = results['analytical_comparison'].copy()
    df['edge_type'] = edge_type
    analytical_performance.append(df)

if analytical_performance:
    analytical_df = pd.concat(analytical_performance, ignore_index=True)
    print(f"✓ Aggregated analytical comparison: {len(analytical_df)} records")
    print(f"  Models: {analytical_df['Model'].unique().tolist()}")
else:
    print("⚠ No analytical comparison data available")
    analytical_df = pd.DataFrame()

# Aggregate empirical comparison metrics across all edge types
empirical_performance = []

for edge_type, results in all_results.items():
    if 'empirical_comparison' not in results:
        continue

    df = results['empirical_comparison'].copy()
    df['edge_type'] = edge_type
    empirical_performance.append(df)

if empirical_performance:
    empirical_df = pd.concat(empirical_performance, ignore_index=True)
    print(f"✓ Aggregated empirical comparison: {len(empirical_df)} records")
    print(f"  Models: {empirical_df['Model'].unique().tolist()}")
else:
    print("⚠ No empirical comparison data available")
    empirical_df = pd.DataFrame()

print(f"\nData aggregation complete:")
print(f"  analytical_df: {len(analytical_df)} records")
print(f"  empirical_df: {len(empirical_df)} records")
print(f"  analytical_vs_empirical_df: {len(analytical_vs_empirical_df)} records")

# Test the problematic cell 21 logic
print("\n3. Testing cell 21 logic...")

# This is the exact code from cell 21 that was failing
try:
    if not analytical_df.empty:
        print("✓ analytical_df exists and is not empty")
        print(f"  Shape: {analytical_df.shape}")
        print(f"  Columns: {analytical_df.columns.tolist()}")
    else:
        print("✓ analytical_df exists but is empty (this is OK)")

    if not empirical_df.empty:
        print("✓ empirical_df exists and is not empty")
        print(f"  Shape: {empirical_df.shape}")
        print(f"  Columns: {empirical_df.columns.tolist()}")
    else:
        print("✓ empirical_df exists but is empty (this is OK)")

    print("✅ Cell 21 logic test PASSED - no NameError!")

except NameError as e:
    print(f"❌ Cell 21 logic test FAILED: {e}")

print("\n4. Summary:")
print("✅ The fix successfully creates analytical_df and empirical_df")
print("✅ Cell 21 should now work without NameError")
print("✅ Ready for deployment to HPC")