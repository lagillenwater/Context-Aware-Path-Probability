#!/usr/bin/env python
"""
Visualize het.io validation results for DWPC and p-values.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_dir = Path("/Users/gillenlu/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Repositories/Context-Aware-Path-Probability/results/dwpc_pvalue_validation")

# Load data
metapaths = ['CbG', 'CtD', 'CbGpPW']
dfs = {}

for mp in metapaths:
    file_path = results_dir / f"pvalue_validation_{mp}_empirical.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        df['metapath'] = mp
        df['metapath_length'] = 1 if mp in ['CbG', 'CtD'] else 2
        dfs[mp] = df

# Combine all data
all_data = pd.concat(dfs.values(), ignore_index=True)

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Het.io Validation: DWPC and P-Value Comparison', fontsize=14, fontweight='bold')

# Plot 1: DWPC comparison (scatter)
ax = axes[0, 0]
for mp in metapaths:
    df = dfs.get(mp)
    if df is not None:
        marker = 'o' if mp in ['CbG', 'CtD'] else 's'
        label = f"{mp} (length {df['metapath_length'].iloc[0]})"
        ax.scatter(df['hetio_dwpc'], df['our_dwpc'],
                  label=label, alpha=0.7, s=100, marker=marker)

# Add diagonal line
max_val = max(all_data['hetio_dwpc'].max(), all_data['our_dwpc'].max())
min_val = min(all_data['hetio_dwpc'].min(), all_data['our_dwpc'].min())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect match')

ax.set_xlabel('Het.io DWPC', fontsize=11)
ax.set_ylabel('Our DWPC', fontsize=11)
ax.set_title('DWPC Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Plot 2: DWPC absolute differences
ax = axes[0, 1]
x_pos = []
heights = []
colors_list = []
labels = []
colors_map = {'CbG': 'C0', 'CtD': 'C1', 'CbGpPW': 'C2'}

for i, mp in enumerate(metapaths):
    df = dfs.get(mp)
    if df is not None:
        for j, row in df.iterrows():
            x_pos.append(i + j*0.3)
            heights.append(row['dwpc_diff'])
            colors_list.append(colors_map[mp])
            if j == 0:
                labels.append(mp)
            else:
                labels.append('')

bars = ax.bar(x_pos, heights, width=0.25, color=colors_list)
ax.set_xlabel('Metapath', fontsize=11)
ax.set_ylabel('|Our DWPC - Het.io DWPC|', fontsize=11)
ax.set_title('DWPC Absolute Differences', fontsize=12, fontweight='bold')
ax.set_xticks([i + 0.15 for i in range(len(metapaths))])
ax.set_xticklabels(metapaths)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=1e-6, color='g', linestyle='--', alpha=0.5, label='Exact match threshold')
ax.set_yscale('log')
ax.legend(fontsize=9)

# Plot 3: P-value comparison
ax = axes[1, 0]
for mp in metapaths:
    df = dfs.get(mp)
    if df is not None:
        marker = 'o' if mp in ['CbG', 'CtD'] else 's'
        label = f"{mp} (length {df['metapath_length'].iloc[0]})"
        ax.scatter(df['hetio_pvalue'], df['our_pvalue'],
                  label=label, alpha=0.7, s=100, marker=marker)

# Add diagonal line
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect match')

ax.set_xlabel('Het.io P-value', fontsize=11)
ax.set_ylabel('Our P-value', fontsize=11)
ax.set_title('P-Value Comparison\n(3 permutations vs het.io\'s 200)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Plot 4: P-value differences
ax = axes[1, 1]
x_pos = []
heights = []
colors_list = []

for i, mp in enumerate(metapaths):
    df = dfs.get(mp)
    if df is not None:
        for j, row in df.iterrows():
            x_pos.append(i + j*0.3)
            heights.append(row['pvalue_diff'])
            colors_list.append(colors_map[mp])

bars = ax.bar(x_pos, heights, width=0.25, color=colors_list)
ax.set_xlabel('Metapath', fontsize=11)
ax.set_ylabel('|Our P-value - Het.io P-value|', fontsize=11)
ax.set_title('P-Value Absolute Differences', fontsize=12, fontweight='bold')
ax.set_xticks([i + 0.15 for i in range(len(metapaths))])
ax.set_xticklabels(metapaths)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Close match threshold')
ax.legend(fontsize=9)

plt.tight_layout()

# Save figure
output_path = results_dir / 'validation_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {output_path}")

# Print summary statistics
print("\n" + "="*80)
print("VALIDATION SUMMARY STATISTICS")
print("="*80)

for mp in metapaths:
    df = dfs.get(mp)
    if df is not None:
        print(f"\n{mp} (n={len(df)} samples):")
        print(f"  DWPC:")
        print(f"    Mean absolute difference: {df['dwpc_diff'].mean():.6e}")
        print(f"    Max absolute difference:  {df['dwpc_diff'].max():.6e}")
        print(f"    Exact matches (< 1e-6):   {(df['dwpc_diff'] < 1e-6).sum()}/{len(df)}")
        print(f"  P-values:")
        print(f"    Mean absolute difference: {df['pvalue_diff'].mean():.4f}")
        print(f"    Max absolute difference:  {df['pvalue_diff'].max():.4f}")
        print(f"    Close matches (< 0.1):    {df['pvalue_close'].sum()}/{len(df)}")

print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)
print(f"Total samples: {len(all_data)}")
print(f"DWPC exact matches: {(all_data['dwpc_diff'] < 1e-6).sum()}/{len(all_data)} ({100*(all_data['dwpc_diff'] < 1e-6).sum()/len(all_data):.1f}%)")
print(f"P-value close matches: {all_data['pvalue_close'].sum()}/{len(all_data)} ({100*all_data['pvalue_close'].sum()/len(all_data):.1f}%)")

plt.show()
