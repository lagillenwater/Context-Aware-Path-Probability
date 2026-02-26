#!/usr/bin/env python
"""Create sample size effects visualization."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load variance analysis data
variance_df = pd.read_csv('results/dwpc_pvalue_validation/dwpc_results/diagnostics/variance_analysis.csv')

# Calculate calibration error and size bins
variance_df['calibration_error'] = np.abs(variance_df['mean_pvalue'] - 0.5)
variance_df['size_bin'] = pd.cut(variance_df['n'], bins=[0, 10, 20, 50, 100],
                                  labels=['<10', '10-20', '20-50', '>50'])

# Create 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Sample Size vs Mean P-value
ax = axes[0, 0]
scatter = ax.scatter(variance_df['n'], variance_df['mean_pvalue'],
                     c=variance_df['cv_obs_dwpc'], s=80, alpha=0.6,
                     cmap='viridis', edgecolors='black', linewidth=0.5)
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Expected (0.5)')
ax.set_xlabel('Sample Size (n)', fontsize=12)
ax.set_ylabel('Mean P-value', fontsize=12)
ax.set_title('Sample Size vs Mean P-value\n(color = Coefficient of Variation)', fontsize=13)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('CV (DWPC)', fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3)

# Add correlation text
corr_n_p = variance_df[['n', 'mean_pvalue']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'r = {corr_n_p:.3f}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 2: Sample Size vs Calibration Error
ax = axes[0, 1]
scatter = ax.scatter(variance_df['n'], variance_df['calibration_error'],
                     c=variance_df['cv_obs_dwpc'], s=80, alpha=0.6,
                     cmap='viridis', edgecolors='black', linewidth=0.5)
ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Good calibration')
ax.set_xlabel('Sample Size (n)', fontsize=12)
ax.set_ylabel('Calibration Error (|mean_p - 0.5|)', fontsize=12)
ax.set_title('Sample Size vs Calibration Error\n(color = Coefficient of Variation)', fontsize=13)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('CV (DWPC)', fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3)

# Add correlation text
corr_n_cal = variance_df[['n', 'calibration_error']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'r = {corr_n_cal:.3f}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 3: Distribution of P-values by Sample Size Bins
ax = axes[1, 0]
for size_bin in ['<10', '10-20', '20-50', '>50']:
    subset = variance_df[variance_df['size_bin'] == size_bin]
    if len(subset) > 0:
        ax.hist(subset['mean_pvalue'], bins=15, alpha=0.5, label=f'n {size_bin}')
ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Expected')
ax.set_xlabel('Mean P-value', fontsize=12)
ax.set_ylabel('Count of Degree Categories', fontsize=12)
ax.set_title('Distribution of P-values by Sample Size', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: CV vs Mean P-value (bubble size = sample size)
ax = axes[1, 1]
scatter = ax.scatter(variance_df['cv_obs_dwpc'], variance_df['mean_pvalue'],
                     s=variance_df['n']*3, alpha=0.5,
                     c=variance_df['n'], cmap='YlOrRd',
                     edgecolors='black', linewidth=0.5)
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Expected (0.5)')
ax.set_xlabel('Coefficient of Variation (DWPC)', fontsize=12)
ax.set_ylabel('Mean P-value', fontsize=12)
ax.set_title('CV vs Mean P-value\n(bubble size = sample size)', fontsize=13)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Sample Size (n)', fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3)

# Add correlation text
corr_cv_p = variance_df[['cv_obs_dwpc', 'mean_pvalue']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'r = {corr_cv_p:.3f}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_file = Path('results/dwpc_pvalue_validation/dwpc_results/diagnostics/sample_size_effects.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {output_file}")
plt.close()

print("\nKey correlations:")
print(f"Sample Size vs Mean P-value: r = {corr_n_p:.3f}")
print(f"Sample Size vs Calibration Error: r = {corr_n_cal:.3f}")
print(f"CV vs Mean P-value: r = {corr_cv_p:.3f}")
