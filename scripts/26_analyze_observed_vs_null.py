#!/usr/bin/env python
"""
Analyze observed vs null DWPC distributions by degree category.

Tests hypothesis: High-degree nodes have lower observed DWPCs than null,
explaining p-values near 1.0.

Terminology:
- "Depleted" (p > 0.7): Observed DWPC is LOWER than null distribution.
  Real network has FEWER or WEAKER paths than expected from permutations.
  Interpretation: Biological networks avoid redundant connections.

- "Enriched" (p < 0.3): Observed DWPC is HIGHER than null distribution.
  Real network has MORE or STRONGER paths than expected from permutations.
  Interpretation: Biological networks maintain specific functional connections.

- P-value interpretation:
  p ≈ 1.0 → observed is at/below null mean (depleted)
  p ≈ 0.5 → observed is at null mean (calibrated)
  p ≈ 0.0 → observed is far above null mean (enriched)
"""

import numpy as np
import pandas as pd
from pathlib import Path

results_dir = Path("results/dwpc_pvalue_validation")

metapaths = ['CbGpPW', 'CtDaG', 'GiGaD', 'CbGpPWpG', 'CtDaGiG', 'CbGpPWpGaD']
categories = ['Low-Low', 'Low-Med', 'Low-High',
              'Med-Low', 'Med-Med', 'Med-High',
              'High-Low', 'High-Med', 'High-High']

results = []

for metapath in metapaths:
    print(f"\n{'='*80}")
    print(f"{metapath}")
    print(f"{'='*80}")

    # Load Scenario A (perm0 observed vs perm1-20 null)
    data_a = np.load(results_dir / f"{metapath}_scenario_a.npz", allow_pickle=True)

    # Keys are like 'observed_Low_High', 'pvalues_Low_High'
    available_keys = [k for k in data_a.keys() if k.startswith('observed_')]

    print(f"\n{'Category':<15} {'n':>5} {'Mean Obs':>12} {'Median Obs':>12} {'Mean p':>10} {'Obs/Null':>20}")
    print("-" * 90)

    for key in available_keys:
        # Extract category name from key
        cat = key.replace('observed_', '').replace('_', '-')

        cat_obs = data_a[key]
        pval_key = f"pvalues_{key.replace('observed_', '')}"

        if pval_key not in data_a:
            continue

        cat_pvals = data_a[pval_key]

        if len(cat_obs) == 0:
            continue

        mean_obs = cat_obs.mean()
        median_obs = np.median(cat_obs)
        mean_p = cat_pvals.mean()

        # Estimate observed/null ratio from p-value
        # If p ≈ 1.0: observed is at/below null (ratio < 1)
        # If p ≈ 0.0: observed is above null (ratio > 1)
        # Rough approximation: ratio ∝ 1 / (p + 0.01)
        # This is very approximate but gives direction

        # Better: use the fact that p = lambda * P(Gamma >= obs)
        # If p ≈ 1, then P(Gamma >= obs) ≈ 1, meaning obs is small relative to gamma
        # If p ≈ 0, then P(Gamma >= obs) ≈ 0, meaning obs is large relative to gamma

        if mean_p > 0.9:
            obs_vs_null = "obs << null (depleted)"
        elif mean_p > 0.6:
            obs_vs_null = "obs < null"
        elif mean_p > 0.4:
            obs_vs_null = "obs ≈ null"
        elif mean_p > 0.1:
            obs_vs_null = "obs > null"
        else:
            obs_vs_null = "obs >> null (enriched)"

        print(f"{cat:<12} {len(cat_obs):>5} {mean_obs:>12.3e} {median_obs:>12.3e} {mean_p:>10.3f} {obs_vs_null:>20}")

        results.append({
            'metapath': metapath,
            'category': cat,
            'n': len(cat_obs),
            'mean_obs_dwpc': mean_obs,
            'median_obs_dwpc': median_obs,
            'std_obs_dwpc': cat_obs.std(),
            'mean_pvalue': mean_p,
            'median_pvalue': np.median(cat_pvals),
            'obs_vs_null': obs_vs_null
        })

# Create summary DataFrame
df = pd.DataFrame(results)

print(f"\n\n{'='*80}")
print("SUMMARY: Path Length Effect")
print(f"{'='*80}")

# Add path length
length_map = {'CbGpPW': 3, 'CtDaG': 3, 'GiGaD': 3,
              'CbGpPWpG': 4, 'CtDaGiG': 4, 'CbGpPWpGaD': 5}
df['path_length'] = df['metapath'].map(length_map)

# Group by path length and category
for length in [3, 4, 5]:
    print(f"\nPath Length {length}:")
    subset = df[df['path_length'] == length]
    for cat in categories:
        cat_data = subset[subset['category'] == cat]
        if len(cat_data) > 0:
            mean_p = cat_data['mean_pvalue'].mean()
            mean_dwpc = cat_data['mean_obs_dwpc'].mean()
            print(f"  {cat:<12}: mean_p={mean_p:.3f}, mean_dwpc={mean_dwpc:.3e}")

print(f"\n\n{'='*80}")
print("SUMMARY: Degree Category Effect")
print(f"{'='*80}")

for cat in categories:
    cat_data = df[df['category'] == cat]
    if len(cat_data) > 0:
        mean_p = cat_data['mean_pvalue'].mean()
        mean_dwpc = cat_data['mean_obs_dwpc'].mean()
        depleted = (cat_data['mean_pvalue'] > 0.7).sum()
        enriched = (cat_data['mean_pvalue'] < 0.3).sum()
        print(f"{cat:<12}: mean_p={mean_p:.3f}, mean_dwpc={mean_dwpc:.3e}, depleted={depleted}/enriched={enriched}")

# Save to CSV
df.to_csv(results_dir / 'observed_vs_null_analysis.csv', index=False)
print(f"\n\nSaved results to {results_dir / 'observed_vs_null_analysis.csv'}")
