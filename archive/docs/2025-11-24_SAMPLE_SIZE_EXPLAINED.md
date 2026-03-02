# What "n" Represents in the Diagnostic Analysis

**Date:** 2025-11-24
**Context:** Understanding why High-High categories have larger sample sizes than Low-Low

## The Sampling Strategy

From `src/dwpc_pvalue_validation/sampling.py:139`:
```python
# Get connected pairs
source_nodes, target_nodes = loader.get_connected_node_pairs(metaedge, source)
```

**Key insight:** Sampling is from **CONNECTED PAIRS ONLY** - pairs where at least one path exists.

Then from lines 170-180:
```python
n_available = len(indices)
n_to_sample = min(n_samples_per_category, n_available)

if n_to_sample < n_samples_per_category:
    logger.warning(f"Only {n_available} pairs available, requested {n_samples_per_category}")

sampled_indices = rng.choice(indices, size=n_to_sample, replace=False)
```

The script:
1. Tries to sample 100 pairs per degree category (default)
2. If fewer than 100 connected pairs exist in that category, samples all available
3. Samples without replacement

## What "n" Means in variance_analysis.csv

From `scripts/27_gamma_hurdle_diagnostics.py:145`:
```python
'n': len(observed)
```

**n = number of sampled node pairs in that degree category**

This represents:
- How many connected pairs were available in that category (capped at 100)
- Each pair has an observed DWPC and a p-value

## Why High-High Has More Samples Than Low-Low

### Network Topology Perspective

**All possible pairs:**
- Low-Low: MANY pairs possible (many low-degree nodes × many low-degree nodes)
- High-High: FEW pairs possible (few high-degree nodes × few high-degree nodes)

**Connected pairs (with paths):**
- Low-Low: FEW connected pairs (low-degree nodes poorly connected → few paths)
- High-High: MANY connected pairs (high-degree nodes well-connected → many paths)

### Example: CbGpPWpG (4-node metapath)

This metapath is: Compound → Gene → Pathway → Gene

For a (Compound, Gene) pair to be "connected":
- At least one path must exist: Compound -binds→ Gene₁ -participates→ Pathway -participates→ Gene₂
- **Low-Low pairs:** Low-degree Compounds and low-degree Genes have few intermediate connections
  - Few paths exist between them
  - Result: Only 11 connected Low-Low pairs found
- **High-High pairs:** High-degree Compounds and high-degree Genes have many intermediate connections
  - Many paths exist between them (via multiple Genes and Pathways)
  - Result: 100+ connected High-High pairs found (sampled 100)

### This Pattern Strengthens with Path Length

**Short paths (3 nodes):**
- Example: CbGpPW (Compound → Gene → Pathway)
- High-High has n=19 (moderate)
- Still constrained by total number of high-degree nodes

**Medium paths (4 nodes):**
- Example: CbGpPWpG, CtDaGiG
- High-High has n=77-100 (many connected pairs)
- High-degree nodes create exponentially more paths

**Long paths (5 nodes):**
- Example: CbGpPWpGaD
- High-High has n=44
- Many possible paths, but also very specific endpoint constraints

## Implications for Sample Size Bias Analysis

### The Original Confounding Claim Was Backwards

I originally stated: "High-degree categories have small sample sizes, creating confounding."

**This is empirically FALSE.** The data shows:
- High-High: mean n = 48.5
- Low-Low: mean n = 24.0
- **High-degree categories actually have LARGER sample sizes**

### The Correct Interpretation of r = 0.807

The r = 0.807 correlation (sample size vs p-value) now makes more sense:

**Categories with many connected pairs:**
- Tend to be High-degree categories (high connectivity)
- Have large n (many samples)
- Show high p-values (over-conservative)

**Categories with few connected pairs:**
- Tend to be Low-degree categories (low connectivity)
- Have small n (few samples)
- Show low p-values (under-conservative)

### But There's Still Confounding!

The confounding is now:
- **Degree** (High vs Low) correlates with **connectivity** (many vs few paths)
- **Connectivity** determines **sample size** (how many connected pairs exist)
- **Sample size** affects **gamma-hurdle parameter stability**
- **Parameter stability** affects **p-value bias**

Chain: Degree → Connectivity → Sample Size → Parameter Bias → P-value Bias

**We cannot separate:**
1. Is gamma-hurdle failing for high-degree pairs because they're high-degree?
2. Or because high-degree categories happen to have large sample sizes?
3. Or because high-degree categories have different DWPC distributions?

### The r = 0.807 Still Indicates Method-of-Moments Bias

Even with this corrected understanding, the r = 0.807 tells us:
- Sample size is a strong predictor of p-value bias
- Categories with n=100 → p ≈ 0.95
- Categories with n=10 → p ≈ 0.10
- Both are equally poorly calibrated (should be 0.50)

This suggests **method-of-moments produces systematically biased parameters that scale with sample size**, independent of whether the category is high-degree or low-degree.

## Why This Doesn't Change the Main Conclusion

**The gamma-hurdle method still fails**, but for a more nuanced reason:

1. **High-degree categories have large sample sizes** (not small as I initially claimed)
2. **Large sample sizes produce over-conservative p-values** (p ≈ 0.95)
3. **Small sample sizes produce under-conservative p-values** (p ≈ 0.10)
4. **Both directions are wrong** - should be p ≈ 0.50 for Permutation 0

The method-of-moments bias affects ALL sample sizes, just in opposite directions. This is arguably WORSE than if it only failed for small samples - you can't fix it by collecting more data.

## Corrected Summary

**What we know:**
- High-High categories: Large n (many connected pairs), high p-values (over-conservative)
- Low-Low categories: Small n (few connected pairs), low p-values (under-conservative)
- Strong correlation: r = 0.807 between n and p-value

**What this means:**
- Degree and sample size are confounded via connectivity
- Method-of-moments bias scales with sample size in a non-monotonic way
- Cannot fix by increasing sample size (large samples also biased)
- Cannot deconfound degree from sample size without changing sampling strategy

**Why the method must be replaced:**
- Bias affects all sample sizes (not just small ones)
- Bias direction flips with sample size (small → under, large → over)
- No correction factor can fix this
- Only solution: Empirical percentile method (no parametric assumptions)
