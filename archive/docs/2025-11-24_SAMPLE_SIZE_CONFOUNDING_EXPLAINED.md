# Sample Size Confounding in Diagnostic Analysis

**Date:** 2025-11-24
**Context:** Explaining the r = 0.807 sample size correlation and its implications

## Important Context

Degree stratification is **diagnostic only** - used to test whether gamma-hurdle p-value calibration varies by node degree. This is NOT (to our knowledge) how production p-values are calculated.

Therefore, the sample size confounding affects:
- Our ability to diagnose where/why gamma-hurdle fails
- Our interpretation of calibration test results

It does NOT affect:
- Production p-values (assuming they use global, non-stratified nulls)
- DWPC calculations
- Biological conclusions from Hetionet analysis

## What is the Sample Size Confounding?

### The Observation

In the diagnostic validation (scripts 23-24 and analysis script 27), we found:
- **Sample size vs mean p-value: r = +0.807** (very strong correlation)
- **Sample size vs calibration error: r = -0.030** (no correlation)

This means:
- Degree categories with n=7-10 samples: mean p ≈ 0.05-0.20
- Degree categories with n=20-40 samples: mean p ≈ 0.40-0.60
- Degree categories with n=80-100 samples: mean p ≈ 0.85-1.00

### Why This Is Confounding

We designed the diagnostic test to answer: **"Does gamma-hurdle calibration vary by node degree?"**

But the test simultaneously varies two things:
1. **Node degree** (Low-Low vs Medium-Medium vs High-High)
2. **Sample size** (varies from n=2 to n=100 due to network topology)

Example degree categories and their sample sizes:

| Metapath | Category | Sample Size | Mean P-value | Interpretation |
|----------|----------|-------------|--------------|----------------|
| CtDaG | Medium-Medium | 2 | 0.014 | Small n, low p |
| CbGpPW | High-High | 19 | 0.442 | Medium n, medium p |
| CbGpPWpG | High-High | 100 | 0.976 | Large n, high p |

When we observe poor calibration in High-High categories, we cannot determine:
- Is it because gamma-hurdle fails for high-degree nodes? (degree effect)
- Is it because we only have n=7-19 samples to fit parameters? (sample size effect)
- Is it both?

**The r = 0.807 correlation suggests sample size dominates over degree as a predictor of p-value bias.**

## Why Does Sample Size Affect P-values?

### Mechanism

The gamma-hurdle method uses method-of-moments parameter estimation:

```
Given n null DWPC values: {x_1, x_2, ..., x_n}

lambda_hat = n_nonzero / n
alpha_hat = (n-1) * sum(x_i) / [n * sum(x_i^2) - (sum(x_i))^2]
beta_hat = (n-1)/n * [n * sum(x_i) / [n * sum(x_i^2) - (sum(x_i))^2]]
```

**Parameter stability depends critically on n:**

1. **Small n (e.g., n=7)**:
   - High variance in parameter estimates
   - Sensitive to outliers
   - Denominator [n * sum(x_i^2) - (sum(x_i))^2] can be unstable
   - Fitted distribution may not represent true null
   - Result: **Biased toward low p-values**

2. **Large n (e.g., n=100)**:
   - More stable parameter estimates
   - Less sensitive to outliers
   - But: Different bias pattern emerges
   - Result: **Biased toward high p-values**

### Why the Bias Direction Flips

This is the puzzling finding: small samples → low p-values, large samples → high p-values.

**Hypothesis 1: Variance underestimation in small samples**
- Small samples underestimate true null variance
- Fitted gamma distribution is too narrow
- Observed values appear in the tail → low p-values

**Hypothesis 2: Variance overestimation in large samples**
- Large samples capture more outliers
- Fitted gamma distribution is too wide
- Observed values appear near the center → high p-values

**Hypothesis 3: Degree correlation**
- High-degree categories genuinely have different DWPC distributions
- Sample size is a proxy for degree effects
- But the r = 0.807 is too strong for this to be the full explanation

**Hypothesis 4: Method-of-moments bias**
- Method-of-moments is known to have bias that scales with sample size
- For gamma distributions specifically, MoM produces systematically biased alpha/beta
- Bias direction depends on distribution skewness and sample size

**Most likely: Combination of all four**, with hypothesis 4 (MoM bias) being primary.

## Why Degree and Sample Size Are Confounded

### The Network Topology Constraint

In Hetionet (or any degree-preserved permutation):
- Number of Low-Low degree pairs = (# low-degree nodes)^2 → LARGE
- Number of High-High degree pairs = (# high-degree nodes)^2 → SMALL

This is fundamental to scale-free networks:
- Most nodes have low degree (power-law tail)
- Few nodes have high degree (hubs)

**Therefore:**
- When we sample 100 pairs from Low-Low: Easy, plenty available
- When we sample 100 pairs from High-High: Difficult, might only have 7 pairs total

**This confounding cannot be eliminated** while using degree stratification, because:
- Degree distribution determines category sizes
- We cannot artificially "create" more high-degree pairs
- Oversampling (sampling same pair multiple times) would violate independence

## Implications for Diagnostic Interpretation

### What We Can Conclude

**Confirmed:**
- Gamma-hurdle method produces systematically biased p-values
- Bias scales strongly with sample size (r = 0.807)
- Calibration fails across all degree categories (97.4% have error > 0.05)

**Uncertain:**
- Does gamma-hurdle fail worse for high-degree pairs specifically?
- Or is apparent degree effect just sample size effect?
- Would increasing sample size fix high-degree calibration?

### What We Cannot Conclude from This Analysis

**Cannot conclude:** "Gamma-hurdle fails specifically for high-degree pairs"
- Because high-degree categories have small sample sizes
- Sample size alone predicts poor calibration
- Need to deconfound degree and sample size

**Cannot conclude:** "Small sample size is the only problem"
- Because even well-powered categories (n=100) show poor calibration
- Large samples biased toward p≈1.0, equally far from correct 0.5
- Method-of-moments bias affects all sample sizes

## How to Deconfound Degree and Sample Size

### Option 1: Control Sample Size Across Categories

**Approach:** Artificially balance sample sizes
- Sample n=7 pairs from ALL categories (use minimum available)
- Or sample n=100 from categories that have sufficient pairs, exclude others

**Limitations:**
- Throws away data
- Might exclude entire degree ranges
- Still leaves categories with n<7

### Option 2: Use Continuous Degree, Not Bins

**Approach:** Model p-value bias as function of continuous degree
- Fit: p-value ~ f(source_degree, target_degree, sample_size)
- Test whether degree has effect after controlling for sample size

**Advantages:**
- Separates degree effect from sample size effect
- Uses all data
- No binning artifacts

### Option 3: Permutation Test for Degree Effect

**Approach:** Bootstrap analysis
1. For each degree category, resample null values with replacement
2. Create multiple bootstrap samples of varying size (n=7, 10, 20, 50, 100)
3. For each bootstrap sample size, fit gamma-hurdle and compute calibration
4. Test whether High-degree vs Low-degree differs after controlling for n

### Option 4: Focus on Permutation 0 Analysis

**Key insight:** In Permutation 0 analysis, degree should have NO effect
- Permutation 0 drawn from same distribution as Permutations 1-20
- True p-value should be 0.5 for all degree categories
- Any deviation is purely methodological failure

**Therefore:** The r = 0.807 sample size correlation in Permutation 0 is sufficient evidence that sample size (via MoM parameter estimation) is the primary issue.

## Other Pipeline Issues Beyond Sample Size Confounding

### 1. Limited Use of Available Permutations

**Issue:** Currently using 20 of 200 available permutations

**Impact:**
- Null sample size: 100 samples × 20 perms = 2,000 values per category
- Possible: 100 samples × 200 perms = 20,000 values per category
- For empirical percentile: Minimum p-value = 1/2,000 = 0.0005 (current) vs 1/20,000 = 0.00005 (possible)

**Severity:** MODERATE
- Current resolution sufficient for most applications
- But using all 200 would improve:
  - Empirical percentile precision
  - Parametric parameter stability (if using parametric methods)
  - Statistical power

**Fix:** Easy - just use all 200 permutations instead of 20

### 2. Small Sample Size Per Category

**Issue:** Only 100 node pairs sampled per category

**Impact:**
- For gamma-hurdle: Unstable parameter estimates, especially for high-variance categories
- For empirical percentile: Coarser resolution (but less critical)

**Severity:** MODERATE
- Contributes to parameter instability
- Interacts with sample size confounding issue above
- But not the root cause of failure

**Fix:** Increase to 500-1000 samples per category

**Note:** This would help with parametric methods, but empirical percentile is less sensitive to this.

### 3. Degree Binning Artifacts

**Issue:** Using tertile bins (Low/Medium/High) creates within-bin heterogeneity

**Example:**
- "High" degree bin might include degrees 20-1000
- These have very different connectivity patterns
- Within-bin variance could be high

**Impact:**
- Fitted gamma-hurdle represents average of heterogeneous subpopulations
- Poor fit to any specific degree value
- Loses information from continuous degree values

**Severity:** MINOR-MODERATE
- Contributes to poor fit quality
- But sample size bias is larger effect

**Fix:**
- Use finer stratification (5-10 bins instead of 3)
- Use continuous degree modeling
- Or use kernel smoothing / k-nearest neighbors

### 4. Potential Permutation Quality Issues

**Issue:** Have not verified that XSwap permutations are high-quality

**Potential problems:**
- Degree sequences might not be exactly preserved (small errors)
- Permutations might have residual correlation structure
- Early permutations (1-20) might be less randomized than later ones

**Impact:**
- If permutations are not fully randomized:
  - Null distributions would be biased
  - P-values would be systematically wrong
- If using perms 1-20 is systematically different from perms 100-120:
  - Our null might not be representative

**Severity:** UNKNOWN (needs investigation)
- XSwap is published method, assumed correct
- But worth verifying for this specific dataset

**Verification needed:**
- Check exact degree sequence preservation
- Test for correlation between permutation index and DWPC statistics
- Compare early permutations (1-20) to late permutations (180-200)

### 5. Within-Pair Correlation Across Permutations

**Issue:** Same node pair measured across 20 permutations might not be independent

**Example:**
For pair (Gene_A, Disease_B):
- Measure DWPC in Permutation 1: 0.0023
- Measure DWPC in Permutation 2: 0.0024
- ...
- Measure DWPC in Permutation 20: 0.0022

**Are these 20 measurements independent?**

**Possible correlation sources:**
- Structural constraints: Both Gene_A and Disease_B have fixed degrees
- Local structure: Some paths might be more "preservable" than others
- XSwap process: Sequential permutations might have carry-over effects

**Impact:**
- If measurements are positively correlated:
  - Variance of null distribution underestimated
  - P-values too liberal (too many false positives)
- Effective sample size < 20 permutations × 100 samples

**Severity:** UNKNOWN (needs investigation)
- Standard permutation testing assumes independence
- Within-pair correlation would violate this

**Verification needed:**
- For subset of pairs, compute correlation of DWPC across permutations
- Test if ICC (intraclass correlation) is significant
- If correlated: Need to adjust for effective sample size

### 6. Zero-Inflation Modeling

**Issue:** Gamma-hurdle assumes zeros are qualitatively different from small non-zeros

**Philosophical question:** Are zeros special?
- Option 1: Yes - zero means "no path exists", fundamentally different from "weak path"
- Option 2: No - zero is just the lower tail of a continuous distribution

**Impact:**
- Hurdle model (Option 1) fits separate models for zeros vs non-zeros
- Adds complexity: 3 parameters (lambda, alpha, beta) instead of 2
- Lambda parameter has high variance with small sample size
- Could introduce additional instability

**Alternative approaches:**
- Continuous distributions that allow zeros (shifted log-normal)
- Zero-inflated gamma (ZIG): Different mixture model
- Ignore zeros entirely: Calculate p-values only for non-zero observed

**Severity:** MINOR
- Contributes to model complexity
- But sample size bias dominates

### 7. Independence Assumptions

**Issue:** Multiple implicit independence assumptions

**Assumptions:**
1. Permutations are independent
2. DWPC values across permutations are independent (see #5 above)
3. Sampled node pairs within a category are independent
4. Different degree categories are independent

**Violations:**
- #1: Permutations share starting network, might have carry-over
- #2: Within-pair correlation (discussed above)
- #3: Node pairs share nodes (hub appears in many pairs)
- #4: Categories share network structure and nodes

**Impact:**
- Variance estimates might be too small
- Confidence intervals too narrow
- P-values too liberal

**Severity:** UNKNOWN
- Standard practice assumes independence
- Violations could be minor or substantial

**Verification needed:**
- Cluster-robust variance estimation
- Permutation-based variance estimation
- Sensitivity analysis

## Summary

### Primary Issue
**Gamma-hurdle method-of-moments produces biased p-values that scale with sample size (r = 0.807)**
- Cannot be fixed by improving parameter estimation
- Cannot be fixed by increasing sample size (large samples also biased, just in opposite direction)
- Must be replaced with empirical percentile method

### Diagnostic Complication
**Sample size confounding makes it difficult to determine if gamma-hurdle fails specifically for high-degree pairs**
- High-degree categories have small sample sizes
- Sample size predicts bias more strongly than degree
- Cannot separate degree effect from sample size effect in current analysis

### Other Pipeline Issues (In Order of Severity)

**High priority:**
1. Limited permutation use (20/200) - Easy fix, worthwhile
2. Small samples per category (n=100) - Moderate effort, helps parametric methods

**Medium priority:**
3. Permutation quality verification - Needs investigation
4. Within-pair correlation - Needs investigation
5. Degree binning artifacts - Inherent to approach, could use finer bins

**Low priority:**
6. Zero-inflation model - Minor complexity, not primary issue
7. Independence assumptions - Standard violations, likely minor impact

### Key Insight

**The r = 0.807 correlation tells us sample size bias is the dominant issue**, regardless of whether it interacts with degree effects. Even if we could perfectly deconfound degree and sample size, the method would still fail because both small samples (n=7) AND large samples (n=100) produce biased p-values - just in opposite directions.

This strongly implicates method-of-moments bias as the root cause, not sample size insufficiency or degree-specific failures.
