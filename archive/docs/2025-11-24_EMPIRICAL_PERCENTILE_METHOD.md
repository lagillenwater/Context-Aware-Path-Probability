# Empirical Percentile P-Value Method

**Date:** 2025-11-24
**Context:** Detailed explanation of the empirical percentile method as replacement for gamma-hurdle

## What is the Empirical Percentile Method?

The empirical percentile method is a **non-parametric** approach to calculating p-values that directly uses the observed null distribution without fitting any parametric model.

**Core idea:** Compare the observed value directly to the empirical null distribution and count how many null values are as extreme or more extreme.

## Mathematical Definition

For an observed DWPC value `x_obs` and a null distribution of N samples `{x_1, x_2, ..., x_N}`:

```
p-value = (number of null values >= x_obs) / N
        = (1 + sum(x_i >= x_obs)) / (N + 1)
```

**The "(N+1)" correction** (optional):
- Some formulations use `(count + 1) / (N + 1)` to avoid p=0
- Conservative adjustment that ensures p-value is never exactly 0
- Het.io paper uses the simpler `count / N` formulation

## Step-by-Step Example

### Scenario: Calculate p-value for a Compound-Disease pair

**Setup:**
- Observed DWPC in Hetionet: 0.0025
- Null distribution from 20 permutations × 100 samples = 2,000 null values
- Degree category: High-High

**Step 1: Collect null distribution**
From permutations 1-20, for this (Compound, Disease) pair's degree category:
```
Null values: [0.0001, 0.0003, 0.0015, 0.0018, 0.0021, 0.0028, 0.0030, ...]
Total: 2,000 values
```

**Step 2: Count extreme values**
```
Count how many null values >= 0.0025 (observed)
Count = 450
```

**Step 3: Calculate p-value**
```
p-value = 450 / 2000 = 0.225
```

**Interpretation:** 22.5% of null permutations have DWPC as high or higher than Hetionet, suggesting this is not an unusually strong connection.

## Comparison to Gamma-Hurdle Method

### Gamma-Hurdle Approach

**Steps:**
1. Collect null DWPC values: {x_1, x_2, ..., x_N}
2. **Fit parametric model:**
   - Separate zeros from non-zeros
   - Estimate λ (proportion non-zero) via method-of-moments
   - Fit gamma(α, β) to non-zero values via method-of-moments
3. **Calculate p-value from fitted model:**
   ```
   p-value = λ × Gamma_survival(x_obs; α, β)
   ```

**Problems:**
- Method-of-moments parameter estimation is biased
- Bias scales with sample size (r = 0.807)
- Assumes data follows zero-inflated gamma distribution
- Small samples → unstable parameters
- Large samples → systematically biased parameters

### Empirical Percentile Approach

**Steps:**
1. Collect null DWPC values: {x_1, x_2, ..., x_N}
2. **Directly count extreme values:**
   ```
   count = sum(x_i >= x_obs)
   ```
3. **Calculate p-value:**
   ```
   p-value = count / N
   ```

**Advantages:**
- No parametric assumptions (distribution-free)
- No parameter estimation (no method-of-moments bias)
- No sample size bias
- Guaranteed correct calibration if null is representative
- Simple to understand and implement

## Why It Avoids Gamma-Hurdle Problems

### Problem 1: Sample Size Bias (r = 0.807)

**Gamma-hurdle:**
- Small samples (n=10): Unstable parameters → low p-values
- Large samples (n=100): Biased parameters → high p-values

**Empirical percentile:**
- Small samples (n=10): Uses 200 null values (20 perms × 10 samples)
- Large samples (n=100): Uses 2,000 null values (20 perms × 100 samples)
- **No parameter estimation** → no bias
- P-value is simply the fraction of null exceeding observed
- Sample size only affects resolution, not bias

### Problem 2: Distributional Assumptions

**Gamma-hurdle:**
- Assumes DWPC follows zero-inflated gamma
- If true distribution is different → poor fit → biased p-values
- Heavy tails, multimodality, or other features not captured

**Empirical percentile:**
- Makes **no assumptions** about distribution shape
- Works for any distribution (gamma, log-normal, multimodal, etc.)
- As long as null represents the null hypothesis, calibration is correct

### Problem 3: Method-of-Moments Bias

**Gamma-hurdle:**
- Method-of-moments is known to produce biased estimates
- Bias depends on sample size, skewness, and outliers
- Cannot be fixed without changing estimation method

**Empirical percentile:**
- **No parameter estimation** at all
- Direct comparison to empirical distribution
- No estimation bias possible

## Implementation Details

### Basic Implementation

```python
def calculate_empirical_pvalue(observed, null_values):
    """
    Calculate empirical percentile p-value.

    Parameters
    ----------
    observed : float
        Observed DWPC value.
    null_values : np.ndarray
        Null DWPC values from permutations.

    Returns
    -------
    pvalue : float
        Empirical p-value.
    """
    n_null = len(null_values)
    n_extreme = np.sum(null_values >= observed)
    pvalue = n_extreme / n_null
    return pvalue
```

### Handling Edge Cases

**1. Observed value is minimum (all null values > observed):**
```
n_extreme = 0
p-value = 0 / N = 0.0
```
This is correct - observed is the least extreme value.

**2. Observed value is maximum (all null values < observed):**
```
n_extreme = N
p-value = N / N = 1.0
```
This is correct - observed is the most extreme value.

**3. Conservative adjustment (optional):**
```python
def calculate_empirical_pvalue_conservative(observed, null_values):
    n_null = len(null_values)
    n_extreme = np.sum(null_values >= observed)
    # Add 1 to numerator and denominator to avoid p=0
    pvalue = (n_extreme + 1) / (n_null + 1)
    return pvalue
```

This ensures minimum p-value = 1/(N+1) instead of 0.

### Degree-Stratified Implementation

```python
def calculate_empirical_pvalues_stratified(
    observed_dwpcs,
    null_dwpcs_by_category,
    categories
):
    """
    Calculate degree-stratified empirical p-values.

    Parameters
    ----------
    observed_dwpcs : np.ndarray
        Observed DWPC values.
    null_dwpcs_by_category : dict
        Null DWPC arrays keyed by category tuple (e.g., ('Low', 'High')).
    categories : np.ndarray
        Category for each observed DWPC.

    Returns
    -------
    pvalues : np.ndarray
        Empirical p-values.
    """
    pvalues = np.zeros(len(observed_dwpcs))

    for category, null_dwpcs in null_dwpcs_by_category.items():
        # Get observations in this category
        mask = categories == category

        if not np.any(mask):
            continue

        # Calculate p-values for this category
        observed_in_category = observed_dwpcs[mask]
        pvalues_in_category = np.array([
            calculate_empirical_pvalue(obs, null_dwpcs)
            for obs in observed_in_category
        ])

        pvalues[mask] = pvalues_in_category

    return pvalues
```

## Resolution and Precision

### Minimum P-Value

The minimum non-zero p-value depends on the number of null samples:

**Current setup (20 permutations × 100 samples):**
- N = 2,000 null values per category
- Minimum p-value = 1 / 2,000 = **0.0005**

**If using all 200 permutations (100 samples):**
- N = 20,000 null values per category
- Minimum p-value = 1 / 20,000 = **0.00005**

**Comparison to gamma-hurdle:**
- Gamma-hurdle can produce any p-value (continuous distribution)
- But biased, so precise but inaccurate
- Empirical percentile has coarser resolution but accurate

### P-Value Resolution

With N null samples, possible p-values are:
```
0/N, 1/N, 2/N, 3/N, ..., N/N
```

**Example with N=2,000:**
```
Possible p-values: 0.0000, 0.0005, 0.0010, 0.0015, ..., 1.0000
Resolution: 0.0005 increments
```

**Is this sufficient?**
- For significance testing at α=0.05: YES (can distinguish p<0.05 from p>0.05)
- For exact p-value reporting: Somewhat coarse
- For ranking: Ties possible

### Increasing Resolution

**Option 1: Use all 200 permutations**
- Current: N = 2,000, resolution = 0.0005
- Possible: N = 20,000, resolution = 0.00005
- 10× improvement in resolution

**Option 2: Increase samples per category**
- Current: 100 samples × 20 perms = 2,000
- Possible: 500 samples × 20 perms = 10,000
- Resolution = 0.0001

**Option 3: Interpolation (advanced)**
- Fit smooth curve through empirical CDF
- Interpolate p-value between discrete steps
- Not strictly non-parametric anymore

## Advantages

### 1. No Parametric Assumptions
- Works for any distribution shape
- No need to choose distribution family
- No model misspecification

### 2. No Parameter Estimation Bias
- Direct comparison to empirical null
- No method-of-moments, no MLE
- Guaranteed unbiased if null is representative

### 3. Simple and Interpretable
- Easy to understand: "What fraction of nulls exceed observed?"
- Easy to explain to reviewers
- No hidden assumptions

### 4. Guaranteed Calibration
- If null permutations truly represent null hypothesis
- Then p-values are uniformly distributed under null
- Calibration is automatic, not dependent on fitting quality

### 5. Robust
- Not sensitive to outliers (unlike parameter estimation)
- Works for small and large samples
- No special handling needed for zeros

## Disadvantages

### 1. Coarse Resolution
- P-values are discrete: 0, 1/N, 2/N, ...
- Cannot distinguish between observations with same rank
- Matters most for very small p-values

**Mitigation:**
- Use more permutations (200 instead of 20)
- For most applications, resolution of 0.0005 is sufficient

### 2. Requires Many Null Samples
- Need sufficient permutations to estimate tails accurately
- Small N → unreliable p-values for extreme observations

**Current status:**
- N = 2,000 per category is reasonable
- N = 20,000 (if using all 200 perms) is excellent

### 3. Minimum P-Value Limitation
- Cannot get p-value smaller than 1/N
- If true p = 0.0001 but N = 2,000, report p = 0.0005

**Mitigation:**
- Use conservative language: "p < 0.0005"
- Or increase N by using all permutations

### 4. Computationally Simple (not really a disadvantage!)
- Some might argue it's "too simple"
- But simplicity is a feature, not a bug

## Expected Validation Results

### Permutation 0 Test (Calibration Check)

If empirical percentile is working correctly:

**Expected p-value distribution:**
- Each degree category should have mean p ≈ 0.50
- Overall distribution should be uniform [0, 1]
- No systematic bias by degree category
- No correlation with sample size (r ≈ 0)

**Comparison to gamma-hurdle results:**

| Category | Gamma-Hurdle Mean P | Empirical Method Mean P (Expected) |
|----------|--------------------|------------------------------------|
| Low-Low | 0.374 | 0.50 ± 0.05 |
| Medium-Medium | 0.430 | 0.50 ± 0.05 |
| High-High | 0.741 | 0.50 ± 0.05 |

**Kolmogorov-Smirnov test:**
- Gamma-hurdle: p < 0.001 (reject uniformity)
- Empirical method: p > 0.05 (accept uniformity)

### Hetionet vs Permutations Test (Biological Signal)

After confirming calibration is correct, compare Hetionet to permutations:

**Expected pattern:**
- Some metapaths show Hetionet mean p < Permutation 0 mean p (biological signal)
- Effect sizes should be clearer with proper calibration
- Can finally trust which paths show real biological enrichment

## Implementation in Current Pipeline

### Minimal Changes Needed

**File:** `src/dwpc_pvalue_validation/pvalue_calculation.py`

Add new function:
```python
def calculate_empirical_pvalue(observed, null_values):
    """Calculate empirical percentile p-value."""
    n_extreme = np.sum(null_values >= observed)
    pvalue = n_extreme / len(null_values)
    return pvalue
```

**File:** `src/dwpc_pvalue_validation/experiment.py`

Add parameter to switch methods:
```python
def run_experiment(
    metapath,
    observed_source,
    null_sources,
    method='gamma_hurdle',  # or 'empirical'
    ...
):
    if method == 'empirical':
        # Use empirical percentile
    else:
        # Use gamma-hurdle (existing code)
```

**File:** `scripts/23_dwpc_pvalue_validation.py`

Add command-line flag:
```python
parser.add_argument(
    '--method',
    choices=['gamma_hurdle', 'empirical'],
    default='gamma_hurdle',
    help='P-value calculation method'
)
```

### Testing Strategy

**Phase 1: Single metapath validation**
```bash
python scripts/23_dwpc_pvalue_validation.py \
    --metapaths CbGpPW \
    --method empirical \
    --n-samples 100 \
    --null-perms 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
```

Check Permutation 0 calibration for this metapath.

**Phase 2: All metapaths**
Run all 6 metapaths with empirical method.

**Phase 3: Compare methods**
Side-by-side comparison:
- Gamma-hurdle p-values vs empirical p-values
- Calibration quality metrics
- Biological signal detection

## Statistical Properties

### Under Null Hypothesis (Permutation 0)

If observed is from same distribution as null:

**Theoretical p-value distribution:**
- Discrete uniform on {0, 1/N, 2/N, ..., 1}
- Mean = 0.50
- Variance = (N+1) / (12N) ≈ 1/12 ≈ 0.083

**Calibration metrics:**
```
Mean p-value: 0.50 (expected)
Median p-value: 0.50 (expected)
Std p-value: 0.289 (expected)
KS test p-value: > 0.05 (expected)
```

### Power Considerations

**Question:** Does empirical percentile have less power than parametric methods?

**Answer:** No, if null is correct.
- Parametric methods can be more powerful IF model is correct
- But gamma-hurdle model is WRONG (we've proven this)
- Wrong model → biased p-values → false conclusions
- Empirical percentile: Correct model-free approach
- Slight loss of resolution but no loss of validity

## Summary

**Empirical percentile method:**
- Directly compares observed to empirical null distribution
- No parametric assumptions
- No parameter estimation
- No sample size bias
- Guaranteed calibration if null is representative

**Key formula:**
```
p-value = (count of null values >= observed) / (total null values)
```

**Advantages over gamma-hurdle:**
- Eliminates method-of-moments bias
- Eliminates sample size correlation (r = 0.807 → r ≈ 0)
- Eliminates distributional assumptions
- Simple and transparent

**Trade-off:**
- Coarser resolution (discrete p-values)
- But: accuracy > precision

**Next step:**
Implement and test on Permutation 0 to verify it produces mean p ≈ 0.50 across all degree categories.
