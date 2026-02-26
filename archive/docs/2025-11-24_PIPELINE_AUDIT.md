# DWPC P-Value Pipeline Audit

**Date:** 2025-11-24
**Purpose:** Systematic review of the DWPC p-value calculation pipeline to identify any issues beyond the gamma-hurdle method

## Pipeline Overview

```
1. Network Data → XSwap Permutations
2. Node Pairs → Degree Stratification → Sampling
3. DWPC Calculation (observed + null permutations)
4. Null Distribution Fitting → P-value Calculation
5. Validation & Calibration Analysis
```

## Component-by-Component Analysis

### 1. Network Permutations (XSwap)

**Purpose:** Generate degree-preserving null networks

**Implementation:**
- XSwap algorithm (Himmelstein et al.)
- Preserves node degree sequence exactly
- 200 permutations generated

**Potential Issues:**
- **Permutation quality**: Are permutations sufficiently randomized?
  - XSwap iterations might be insufficient
  - Could have residual correlation structure
- **Edge case handling**: Are self-loops and multi-edges handled correctly?
- **Verification**: Have we confirmed degree sequences are exactly preserved?

**Status:** ASSUMED CORRECT (implemented by Himmelstein lab, used in published paper)

**Verification needed:**
- [ ] Check degree sequence preservation across permutations
- [ ] Compute autocorrelation of permuted edge positions
- [ ] Verify XSwap iteration count is sufficient

### 2. Degree Stratification

**Purpose:** Bin node pairs by source and target degree quantiles

**Implementation:** `src/dwpc_pvalue_validation/sampling.py:18-46`
```python
def compute_degree_bins(degrees, quantiles):
    bins = np.quantile(degrees, quantiles)
    # Returns bin edges: [0.0, 0.33, 0.67, 1.0] → Low/Medium/High
```

**Key Details:**
- Uses quantiles: [0.0, 0.33, 0.67, 1.0] for 3 categories
- Bins computed from **all nodes**, not just connected pairs
- Results in 9 categories: Low-Low, Low-Medium, ..., High-High
- Categories have unequal sample sizes based on actual degree distribution

**Potential Issues:**
- **Bin heterogeneity**: Within each bin, degrees can span wide ranges
  - Example: "High" might include degrees 20-1000
  - Could create heterogeneous null distributions within categories
- **Quantile choice**: Why tertiles? Are they optimal?
  - Binning loses continuous degree information
  - Could use finer stratification (quartiles, quintiles)
- **Bin edge effects**: Pairs near bin boundaries might be more similar to adjacent bins

**Status:** REASONABLE BUT NOT OPTIMAL

**Alternative approaches:**
- Use continuous degree values instead of binning
- Use finer stratification (5+ categories)
- Use log-scaled degree bins for heavy-tailed distributions

### 3. Stratified Sampling

**Purpose:** Sample node pairs from each degree category

**Implementation:** `src/dwpc_pvalue_validation/sampling.py:94-170`
```python
def sample_node_pairs(
    metaedge,
    source="true",
    n_samples_per_category=100,
    ...
)
```

**Key Details:**
- Samples 100 pairs per category (configurable)
- Uses random sampling within each category
- Samples from **observed connected pairs** only

**Potential Issues:**
- **Sample size**: 100 samples per category might be insufficient for:
  - Stable gamma-hurdle parameter estimation
  - Capturing distribution tails
  - Categories with high variance
- **Sample representativeness**: Are 100 random samples representative?
  - Could miss rare but important edge types
  - No stratification within categories
- **Category size**: What if a category has < 100 pairs?
  - Code might fail or undersample
  - Small categories less reliable

**Status:** REASONABLE FOR PILOT, NEEDS SCALING

**Recommendations:**
- Increase to 500-1000 samples per category
- Verify all categories have sufficient pairs
- Consider adaptive sampling (more samples for high-variance categories)

### 4. DWPC Calculation

**Purpose:** Compute degree-weighted path counts

**Implementation:** Validated by script 25 against het.io Neo4j database

**Status:** CONFIRMED CORRECT

**Evidence:**
- Script 25 results match het.io exactly
- No issues identified

### 5. Null Distribution Generation

**Purpose:** Collect null DWPC values from permuted networks

**Implementation:** Scripts 23-24
- For each degree category:
  - Sample 100 node pairs
  - Compute DWPC on 20 permutations
  - Collect 100 × 20 = 2,000 null values per category

**Potential Issues:**
- **Number of permutations**: Using only 20 of 200 available
  - Could use all 200 for more stable estimates
  - Current: 2,000 null samples per category
  - Possible: 20,000 null samples per category
- **Permutation selection**: Are perms 1-20 representative?
  - If early permutations are less randomized, could introduce bias
  - Should verify no systematic differences across permutation indices
- **Independence**: Are null values truly independent?
  - Same node pair measured across 20 permutations
  - Permutations might have residual correlation
  - Within-pair correlation could affect variance estimates

**Status:** ADEQUATE BUT COULD BE IMPROVED

**Recommendations:**
- Use all 200 permutations instead of 20
- Verify no systematic trends across permutation indices
- Test whether within-pair correlation affects parameter estimates

### 6. Gamma-Hurdle Fitting

**Purpose:** Fit parametric distribution to null DWPC values

**Implementation:** `src/dwpc_pvalue_validation/pvalue_calculation.py:17-105`

**Status:** CONFIRMED BROKEN (see session summary for details)

**Issues:**
- Sample size bias (r = 0.807)
- Method-of-moments instability
- Poor calibration (97.4% of categories have error > 0.05)

### 7. P-Value Calculation

**Purpose:** Calculate p-value for observed DWPC given fitted distribution

**Implementation:** `src/dwpc_pvalue_validation/pvalue_calculation.py:108-165`
```python
pvalue = lambda_param * gamma_survival
```

**Potential Issues:**
- **Formula correctness**: Is this the correct formula?
  - P(DWPC >= x) = lambda × P(X >= x | X > 0)
  - Assumes independence of hurdle and gamma components
  - This is correct IF gamma-hurdle is appropriate model
- **Zero handling**: Zeros assigned p = 1.0
  - Is this correct interpretation?
  - Zeros are least extreme values, so p = 1.0 makes sense
- **Numerical precision**: Small DWPC values (1e-10) might cause issues
  - Gamma survival function evaluation at extreme values
  - Could have underflow/overflow

**Status:** FORMULA CORRECT, BUT RELIES ON BROKEN GAMMA-HURDLE FIT

## Other Potential Issues

### A. Degree Category Sample Size Confounding (DIAGNOSTIC ONLY)

**Important:** Degree stratification appears to be diagnostic only, not used in production p-value calculation. This confounding affects our ability to diagnose calibration issues, not production p-values themselves.

**Issue:** In the diagnostic validation (scripts 23-24), degree categories have inherently different sample sizes based on network topology.

**Example:**
- Low-Low category: Many low-degree nodes → large sample size (n=100)
- High-High category: Few high-degree nodes → small sample size (n=7)

**Impact on diagnostics:**
- Sample size directly affects gamma-hurdle parameter stability
- Creates confounding between degree and sample size in our ability to test calibration
- When we see poor calibration in High-High categories, we cannot distinguish:
  - Is gamma-hurdle failing for high-degree pairs specifically?
  - Is gamma-hurdle failing due to small sample size (n=7)?
  - Both?

**Evidence:** Script 27 diagnostics show r = 0.807 correlation between sample size and p-value

**Does this affect production?**
- NO - if production uses global (non-stratified) null distributions
- YES - if production also uses degree-stratified nulls
- UNCLEAR - need to verify production pipeline methodology

### B. Degree Stratification Philosophy

**Question:** Is degree stratification the right approach?

**Current approach:**
- Stratify by (source_degree_bin, target_degree_bin)
- Fit separate null distribution for each stratum
- Compare observed to stratum-specific null

**Alternative approaches:**

**Option 1: Continuous degree regression**
- Model p-value as function of continuous (deg_source, deg_target)
- Fit single parametric model across all pairs
- No binning artifacts, no sample size confounding
- But: requires more complex modeling

**Option 2: Global null with degree covariates**
- Fit single null distribution to all pairs
- Include degree as covariate in model
- Example: log(DWPC) ~ Gamma(alpha(deg_s, deg_t), beta(deg_s, deg_t))

**Option 3: Non-parametric local smoothing**
- For each observed pair, find k nearest neighbors by degree
- Use empirical null from neighbors across permutations
- No parametric assumptions, adaptive to local structure

**Current status:** Degree stratification is reasonable but creates artifacts

### C. Zero-Inflation Handling

**Issue:** Many DWPC values are exactly zero (no paths exist)

**Current approach:** Gamma-hurdle model (mixture of point mass at 0 + gamma)

**Potential Issues:**
- Hurdle model assumes zeros are qualitatively different from small non-zeros
- But zeros might just be the tail of a continuous distribution
- Lambda parameter (proportion non-zero) has high variance for small samples

**Alternative approaches:**
- Zero-inflated gamma (ZIG): Models zeros as part of distribution
- Continuous distributions that allow zeros: Log-normal with shift
- Ignore zeros completely: Calculate p-values only for non-zero observed

**Current status:** Zero-inflation modeling adds complexity and potential instability

### D. Independence Assumptions

**Issue:** Multiple assumptions of independence in the pipeline

**Independence assumptions:**
1. Permutations are independent of each other
2. DWPC values across permutations are independent
3. Sampled node pairs within a category are independent
4. Categories are independent (no spillover effects)

**Potential violations:**
- Permutations might have residual correlation structure
- Same node pair measured across permutations is not independent
- High-degree hubs appear in multiple categories
- Categories share edge information (same network)

**Impact:**
- Variance estimates might be biased
- P-values might be overly confident
- Standard errors underestimated

**Verification needed:**
- [ ] Compute autocorrelation of permuted networks
- [ ] Test independence of DWPC values across permutations
- [ ] Check for correlation between categories

## Summary of Findings

### Confirmed Issues
1. **Gamma-hurdle method**: Sample size bias (r = 0.807), must be replaced
2. **Sample size confounding**: Inherent to degree stratification approach
3. **Limited null samples**: Using 20/200 permutations, 100 samples per category

### Potential Issues Needing Investigation
1. **Permutation quality**: Verify XSwap randomization is sufficient
2. **Bin heterogeneity**: Check variance within degree bins
3. **Independence violations**: Test correlation assumptions
4. **Zero-inflation**: Verify hurdle model is appropriate

### Design Decisions to Reconsider
1. **Degree binning vs continuous**: Could avoid binning artifacts
2. **Stratified vs global null**: Trade-offs between specificity and stability
3. **Parametric vs non-parametric**: Empirical percentile avoids all distributional assumptions

## Recommendations

### Immediate Priority
**Replace gamma-hurdle with empirical percentile method**
- Eliminates all parametric assumptions
- Eliminates sample size bias
- Simple to implement and interpret
- Only limitation: Coarser p-value resolution (min p = 1/2000)

### Short-Term Improvements
1. **Use all 200 permutations** instead of 20
   - Increases null samples from 2,000 to 20,000 per category
   - Improves empirical percentile resolution (min p = 1/20,000 = 0.00005)
2. **Increase samples per category** to 500-1000
   - Better represents category distributions
   - More stable parameter estimates (if using parametric methods)
3. **Verify permutation quality**
   - Check degree sequence preservation
   - Test for residual correlation structure

### Long-Term Considerations
1. **Alternative stratification approaches**
   - Test continuous degree regression
   - Compare global vs stratified nulls
2. **Independence testing**
   - Verify permutation independence
   - Test category independence
3. **Benchmark alternative methods**
   - Compare empirical vs parametric approaches
   - Evaluate calibration quality across methods

## Conclusion

Beyond the confirmed gamma-hurdle failure, the pipeline has several design decisions that could be optimized:

1. **Degree stratification creates inherent sample size confounding** that cannot be eliminated without changing the approach
2. **Limited use of available data** (20/200 permutations, 100 samples/category) reduces statistical power
3. **Parametric assumptions** (gamma-hurdle, zero-inflation) add complexity and failure modes

**The empirical percentile method eliminates most of these issues** by avoiding parametric assumptions entirely. However, the fundamental sample size confounding from degree stratification will persist unless we move to continuous degree modeling or global null distributions.

The pipeline is fundamentally sound in its logic (DWPC calculation, permutation testing, degree stratification), but the implementation choices (gamma-hurdle, limited samples, binning) introduce unnecessary complications and biases.
