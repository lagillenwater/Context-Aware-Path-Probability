# Permutation 0 as Proxy for Null Distribution - Results

**Date**: 2025-11-03
**Status**: Complete
**Analysis**: Test if permutation 0 can serve as proxy for null distribution (perms 1-20 mean)

---

## Executive Summary

**Question**: Can we learn transformation from original → perm 0, then use perm 0 as proxy for mean(perms 1-20)?

**Phase 1 Result**: **FAIL** - Permutation 0 is NOT a good proxy
- Correlation: r(perm 0, mean perms 1-20) = 0.8386
- Threshold: 0.9
- Decision: Skip Phase 2 (learning transformation)

**Conclusion**: Permutation 0 differs sufficiently from null mean that using it as proxy would introduce error

---

## Methodology

### Phase 1: Validate Perm 0 as Proxy

**Test**: Do pathway counts in permutation 0 correlate with mean(perms 1-20)?

**Data**:
- Sample: 10,000 pairs from original graph
- Perm 0 counts: Computed for each pair
- Perms 1-20 mean: Average pathway count across 20 permutations

**Success criterion**: r > 0.9

---

## Results

### Phase 1: Correlation Analysis

| Metric | Value |
|--------|-------|
| Correlation | **0.8386** |
| RMSE | 0.3835 |
| Bias | -0.0057 |
| Perm 0 mean | 0.16 pathways/pair |
| Perms 1-20 mean | 0.17 pathways/pair |

**Result**: r = 0.8386 < 0.9 → **FAIL**

### Interpretation

**r = 0.84 is good but not excellent**:
- Strong positive correlation
- But 16% of variance unexplained
- Too much error for high-precision null modeling

**Residual analysis**:
- Bias nearly zero (-0.0057)
- No systematic offset between perm 0 and mean
- But substantial scatter (RMSE = 0.38)

**Visualization findings**:
- Most pairs cluster near perfect agreement line
- But outliers show perm 0 can differ by ±2-4 pathways
- For pairs with high pathway counts, perm 0 less reliable

---

## Why Permutation 0 Differs from Null Mean

### Hypothesis 1: Residual Structure

**Permutation 0**:
- Only 1 XSwap iteration from original
- May retain some original graph structure
- Not fully randomized

**Permutations 1-20**:
- Many XSwap iterations
- Fully randomized (within degree constraints)
- True null distribution

**Evidence**:
- Perm 0 mean (0.16) slightly lower than perms 1-20 mean (0.17)
- Suggests perm 0 has slightly less connectivity
- First permutation may "overcorrect" from original

### Hypothesis 2: Stochastic Variation

**Permutation 0**:
- Single realization of random process
- Has its own random fluctuations

**Perms 1-20 mean**:
- Average of 20 realizations
- Fluctuations averaged out
- More stable estimate

**Evidence**:
- RMSE = 0.38 shows substantial pair-level variation
- This is noise inherent to finite sampling

---

## Implications

### For Transformation Learning Approach

**Would not have worked even if we proceeded to Phase 2**:
- Learning original → perm 0 transformation
- Then using perm 0 as proxy for null mean
- Error from r=0.84 would propagate:
  - Imperfect prediction of perm 0 from original
  - Imperfect correspondence of perm 0 to null mean
  - Compound errors likely r < 0.7 overall

### For Phase 2 Correction (October 31 work)

**This explains why Phase 2 correction failed after bug fix**:
- Phase 2 used perm 0 to correct baseline predictions
- But perm 0 ≠ null mean (r = 0.84)
- Adding perm 0 signal to baseline added noise, not signal
- Correction decreased performance (r = 0.99 → 0.98)

---

## Comparison to Previous Findings

### Original vs Null (from Jaccard experiment)

| Comparison | Correlation |
|------------|-------------|
| Original vs Perm 0 | Not tested |
| Original vs Perms 1-20 mean | -0.007 (complete failure) |
| **Perm 0 vs Perms 1-20 mean** | **0.84 (moderate)** |

**Key insight**:
- Perm 0 is much closer to null than original is
- But still not close enough to be good proxy
- r = 0.84 vs needed r > 0.9

### Transformation Feasibility

**What we learned**:
1. Original → null: Impossible (r = -0.007)
2. Perm 0 → null: Possible but imprecise (r = 0.84)
3. Need direct null modeling (train on perms 1-20)

---

## Alternative: Use Perm 0 for Calibration?

### Potential Approach

Instead of using perm 0 as proxy, use it for **calibration**:

1. Train model on original → original
2. Predict on original features
3. Calibrate: `calibrated = α × predicted + β`
   - Learn α, β from perm 0 data
   - Adjust scale and offset

**Expected result**: Still poor
- r = 0.84 ceiling from perm 0 ≠ null
- Calibration can't add information, only adjust scale
- Likely r ≈ 0.5-0.7 after calibration

**Not recommended**: Better to train directly on permutations

---

## Conclusions

### Main Findings

1. **Permutation 0 moderately correlates with null (r = 0.84)**
   - Good but not excellent
   - Too much error for high-precision modeling

2. **Cannot use perm 0 as proxy for null mean**
   - Fails r > 0.9 threshold
   - Would introduce ~16% unexplained variance

3. **Explains Phase 2 correction failure**
   - Perm 0 added noise, not signal
   - This is why correction hurt performance after bug fix

### Recommendations

**Do NOT pursue transformation learning approach**:
- Original → perm 0: Hard (biological vs partially random)
- Perm 0 → null: Imprecise (r = 0.84)
- Compound error too large

**DO use direct permutation-based training**:
- Train on mean(perms 1-10)
- Validate on mean(perms 11-20)
- Both from same distribution (true null)
- Expected: r = 0.80-0.90 (honest, no data leakage)

### Why 20 Permutations Are Needed

**Cannot reduce to 1 permutation (perm 0)**:
- Single permutation has stochastic noise
- Mean of 20 permutations averages out noise
- r(perm 0, mean perms) = 0.84 shows this noise
- Need multiple permutations for stable null estimate

**Minimum permutations**:
- At least 10 for training
- At least 10 for validation
- Total: 20 permutations minimum

---

## Files Generated

- Script: `test_src/run_pair_level_perm0_proxy.py`
- Results: `results/pair_level_perm0_proxy/phase1_results.csv`
- Visualization: `results/pair_level_perm0_proxy/phase1_perm0_proxy_test.png`
- This document: `docs/2025-11-03_PERM0_PROXY_TEST_RESULTS.md`

---

## Summary Table of All Approaches Tested Today

| Approach | Train Source | Val Source | Val r | Can Avoid Perms? |
|----------|--------------|------------|-------|------------------|
| Jaccard | Original | Perms 1-20 | -0.007 | **NO** |
| Degrees only | Original | Perms 1-20 | -0.013 | **NO** |
| Perm 0 proxy | - | - | 0.84* | **NO** |
| Phase 2 (buggy) | Perms 1-20 | Perms 1-20 | 0.99 | No (data leakage) |
| Bin-level | Perms? | Perms? | 0.99+ | No (uses perms) |

*Correlation between perm 0 and null mean, not a trained model

**Final conclusion**: Cannot avoid using permutations for null modeling. Minimum requirement is 20 permutations (10 train, 10 validation).
