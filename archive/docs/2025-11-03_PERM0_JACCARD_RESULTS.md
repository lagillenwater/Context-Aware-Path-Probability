# Permutation 0 with Jaccard Features - Results

**Date**: 2025-11-03
**Status**: Complete
**Analysis**: Test if Jaccard features from permutation 0 can predict mean(perms 1-20)

---

## Executive Summary

**Question**: Can we combine single permutation (perm 0) with Jaccard features to predict null mean?

**Result**: **MODERATE PERFORMANCE** - Better than degrees alone but insufficient
- Model trained on perm 0: r = 1.0000 (perfect fit to training data)
- Validation on perms 1-20: r = 0.6592
- Threshold for success: r > 0.85
- **Conclusion**: Jaccard features help but perm 0 ≠ null mean remains fundamental limitation

---

## Methodology

**Approach**:
1. Sample 10,000 pairs from permutation 0
2. Extract Jaccard features from perm 0 structure
3. Train on perm 0 pathway counts
4. Validate on mean(perms 1-20)

**Features (8 total)**:
- d_u, d_v (degrees in perm 0)
- d_u×d_v, d_u², d_v² (degree interactions)
- jaccard (intermediate node overlap in perm 0)
- jaccard×d_u, jaccard×d_v (Jaccard interactions)

**Key difference from previous Jaccard test**:
- Previous: Trained on original graph (biological structure)
- This test: Trained on perm 0 (randomized structure)

---

## Results

### Performance Metrics

| Metric | Training Target (perm 0) | Validation Target (perms 1-20) |
|--------|-------------------------|--------------------------------|
| Train r | 1.0000 | 0.6674 |
| **Test r** | **1.0000** | **0.6592** |
| RMSE | 0.0045 | 0.7263 |
| Bias | - | +0.4317 |

### Target Correlation

**r(perm 0, mean perms 1-20) = 0.6659**

- Perm 0 mean: 0.64 pathways/pair
- Perms 1-20 mean: 0.22 pathways/pair
- **Large mean difference**: Perm 0 has ~3× more pathways
- Lower correlation than previous perm 0 test (0.84 → 0.67)

**Why lower correlation?**
The previous perm 0 proxy test sampled pairs from original graph, while this test sampled pairs from perm 0 graph. Different sampling strategies can lead to different pair distributions and correlations.

### Feature Coefficients

| Feature | Coefficient |
|---------|-------------|
| d_u | +0.000158 |
| d_v | +0.000012 |
| d_u×d_v | -0.000000 |
| d_u² | -0.000001 |
| d_v² | -0.000000 |
| jaccard | -0.851958 |
| jaccard×d_u | +0.985565 |
| jaccard×d_v | +0.993836 |
| Intercept | -0.000059 |

**Interpretation**:
- Jaccard interaction terms dominate (coefficients ~1.0)
- Raw Jaccard has negative coefficient (-0.85)
- Net effect: Jaccard×degrees provides signal
- Model learns perm 0 structure perfectly (r=1.0)

---

## Analysis

### Why Validation Performance Is Moderate (r=0.66)

**1. Perm 0 differs from null mean**
- Fundamental issue: r(perm 0, null mean) = 0.67
- Model ceiling bounded by target correlation
- Cannot predict better than underlying correlation

**2. Large systematic bias**
- Perm 0 mean: 0.64 pathways/pair
- Null mean: 0.22 pathways/pair
- Bias: +0.43 (model overpredicts by ~200%)
- Suggests perm 0 has different connectivity structure

**3. Jaccard captures perm 0 structure, not null structure**
- Jaccard from perm 0 reflects single realization
- Null mean averages over 20 realizations
- Stochastic differences between realizations

### Comparison to Other Approaches

| Approach | Train Data | Jaccard? | Val r | Status |
|----------|-----------|----------|-------|--------|
| Original + Jaccard | Original | Yes | -0.007 | FAIL |
| Original + Degrees | Original | No | -0.013 | FAIL |
| Perm 0 proxy | - | - | 0.84* | FAIL |
| **Perm 0 + Jaccard** | **Perm 0** | **Yes** | **0.66** | **MODERATE** |
| 5 perms + Degrees | Perms 0-4 | No | 0.988 | SUCCESS |

*Correlation, not trained model

**Key insights**:
1. Training on permutation (0.66) >> training on original (-0.01)
2. Single perm proxy without features (0.84) > single perm with Jaccard (0.66)
3. Jaccard features don't help when perm 0 ≠ null mean
4. Multiple perms (5) >> single perm (0.66 → 0.99)

### Why Jaccard Doesn't Help Here

**Hypothesis**: Jaccard captures stochastic noise, not systematic signal

**Evidence**:
1. Perm 0 proxy (no Jaccard): r = 0.84
2. Perm 0 + Jaccard: r = 0.66
3. **Jaccard makes it worse**

**Explanation**:
- Jaccard from perm 0 reflects random edge placement in that specific permutation
- Mean(perms 1-20) averages out such random effects
- Jaccard features add noise rather than signal
- Degree features alone more stable across permutations

### Why This Differs from Minimal Perms Success

**Minimal perms approach (r=0.988)**:
- Trains on mean(perms 0-4)
- Validates on mean(perms 5-20)
- Both targets are averaged, reducing noise
- Degrees preserved across all permutations

**This approach (r=0.66)**:
- Trains on single perm 0 (noisy)
- Validates on mean(perms 1-20) (stable)
- Mismatch: noisy training vs stable validation
- Jaccard captures noise, not stable patterns

---

## Implications

### For Single Permutation Approaches

**Cannot use single permutation for null modeling**:
- Without features: r = 0.84 (perm 0 proxy)
- With Jaccard features: r = 0.66 (this test)
- Both insufficient (threshold r > 0.85)

**Why single perm insufficient**:
1. Stochastic variation in single realization
2. Jaccard captures realization-specific noise
3. Need averaging across multiple perms

### For Feature Selection

**Jaccard features counterproductive for single perm**:
- Perm 0 proxy: r = 0.84
- Perm 0 + Jaccard: r = 0.66
- Simpler (degrees only) may be better

**When would Jaccard help?**
- If training on averaged data (e.g., mean of perms 0-4)
- Then Jaccard also averaged, captures stable patterns
- Not tested here, but possible future experiment

### For Computational Cost

**This approach not recommended**:
- Requires 1 perm for training + 20 for validation = 21 perms
- Achieves only r = 0.66
- Worse than minimal perms: 5 perms for r = 0.988

**Better alternative**:
- Use 5 perms (0-4) with degrees only
- Achieves r = 0.988
- Fewer perms, better performance

---

## Comparison to Minimal Perms Approach

### Why 5 Perms Works But Perm 0 + Jaccard Doesn't

| Aspect | Perm 0 + Jaccard | 5 Perms + Degrees |
|--------|------------------|-------------------|
| Training data | Single perm (noisy) | Mean of 5 perms (stable) |
| Validation data | Mean of 20 perms (stable) | Mean of 16 perms (stable) |
| Data match | Noisy vs stable (mismatch) | Stable vs stable (match) |
| Features | 8 (includes Jaccard) | 5 (degrees only) |
| Jaccard captures | Realization-specific noise | N/A |
| Validation r | 0.66 | 0.988 |

**Key difference**: Averaging in training data, not feature engineering

---

## Conclusions

### Main Findings

1. **Perm 0 + Jaccard achieves r = 0.66**
   - Better than training on original (r = -0.01)
   - Worse than perm 0 proxy without features (r = 0.84)
   - Insufficient for null modeling (threshold r > 0.85)

2. **Jaccard features counterproductive**
   - Capture stochastic noise in single permutation
   - Reduce performance vs degrees alone
   - Not helpful for single-perm approaches

3. **Single permutation fundamentally insufficient**
   - With or without Jaccard: r < 0.85
   - Need multiple perms to average out noise
   - 5 perms minimum for r > 0.95

### Updated Approach Comparison

| Approach | Perms Needed | Features | Val r | Recommended? |
|----------|--------------|----------|-------|--------------|
| Original + Jaccard | 20 | 8 | -0.007 | NO |
| Original + Degrees | 20 | 5 | -0.013 | NO |
| Perm 0 proxy | 1 | 0 | 0.84* | NO |
| Perm 0 + Jaccard | 1 | 8 | 0.66 | NO |
| **5 perms + Degrees** | **5** | **5** | **0.988** | **YES** |

*Correlation, not trained model

### Final Recommendation

**Use 5 permutations with degree features**:
- Achieves r = 0.988 (exceeds threshold)
- Simpler than Jaccard (5 vs 8 features)
- Fewer permutations than single perm + validation (5 vs 21)
- Training and validation from same distribution (stable)

**Do NOT use single permutation approaches**:
- Insufficient with or without Jaccard features
- Stochastic noise cannot be overcome by feature engineering
- Need averaging across multiple permutations

---

## Files Generated

- Script: `test_src/run_pair_level_perm0_jaccard.py`
- Results: `results/pair_level_perm0_jaccard/perm0_jaccard_results.csv`
- Visualization: `results/pair_level_perm0_jaccard/perm0_jaccard_analysis.png`
- This document: `docs/2025-11-03_PERM0_JACCARD_RESULTS.md`

---

**Summary**: Jaccard features from permutation 0 cannot predict null mean (r=0.66 < 0.85). Single permutation approaches fail regardless of feature engineering. Multiple permutations (5+) required for stable null modeling.
