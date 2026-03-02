# Pair-Level Jaccard Similarity Results

**Date**: 2025-11-03
**Status**: Pending Execution
**Analysis**: Pair-level pathway prediction with Jaccard features

---

## Executive Summary

**Approach**: Linear Regression with Jaccard + degree features (8 total)
- Train on original Hetionet pairs
- Test on held-out original pairs
- Validate on permutation averages

**Results**: FAILURE - Model does not generalize from original to permutations

**Outcome**:
- Train/test on original: r = 0.9965 (excellent)
- Validation on permutations: r = -0.0071 (complete failure)

**Conclusion**: Original graph structure fundamentally different from null distribution

---

## Methodology

See `docs/2025-11-03_PAIR_LEVEL_JACCARD_PLAN.md` for full methodology.

**Key points**:
- 50,000 pairs sampled from original graph
- 80/20 train/test split
- 8 features: degrees + Jaccard similarity + interactions
- Validation: Compare predictions vs permutations 1-20 means

---

## Results

### Performance Metrics

| Metric | Train | Test | Validation |
|--------|-------|------|------------|
| r | 0.9957 | **0.9965** | **-0.0071** |
| RMSE | 0.1807 | 0.1620 | 2.1734 |
| Bias | +0.0000 | -0.0004 | +1.0185 |

**Sample sizes**:
- Train: 40,000 pairs
- Test: 10,000 pairs
- Validation: 10,000 pairs

**Pathway count statistics**:
- Original graph mean: 1.07 pathways/pair
- Permutation mean: 0.05 pathways/pair
- **XSwap destroys ~95% of pathways**

### Feature Importance

Model coefficients:

| Feature | Coefficient |
|---------|-------------|
| d_u | +0.007031 |
| d_v | -0.000018 |
| d_u×d_v | +0.000001 |
| d_u² | -0.000011 |
| d_v² | +0.000000 |
| **jaccard** | **-0.578544** |
| **jaccard×d_u** | **+0.654815** |
| **jaccard×d_v** | **+0.960538** |
| Intercept | +0.002908 |

**Interpretation**:
- Jaccard interaction terms dominate (largest coefficients)
- Model heavily relies on Jaccard similarity
- This works for original graph but fails for permutations

---

## Visualizations

[TO BE COMPLETED]

Figure: `results/pair_level_jaccard/jaccard_analysis.png`

6-panel visualization showing:
- Row 1: Predicted vs observed (train, test, validation)
- Row 2: Residual analysis (train, test, validation)

---

## Interpretation

### Success Criteria

- Test r > 0.85: **PASS** (r = 0.9965)
- Validation r > 0.85: **FAIL** (r = -0.0071)

### Key Findings

1. **Original graph structure != null distribution**
   - Model perfectly learns original graph (r = 0.9965)
   - Completely fails on permutations (r = -0.0071)
   - Jaccard similarity in original graph does not transfer to permutations

2. **XSwap dramatically reduces pathway counts**
   - Original: 1.07 pathways/pair (mean)
   - Permutations: 0.05 pathways/pair (mean)
   - 95% reduction in pathways
   - Degree-preserving randomization destroys biological clustering

3. **Biological signal dominates**
   - Original graph has strong clustering (PMI ≈ 7 from previous work)
   - Genes in same pathway cluster together
   - XSwap breaks these clusters while preserving degrees
   - Model trained on clustering cannot predict random structure

4. **Jaccard features capture biological structure**
   - Largest coefficients: jaccard×d_v (+0.96), jaccard×d_u (+0.65)
   - Model learns: high Jaccard → high pathway count (in original)
   - But in permutations: Jaccard is random noise
   - Biological Jaccard != random Jaccard

### Why This Approach Failed

**Fundamental mismatch**:
- Training data: Original graph with biological clustering
- Validation data: Random graph with degrees preserved but no clustering
- These are different distributions

**The problem**: We want to predict null (random) from biological (clustered)
- Like training on signal+noise, testing on noise alone
- Model learns signal, which doesn't exist in null

**Mathematical interpretation**:
```
Original: pathway_count = f(degrees, biological_clustering)
Null:     pathway_count = g(degrees)  # no clustering

Model learns f but we need g
```

---

## Comparison to Previous Approaches

| Approach | Train r | Val r | Data Leakage | Generalizes to Null? |
|----------|---------|-------|--------------|---------------------|
| Compositional | - | 0.35 | No | Poor (independence) |
| Phase 2 (buggy) | 0.99 | 0.99 | **Yes** | Unknown (circular) |
| **Jaccard** | **0.9965** | **-0.0071** | **No** | **No (bio≠null)** |
| Bin-level Phase 5b | 0.99+ | 0.99+ | ? | Needs checking |

**Key insight**: All successful previous approaches either:
1. Had data leakage (Phase 2)
2. Trained on permutations (bin-level?)
3. Used compositional formula (failed at r=0.35)

**This approach proves**: Cannot predict null from original structure alone

---

## Conclusions

### Main Conclusion

**Training on original Hetionet CANNOT predict permutation null distributions.**

**Why**:
1. Original graph has biological clustering (modules, pathways, functional groups)
2. XSwap preserves degrees but destroys clustering
3. Models trained on clustering cannot generalize to random structure
4. Jaccard similarity captures biological overlap, not random overlap

### Implications

**For anomaly detection**:
- Cannot use original-trained models to predict null expectations
- Must train on permutations themselves
- This requires running permutations (defeating the goal of avoiding them)

**For null modeling**:
- Need to model random graph structure, not biological structure
- Options:
  1. Train on permutation data (Phase 2 approach, but fix data leakage)
  2. Use analytical formulas (compositional failed, DP might fail too)
  3. Use bin-level predictions (already works but can't identify specific pairs)

### What We Learned

1. **Biological clustering is strong**: 95% of pathways disappear in XSwap
2. **Structure != degrees**: Degree features alone insufficient without clustering
3. **Jaccard captures biology**: Works perfectly for original, fails for null
4. **Original vs null are different problems**: Cannot solve one from the other

---

## Next Steps

### Recommended Approach: Return to Permutation-Based Training

Since we cannot predict null from original structure, we must train on permutations:

**Option A: Fix Phase 2 data leakage**
- Train on mean(perms 1-10)
- Validate on mean(perms 11-20)
- Use only degree features (5 features)
- Expected: r = 0.80-0.90 (lower than buggy r=0.99 but honest)

**Option B: Use bin-level approach**
- Already working (r > 0.99)
- Limitation: Cannot identify specific anomalous pairs
- Use for screening, then targeted analysis

**Option C: Hybrid approach**
- Bin-level for bulk scoring
- Permutation-based pair-level for candidates
- Best of both worlds

### Not Recommended

- Adding more link prediction features (won't solve bio≠null problem)
- Training on more original data (fundamental mismatch)
- Compositional formulas (already failed at r=0.35)

### Critical Question

**Do we really need pair-level predictions?**

If goal is anomaly detection:
- Bin-level (r > 0.99) might be sufficient
- Within-bin variance is high but outliers still detectable
- Computationally much faster

If goal is biological discovery:
- Need pair-specific predictions
- Must accept 10-20 permutations requirement
- Cannot avoid this computational cost

---

## Files Generated

- Script: `test_src/run_pair_level_jaccard.py`
- Results: `results/pair_level_jaccard/jaccard_results.csv`
- Visualization: `results/pair_level_jaccard/jaccard_analysis.png`
- Plan: `docs/2025-11-03_PAIR_LEVEL_JACCARD_PLAN.md`
- This document: `docs/2025-11-03_PAIR_LEVEL_JACCARD_RESULTS.md`
