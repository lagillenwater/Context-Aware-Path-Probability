# Corrected Summary: 2025-11-03

## Critical Bug Discovered and Fixed

**Bug**: Edge matrices stored as `dtype=bool` instead of numeric types
**Impact**: Boolean matrix multiplication performed logical OR instead of counting pathways
**Result**: All pathway counts were 0 or 1 instead of actual counts (0 to 48+)

**Discovery**: User noticed maximum pathway count of 1 in all visualizations, which is physically impossible for high-degree node pairs.

---

## Original Results (INVALID - Boolean Bug)

**Task**: Predict binary pathway presence/absence

| Metapath | Baseline r | Corrected r | Improvement |
|----------|-----------|-------------|-------------|
| CbGpPW | 0.8846 | 0.9363 | +5.8% |
| CtDaG | 0.9027 | 0.9489 | +5.1% |
| CrCbG | 0.9118 | 0.9692 | +6.3% |

**Original conclusion**: Degree-aware correction essential to reach r > 0.90.

---

## Corrected Results (VALID - Integer Matrices)

**Task**: Predict actual pathway counts (0 to 48+ for CbGpPW)

| Metapath | Baseline r | Corrected r | Change |
|----------|-----------|-------------|--------|
| CbGpPW | **0.9910** | 0.9859 | -0.5% (worse) |
| CtDaG | **0.9769** | 0.9746 | -0.2% (worse) |
| CrCbG | **0.9913** | 0.9898 | -0.1% (worse) |

**Corrected conclusion**: Baseline Linear Regression alone achieves r = 0.99. Correction not needed.

---

## Why Results Changed Dramatically

### Prediction Task

**With bug (boolean)**:
- Target: y ∈ {0, 1} (binary presence/absence)
- Feature informativeness: Weak (r = 0.88)
- Correction benefit: Helps reach r = 0.94

**Without bug (integer)**:
- Target: y ∈ {0, 1, 2, ..., 48} (actual counts)
- Feature informativeness: Strong (r = 0.99)
- Correction benefit: None (decreases performance)

### Why Correction Hurts

**Permutation 000 correlation with validation**:
- Binary task: r = 0.56-0.71 (low but correction still helps)
- Count task: r = 0.76-0.86 (higher but adding to r=0.99 baseline hurts)

**Effect**: Adding noise (r=0.80) to excellent signal (r=0.99) decreases performance.

### Heteroscedasticity

**With bug**:
- Correction reduces heteroscedasticity (0.62 → 0.55)

**Without bug**:
- Correction increases heteroscedasticity (0.55 → 0.72)

---

## Pathway Count Statistics (Corrected)

### CbGpPW (Compound-binds-Gene-participates-Pathway)

- Range: 0 to 48 shared genes
- Mean (nonzero pairs): 2.09 pathways
- Median: 1 pathway
- 95th percentile: 6 pathways
- 99th percentile: 12 pathways

**Distribution**: Highly skewed, most pairs have 1-2 shared genes, but high-degree pairs can have 48+.

---

## Corrected Conclusions

### What We Learned

**Valid findings**:
- Degree features (d_u, d_v, d_u×d_v, d_u², d_v²) highly predictive
- Simple Linear Regression achieves r = 0.98-0.99
- All metapaths exceed r > 0.90 target
- Method generalizes across diverse metapaths

**Invalid findings** (artifacts of bug):
- Baseline inadequate (r = 0.88) - FALSE
- Correction needed to reach r > 0.90 - FALSE
- Two-stage model improves performance - FALSE

### Simplified Approach for Anomaly Detection

**Use baseline Linear Regression**:
1. Train on 5 degree features
2. Achieves r = 0.99 without correction
3. Faster and simpler (single stage)

**Tomorrow's workflow**:
1. Predict expected counts: E[y] = baseline model
2. Compute variance: Var[y] from permutations 1-20
3. Calculate z-scores: z = (y_obs - E[y]) / sqrt(Var[y])
4. Identify anomalies: |z| > 3

---

## Files Updated

**Documentation**:
- `docs/2025-11-03_BUG_CORRECTION.md` - Detailed bug analysis
- `docs/2025-11-03_PAIR_LEVEL_PHASE2_RESULTS.md` - Updated with corrected results
- `docs/2025-11-03_CORRECTED_SUMMARY.md` - This document

**Scripts corrected**:
- `test_src/run_pair_level_phase2.py`
- `test_src/run_pair_level_phase2_all_metapaths.py`
- `test_src/run_pair_level_phase2_detailed_analysis.py`

**Fix applied**: Added `.astype(np.int32)` to `load_edge_matrix()` function.

**Results regenerated**:
- All visualization PNGs corrected
- All CSV result files corrected

---

## Mathematical Formulation (Still Valid)

The mathematical formulation of the Linear Regression model remains correct:

**Baseline model**:
$$\hat{y} = \mathbf{X}\boldsymbol{\beta} + \beta_0$$

where $\mathbf{x} = [d_u, d_v, d_u \cdot d_v, d_u^2, d_v^2]^T$

**What changed**: Only the target variable y
- Before (bug): y ∈ {0, 1}
- After (fix): y ∈ {0, 1, 2, ..., 48}

The model itself is unchanged, only the data type of the matrices.

---

## Lesson Learned

**Data type validation is critical**:

```python
# Add validation to loading function
def load_edge_matrix(edge_type, perm_id='original'):
    matrix = sp.load_npz(filepath)

    # Validate dtype
    if matrix.dtype == bool:
        warnings.warn(f"Converting {edge_type} from bool to int32")
        matrix = matrix.astype(np.int32)

    # Sanity check pathway counts
    if edge_type not in ['edge_list']:
        pathway_matrix = compute_test_pathways(matrix)
        assert pathway_matrix.data.max() > 1, \
            f"Pathway counts suspiciously low (max={pathway_matrix.data.max()})"

    return matrix
```

**Always visualize data ranges**: The bug was caught by noticing max pathway count = 1 in plots.

---

## Impact on Tomorrow's Work

**Positive impacts**:
1. Simpler approach (no correction needed)
2. Better performance (r = 0.99 vs 0.95)
3. Faster (single-stage model)
4. More interpretable (fewer components)

**No negative impacts**: Can proceed directly to anomaly detection with confidence.

---

## Final Status

**Pair-level null distribution prediction**: COMPLETE
- Baseline Linear Regression: r = 0.98-0.99
- All metapaths exceed target (r > 0.90)
- Correction method not needed
- Ready for anomaly detection

**Next steps**:
1. Variance modeling or direct computation
2. Z-score calculation
3. Anomaly identification
4. Biological interpretation
