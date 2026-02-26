# Critical Bug Fix: Boolean vs Integer Matrix Multiplication

**Date**: 2025-11-03
**Status**: CORRECTED
**Severity**: High - Results completely invalidated

---

## Bug Description

### Problem

Edge matrices were stored as `dtype=bool` instead of numeric types. When multiplying boolean sparse matrices to compute pathway counts, the operation performed logical OR instead of counting shared intermediates.

**Result**: All pathway counts were either 0 (no path) or 1 (path exists), regardless of the actual number of shared intermediate nodes.

### Discovery

During documentation review, user noticed that all visualizations showed maximum true pathway count of 1.0, which is impossible for high-degree node pairs that should have dozens of shared intermediates.

### Root Cause

```python
# Original (WRONG)
edge1 = sp.load_npz('data/edges/CbG.sparse.npz')  # dtype=bool
edge2 = sp.load_npz('data/edges/GpPW.sparse.npz')  # dtype=bool
pathway_matrix = edge1 @ edge2  # dtype=bool, max value = 1

# Fixed (CORRECT)
edge1 = sp.load_npz('data/edges/CbG.sparse.npz').astype(np.int32)
edge2 = sp.load_npz('data/edges/GpPW.sparse.npz').astype(np.int32)
pathway_matrix = edge1 @ edge2  # dtype=int32, max value = 48 for CbGpPW
```

### Verification

```python
# CbGpPW pathway counts
# BEFORE (bug): max=1, mean=1.0 (all nonzero values = True)
# AFTER (fix): max=48, mean=2.09, 95th percentile=6, 99th percentile=12
```

---

## Impact on Results

### Original Results (INVALID - based on boolean bug)

**Task**: Predict binary pathway presence/absence (0 or 1)

| Metapath | Baseline r | Corrected r | Improvement |
|----------|-----------|-------------|-------------|
| CbGpPW | 0.8846 | 0.9363 | +5.8% |
| CtDaG | 0.9027 | 0.9489 | +5.1% |
| CrCbG | 0.9118 | 0.9692 | +6.3% |

**Interpretation (WRONG)**:
- Baseline predicts binary presence with r = 0.88-0.91
- Degree-aware correction pushes to r = 0.94-0.97
- Correction is essential for achieving r > 0.90

### Corrected Results (VALID - after bug fix)

**Task**: Predict actual pathway counts (0 to 48+ for CbGpPW)

| Metapath | Baseline r | Corrected r | Improvement |
|----------|-----------|-------------|-------------|
| CbGpPW | **0.9910** | 0.9859 | **-0.5%** |
| CtDaG | **0.9769** | 0.9746 | **-0.2%** |
| CrCbG | **0.9913** | 0.9898 | **-0.1%** |

**Interpretation (CORRECT)**:
- Baseline predicts actual pathway counts with r = 0.98-0.99
- Degree-aware correction **decreases** performance
- Correction is **not needed** - baseline already exceeds target

---

## Why Results Changed So Dramatically

### Binary Prediction (bug) vs Count Prediction (correct)

**With boolean matrices**:
- Target: y ∈ {0, 1}
- Task difficulty: Binary classification
- Feature informativeness: Degrees weakly predict presence (r = 0.88)
- Correction benefit: Systematic bias in binary predictions, correction helps

**With integer matrices**:
- Target: y ∈ {0, 1, 2, ..., 48}
- Task difficulty: Count regression
- Feature informativeness: Degrees strongly predict counts (r = 0.99)
- Correction benefit: None - permutation 000 adds noise (r_perm0_val = 0.76-0.86)

### Why Correction Helped with Bug, Hurts Without

**With bug (binary)**:
- Permutation 000 correlation with validation: r ≈ 0.56-0.71
- Despite low correlation, correction learned systematic structure
- Binary task benefits from any additional signal

**Without bug (counts)**:
- Permutation 000 correlation with validation: r ≈ 0.76-0.86
- Higher correlation BUT baseline already r = 0.99
- Adding perm 000 noise (r = 0.80) to baseline (r = 0.99) decreases performance
- Correction increases heteroscedasticity: 0.55 → 0.72 for CbGpPW

---

## Heteroscedasticity Analysis

### Before Bug Fix (binary prediction)

Heteroscedasticity (correlation between |residuals| and predicted values):
- CbGpPW: 0.619 baseline → 0.549 corrected (improvement)
- CtDaG: 0.660 baseline → 0.561 corrected (improvement)
- CrCbG: 0.609 baseline → 0.529 corrected (improvement)

**Interpretation**: Correction reduced heteroscedasticity

### After Bug Fix (count prediction)

Heteroscedasticity:
- CbGpPW: 0.551 baseline → **0.722 corrected** (worse)
- CtDaG: 0.573 baseline → **0.738 corrected** (worse)
- CrCbG: 0.633 baseline → 0.647 corrected (slightly worse)

**Interpretation**: Correction **increases** heteroscedasticity

---

## Pathway Count Statistics (After Fix)

### CbGpPW (Compound-binds-Gene-participates-Pathway)

**Original graph pathway counts** (nonzero only):
- Count: 71,653 pairs with pathways
- Min: 1
- Max: 48
- Mean: 2.09
- Median: 1
- 75th percentile: 2
- 90th percentile: 4
- 95th percentile: 6
- 99th percentile: 12

**Interpretation**: Most pairs have 1-2 shared genes, but some high-degree pairs have up to 48 shared genes.

### Distribution Characteristics

- **Highly skewed**: Median = 1, but long tail extends to 48
- **Degree-dependent**: High-degree pairs have more pathways
- **Predictable from degrees**: Linear regression on degrees achieves r = 0.99

---

## Corrected Conclusions

### What Changed

**Original conclusion (INVALID)**:
- Baseline Linear Regression achieves r = 0.88-0.91
- Degree-aware correction essential to reach r = 0.94-0.97
- Two-stage model required for accurate null prediction

**Corrected conclusion (VALID)**:
- Baseline Linear Regression achieves r = 0.98-0.99
- Degree-aware correction **not needed** (decreases performance)
- Simple single-stage model sufficient for accurate null prediction

### What Stayed the Same

**Still true**:
- Degree features (5 features: d_u, d_v, d_u × d_v, d_u², d_v²) are highly predictive
- All metapaths exceed r > 0.90 target
- Can proceed to anomaly detection

**No longer true**:
- Correction is needed
- Permutation 000 provides useful signal for pair-level prediction
- Two-stage model improves performance

### Implications for Tomorrow's Work

**Simplified approach**:
1. Use baseline Linear Regression (5 degree features)
2. No correction stage needed
3. Already achieves r = 0.98-0.99
4. Faster and simpler implementation

**Anomaly detection**:
- Predict expected pathway count: E[y] from baseline model
- Compute variance from permutations: Var[y] from perms 1-20
- Calculate z-score: z = (y_observed - E[y]) / sqrt(Var[y])
- Identify anomalies: |z| > 3

---

## Files Corrected

**Scripts updated** (added `.astype(np.int32)` to `load_edge_matrix()`):
- `test_src/run_pair_level_phase2.py`
- `test_src/run_pair_level_phase2_all_metapaths.py`
- `test_src/run_pair_level_phase2_detailed_analysis.py`

**Results regenerated**:
- `results/pair_level_phase2_detailed/CbGpPW_detailed_analysis.png` (corrected)
- `results/pair_level_phase2_detailed/CtDaG_detailed_analysis.png` (corrected)
- `results/pair_level_phase2_detailed/CrCbG_detailed_analysis.png` (corrected)
- `results/pair_level_phase2_detailed/detailed_analysis_summary.csv` (corrected)
- `results/pair_level_phase2_all/phase2_all_metapaths_results.csv` (corrected)
- `results/pair_level_phase2_all/phase2_all_metapaths_comparison.png` (corrected)

**Documentation to update**:
- `docs/2025-11-03_PAIR_LEVEL_PHASE2_RESULTS.md` (invalidated, needs major revision)
- `docs/2025-11-03_SUMMARY.md` (invalidated, needs major revision)

---

## Lessons Learned

### Data Type Validation

**Problem**: Assumed edge matrices were numeric types
**Solution**: Always verify dtype after loading sparse matrices

```python
# Add validation
edge_matrix = sp.load_npz(filepath)
assert edge_matrix.dtype in [np.int32, np.float32, np.float64], \
    f"Expected numeric dtype, got {edge_matrix.dtype}"
```

### Sanity Checks

**Problem**: Did not verify pathway count ranges
**Solution**: Always check summary statistics of computed values

```python
# Add sanity checks
pathway_counts = edge1 @ edge2
print(f"Pathway count range: {pathway_counts.data.min()} - {pathway_counts.data.max()}")
assert pathway_counts.data.max() > 1, "Pathway counts suspiciously low"
```

### Visualization Review

**Success**: User caught the bug by noticing max pathway count = 1 in plots
**Lesson**: Always review visualizations for physically impossible values

---

## Status

- [COMPLETE] Bug identified
- [COMPLETE] Root cause diagnosed
- [COMPLETE] Fix implemented in all scripts
- [COMPLETE] Results regenerated with corrected code
- [IN PROGRESS] Documentation being updated
- [PENDING] Final validation of corrected results

---

**Conclusion**: The boolean dtype bug caused us to solve a binary classification problem (r = 0.88) when we should have been solving a count regression problem (r = 0.99). The correction method was effective for the binary problem but unnecessary for the count problem. With corrected data, simple baseline Linear Regression already achieves r = 0.98-0.99 without any correction needed.
