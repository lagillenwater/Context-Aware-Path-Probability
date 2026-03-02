# Phase 2: Critical Findings and Issues

## Summary of Work Done

### Completed:
1. ✓ Data exploration for 5 metapaths
2. ✓ Feature extraction verification (Sets A-F)
3. ✓ Training data preparation (30 datasets)
4. ✓ Hybrid evaluation pipeline implementation
5. ✓ Residual analysis and visualizations

### Results:
- **Validation r = 0.71** (on 18 bin combinations)
- **Test r = NaN** (on 28,277 individual pairs)
- **Test predictions are CONSTANT** (no variance)

---

## The Fundamental Problem

### Training vs Testing Mismatch

**Training:**
- 72 degree bin combinations (aggregated statistics)
- Features: degree bins + intermediate node signature histogram
- Target: mean pathway count per bin
- Model learns: bin 3×5 → pathway count ~0.05

**Testing:**
- 28,277 individual (source, target) node pairs
- Features: SAME (degree bins + histogram)
- Target: actual pathway count for specific pair
- Model predicts: **CONSTANT VALUE** for all pairs

###Why Predictions Are Constant

**Root cause:** All test pairs within the same degree bin combination have **IDENTICAL FEATURES**.

Example:
```
Bin 3×5 contains 1000 node pairs:
  (compound_50, pathway_100)
  (compound_51, pathway_105)
  (compound_52, pathway_110)
  ...

All 1000 pairs have SAME features:
  feat_source_bin = 3
  feat_target_bin = 5
  feat_hist_in0_out0 = 795
  ... (same histogram)

Therefore model predicts SAME value for all 1000 pairs!
```

**This is by design** - the model is trained to predict bin-level means, not pair-specific counts.

---

## Why Validation Worked But Testing Failed

**Validation (r=0.71):**
- Predicting for held-out degree BINS
- Each bin is a different prediction task
- 18 bins → 18 different predictions → variance

**Testing (r=NaN):**
- Predicting for individual PAIRS
- Pairs in same bin get same prediction → constant within bins
- If test pairs are concentrated in few bins → all get similar predictions
- → No variance → r = NaN

---

## What The Benchmark (r=0.94) Actually Did

Looking at the documented benchmark that achieved r=0.94, it likely did ONE of:

1. **Trained on individual pairs** (not bins)
   - Millions of training samples
   - Node-specific features (PageRank, betweenness, node IDs)
   - Would violate null model assumption

2. **Evaluated on bin-level predictions** (not pairs)
   - Test on held-out bins
   - r=0.94 for bin means
   - NOT individual pair predictions

3. **Used different features** that vary within bins
   - Actual node degrees (not bins)
   - Edge-specific features
   - Local topology beyond degree

---

## Possible Solutions

### Option 1: Change Task to Bin-Level Prediction
**What:** Evaluate on held-out degree bins (not individual pairs)

**Pros:**
- Matches training task
- Already achieved r=0.71 validation

**Cons:**
- Not what user requested ("evaluate on pairs")
- Limited test samples (~20 held-out bins)
- Not practically useful for individual pair predictions

---

### Option 2: Add Pair-Specific Features
**What:** Include features that vary within degree bins

**Examples:**
- Actual node degrees (not bins)
- Local clustering coefficients
- Neighbor overlap
- Edge weights / timestamps

**Pros:**
- Model can differentiate pairs within same bin
- More realistic for pair prediction

**Cons:**
- Violates current feature design (only uses bin + histogram)
- Requires major feature extraction rewrite
- May overfit with small training data

---

### Option 3: Train on Individual Pairs
**What:** Generate millions of training samples from original Hetionet

**Approach:**
- Sample 100k pairs from original graph
- Extract features + pathway counts
- Train on individual pairs (not bins)
- Test on permutation pairs

**Pros:**
- Solves constant prediction issue
- Matches testing format

**Cons:**
- Huge dataset (~millions of pairs)
- Training time >> current
- Overfitting risk with 116 features
- May memorize original graph structure

---

### Option 4: Hybrid Binning Strategy
**What:** Use finer-grained bins or continuous degree features

**Approach:**
- Replace binary bins (0-9) with actual degree values
- Use 100 bins instead of 10
- Add degree × degree interaction terms

**Pros:**
- More granularity within coarse bins
- Keeps bin-based training

**Cons:**
- Still fundamentally bin-based
- May not solve constant prediction issue
- Sparse bins with 100×100 = 10k combinations

---

## Recommendation

**We have discovered a fundamental flaw in the hybrid approach:**

The current design trains on aggregated bins but tests on individual pairs using features that don't distinguish pairs within the same bin. This makes individual pair prediction impossible - the model can ONLY predict bin-level means.

**Two paths forward:**

### Path A: Acknowledge Limitation, Report Bin-Level Results
- Evaluate feature sets (A-F) on held-out bins
- Report: "r=0.71 for bin-level prediction with 31 features"
- Honest about limitations
- Quick to complete

### Path B: Redesign for Pair-Level Prediction
- Add pair-specific features (actual degrees, local topology)
- Train on sample of individual pairs
- Much more work, uncertain outcome
- May achieve r≥0.95 but violates original design

**Given the coding errors in Phase 1 and this fundamental issue in Phase 2, I recommend:**

1. **Stop and discuss with user** before proceeding
2. Clarify what the original r=0.94 benchmark actually measured
3. Decide if bin-level or pair-level prediction is the actual goal
4. Potentially simplify to linear regression baseline

---

## Files Generated

**Data:**
- `results/phase2_training_data/*.csv` (30 files)

**Scripts:**
- `explore_metapath_data.py`
- `test_feature_extraction.py`
- `prepare_training_data.py`
- `run_phase2_hybrid_evaluation.py`
- `run_full_test_evaluation.py`

**Visualizations:**
- `results/phase2_visualizations/CbGpPW/CbGpPW_A_diagnostics.png`
- `results/phase2_visualizations/CbGpPW/CbGpPW_A_performance.png`

**Documentation:**
- `PHASE2_FEATURE_SUMMARY.md`
- `PHASE2_FINDINGS.md` (this file)
