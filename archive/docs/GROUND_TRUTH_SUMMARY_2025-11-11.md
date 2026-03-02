# Ground Truth Summary: What Actually Happened
## Based on Comprehensive Audit - 2025-11-11

This document establishes facts based on EVIDENCE (executed notebooks, results files, scripts that ran) vs CLAIMS (documentation without supporting evidence).

---

## VALIDATED FACTS (Evidence Found)

### 1. Compositional Null FAILS
**Evidence:**
- Notebook 17 executed → r=0.35 across 7 metapaths
- Notebook 17b executed → PMI≈7, edges are dependent
- Experiment 2J corrected results: r plateaus at 0.80 ceiling
- Results file: `experiment2j_corrected_results.csv`

**Conclusion:** Compositional approaches definitively don't work for pathway prediction.

### 2. Validation Method Matters Critically
**Evidence:**
- Experiment 2L original: r=0.9076 (validated on mean of perms 11-20)
- Experiment 2L revised: r=0.692 (validated on individual perms 11-20)
- Results files: `experiment2l_results.csv` vs `experiment2l_revised_summary.csv`
- Difference: ~0.20 correlation points inflation from mean validation

**Conclusion:** Mean validation artificially inflates performance. Most previous "successes" may be inflated.

### 3. Nov 3 Minimum Permutations Work
**Evidence:**
- Results file: `minimum_perms_comparison/minimum_perms_comparison.csv`
- CbGpPW: r_val_degree=0.984 with 1 perm
- CtDaG: r_val_degree=0.967 with 1 perm
- CrCbG: r_val_degree=0.988 with 1 perm
- CbGaD: r_val_degree=0.976 with 1 perm
- CpDaG: r_val_degree=0.945 with 1 perm (failed threshold)

**VALIDATION METHOD CONFIRMED:**
- SESSION_SUMMARY explicitly states: "Validation target: Mean pathway counts from perms 6-20"
- This is MEAN validation, same issue as Experiment 2L
- Results likely inflated by ~0.20 points based on Exp 2L finding

**Status:** NEEDS RE-VALIDATION with individual perm methodology to get true performance.

### 4. Experiment 2 Series Ran (Nov 4)
**Evidence:**
- Results files exist for exp2a through exp2h (all dated Nov 4)
- exp2g: r_test=0.969, "Focused composition model" - marked SUCCESS
- exp2h: r_test=0.979, "Analytical prior composition" - marked WEAK (unclear why)
- Multiple experiments testing different compositional approaches

**Validation Method:** Unknown - need to check if these also used mean validation

**Status:** Results exist with high correlations but validation method needs verification.

### 5. Experiment 2L Shows Severe Overfitting
**Evidence:**
- Training r=0.98 on perm 0
- Validation r_mean=0.692 on individual perms 11-20
- Feature concentration: 92% importance in single feature (sum_CbGiG)

**Conclusion:** Topology-dependent aggregation features don't generalize across permutations.

### 6. Original Graph Training Works (Oct 31)
**Evidence:**
- Oct 31 RESULTS.md documents comprehensive testing
- Phase 0: Original vs perm average r=0.9559 (bin correlation)
- Phase 1: DegreeSignatureNN on original achieves r=0.9266
- Phase 2: Feature Set E (216 features) achieves r=0.9587
- Phase 5b: Degree-aware correction achieves r=0.9964-0.9987 (!!!)

**Validation Method:** Validated on mean(perms 0-19) - same mean validation issue

**Critical Finding from Phase 1b:**
- Training on perm 0 completely FAILS (r=-0.09)
- Despite bin correlation of r=0.9967
- Proves permutation training doesn't work, must use original graph

**Conclusion:** Training on original Hetionet works (r>0.95), training on perms fails (r<0).

### 7. Assortativity Lost in Permutations (Nov 1)
**Evidence:**
- Nov 1 ASSORTATIVITY_RESULTS.md documents systematic analysis
- Original Hetionet: Assortative source-intermediate connections (r=+0.20)
- Permutation 000: Assortativity destroyed (r=+0.04, Δr=-0.16)
- Edge-level disassortativity weakened by 20-22%

**Key Finding:**
- XSwap preserves degrees but NOT assortativity
- This explains 10-20% unexplained variance in models
- Pathway-level structure is biologically meaningful and lost during randomization

**Conclusion:** Permutations lose important structural properties beyond degree distribution.

---

## UNVALIDATED CLAIMS (No Evidence Found)

### 1. Pipeline 18 Achieves r=0.88
**Claim:** Multiple Oct 30 docs mention "r=0.88 baseline from Pipeline 18"

**Evidence Searched:**
- Notebooks 18a-18h: Created but NONE executed
- No results files with "pipeline_18" or "notebook_18" prefix
- No executed notebooks in `notebooks/executed/` matching 18a-18h

**Status:** NEVER ACTUALLY RUN. The r=0.88 was aspirational/planned, not achieved.

**Impact:** Major - this became a reference baseline in subsequent docs despite never being validated.

### 2. Phase 5b Two-Stage Correction - ACTUALLY VALIDATED
**Previous Status:** Listed as "UNKNOWN - mentioned in docs but can't find evidence"

**UPDATE - EVIDENCE FOUND:**
- Oct 31 RESULTS.md Phase 5b section documents complete implementation
- DegreeAwareCorrectionModel implemented in `src/degree_aware_correction.py`
- Results files exist in `results/phase5b_degree_aware_correction/`
- **Performance achieved:**
  - CbGpPW: r=0.9964 (improvement from 0.9550)
  - CtDaG: r=0.9972 (improvement from 0.9866)
  - CrCbG: r=0.9987 (improvement from 0.9863)

**Validation Method:** Validated on mean(perms 0-19) - mean validation

**Status:** VALIDATED - Phase 5b DID run and achieved r>0.99 as claimed. Two-stage degree-aware correction works. But likely also inflated by mean validation.

### 3. Binning Approach - RESOLVED
**Claim:** User mentioned "binning approach from notebook 18 was determined not to work"

**Evidence Found:**
- Pipeline 18a WAS executed on HPC for 8 metapaths (found in ~/Downloads/)
- Successfully created 10×10 binned training data
- Computed statistics per bin: mean, std, n_pairs_in_bin
- Results show: within-bin std = 0.87, mean = 1.41 (CV = 62 percent)
- Some bins contain 4,645 pairs with highly variable pathway counts

**Why it failed:**
- Bin-level prediction works (Oct 31: r>0.99 for bin averages)
- But high within-bin variation (CV=62 percent) means:
  - Cannot distinguish anomalous pairs from normal pairs within same bin
  - Bin average is useless for pair-specific anomaly detection
- Pipeline 18b-h were never executed (abandoned after discovering variation)

**Resolution:** Different tasks require different approaches
- Bin-level prediction: Works (Oct 31 showed this)
- Pair-level prediction: Required for anomaly detection (Nov 3 onwards)

**Status:** RESOLVED - No contradiction, binning works for bins but not for pairs.

---

## KEY CONTRADICTIONS

### Contradiction 1: Pipeline 18 Status - PARTIALLY RESOLVED
**Doc Claims:** "Pipeline 18 achieves r=0.88" (Oct 30 REFINEMENT_PLAN)
**Reality:** Pipeline 18a executed on HPC, 18b-h never executed
**Resolution:**
- 18a ran successfully, created binned data
- Discovered within-bin CV=62 percent (too high for pair-level)
- 18b-h abandoned, so r=0.88 was never actually measured
- Claim was aspirational, not achieved

### Contradiction 2: Nov 3 Validation Method
**Doc Claims:** "1 permutation sufficient, r>0.95" (Nov 3 SESSION_SUMMARY)
**Evidence:** Results show r>0.95 but methodology likely uses mean validation
**Concern:** May be inflated by ~0.20 points (same issue as Exp 2L)
**Status:** Needs re-validation with individual perms

### Contradiction 3: Training Data Source
**Oct 31:** Training on original graph works (r=0.93)
**Nov 3:** Training on perm 0 works (r>0.95)
**Oct 31:** Training on single perm fails (r=-0.09)
**Conflict:** Nov 3 contradicts Oct 31 Phase 1b finding
**Likely Explanation:** Nov 3 validated on mean, masking poor generalization

### Contradiction 4: Binning Efficacy - RESOLVED
**Docs:** Pipeline 18 and Oct 31 use 10×10 binning successfully
**User:** Says binning approach failed
**Resolution:** Both statements are correct for different tasks:
- Bin-level prediction: Binning works (Oct 31: r>0.99)
- Pair-level prediction: Binning fails (within-bin CV=62 percent)
- Anomaly detection requires pair-level, so binning was abandoned
**Status:** RESOLVED

---

## TIMELINE OF ACTUAL WORK

### Oct 30 (Planning Phase)
- Created refinement plan referencing Pipeline 18
- Set goal: improve from r=0.88 to r≥0.90
- **No experiments actually ran**

### Oct 31 (Comprehensive Phase Testing - Bin-Level)
- Phase 0: Original vs perm average correlation r=0.9559 (VALIDATED)
- Phase 1: DegreeSignatureNN on original achieves r=0.9266 (VALIDATED)
- Phase 1b: Training on perm 0 FAILS r=-0.09 (CRITICAL FINDING)
- Phase 2: Feature Set E optimization achieves r=0.9587 (VALIDATED)
- Phase 3: 10×10 binning optimal for bin-level (VALIDATED)
- Phase 4: Linear Regression competitive with NN (VALIDATED)
- Phase 5b: Degree-aware correction achieves r>0.99 (VALIDATED)
- **All validated on mean(perms 0-19) - mean validation issue**
- **All predict bin averages, not individual pairs**

### Pipeline 18a Execution (Between Oct 31 and Nov 3)
- Executed on HPC for 8 metapaths (evidence in ~/Downloads/)
- Created 10×10 binned training data (100 bins per metapath)
- Discovered critical flaw: within-bin CV = 62 percent
- Example CbGpPW: bin mean=1.41, std=0.87, n_pairs up to 4,645
- High variation means bin averages cannot identify anomalous pairs
- Pipeline 18b-h abandoned (training scripts never executed)
- **Motivated switch to pair-level prediction**

### Nov 1 (Assortativity Analysis)
- Comprehensive assortativity analysis completed
- Original graph: assortative source-intermediate (r=+0.20)
- Perm 0: assortativity destroyed (r=+0.04, Δr=-0.16)
- Explains 10-20% unexplained variance in models
- **Results documented in ASSORTATIVITY_RESULTS.md**

### Nov 3 (Minimum Permutations)
- Ran minimum_perms_comparison script on 5 metapaths
- Results: 1 perm achieves r>0.95 for 4/5 metapaths
- **VALIDATED on mean(perms 6-20) - MEAN VALIDATION**
- Likely inflated by ~0.20 points (same as Exp 2L)
- Needs re-validation with individual perms

### Nov 4 (Experiment 2 Series)
- Ran exp2a through exp2h (Python scripts)
- All have results files dated Nov 4
- **Results exist, need detailed analysis**

### Nov 5 (Hierarchical Prediction)
- Exp 2J: Composition ceiling confirmed (r=0.80)
- Exp 2L: Aggregation "breakthrough" (r=0.9076)
- **Both have results files**

### Nov 11 (Validation Discovery)
- Exp 2L revised: Discovered mean validation inflation
- True performance: r=0.692 (not 0.9076)
- **Critical finding that invalidates many previous claims**

---

## CURRENT STATE ASSESSMENT

### What We KNOW Works
1. **Training on original Hetionet graph** (Oct 31 Phase 1-5b)
   - Phase 1: DegreeSignatureNN r=0.9266 (validated on mean)
   - Phase 2: Feature Set E r=0.9587 (validated on mean)
   - Phase 5b: Degree-aware correction r>0.99 (validated on mean)
   - Phase 1b proves: Training on perm 0 FAILS (r=-0.09)

2. **Compositional null does NOT work** (Notebooks 17, 17b, Exp 2J)
   - Notebook 17: r=0.35 across 7 metapaths
   - Notebook 17b: PMI≈7, edges are dependent
   - Exp 2J: r plateaus at 0.80 ceiling

3. **Mean validation inflates metrics by ~0.20 points** (Exp 2L discovery)
   - Original Exp 2L: r=0.9076 on mean(perms 11-20)
   - Revised Exp 2L: r=0.692 on individual perms
   - Inflation: 0.216 correlation points

4. **Assortativity is lost in permutations** (Nov 1)
   - Original: r=+0.20 source-intermediate
   - Perm 0: r=+0.04 (82% reduction)
   - Explains 10-20% unexplained variance

### What We THINK Works (Need Re-Validation)
1. **Oct 31 Phase 1-5b results** - ALL validated on mean(perms)
   - Likely inflated by ~0.20 points
   - Need individual perm validation
   - But Phase 1b proves original graph training works better than perm training

2. **Nov 3 degree models** (1 perm, r>0.95)
   - CONFIRMED: Validated on mean(perms 6-20)
   - Likely inflated by ~0.20 points
   - True performance may be r~0.75-0.85

3. **Experiment 2 series** (Nov 4, exp2a-exp2h)
   - Exp 2G: r_test=0.969 ("SUCCESS")
   - Exp 2H: r_test=0.979 ("WEAK")
   - Validation method unknown - need to check scripts

### What We DON'T Know
1. **True performance with individual perm validation**
   - Oct 31 results all use mean validation
   - Nov 3 results use mean validation
   - Need to re-run with individual perms to get true performance

2. **Why user says binning failed**
   - Oct 31 Phase 3: 10×10 binning optimal (r=0.9587)
   - Pipeline 18 uses 10×10 binning
   - User says "binning approach was determined not to work"
   - Contradiction needs clarification

3. **What the TRUE baseline performance is**
   - Oct 31 Phase 5b: r>0.99 (but mean validation)
   - Nov 3: r>0.95 (but mean validation)
   - True performance on individual perms unknown
   - Likely r~0.75-0.85 after correcting for inflation

### What Was Never Done
1. Pipeline 18 notebooks (18a-18h) - never executed
2. Many planned analyses from Nov 5 and Nov 11
3. GNN approach (Exp 2M) - incomplete
4. Most proposed scaling tests

---

## CRITICAL QUESTIONS FOR USER

Since user provided context that:
1. "Binning approach from notebook 18 was determined not to work"
2. "Phase 5b was abandoned for a different reason"

**Questions:**
1. What specifically about binning failed? Docs show 10×10 binning optimal.
2. Why was Phase 5b abandoned if it achieved r>0.99?
3. Are you aware that Pipeline 18 notebooks never actually executed?
4. Do you recall if Nov 3 work validated on means or individuals?

---

## RECOMMENDATIONS

### Immediate Priority: Re-validate Nov 3 with Mean + Variance Prediction
**Why this is the top priority:**
- **Pair-level prediction is required** for anomaly detection (not bin-level)
- Oct 31 Phase 5b predicts bin averages (not suitable due to within-bin CV=62 percent)
- Nov 3 provides pair-level predictions with 5 degree features
- Need to extend to both mean AND variance for z-score calculation

**Approach:**
- Train on K permutations (test K = 2, 3, 4, 5, 7, 9)
- Compute mu_train = mean(perms 0 to K-1), sigma_train = std(perms 0 to K-1)
- Train model_mean and model_std on degree features
- Validate on perms 10-14 (hyperparameters only)
- Test on perms 15-20 (individual evaluation, z-score calibration)
- Find minimum K where performance plateaus

**Expected outcome:**
- Mean prediction: r ~ 0.75-0.85 on individual test perms
- Variance estimation: stable with K >= 5-7 perms
- Z-score calibration: mean(abs(z)) ~ 0.8, std(z) ~ 1.0
- 95 percent reduction vs 200 perms if K=5-7 sufficient

### Secondary Priority
**Analyze Experiment 2 series results (Nov 4)**
- Multiple experiments ran with results files
- Check validation methodology
- May contain insights we're missing

### Questions to Answer
1. What is the TRUE baseline performance with correct validation?
2. Does ANY approach achieve r>0.85 on individual permutations?
3. If prediction struggles, what's the computational cost of enumeration?

---

## MAJOR INSIGHTS FROM COMPREHENSIVE AUDIT

### 1. The Validation Method Problem is WIDESPREAD
**Nearly ALL work from Oct 30 onward uses mean validation:**
- Oct 31 Phases 0-5b: Validated on mean(perms 0-19)
- Nov 3: Validated on mean(perms 6-20)
- Nov 5 Exp 2L: Validated on mean(perms 11-20)
- Likely Nov 4 Experiment 2 series as well

**Impact:** Most reported r values are inflated by ~0.20 points

### 2. Oct 31 Phase 1b is CRITICAL FINDING
**Training on permutations FAILS:**
- Training on perm 0 achieves r=-0.09
- Despite bin correlation of r=0.9967
- Proves that permutation structure doesn't enable learning
- Must train on original graph

**This resolves multiple contradictions:**
- Why can't train on perms 1-20 (no biological structure)
- Why original graph training works (has biological patterns)
- Why Nov 3 had to train on perm 0 mean (not individual perm)

### 3. Oct 31 Phase 5b May Be the Best Approach
**If it holds up on individual perms:**
- Two-stage degree-aware correction
- Achieves r>0.99 (even if inflated to r=0.80-0.85, still excellent)
- No validation data needed (uses perm 0 for correction)
- Addresses heteroscedasticity
- Already implemented and tested

**Why it may work better than Nov 3:**
- Oct 31 uses 216 features (Feature Set E) + correction
- Nov 3 uses only 5 degree features
- Oct 31 more sophisticated modeling
- Both likely inflated, but Oct 31 started higher

### 4. The User's Question About Binning Now Makes Sense
**User said "binning approach from notebook 18 was determined not to work"**

**Possible interpretation:**
- Pipeline 18 notebooks NEVER RAN (confirmed)
- But Oct 31 Phase 3 shows 10×10 binning WORKS
- Perhaps user meant "Pipeline 18 approach" failed because it was never executed
- Not that binning itself failed

**Need clarification from user.**

---

## DOCUMENT STATUS
- **Completion:** ~85% - Major findings established, cross-references complete
- **Confidence:** HIGH for validated facts, HIGH for validation methodology issue
- **Next Steps:**
  1. Re-run Oct 31 Phase 5b with individual perm validation
  2. Re-run Nov 3 with individual perm validation
  3. Clarify binning question with user

**Last Updated:** 2025-11-11 (Post-comprehensive audit)
