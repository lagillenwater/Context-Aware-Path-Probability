# Pair-Level Pathway Null Prediction Analysis Plan

**Date**: 2025-10-31
**Goal**: Predict pathway null distributions at pair level for specific anomalous pathway identification
**Target**: r > 0.90 for pair-level null prediction, enabling z-score based anomaly detection
**Context**: See `2025-10-31_ANALYSIS_PLAN.md` and `2025-10-31_RESULTS.md` for bin-level approach

---

## Executive Summary

**Problem with Current Binned Approach**:
- Achieves r > 0.99 accuracy at **bin level** (degree classes)
- But individual pairs deviate 100-188% from bin mean
- Only 1-40% of pairs within 2x of bin prediction
- **Cannot identify specific anomalous pathways** (the actual goal)

**Solution: Pair-Level Prediction**:
- Extract features for individual (source, target) pairs
- Train on pair-level samples from original graph
- Predict pair-specific null expectations
- Enable z-score calculation for each pair: z = (observed - expected) / sqrt(variance)
- Identify specific anomalous pathways (like Himmelstein et al. 2017)

**Key Insight from Bin-Level Work**:
- Linear Regression with Feature Set E works very well (r > 0.99)
- Degree-aware correction using permutation 0 is effective
- Training on original graph is valid and efficient
- These lessons transfer to pair-level approach

---

## Critical Architectural Decision

**TRAINING DATA SOURCE**: Individual (source, target) pairs from original Hetionet graph
**VALIDATION DATA**: Pair-level pathway counts from permutations 1-19
**CORRECTION DATA**: Permutation 0 for degree-aware correction
**VARIANCE ESTIMATION**: Variation across permutations for each pair

**Key Hypothesis to Validate**:
> Training on sampled pairs from original graph can predict pair-level pathway counts in permutations with r > 0.90

---

## Development Workflow

**PRIMARY WORKFLOW**: `src/` module development + `test_src/` rapid evaluation scripts
**NO NOTEBOOKS** during development phase (use for final validation only)
**CODE MANAGEMENT**: Create `pair_level_*` modules (preserve bin-level code)

**Workflow for Each Phase**:
1. Develop and test locally on single metapath (CbGpPW)
2. Validate on 1,000-10,000 sampled pairs
3. Scale to full metapath analysis
4. Extend to additional metapaths

---

## Metapath Selection

**2-hop metapaths** (5 total for initial validation):
1. **CbGpPW** - Compound-binds-Gene-participates-Pathway (baseline: sparse, asymmetric)
2. **CtDaG** - Compound-treats-Disease-associates-Gene (moderate density)
3. **CrCbG** - Compound-resembles-Compound-binds-Gene (sparse, compound-compound)
4. **CbGaD** - Compound-binds-Gene-associates-Disease (moderate, asymmetric)
5. **GiGiG** - Gene-interacts-Gene-interacts-Gene (dense, symmetric)

**3-hop metapaths** (Phase 6+):
- To be determined based on 2-hop success

---

## Phase 0: Validate Pair-Level Hypothesis - CRITICAL

**Duration**: 2-3 days
**Priority**: BLOCKING - Must complete before proceeding
**Goal**: Test if pair-level features from original graph can predict pair-level permutation pathways

### Objectives

1. Sample 10,000 random (source, target) pairs from CbGpPW
2. Extract pair-specific features from original graph
3. Extract pair-level pathway counts from permutations 1-19
4. Train simple Linear Regression: pair features → permutation average
5. Compute correlation: predicted vs actual
6. **Decision criterion**: If r > 0.85, proceed with pair-level approach

### Key Differences from Bin-Level Phase 0

**Bin-Level** (what we did):
- Aggregated all pairs in bin (3, 5)
- Computed bin mean pathway count
- One prediction per bin (100 total)

**Pair-Level** (what we need):
- Individual pair (u=compound_42, v=pathway_137)
- Compute pathway count for THIS SPECIFIC PAIR
- One prediction per pair (thousands to millions total)

### Implementation

**New modules**:
- `src/pair_level_features.py` - Feature extraction for individual pairs
- `src/pair_level_sampling.py` - Sampling strategies for training
- `test_src/test_pair_level_hypothesis.py` - Quick validation script

**Key functions**:
```python
def extract_pair_features(
    edge1_matrix,
    edge2_matrix,
    source_idx,
    target_idx,
    feature_set='E'
) -> np.ndarray:
    """
    Extract features for a single (source, target) pair.

    Parameters
    ----------
    edge1_matrix : scipy.sparse matrix
        First edge type matrix
    edge2_matrix : scipy.sparse matrix
        Second edge type matrix
    source_idx : int
        Index of source node
    target_idx : int
        Index of target node
    feature_set : str
        Feature set to extract ('A', 'B', 'C', 'D', 'E', 'F')

    Returns
    -------
    np.ndarray
        Feature vector for this pair

    Notes
    -----
    Feature Set E (recommended based on bin-level results):
    - deg_source (actual degree, not binned)
    - deg_target (actual degree, not binned)
    - deg_source^2, deg_target^2
    - deg_source * deg_target
    - sqrt(deg_source), sqrt(deg_target)
    - log(deg_source + 1), log(deg_target + 1)
    - intermediate_signature: 10x10 histogram of intermediate node degrees
      for paths from THIS source to THIS target
    - Summary stats: mean, std, min, max of intermediate degrees
    - Total: ~120 features (vs 216 for bin-level)
    """

def compute_pair_intermediate_signature(
    edge1_matrix,
    edge2_matrix,
    source_idx,
    target_idx,
    n_bins=10
) -> np.ndarray:
    """
    Compute intermediate node signature for specific pair.

    For paths source_idx -> intermediate -> target_idx:
    - Find all intermediate nodes on paths
    - Compute (in_degree, out_degree) for each
    - Create 10x10 histogram

    Returns
    -------
    np.ndarray
        Flattened 10x10 histogram (100 features)
    """

def sample_pairs_stratified(
    edge1_matrix,
    edge2_matrix,
    n_samples=10000,
    strategy='degree_stratified',
    random_state=42
) -> List[Tuple[int, int]]:
    """
    Sample (source, target) pairs for training.

    Strategies:
    - 'random': Uniform random sampling
    - 'degree_stratified': Sample from each degree quartile
    - 'pathway_stratified': Sample high/medium/low pathway counts
    - 'balanced': Combine degree and pathway stratification

    Returns
    -------
    List[Tuple[int, int]]
        List of (source_idx, target_idx) pairs
    """

def compute_pair_pathway_counts(
    edge1_type,
    edge2_type,
    pair_indices,
    perm_ids,
    data_dir
) -> Dict:
    """
    Compute pathway counts for specific pairs across permutations.

    Parameters
    ----------
    edge1_type : str
        First edge type
    edge2_type : str
        Second edge type
    pair_indices : List[Tuple[int, int]]
        List of (source_idx, target_idx) pairs
    perm_ids : List[int]
        Permutation IDs
    data_dir : Path
        Data directory

    Returns
    -------
    dict
        {
            'pair_indices': List[Tuple[int, int]],
            'original_counts': np.ndarray (n_pairs,),
            'perm_counts': np.ndarray (n_pairs, n_perms),
            'perm_means': np.ndarray (n_pairs,),
            'perm_stds': np.ndarray (n_pairs,)
        }
    """
```

### Analysis Steps

1. Load original Hetionet graph (CbGpPW)
2. Compute pathway matrix: CbG @ GpPW
3. Sample 10,000 random (source, target) pairs
4. For each pair:
   - Extract features from original graph
   - Compute pathway count in original graph
   - Compute pathway counts in permutations 1-19
   - Compute mean and variance across permutations
5. Train Linear Regression: features → permutation mean
6. Validate: correlation, RMSE, bias
7. Analyze failure modes (if r < 0.85)

### Success Criteria

- **r > 0.85**: Hypothesis confirmed, proceed with Phase 1
- **0.75 < r < 0.85**: Marginal, investigate further
- **r < 0.75**: Hypothesis FAILED
  - Check if compositional approach (notebook 20) is actually better
  - May need to use permutations 1-20 for training instead of original graph
  - Reconsider entire approach

### Decision Point

**IF r > 0.85**:
- Proceed with pair-level approach (Phase 1)
- Document correlation and residual patterns
- Note: Pair-level prediction from original graph works

**IF r < 0.75**:
- STOP and investigate
- Compare to compositional approach (notebook 20)
- Consider using permutations for training
- Discuss with team before proceeding

### Deliverables

- `src/pair_level_features.py` - Pair-level feature extraction
- `src/pair_level_sampling.py` - Sampling strategies
- `test_src/test_pair_level_hypothesis.py` - Validation script
- `results/pair_level_hypothesis/CbGpPW_hypothesis_validation.csv` - Results
- `results/pair_level_hypothesis/CbGpPW_hypothesis_validation.png` - Scatter plot
- `results/pair_level_hypothesis/hypothesis_summary.json` - Decision metrics

---

## Phase 1: Baseline Pair-Level Model

**Duration**: 3-4 days
**Priority**: HIGH - Establish baseline performance
**Goal**: Train and validate pair-level Linear Regression model

### Objectives

1. Implement efficient pair-level feature extraction
2. Design sampling strategy for training data
3. Train Linear Regression on 10,000-100,000 pairs
4. Validate on held-out pairs from permutations
5. Establish baseline performance (target: r > 0.85)

### Implementation

**New modules**:
- `src/pair_level_models.py` - Model definitions
- `src/pair_level_training.py` - Training loop
- `src/pair_level_evaluation.py` - Evaluation metrics
- `test_src/test_pair_level_baseline.py` - Quick training and validation

**Key Design Decisions**:

**1. Sampling Strategy**:
```python
# Option A: Random sampling
pairs = sample_random_pairs(pathway_matrix, n=100000)

# Option B: Stratified by degree
pairs = sample_degree_stratified_pairs(
    edge1, edge2,
    n_per_quartile=2500  # 4x4 = 16 quartiles, 40k pairs
)

# Option C: Stratified by pathway count (recommended)
pairs = sample_pathway_stratified_pairs(
    pathway_matrix,
    zero_count=50000,        # Many pairs have 0 pathways
    low_count=30000,         # 1-10 pathways
    medium_count=15000,      # 10-100 pathways
    high_count=5000          # 100+ pathways
)
```

**Recommendation**: Option C (pathway-stratified)
- Original graph is very sparse (most pairs have 0 pathways)
- Random sampling would oversample zeros
- Need representation across pathway count range
- Ensures model learns both sparse and dense regions

**2. Feature Extraction Efficiency**:
```python
def extract_features_batch(
    edge1_matrix,
    edge2_matrix,
    pair_indices,
    feature_set='E'
) -> np.ndarray:
    """
    Extract features for batch of pairs efficiently.

    Optimizations:
    - Compute degrees once for all nodes
    - Vectorize degree feature computation
    - Compute intermediate signatures in batch
    - Use sparse matrix operations

    Returns
    -------
    np.ndarray
        Feature matrix (n_pairs, n_features)
    """
```

**3. Training Data Size**:
- Start with 10,000 pairs (fast iteration)
- Scale to 100,000 pairs (better coverage)
- Monitor training time and memory
- Target: < 10 minutes training time

### Architecture

**Baseline Model**: Linear Regression
- Input: ~120 features (Feature Set E adapted for pairs)
- Output: Predicted pathway count
- Training: sklearn LinearRegression
- Rationale: Worked well for bin-level (r > 0.99)

**Feature Set E for Pairs** (~120 features):
- Exact degrees (2): deg_source, deg_target
- Degree polynomials (4): source^2, target^2, sqrt(source), sqrt(target)
- Degree products (1): source * target
- Log transforms (2): log(source+1), log(target+1)
- Intermediate signature (100): 10x10 histogram
- Summary stats (5): mean, std, min, max, median of intermediate degrees
- Total: ~114 features (less than bin-level's 216 due to no bin-specific terms)

### Success Criteria

- Model trains successfully on 100,000 pairs
- Validation on held-out pairs achieves r >= 0.85
- Training time < 10 minutes
- Predictions generalize to permutations (r >= 0.80)

### Deliverables

- `src/pair_level_features.py` - Efficient feature extraction
- `src/pair_level_models.py` - Model definitions
- `src/pair_level_training.py` - Training utilities
- `src/pair_level_evaluation.py` - Evaluation utilities
- `test_src/test_pair_level_baseline.py` - Quick evaluation
- `results/pair_level_baseline/CbGpPW_baseline_metrics.json` - Performance
- `results/pair_level_baseline/CbGpPW_model.pkl` - Trained model

---

## Phase 2: Degree-Aware Correction for Pairs

**Duration**: 2-3 days
**Priority**: HIGH - Critical for unbiased null estimation
**Goal**: Adapt degree-aware correction to pair level

### Objectives

1. Analyze pair-level bias patterns
2. Implement two-stage correction using permutation 0
3. Test correction effectiveness
4. Achieve near-zero bias (|bias| < 0.01)

### Key Insight from Bin-Level Work

**Bin-Level Correction** (what we did):
- Base model predicts bin mean from original graph
- Correction model learns: (original_bin, perm0_bin) → correction
- Achieves r > 0.99 with near-zero bias

**Pair-Level Correction** (what we need):
- Base model predicts pair pathway count from original graph
- Correction model learns: (original_pair, perm0_pair) → correction
- Should achieve similar performance

### Implementation

**Extend existing module**:
- `src/degree_aware_correction.py` - Already exists, adapt for pairs

**Two-Stage Training**:
```python
# Stage 1: Train on original graph pairs
base_model = LinearRegression()
base_model.fit(X_pairs, y_original_pairs)

# Stage 2: Learn correction using permutation 0
y_pred = base_model.predict(X_pairs)
y_perm0_pairs = extract_pathway_counts(edge1_perm0, edge2_perm0, pair_indices)
correction_target = y_perm0_pairs - y_pred

correction_features = extract_correction_features(X_pairs, y_pred)
correction_model = LinearRegression()
correction_model.fit(correction_features, correction_target)
```

**Correction Features** (15 features, same as bin-level):
- Degree features (7): deg_source, deg_target, source*target, source^2, target^2, sqrt(source), sqrt(target)
- Prediction features (3): y_pred, y_pred^2, log(y_pred + 1)
- Interaction terms (5): y_pred * deg_source, y_pred * deg_target, etc.

### Analysis Steps

1. Train base model on 100,000 pairs
2. Compute predictions and residuals
3. Analyze bias as function of:
   - Predicted pathway count
   - Source degree
   - Target degree
4. Train correction model on permutation 0
5. Validate on permutations 1-19
6. Measure bias reduction

### Success Criteria

- Bias reduced to |bias| < 0.01
- Correlation maintained or improved (r >= baseline)
- RMSE reduced by >= 30%
- Works across different metapaths

### Deliverables

- `test_src/test_pair_level_correction.py` - Correction testing
- `results/pair_level_correction/CbGpPW_correction_metrics.json` - Performance
- `results/pair_level_correction/CbGpPW_bias_analysis.png` - Diagnostic plots

---

## Phase 3: Variance Estimation

**Duration**: 2-3 days
**Priority**: CRITICAL - Required for z-score calculation
**Goal**: Estimate variance of null distribution for each pair

### Objectives

1. Compute variance across permutations for each pair
2. Model variance as function of expected pathway count and degrees
3. Enable z-score calculation: z = (observed - expected) / sqrt(variance)
4. Validate variance estimates are well-calibrated

### Background

**Why Variance Estimation is Critical**:
- Z-score requires: z = (obs - exp) / sqrt(var)
- Variance differs by pair (heteroscedastic)
- High pathway count → high variance
- Need to predict variance for pairs not in permutations

### Implementation

**New module**:
- `src/pair_level_variance.py` - Variance estimation

**Two Approaches**:

**Approach 1: Empirical Variance from Permutations**
```python
# For pairs in training set:
for pair in training_pairs:
    perm_counts = [pathway_count(pair, perm_i) for perm_i in perms_1_19]
    variance[pair] = np.var(perm_counts)

# Model variance as function of features:
variance_model = LinearRegression()
variance_model.fit(X_pairs, variance_values)

# For new pairs:
predicted_variance = variance_model.predict(X_new_pair)
```

**Approach 2: Analytical Variance (from compositional assumption)**
```python
# Variance of sum of Bernoulli random variables
# Var(path_count) ≈ n_paths * p * (1-p)
# where p = edge1_prob * edge2_prob

# This may not work well (compositional failed in Phase 17)
```

**Recommendation**: Approach 1 (empirical)
- More accurate (doesn't assume compositional)
- Validated by actual permutation data
- Can model complex variance patterns

### Analysis Steps

1. For 100,000 training pairs:
   - Compute pathway counts across permutations 1-19
   - Compute empirical variance for each pair
2. Analyze variance patterns:
   - Variance vs expected pathway count
   - Variance vs degrees
   - Heteroscedasticity check
3. Train variance model: features → variance
4. Validate:
   - Predicted variance vs empirical variance
   - Check z-score distribution (should be N(0,1) under null)

### Success Criteria

- Variance model achieves r > 0.70 (variance harder to predict than mean)
- Z-scores on null data approximately N(0,1)
- No systematic over/under-estimation of variance
- Enables proper FDR control in anomaly detection

### Deliverables

- `src/pair_level_variance.py` - Variance estimation module
- `test_src/test_pair_level_variance.py` - Variance validation
- `results/pair_level_variance/CbGpPW_variance_model.pkl` - Trained variance model
- `results/pair_level_variance/CbGpPW_variance_validation.png` - Diagnostic plots

---

## Phase 4: Multi-Metapath Validation

**Duration**: 3-4 days
**Priority**: HIGH - Required for generalization
**Goal**: Validate pair-level approach on 5 diverse metapaths

### Objectives

1. Apply pair-level pipeline to 5 metapaths
2. Document performance across metapaths
3. Identify metapath characteristics predicting success
4. Ensure approach generalizes

### Implementation

**Workflow**:
1. For each metapath:
   - Sample 100,000 pairs
   - Extract features
   - Train base model + correction
   - Estimate variance
   - Validate on permutations
2. Compare performance across metapaths
3. Analyze what makes some metapaths easier/harder

**Metapaths to Test**:
- CbGpPW (baseline, sparse)
- CtDaG (moderate, tested in bin-level)
- CrCbG (sparse, tested in bin-level)
- CbGaD (moderate, new)
- GiGiG (dense, symmetric, new)

### Success Criteria

- At least 4/5 metapaths achieve r > 0.85
- At least 3/5 metapaths achieve r > 0.90
- Bias < 0.01 for all metapaths
- Identify predictive characteristics

### Deliverables

- `test_src/test_pair_level_multi_metapath.py` - Multi-metapath validation
- `results/pair_level_multi_metapath/comparison.csv` - Performance by metapath
- `results/pair_level_multi_metapath/comparison.png` - Visualization

---

## Phase 5: Pair-Level Anomaly Detection

**Duration**: 2-3 days
**Priority**: HIGH - The actual goal
**Goal**: Identify specific anomalous pathways in Hetionet

### Objectives

1. Compute expected pathway counts for ALL pairs
2. Compute variance estimates for ALL pairs
3. Calculate z-scores: z = (observed - expected) / sqrt(variance)
4. Apply FDR correction
5. Identify and characterize significant anomalies
6. Compare to notebook 20 results

### Implementation

**Workflow** (adapted from notebook 20):
```python
# For each metapath:

# 1. Load models
base_model = load_model(f'{metapath}_base.pkl')
correction_model = load_model(f'{metapath}_correction.pkl')
variance_model = load_model(f'{metapath}_variance.pkl')

# 2. For all pairs with non-zero observed or expected:
all_pairs = get_all_pairs_with_pathways(metapath)

for pair in all_pairs:
    # Extract features
    features = extract_pair_features(pair)

    # Predict expected
    y_base = base_model.predict(features)
    correction = correction_model.predict([features, y_base])
    expected = y_base + correction

    # Predict variance
    variance = variance_model.predict(features)

    # Compute z-score
    z = (observed[pair] - expected) / sqrt(variance)
    p_value = 2 * (1 - norm.cdf(abs(z)))

# 3. FDR correction
p_adjusted = fdr_correction(p_values)

# 4. Identify significant anomalies
significant = p_adjusted < 0.05
```

### Computational Considerations

**Challenge**: May need to score millions of pairs
- CbGpPW: ~10,000 compounds × ~1,400 pathways = 14M pairs
- Most are zero, but still need to check

**Optimization**:
- Only score pairs with observed > 0 OR expected > 0.01
- Batch feature extraction (process 10,000 pairs at a time)
- Use sparse matrix operations
- Target: < 1 hour per metapath

### Analysis

1. **Identify top anomalies**:
   - Top 100 enriched pathways (z > 0, p_adj < 0.05)
   - Top 100 depleted pathways (z < 0, p_adj < 0.05)

2. **Characterize anomalies**:
   - Degree distribution of anomalous pairs
   - Pathway count distribution
   - Metapath patterns

3. **Biological interpretation**:
   - Map source/target indices to node names
   - Identify meaningful patterns
   - Compare to known biology

4. **Compare to notebook 20**:
   - How many anomalies overlap?
   - Which approach finds more/fewer?
   - Which has better specificity?

### Success Criteria

- Successfully score all pairs in < 1 hour per metapath
- Identify statistically significant anomalies (FDR < 0.05)
- Find specific pathway examples (source → target)
- Results are biologically interpretable

### Deliverables

- `src/pair_level_anomaly_detection.py` - Anomaly detection module
- `test_src/test_pair_level_anomaly_detection.py` - Detection script
- `results/pair_level_anomaly_detection/all_z_scores.csv` - All pairs
- `results/pair_level_anomaly_detection/significant_FDR05.csv` - Significant only
- `results/pair_level_anomaly_detection/top_anomalies_enriched.csv` - Top enriched
- `results/pair_level_anomaly_detection/top_anomalies_depleted.csv` - Top depleted
- `results/pair_level_anomaly_detection/comparison_to_notebook20.md` - Method comparison

---

## Phase 6: Comparison to Published Methods

**Duration**: 2-3 days
**Priority**: MEDIUM - Publication requirement
**Goal**: Compare to Himmelstein et al. 2017 and notebook 20

### Objectives

1. Compare pair-level ML to compositional approach (notebook 20)
2. Compare to DWPC approach (if feasible)
3. Document advantages and limitations
4. Validate biological insights

### Comparisons

**Method 1: Our Pair-Level ML**
- Expected: ML model (r > 0.90)
- Variance: ML model
- Pros: High accuracy, uses pathway patterns
- Cons: Requires training, computationally intensive

**Method 2: Compositional (Notebook 20)**
- Expected: edge_prob1 @ edge_prob2
- Variance: Across permutations
- Pros: Simple, no training
- Cons: Low accuracy (r = 0.35 from Phase 17)

**Method 3: DWPC (Himmelstein et al.)**
- Expected: Degree-weighted path count
- Variance: Across permutations
- Pros: Published, validated
- Cons: Requires implementing DWPC

### Analysis

1. **Accuracy comparison**:
   - Correlation of expected counts
   - RMSE, bias
   - Calibration of z-scores

2. **Anomaly detection comparison**:
   - Overlap in significant hits
   - Unique hits per method
   - False positive/negative analysis (if possible)

3. **Computational cost**:
   - Training time
   - Inference time
   - Memory requirements

### Deliverables

- `results/method_comparison/accuracy_comparison.csv` - Quantitative comparison
- `results/method_comparison/anomaly_overlap.csv` - Venn diagram data
- `results/method_comparison/method_comparison_report.md` - Detailed report

---

## Risk Assessment and Mitigation

### High-Risk Phases

**Phase 0 (Pair-Level Hypothesis)**:
- **Risk**: Pair-level prediction may not work (r < 0.75)
- **Mitigation**:
  - Compare to compositional baseline
  - Try different sampling strategies
  - Consider using permutations for training
- **Impact**: Would require complete rethinking of approach

**Phase 3 (Variance Estimation)**:
- **Risk**: Variance may be unpredictable or heteroscedastic
- **Mitigation**:
  - Try multiple variance models
  - Use quantile regression for robust estimation
  - Consider empirical calibration
- **Impact**: Poor variance → incorrect z-scores → bad FDR control

### Medium-Risk Phases

**Phase 1 (Baseline Model)**:
- **Risk**: 100,000 pairs may not be enough training data
- **Mitigation**:
  - Start with 10,000, scale up
  - Monitor learning curves
  - Use stratified sampling
- **Impact**: Would need more pairs or better features

**Phase 5 (Anomaly Detection)**:
- **Risk**: Computational cost too high (millions of pairs)
- **Mitigation**:
  - Optimize feature extraction
  - Use sparse matrix operations
  - Batch processing
- **Impact**: May need HPC resources

### Low-Risk Phases

**Phase 2 (Correction)**: Worked well at bin level, should transfer
**Phase 4 (Multi-Metapath)**: Just replication across metapaths
**Phase 6 (Comparison)**: Just analysis, no new development

---

## Decision Points and Contingencies

### After Phase 0
**IF r > 0.85**: Proceed with pair-level approach (Phase 1)
**IF 0.75 < r < 0.85**:
- Investigate feature engineering
- Try different sampling strategies
- Compare to compositional baseline
**IF r < 0.75**:
- STOP and reassess
- Compare to notebook 20 (compositional)
- May need to use permutations for training

### After Phase 1
**IF r < 0.80**:
- Increase training data size
- Add more features
- Try different models
**IF 0.80 < r < 0.85**: Acceptable, proceed
**IF r >= 0.85**: Excellent, proceed

### After Phase 3
**IF variance model r < 0.60**:
- Try different variance models
- Use empirical calibration
- Consider fixed variance by degree class
**IF variance z-scores poorly calibrated**:
- Investigate heteroscedasticity
- Use quantile regression
- Consider transformation

### After Phase 4
**IF < 50% metapaths succeed**:
- Identify failure modes
- Document limitations
- Scope paper accordingly
**IF > 75% succeed**:
- Strong generalization
- Proceed to longer paths

---

## Success Metrics

### Minimum Viable Product (MVP)
- Phase 0: Pair-level hypothesis validated (r > 0.85)
- Phase 1: Baseline model working (r > 0.80)
- Phase 3: Variance estimation functional
- Phase 5: Can identify specific anomalous pathways

### Target Success (Publication Quality)
- Phase 1: Pair-level model achieves r > 0.90
- Phase 2: Correction reduces bias to < 0.01
- Phase 3: Variance well-calibrated (z-scores ~ N(0,1))
- Phase 4: Works on 4/5 metapaths
- Phase 5: Identifies biologically meaningful anomalies
- Phase 6: Outperforms compositional baseline

### Stretch Goals
- r > 0.95 on pair-level predictions
- Works on all 5 metapaths with r > 0.90
- Scales to 3-hop metapaths
- Better than DWPC approach

---

## Timeline and Milestones

### Week 1
- **Days 1-2**: Phase 0 (Pair-level hypothesis validation)
  - **Milestone**: Hypothesis validated or rejected
- **Days 3-5**: Phase 1 (Baseline model)
  - **Milestone**: Working pair-level model (r > 0.80)

### Week 2
- **Days 1-2**: Phase 2 (Degree-aware correction)
  - **Milestone**: Bias reduced to < 0.01
- **Days 3-5**: Phase 3 (Variance estimation)
  - **Milestone**: Variance model functional

### Week 3
- **Days 1-3**: Phase 4 (Multi-metapath validation)
  - **Milestone**: Generalization confirmed
- **Days 4-5**: Phase 5 (Anomaly detection - setup)

### Week 4
- **Days 1-3**: Phase 5 (Anomaly detection - analysis)
  - **Milestone**: Specific anomalies identified
- **Days 4-5**: Phase 6 (Method comparison)
  - **Milestone**: Comparison to baselines complete

---

## Resource Requirements

### Compute
- **Local development**: MacBook with 16-32GB RAM (sufficient for sampling)
- **HPC cluster**: Alpine (for full-metapath scoring)
- **GPU**: Not required
- **Estimated HPC time**:
  - Phase 1: 10 jobs × 2 hours = 20 CPU-hours (training on 100k pairs per metapath)
  - Phase 4: 5 jobs × 4 hours = 20 CPU-hours (multi-metapath)
  - Phase 5: 5 jobs × 8 hours = 40 CPU-hours (scoring millions of pairs)
  - **Total**: ~80 CPU-hours

### Data
- **Permutations 1-19**: Already downloaded
- **Additional storage**: ~10GB for pair-level results
- **Memory**: Up to 32GB for feature extraction on large metapaths

### Code Organization
```
src/
├── pair_level_features.py          # NEW: Pair-level feature extraction
├── pair_level_sampling.py          # NEW: Sampling strategies
├── pair_level_models.py            # NEW: Model definitions
├── pair_level_training.py          # NEW: Training utilities
├── pair_level_evaluation.py        # NEW: Evaluation metrics
├── pair_level_variance.py          # NEW: Variance estimation
├── pair_level_anomaly_detection.py # NEW: Anomaly detection
├── degree_aware_correction.py      # EXISTING: Adapt for pairs
└── pathway_features_v2.py          # EXISTING: Keep for bin-level

test_src/
├── test_pair_level_hypothesis.py      # NEW: Phase 0
├── test_pair_level_baseline.py        # NEW: Phase 1
├── test_pair_level_correction.py      # NEW: Phase 2
├── test_pair_level_variance.py        # NEW: Phase 3
├── test_pair_level_multi_metapath.py  # NEW: Phase 4
└── test_pair_level_anomaly_detection.py # NEW: Phase 5
```

---

## Key Differences from Bin-Level Approach

| Aspect | Bin-Level (Completed) | Pair-Level (This Plan) |
|--------|----------------------|------------------------|
| **Training samples** | 100 bins | 100,000 pairs |
| **Feature extraction** | Aggregate all pairs in bin | Individual pair features |
| **Intermediate signature** | All intermediates for bin | Intermediates for THIS pair |
| **Prediction resolution** | Degree class (bin) | Specific (source, target) |
| **Accuracy** | r > 0.99 at bin level | Target: r > 0.90 at pair level |
| **Within-bin variation** | Not modeled (100-188% CV) | Not applicable (pair-level) |
| **Anomaly detection** | Degree class anomalies | **Specific pathway anomalies** |
| **Training time** | < 1 second | ~10 minutes (100k pairs) |
| **Inference time** | Instant (100 bins) | ~1 hour (millions of pairs) |
| **Memory** | Low (~10 MB) | Medium (~1 GB) |
| **Use case** | Degree-class analysis | **Specific pathway identification** |

---

## Expected Outcomes

### Best Case
- Pair-level prediction: r > 0.95
- Works on all 5 metapaths
- Identifies hundreds of specific anomalous pathways
- Outperforms compositional and DWPC approaches
- Biological insights validated
- Publication-quality results

### Realistic Case
- Pair-level prediction: r = 0.85-0.92
- Works on 4/5 metapaths
- Identifies dozens of specific anomalous pathways
- Better than compositional, competitive with DWPC
- Some biological insights
- Solid contribution

### Worst Case
- Pair-level prediction: r = 0.75-0.80
- Works on 3/5 metapaths
- High within-pair variation (like within-bin)
- Comparable to compositional
- May need to fall back to bin-level + weighting
- Document limitations, explain failure modes

---

## Lessons Learned from Bin-Level Approach

### What Worked Well (Keep)
1. **Linear Regression**: Simple, fast, interpretable, effective
2. **Feature Set E**: Polynomial degree terms + intermediate signature
3. **Degree-aware correction**: Two-stage with permutation 0
4. **Training on original graph**: Efficient, works well
5. **Validation on permutations 1-19**: Good test of generalization

### What Didn't Work (Avoid)
1. **Ridge regularization**: Not necessary, plain LinearRegression better
2. **Custom loss functions**: Bias-aware and asymmetric losses failed
3. **Aggregation at bin level**: Lost pair-level resolution (the actual goal)
4. **Overfitting concern**: Training r = 1.0 was not harmful (red herring)

### Key Insights to Apply
1. **Heteroscedasticity is real**: Bias/variance vary with pathway count
2. **Systematic bias exists**: Original graph < permutation average (~25%)
3. **Correction is essential**: Need to learn original → permutation transformation
4. **Sampling matters**: Stratified sampling prevents bias toward common patterns
5. **Within-group variation**: High variation within aggregates (bins) suggests need for finer resolution

---

## Next Steps After Completion

### If Successful
1. **Extend to 3-hop metapaths**:
   - CbGiGpPW (Compound → Gene → Gene → Pathway)
   - Same approach, longer intermediate signatures
   - May need feature reduction (PCA on intermediate signature)

2. **Build production API**:
   - Load trained models
   - Score new pairs on demand
   - Return expected, variance, z-score

3. **Biological validation**:
   - Work with domain experts
   - Validate top anomalies
   - Publish findings

4. **Compare to graph embedding approaches**:
   - Node2Vec, DeepWalk, etc.
   - May capture additional structure

### If Unsuccessful (r < 0.75)
1. **Diagnose failure**:
   - Is it features? sampling? model?
   - Compare to compositional baseline
   - Analyze residuals

2. **Try alternatives**:
   - Use permutations 1-20 for training (not original)
   - Implement DWPC (degree-weighting)
   - Graph neural networks

3. **Fall back to bin-level + adjustment**:
   - Use bin predictions
   - Weight by actual degrees within bin
   - Acknowledge limitations

---

## Conclusion

This plan pivots from bin-level (degree class) to pair-level (specific pathway) prediction to achieve the actual goal: **identifying specific anomalous pathways in Hetionet**.

**Key Innovation**:
- Leverage ML accuracy (r > 0.99 from bin-level work)
- Apply to individual pairs (not aggregates)
- Enable specific pathway anomaly detection (like Himmelstein et al.)

**Expected Impact**:
- First ML-based approach for pair-level pathway null prediction
- More accurate than compositional assumption (r = 0.35)
- Enables biological discovery through anomaly detection
- Computationally efficient (no need for 200 permutations)

**Next Step**: Execute Phase 0 to validate pair-level hypothesis before committing to full development.

---

**Document Status**: Complete
**Related Documents**:
- `2025-10-31_ANALYSIS_PLAN.md` (bin-level approach, superseded for anomaly detection)
- `2025-10-31_RESULTS.md` (bin-level results, lessons learned)
- `notebooks/20_anomaly_detection.ipynb` (target functionality to replicate)
- `test_src/test_within_bin_variance.py` (motivation for pair-level approach)

**Approval**: Pending user review
