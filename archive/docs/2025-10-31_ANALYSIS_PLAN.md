# Revised Null Distribution Prediction Analysis Plan

**Date**: 2025-10-31
**Goal**: Train on original Hetionet to predict average pathway counts across permutations
**Target**: r > 0.90 for null modeling (accept r > 0.80 for anomaly detection)
**Context**: See `2025-10-30_RESOLUTION_AND_ARCHITECTURE_FINDINGS.md` for background

---

## Critical Architectural Decision

**TRAINING DATA SOURCE**: Original Hetionet graph (permutation 000)
**VALIDATION DATA**: Average over permutations 000-019
**RATIONALE**: If XSwap preserves degree structure and edge correlations, then the original graph should contain sufficient information to predict permutation averages

**Key Hypothesis to Validate**:
> Training on original graph degree bins can predict average pathway counts across permutations with r > 0.85

---

## Development Workflow

**PRIMARY WORKFLOW**: `src/` module development + `test_src/` rapid evaluation scripts
**NO NOTEBOOKS** during development phase (use for final validation only)
**CODE MANAGEMENT**: Add new functions/classes (preserve old code), clean up later

**Workflow for Phases 2-5**:
1. Develop and test locally on single metapath (CbGpPW)
2. Create HPC SLURM array script for batch processing
3. Create result summarization script

---

## Metapath Selection

**2-hop metapaths** (5 total):
1. **CbGpPW** - Compound-binds-Gene-participates-Pathway (baseline: sparse, asymmetric)
2. **GiGiG** - Gene-interacts-Gene-interacts-Gene (dense, symmetric, wide degree range)
3. **CtDaG** - Compound-treats-Disease-associates-Gene (moderate density)
4. **CbGaD** - Compound-binds-Gene-associates-Disease (moderate, asymmetric)
5. **CrCbG** - Compound-resembles-Compound-binds-Gene (sparse, compound-compound)

**3-hop metapaths** (Phase 5):
- **CbGiGpPW** - Compound-binds-Gene-interacts-Gene-participates-Pathway
- **CtDaGiG** - Compound-treats-Disease-associates-Gene-interacts-Gene
- **CbGpPWpD** - Compound-binds-Gene-participates-Pathway-participates-Disease

---

## Phase 0: Validate Core Hypothesis - CRITICAL

**Duration**: 1-2 days
**Priority**: BLOCKING - Must complete before proceeding
**Goal**: Test if original graph bins predict permutation average bins

### Objectives

1. Extract degree-binned pathway counts from original Hetionet (perm 000)
2. Extract degree-binned pathway counts from permutations 000-019
3. Compute average pathway counts per bin across permutations
4. Correlate original bins with permutation averages
5. **Decision criterion**: If r > 0.85, proceed with original graph training

### Background: Permutation Similarity Analysis

The `permutation_similarity_analysis.ipynb` notebook validates XSwap properties:

**Confirmed properties**:
- Perfect degree preservation (100% match across all permutations)
- Identical degree distributions (KS statistic approximately 0)
- Perfect edge count preservation
- No systematic differences between training/validation groups

**Critical finding**: Degree correlation variation
- Hetionet has specific bipartite degree correlation (r = some value)
- Permutations show variation in degree correlation (std approximately 0.001-0.05)
- This variation is EXPECTED and GOOD - confirms edge randomization working
- High-degree source nodes may connect to different degree targets across permutations

**What's missing**: The permutation similarity analysis validates degree structure preservation but doesn't test if pathway count patterns in the original graph match average pathway count patterns across permutations.

**Phase 0 hypothesis**: If edge correlation patterns in Hetionet are representative of the permutation average, then training on the original graph will successfully predict permutation averages. If not, we'll need permutations for training.

### Implementation

**New modules**:
- `src/permutation_validation.py` - Hypothesis testing functions
- `test_src/test_hypothesis_original_vs_perms.py` - Quick validation script

**Key functions**:
```python
def extract_pathway_bins_from_graph(
    edge1_matrix, edge2_matrix, n_bins=10
) -> pd.DataFrame:
    """
    Extract pathway counts by source/target degree bin from a single graph.

    Parameters
    ----------
    edge1_matrix : scipy.sparse matrix
        First edge type matrix
    edge2_matrix : scipy.sparse matrix
        Second edge type matrix
    n_bins : int
        Number of bins for degree discretization

    Returns
    -------
    pd.DataFrame
        Columns: source_bin, target_bin, pathway_count
    """

def extract_pathway_bins_from_permutations(
    edge1_type, edge2_type, perm_ids, n_bins=10
) -> pd.DataFrame:
    """
    Extract pathway counts averaged across permutations.

    Parameters
    ----------
    edge1_type : str
        First edge type code (e.g., 'CbG')
    edge2_type : str
        Second edge type code (e.g., 'GpPW')
    perm_ids : list of int
        Permutation IDs to average over
    n_bins : int
        Number of bins for degree discretization

    Returns
    -------
    pd.DataFrame
        Columns: source_bin, target_bin, mean_pathway_count, std_pathway_count
    """

def test_original_vs_permutation_correlation(
    original_bins, permutation_bins
) -> dict:
    """
    Compute correlation between original and permutation average bins.

    Parameters
    ----------
    original_bins : pd.DataFrame
        Pathway counts from original graph
    permutation_bins : pd.DataFrame
        Average pathway counts from permutations

    Returns
    -------
    dict
        {
            'correlation': float,
            'rmse': float,
            'mae': float,
            'hypothesis_valid': bool (r > 0.85)
        }
    """
```

### Analysis Steps

1. Load original Hetionet graph (perm 000)
2. Compute 2-hop pathway matrix (edge1 @ edge2)
3. Bin node pairs by (source_degree, target_degree)
4. Compute mean pathway count per bin in original graph
5. Repeat for permutations 000-019
6. Average pathway counts across permutations for each bin
7. Correlate original bins with permutation average bins
8. Analyze residuals and patterns

### Success Criteria

- **r > 0.85**: Hypothesis confirmed, proceed with Phase 1
- **0.75 < r < 0.85**: Marginal, investigate further
- **r < 0.75**: Hypothesis FAILED, must train on permutations

### Decision Point

**IF r > 0.85**:
- Proceed with original graph training (Phase 1)
- Document correlation and residual patterns
- Note: Original graph is representative of permutation average

**IF r < 0.75**:
- STOP and re-architect
- Must use permutations 000-019 for training
- Update all subsequent phases to extract features from permutations
- Training will be slower (20x more graphs to process)

**IF 0.75 < r < 0.85**:
- Investigate residual patterns
- Check if specific degree bins fail
- Consider hybrid approach (use permutations for low-degree bins)
- Discuss with team before proceeding

### Deliverables

- `src/permutation_validation.py` - Hypothesis testing module
- `test_src/test_hypothesis_original_vs_perms.py` - Quick validation script
- `results/hypothesis_validation/CbGpPW_original_vs_perms.csv` - Correlation data
- `results/hypothesis_validation/CbGpPW_original_vs_perms.png` - Scatter plot
- `results/hypothesis_validation/hypothesis_validation_summary.json` - Decision metrics

---

## Phase 1: Rapid Baseline Establishment

**Duration**: 2-3 days
**Priority**: HIGH - Establish baseline performance
**Goal**: Reproduce r approximately 0.88 baseline with original graph training

### Objectives

1. Create new feature extraction from original graph
2. Train DegreeSignatureNN on original graph bins
3. Validate on permutation averages (000-019)
4. Establish true baseline performance

### Implementation

**New modules** (don't modify existing code):
- `src/pathway_features_v2.py` - Feature extraction from original graph
- `src/pathway_models_v2.py` - Model definitions
- `src/pathway_training_v2.py` - Training loop
- `src/pathway_evaluation_v2.py` - Evaluation metrics

**Evaluation script**:
- `test_src/test_baseline_v2.py` - Quick training and validation

**Key functions**:
```python
def extract_features_from_original(
    edge1_type, edge2_type, n_bins=10, feature_set='A'
) -> tuple:
    """
    Extract features from original Hetionet graph.

    Parameters
    ----------
    edge1_type : str
        First edge type code
    edge2_type : str
        Second edge type code
    n_bins : int
        Number of bins for degree discretization
    feature_set : str
        Feature set to extract ('A', 'B', 'C', 'D', 'E', 'F')

    Returns
    -------
    X_train : np.ndarray
        Feature matrix (n_bins x n_bins, n_features)
    y_train : np.ndarray
        Target pathway counts (bin-level means)
    metadata : dict
        Bin edges, sample counts, etc.
    """

def validate_on_permutations(
    model, edge1_type, edge2_type, perm_ids, n_bins=10
) -> dict:
    """
    Validate model by predicting permutation average bins.

    Parameters
    ----------
    model : trained model
        Trained DegreeSignatureNN
    edge1_type : str
        First edge type code
    edge2_type : str
        Second edge type code
    perm_ids : list of int
        Permutation IDs for validation
    n_bins : int
        Number of bins

    Returns
    -------
    dict
        Metrics: validation_r, rmse, mae, per_bin_errors
    """
```

### Architecture

**DegreeSignatureNN** (baseline feature set A):
- Input: 102 features
  - 2 degree bins (source, target)
  - 100 intermediate signature bins (10×10 histogram)
- Architecture: 102 → 128 → 64 → 32 → 1
- Activation: ReLU + Dropout(0.1) + Softplus output
- Parameters: approximately 17K

**Training details**:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32 (or all bins if fewer than 100)
- Epochs: 500 with early stopping
- Loss: MSE
- Regularization: Dropout(0.1)

### Success Criteria

- Model trains successfully on original graph
- Validation on perms 000-019 achieves r >= 0.85
- If r approximately 0.88, baseline reproduced
- If r < 0.80, investigate issues

### Deliverables

- `src/pathway_features_v2.py` - Feature extraction module
- `src/pathway_models_v2.py` - Model definitions
- `src/pathway_training_v2.py` - Training utilities
- `src/pathway_evaluation_v2.py` - Evaluation utilities
- `test_src/test_baseline_v2.py` - Quick evaluation script
- `results/baseline_v2/CbGpPW_baseline_metrics.json` - Performance metrics
- `results/baseline_v2/CbGpPW_model.pt` - Trained model

---

## Phase 2: Binning Resolution Optimization

**Duration**: 1-2 days
**Priority**: MEDIUM - Quick parameter tuning
**Goal**: Find optimal bin size for null prediction

### Objectives

1. Test bin sizes: 5×5, 10×10, 15×15, 20×20
2. Evaluate trade-off: accuracy vs sample size
3. Identify optimal resolution
4. Validate on CbGpPW and GiGiG

### Implementation

**Add to existing modules**:
- `src/pathway_features_v2.py`:
  - `get_optimal_bins(edge1_type, edge2_type) -> int`
  - `extract_features_variable_bins(n_bins) -> tuple`

**Evaluation script**:
- `test_src/test_bin_optimization.py` - Rapid iteration over bin sizes

**Workflow**:
1. Test locally on CbGpPW
2. Identify optimal bin size
3. Validate on GiGiG
4. Document findings

### Expected Trade-offs

**Coarser bins (5×5)**:
- More samples per bin (better statistics)
- Less resolution (less granular predictions)
- Lower risk of overfitting

**Finer bins (20×20)**:
- Higher resolution (more granular predictions)
- Fewer samples per bin (sparse data)
- Higher risk of overfitting
- May not improve null prediction (high within-bin variance)

### Success Criteria

- Test 4 bin sizes (5, 10, 15, 20)
- Identify optimal n_bins (best validation r)
- Improvement: validation_r increases by >= 0.02 OR confirm 10×10 is optimal
- Consistent across CbGpPW and GiGiG

### Deliverables

- `test_src/test_bin_optimization.py` - Bin size comparison script
- `results/bin_optimization/bin_comparison.csv` - Performance by bin size
- `results/bin_optimization/bin_comparison.png` - Visualization

---

## Phase 3: Feature Enhancement Testing

**Duration**: 2-3 days
**Priority**: HIGH - Highest potential for improvement
**Goal**: Test enhanced feature sets to reach r > 0.90

### Feature Sets

All features extracted from **original graph only**:

| Set | Features | Description |
|-----|----------|-------------|
| **A (Baseline)** | 102 | 2 degree bins + 100 intermediate signatures |
| **B** | 104 | + log(degree) transforms |
| **C** | 109 | + summary stats (mean, std, min, max, median) |
| **D** | 111 | + neighbor context |
| **E** | 113 | + polynomial terms (degree², degree³) |
| **F** | 116 | + interaction terms (source×target features) |

### Feature Descriptions

**Set A (Baseline)**:
- source_degree_bin (1 feature)
- target_degree_bin (1 feature)
- intermediate_signature: 10×10 histogram of intermediate node degrees (100 features)

**Set B (+ Log Transforms)**:
- All Set A features
- log(source_degree + 1) (1 feature)
- log(target_degree + 1) (1 feature)

**Set C (+ Summary Statistics)**:
- All Set B features
- Intermediate degree summary: mean, std, min, max, median (5 features)

**Set D (+ Neighbor Context)**:
- All Set C features
- Source neighbor context: mean degree of source's neighbors (1 feature)
- Target neighbor context: mean degree of target's neighbors (1 feature)

**Set E (+ Polynomial Terms)**:
- All Set D features
- source_degree² (1 feature)
- target_degree² (1 feature)

**Set F (+ Interaction Terms)**:
- All Set E features
- source_degree × target_degree (1 feature)
- source_degree × intermediate_mean (1 feature)
- target_degree × intermediate_mean (1 feature)

### Implementation

**Add to src/pathway_features_v2.py**:
```python
def extract_features_setA(edge1, edge2, n_bins=10) -> np.ndarray:
    """Baseline: degree bins + intermediate signature"""

def extract_features_setB(edge1, edge2, n_bins=10) -> np.ndarray:
    """Set A + log transforms"""

def extract_features_setC(edge1, edge2, n_bins=10) -> np.ndarray:
    """Set B + summary statistics"""

def extract_features_setD(edge1, edge2, n_bins=10) -> np.ndarray:
    """Set C + neighbor context"""

def extract_features_setE(edge1, edge2, n_bins=10) -> np.ndarray:
    """Set D + polynomial terms"""

def extract_features_setF(edge1, edge2, n_bins=10) -> np.ndarray:
    """Set E + interaction terms"""
```

**Evaluation script**:
- `test_src/test_feature_sets.py` - Compare all feature sets locally

**Workflow**:
1. **Local testing** (CbGpPW):
   - Train models with sets A-F
   - Validate on perms 000-019
   - Identify best feature set
2. **HPC batch processing**:
   - Create `scripts/03_feature_ablation_array.sh`
   - Run all 5 metapaths × 6 feature sets = 30 jobs
3. **Summarization**:
   - Create `scripts/03_feature_ablation_summary.sh`
   - Generate comparison plots and tables

### Success Criteria

- Test all 6 feature sets (A-F)
- Best set achieves validation_r > 0.90 on CbGpPW
- Improvement is statistically significant (p < 0.05)
- Generalizes to at least 3/5 metapaths

### Deliverables

- `src/pathway_features_v2.py` - Feature extraction functions (sets A-F)
- `test_src/test_feature_sets.py` - Local comparison script
- `scripts/03_feature_ablation_array.sh` - HPC array job
- `scripts/03_feature_ablation_summary.sh` - Result summarization
- `results/feature_ablation/feature_comparison.csv` - Performance metrics
- `results/feature_ablation/feature_comparison.png` - Visualization

---

## Phase 5: Multi-Metapath Validation

**Duration**: 2-3 days
**Priority**: HIGH - Required for publication
**Goal**: Validate approach on 5 diverse 2-hop metapaths

### Objectives

1. Apply best configuration (bin size + feature set) to all metapaths
2. Document which metapaths achieve r > 0.90
3. Identify metapath characteristics predicting success
4. Ensure generalization beyond CbGpPW

### Implementation

**Workflow**:
1. **Local testing** (one metapath):
   - Verify pipeline works end-to-end
   - Debug any issues
2. **HPC batch processing**:
   - Create `scripts/04_multi_metapath_array.sh`
   - Run 5 metapaths in parallel
3. **Summarization**:
   - Create `scripts/04_multi_metapath_summary.sh`
   - Generate comparison plots
   - Analyze metapath characteristics

**Add to src/pathway_evaluation_v2.py**:
```python
def characterize_metapath(edge1_type, edge2_type) -> dict:
    """
    Extract metapath characteristics.

    Returns
    -------
    dict
        {
            'density': float (edge density),
            'degree_range': float (max - min degree),
            'symmetry': bool (edge1 == edge2),
            'mean_pathway_count': float,
            'std_pathway_count': float
        }
    """

def analyze_metapath_patterns(results_df) -> dict:
    """
    Identify which characteristics predict success.

    Parameters
    ----------
    results_df : pd.DataFrame
        Columns: metapath, validation_r, density, degree_range, etc.

    Returns
    -------
    dict
        Correlation between characteristics and validation_r
    """
```

### Success Criteria

- Test all 5 metapaths (CbGpPW, GiGiG, CtDaG, CbGaD, CrCbG)
- At least 4/5 achieve r > 0.85
- At least 3/5 achieve r > 0.90
- Identify predictive metapath characteristics

### Deliverables

- `test_src/test_single_metapath.py` - Local single metapath pipeline
- `scripts/04_multi_metapath_array.sh` - HPC array job
- `scripts/04_multi_metapath_summary.sh` - Result summarization
- `results/multi_metapath/metapath_comparison.csv` - Performance by metapath
- `results/multi_metapath/metapath_comparison.png` - Visualization
- `results/multi_metapath/metapath_characteristics.csv` - Characteristic analysis

---

## Phase 6: Longer Path Extension (3-5 hops)

**Duration**: 3-4 days
**Priority**: HIGH - Core objective
**Goal**: Scale approach to 3-hop, 4-hop, and 5-hop metapaths

### Challenge: Feature Explosion

**Problem**:
- 2-hop: 102 features (2 bins + 100 intermediate)
- 3-hop: 202 features (2 bins + 200 intermediate)
- 4-hop: 302 features
- 5-hop: 402 features

**Risk**: Overfitting with 400+ features on approximately 100 samples

### Feature Reduction Strategies

**Strategy 1: PCA on intermediate signatures**
```python
pca = PCA(n_components=50)
inter_sig_reduced = pca.fit_transform(all_inter_sigs)
# 3-hop: 2 + 50 = 52 features
```

**Strategy 2: Pooled signature**
```python
all_intermediate_degrees = concat([inter1, inter2, inter3])
pooled_sig = histogram_10x10(all_intermediate_degrees)
# 3-hop: 2 + 100 = 102 features (same as 2-hop)
```

**Strategy 3: LSTM encoding**
```python
lstm = LSTM(input_dim=100, hidden_dim=64)
encoded = lstm([inter1_sig, inter2_sig, inter3_sig])
# 3-hop: 2 + 64 = 66 features
```

### Implementation

**Add to src/pathway_features_v2.py**:
```python
def extract_features_3hop(edge1, edge2, edge3, reduction='pooled') -> np.ndarray:
    """
    Extract features for 3-hop paths with dimensionality reduction.

    Parameters
    ----------
    edge1, edge2, edge3 : scipy.sparse matrices
        Edge type matrices
    reduction : str
        'pooled', 'pca', or 'lstm'

    Returns
    -------
    np.ndarray
        Feature matrix
    """

def extract_features_4hop(...) -> np.ndarray:
    """Extract features for 4-hop paths"""

def extract_features_5hop(...) -> np.ndarray:
    """Extract features for 5-hop paths"""
```

**Workflow**:
1. **Local testing** (3-hop CbGiGpPW):
   - Test all 3 reduction strategies
   - Identify best approach
2. **HPC batch processing**:
   - Create `scripts/05_longer_paths_array.sh`
   - Test 3 metapaths × 3 path lengths = 9 jobs
3. **Summarization**:
   - Create `scripts/05_longer_paths_summary.sh`
   - Document scaling behavior

### Success Criteria

- 3-hop metapaths achieve r > 0.85
- Feature reduction maintains performance
- Scalable to 5-hop (training <2 hours, memory <64GB)
- Approach generalizes across path lengths

### Deliverables

- `src/pathway_features_v2.py` - 3-hop, 4-hop, 5-hop feature extraction
- `test_src/test_longer_paths.py` - Local 3-hop testing
- `scripts/05_longer_paths_array.sh` - HPC array job
- `scripts/05_longer_paths_summary.sh` - Result summarization
- `results/longer_paths/path_length_comparison.csv` - Performance by path length
- `results/longer_paths/path_length_comparison.png` - Visualization

---

## Phase 4: Control Experiments

**Duration**: 1-2 days
**Priority**: MEDIUM - Publication requirement
**Goal**: Compare to baseline methods

### Comparisons

**Negative control**: Random predictions
**Weak baseline**: Degree product (P proportional to deg_source × deg_target)
**ML baselines** (from Pipeline 18):
- Linear Regression
- Negative Binomial GLM
- Random Forest
- Polynomial Logistic Regression

**Our method**: DegreeSignatureNN

### Implementation

**Add to src/pathway_evaluation_v2.py**:
```python
def train_baseline_models(X_train, y_train) -> dict:
    """
    Train all baseline models.

    Returns
    -------
    dict
        {
            'random': RandomModel(),
            'degree_product': DegreeProductModel(),
            'linear_regression': LinearRegression(),
            'negbin_glm': NegativeBinomialGLM(),
            'random_forest': RandomForest()
        }
    """

def compare_methods(models, X_val, y_val) -> pd.DataFrame:
    """
    Compare all methods and rank.

    Returns
    -------
    pd.DataFrame
        Columns: method, validation_r, rmse, mae, rank
    """
```

**Evaluation script**:
- `test_src/test_control_experiments.py` - Compare all methods

**Workflow**:
1. **Local testing** (CbGpPW):
   - Train all baselines
   - Compare performance
2. **HPC batch processing**:
   - Create `scripts/04_control_experiments_array.sh`
   - Run 5 metapaths × 6 methods = 30 jobs
3. **Summarization**:
   - Create `scripts/04_control_experiments_summary.sh`
   - Generate comparison tables

### Success Criteria

- Our method beats all baselines by >= 0.1 in r
- Statistical significance (p < 0.05, paired t-test)
- Consistent across 4/5 metapaths
- Clear visualization of comparison

### Deliverables

- `test_src/test_phase4_control_experiments.py` - Control comparison script
- `scripts/04_control_experiments_array.sh` - HPC array job
- `scripts/04_control_experiments_summary.sh` - Result summarization
- `results/phase4_control_experiments/method_comparison.csv` - Performance by method
- `results/phase4_control_experiments/method_comparison.png` - Visualization

---

## Risk Assessment and Mitigation

### High-Risk Phases

**Phase 0 (Hypothesis Validation)**:
- **Risk**: Original graph bins may not predict permutation averages (r < 0.75)
- **Mitigation**: If fails, switch to training on permutations 000-019
- **Impact**: Major re-architecture of all phases

**Phase 5 (Longer Paths)**:
- **Risk**: Feature explosion causes overfitting
- **Mitigation**: Test 3 reduction strategies, use regularization
- **Impact**: May limit to 2-3 hop paths

### Medium-Risk Phases

**Phase 3 (Feature Enhancement)**:
- **Risk**: Complex features overfit
- **Mitigation**: Dropout, L2 regularization, feature selection
- **Impact**: Stick with baseline features

**Phase 4 (Multi-Metapath)**:
- **Risk**: Some metapaths may fail
- **Mitigation**: Document failure modes, identify characteristics
- **Impact**: Limit scope to successful metapath types

### Low-Risk Phases

**Phase 1 (Baseline)**: Low risk, just implementation
**Phase 2 (Binning)**: Quick test, low cost
**Phase 6 (Controls)**: Just evaluation, baselines exist

---

## Decision Points and Contingencies

### After Phase 0
**IF r > 0.85**: Proceed with original graph training (Phase 1)
**IF 0.75 < r < 0.85**: Investigate patterns, may still proceed
**IF r < 0.75**: STOP, re-architect to use permutations for training

### After Phase 1
**IF r < 0.80**: Debug issues, fix bugs before proceeding
**IF 0.80 < r < 0.85**: Continue but lower expectations
**IF r >= 0.85**: Proceed as planned

### After Phase 2
**IF no improvement**: Skip bin optimization, use 10×10
**IF significant improvement**: Apply optimal bins to all phases

### After Phase 3
**IF best r < 0.88**: Features don't help, stick with baseline
**IF 0.88 < r < 0.90**: Modest improvement, acceptable
**IF r > 0.90**: SUCCESS, proceed to validation

### After Phase 4
**IF <50% metapaths r > 0.85**: Approach not general, document limits
**IF 50-75% succeed**: Identify failure modes, scope paper
**IF >75% succeed**: Strong generalization, proceed to longer paths

### After Phase 5
**IF 3-hop fails**: Stop at 2-hop, document limitations
**IF 3-hop succeeds but 4-hop fails**: Limit to 3-hop
**IF 5-hop succeeds**: MAJOR SUCCESS, headline result

---

## Success Metrics

### Minimum Viable Product (MVP)
- Phase 0: Hypothesis validated (r > 0.85)
- Phase 1: Baseline established (r > 0.80)
- Phase 4: Works on 3+ metapaths (r > 0.85)
- Phase 6: Beats all baselines

### Target Success (Publication Quality)
- Phase 3: Feature enhancement achieves r > 0.90
- Phase 4: 4/5 metapaths achieve r > 0.85
- Phase 5: 3-hop paths work (r > 0.85)
- Phase 6: Statistical significance vs baselines

### Stretch Goals
- r > 0.95 achieved on multiple metapaths
- 5/5 metapaths exceed r > 0.90
- 5-hop paths work with r > 0.85
- Novel methodological contribution

---

## Timeline and Milestones

### Week 1
- **Days 1-2**: Phase 0 (Hypothesis validation)
  - **Milestone**: Hypothesis confirmed or rejected
- **Days 3-5**: Phase 1 (Baseline establishment)
  - **Milestone**: Baseline r approximately 0.85-0.88 achieved

### Week 2
- **Days 1-2**: Phase 2 (Bin optimization)
  - **Milestone**: Optimal bin size identified
- **Days 3-5**: Phase 3 (Feature enhancement - local)
  - **Milestone**: Best feature set identified

### Week 3
- **Days 1-3**: Phase 3 (Feature enhancement - HPC)
  - **Milestone**: r > 0.90 achieved (goal)
- **Days 4-5**: Phase 4 (Multi-metapath - local)

### Week 4
- **Days 1-3**: Phase 4 (Multi-metapath - HPC)
  - **Milestone**: Generalization validated
- **Days 4-5**: Phase 5 (Longer paths - local)

### Week 5
- **Days 1-3**: Phase 5 (Longer paths - HPC)
  - **Milestone**: 3-hop scalability assessed
- **Days 4-5**: Phase 6 (Control experiments)
  - **Milestone**: Baselines compared, statistical significance

---

## Resource Requirements

### Compute
- **Local development**: MacBook with 16-32GB RAM
- **HPC cluster**: Alpine (CU Boulder)
- **GPU**: Not required (small models, fast training)
- **Estimated HPC time**:
  - Phase 3: 30 jobs × 1 hour = 30 CPU-hours
  - Phase 4: 5 jobs × 2 hours = 10 CPU-hours
  - Phase 5: 9 jobs × 3 hours = 27 CPU-hours
  - Phase 6: 30 jobs × 1 hour = 30 CPU-hours
  - **Total**: approximately 100 CPU-hours

### Data
- **Permutations 000-019**: Already downloaded
- **Storage**: approximately 20GB for all data and results

---

## Key Differences from Previous Plan

### Architecture Changes
1. **Training data**: Original graph (not permutations)
2. **Validation data**: Perms 000-019 (not 21-30)
3. **Phase 0 added**: Hypothesis validation (critical)

### Workflow Changes
1. **No notebooks**: Use `src/` + `test_src/` during development
2. **Add, don't delete**: Preserve old code, create v2 modules
3. **Local to HPC to Summary**: 3-step workflow for phases 3-5

### Scope Changes
1. **Control experiments included**: Phase 6 (not skipped)
2. **Anomaly detection excluded**: Defer until after modeling
3. **Target adjusted**: r > 0.9 for modeling, r > 0.8 acceptable for anomaly detection

---

## Conclusion

This plan provides a systematic approach to validate and refine the null distribution prediction model. The critical innovation is training on the original Hetionet graph to predict permutation averages, which requires Phase 0 validation.

**Expected outcomes**:
- **Best case**: r > 0.95, works for 5-hop paths, general across metapaths
- **Realistic**: r approximately 0.90, works for 3-hop paths, 4/5 metapaths succeed
- **Worst case**: r approximately 0.85, 2-hop only, document limitations

**Next step**: Execute Phase 0 to validate core hypothesis before committing to development.

---

**Document Status**: Complete
**Related Documents**:
- `2025-10-30_RESOLUTION_AND_ARCHITECTURE_FINDINGS.md` (background)
- `2025-10-30_REFINEMENT_PLAN.md` (superseded by this plan)
- `permutation_similarity_analysis.ipynb` (XSwap validation)

**Approval**: Pending user review
