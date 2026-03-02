# Training Pipeline Fix: Summary

## Critical Error Identified and Fixed

### The Problem

The original implementation trained models on **empirical frequencies** (continuous probabilities aggregated from 200 permutations) instead of **single-permutation binary labels**. This fundamentally misunderstood the research question.

**Incorrect Approach (Fixed)**:
```python
# WRONG: Training on 200-permutation aggregates
empirical_df = pd.read_csv('edge_frequency_by_degree_CbG.csv')
y_train = empirical_df['frequency'].values  # Continuous probabilities 0-1
```

**Correct Approach (Implemented)**:
```python
# CORRECT: Training on single permutation binary labels
X, y_binary = prepare_edge_features_and_labels('001.hetmat/edges/CbG.sparse.npz')
# y_binary: Binary labels {0, 1} for individual node pairs
# Then validate predictions against empirical frequencies
```

## Research Question

**Can we train a model on a single permutation and still accurately predict multi-permutation empirical frequencies?**

This is important because computing 200-permutation empirical frequencies is expensive for large graphs. If single-permutation training can predict multi-permutation frequencies accurately, it provides a scalable solution.

## Implementation Changes

### 1. New Function: load_single_permutation_data()

Located in: `src/evaluate_feature_reduction.py:135-180`

Loads edge matrix from single permutation (001.hetmat) and creates binary training data using `prepare_edge_features_and_labels()`.

**Returns**:
- X: (N, 2) array of degree pairs
- y: (N,) binary labels {0, 1}
- edge_matrix: Sparse adjacency matrix

### 2. New Function: validate_against_empirical_frequencies()

Located in: `src/evaluate_feature_reduction.py:183-260`

Groups individual predictions by (source_degree, target_degree) and compares averaged predictions to empirical frequencies.

**Process**:
1. Create lookup dictionary from empirical frequencies
2. Group predictions by degree pair
3. Average predictions within each group
4. Calculate correlation, bias, RMSE vs empirical frequencies

### 3. Updated: evaluate_single_edge_type()

Located in: `src/evaluate_feature_reduction.py:263-554`

**Key changes**:
- Loads binary labels from single permutation for training
- Trains models on binary labels (task: predict edge existence)
- Validates predictions against empirical frequencies (gold standard)
- Reports both binary correlation (training task) and empirical correlation (research question)

## Results Comparison

### CbG (Compound-binds-Gene) Test Results

**Dataset**:
- Single permutation: 155,355 samples (11,571 positive, 143,784 negative)
- Empirical frequencies: 4,100 degree combinations
- Matched for validation: 2,096 degree pairs

#### Incorrect Implementation (Before Fix)
```
SimpleNN trained on empirical frequencies:
- Test r: 0.9724  ← WRONG: trained on continuous probabilities
- Task: Regression on empirical frequencies (easy)
```

#### Correct Implementation (After Fix)
```
SimpleNN trained on binary labels:
- Test r (on binary labels): 0.5766  ← Binary classification task
- Test r (vs empirical): 0.7587      ← Research question answer
- Task: Binary classification validated against empirical frequencies (hard)
```

### Why Results Are Different

**Incorrect approach**:
- Training on continuous probabilities (0-1) from 200 permutations
- Predicting continuous probabilities directly
- Easier task: regression on aggregated data
- Artificially high correlation (r=0.97)

**Correct approach**:
- Training on binary labels (0 or 1) from single permutation
- Predicting individual edge probabilities
- Harder task: generalize from binary to continuous
- Realistic correlation (r=0.76)

### Full Results (CbG, Minimal Tier)

| Model | Binary Train r | Binary Test r | Empirical Test r | Bias | RMSE | Training Time |
|-------|----------------|---------------|------------------|------|------|---------------|
| Analytical | N/A | N/A | 0.9949 | -0.0017 | 0.0134 | 0s |
| SimpleNN (2 feat) | 0.5900 | 0.5766 | 0.7587 | +0.3470 | 0.4257 | 24.7s |
| Minimal (13 feat) | 0.6179 | 0.6071 | 0.8233 | +0.3163 | 0.3682 | 73.2s |

**Key findings**:
1. Analytical formula still performs best (r=0.9949) because it doesn't require training
2. SimpleNN achieves r=0.76 when trained on binary labels and validated on empirical frequencies
3. Minimal features improve to r=0.82 (8% improvement over SimpleNN)
4. Both ML models show positive bias (+0.32-0.35), indicating systematic overestimation

## Validation

Residual plots confirm correct implementation:

**Files created**:
- `results/feature_reduction_evaluation/CbG/analytical_residuals.png`
- `results/feature_reduction_evaluation/CbG/simplenn_residuals.png`
- `results/feature_reduction_evaluation/CbG/minimal_residuals.png`

**Plots show**:
- X-axis: Empirical frequency (200-permutation gold standard)
- Y-axis: Predicted probability (grouped average from test set)
- 2,096 degree pairs matched between predictions and empirical frequencies

## Interpretation

### What This Means for the Research

**Positive results**:
1. Single-permutation training CAN predict empirical frequencies (r=0.76-0.82)
2. Theory-guided features improve prediction (minimal features: r=0.82 vs SimpleNN: r=0.76)
3. Validation methodology is now scientifically sound

**Challenges identified**:
1. ML models underperform analytical formula (r=0.82 vs r=0.99)
2. Positive bias suggests models overestimate rare edges
3. Binary classification task is harder than direct regression on empirical frequencies

### Why Analytical Formula Still Wins

The analytical formula:
- Directly computes expected frequency from degree statistics
- Based on XSwap Markov chain stationary distribution
- No training required, no overfitting risk
- Achieves r=0.9949 on CbG

ML models:
- Must learn from binary labels (noisy single-sample observations)
- Generalize to continuous probabilities (averaging effect)
- Limited by training data quality and model capacity
- Achieve r=0.76-0.82 on CbG

## Next Steps

### 1. Apply Fix to evaluate_theory_guided_models.py

The same error exists in `src/evaluate_theory_guided_models.py`. Apply identical fix:
- Add `load_single_permutation_data()` function
- Add `validate_against_empirical_frequencies()` function
- Update `evaluate_edge_type()` to train on binary labels

### 2. Re-run Evaluation on All Edge Types

Test corrected implementation on all 5 edge types:
- CbG (Compound-binds-Gene)
- CtD (Compound-treats-Disease)
- GpPW (Gene-participates-Pathway)
- AeG (Anatomy-expresses-Gene)
- DdG (Disease-downregulates-Gene)

Expected behavior:
- ML correlations will be LOWER than previous incorrect results
- Results will be scientifically valid
- Can now properly assess when ML is beneficial vs analytical formula

### 3. Update Documentation

Files to revise:
- `FEATURE_REDUCTION_RECOMMENDATIONS.md` - Update with correct results
- `MINIMAL_FEATURE_SET_DOCUMENTATION.md` - Update performance metrics
- `THEORY_GUIDED_APPROACH.md` - Clarify training vs validation methodology

### 4. Investigate Bias

Both SimpleNN and minimal features show positive bias (+0.32-0.35):
- Systematic overestimation of edge probabilities
- May be due to class imbalance (7.4% positive class)
- Consider calibration techniques or weighted loss functions

## Files Modified

1. `src/evaluate_feature_reduction.py`
   - Added `load_single_permutation_data()` (lines 135-180)
   - Added `validate_against_empirical_frequencies()` (lines 183-260)
   - Updated `evaluate_single_edge_type()` (lines 263-554)
   - Updated imports to include `prepare_edge_features_and_labels`

2. `test_corrected_training.py`
   - Created test script to verify fix on single edge type

## Test Results

**Test command**:
```bash
python test_corrected_training.py
```

**Status**: PASSED

**Output**:
```
Edge type: CbG
Difficulty: medium
Recommended tier: standard

Analytical r: 0.9949
SimpleNN r: 0.7587
Minimal tier r: 0.8233

Matched degree pairs (analytical): 2096
Matched degree pairs (SimpleNN): 2096
Matched degree pairs (minimal): 2096
```

## Conclusion

The training pipeline has been corrected to implement the proper research question: **training on single-permutation binary labels and validating against multi-permutation empirical frequencies**.

This fix:
1. Corrects a fundamental methodological error
2. Produces scientifically valid results
3. Reveals the true difficulty of the problem
4. Enables proper comparison of ML vs analytical approaches

The lower ML performance (r=0.76-0.82 vs previous incorrect r=0.97) reflects the genuine challenge of learning from single-permutation noisy observations and generalizing to multi-permutation frequencies.
