# Pair-Level Jaccard Similarity Approach

**Date**: 2025-11-03
**Status**: Planned
**Goal**: Train Linear Regression with Jaccard features on original Hetionet, validate on permutation data

---

## Motivation

Previous pair-level approaches (Phase 2) had **data leakage**:
- Trained on mean(permutations 1-20)
- Evaluated on mean(permutations 1-20)
- Result: r = 0.99 but circular

**New approach**: Train on original Hetionet, validate on permutations
- No data leakage
- Tests if model generalizes from original to null distribution
- Uses Jaccard similarity to capture intermediate node overlap

---

## Research Question

**Can we predict null pathway distributions by training on original Hetionet structure?**

Specifically:
1. Train model on original graph pairs → pathway counts
2. Test on held-out original graph pairs
3. Validate: Do predictions generalize to permutation averages?

---

## Methodology

### Features (8 total)

**Endpoint degrees (5)**:
1. d_u - Source node degree
2. d_v - Target node degree
3. d_u × d_v - Degree product
4. d_u² - Source degree squared
5. d_v² - Target degree squared

**Jaccard similarity (1)**:
6. jaccard = |neighbors_u ∩ neighbors_v| / |neighbors_u ∪ neighbors_v|

**Interaction terms (2)**:
7. jaccard × d_u
8. jaccard × d_v

### Why Jaccard?

**Captures intermediate overlap**:
- Pathway count = number of shared intermediate nodes
- Jaccard directly measures overlap (normalized by union)
- Mathematical relationship: pathway_count ≈ jaccard × (d_u + d_v)

**Avoids independence assumption**:
- Compositional approach (r=0.35) assumed edges independent
- Jaccard computes actual overlap from graph structure
- No edge probability multiplication

**Standard in link prediction literature**:
- Proven feature for predicting connections
- Interpretable: ratio of overlap to total neighbors

### Model

**Linear Regression**:
```
pathway_count = β₀ + β₁·d_u + β₂·d_v + β₃·(d_u×d_v) + β₄·d_u² + β₅·d_v²
                + β₆·jaccard + β₇·(jaccard×d_u) + β₈·(jaccard×d_v)
```

**Why linear?**:
- Fast training (<1s)
- Interpretable coefficients
- No hyperparameters to tune
- Previous work showed Linear Regression competitive with complex models

---

## Data Sources

### Training and Testing

**Source**: Original Hetionet
- Sample 50,000 pairs (stratified by pathway count)
- 80/20 train/test split
- Features: Extracted from original graph structure
- Targets: Actual pathway counts in original graph

**No permutations used for training.**

### Validation

**Source**: Permutations 1-20
- Use same test pairs as above
- Features: Same as test (from original Hetionet degrees)
- Targets: Mean pathway counts across permutations 1-20
- Question: Do original-trained predictions match permutation averages?

---

## Sampling Strategy

**Stratified sampling**:
- 50% pairs with pathways (nonzero counts)
- 50% random pairs (mostly zero counts)
- Prevents model from only learning zeros

**Why 50,000 pairs?**:
- Large enough for stable estimates
- Small enough for fast computation
- Previous work used this size successfully

---

## Evaluation Metrics

### Train Set
- Correlation (r)
- RMSE
- Bias (mean residual)
- Sanity check: Should achieve r > 0.95 (model fitting training data)

### Test Set (Held-Out Original Pairs)
- Correlation (r)
- RMSE
- Bias
- **Target**: r > 0.85

### Validation Set (Permutations)
- Correlation between predictions (from original) and permutation means
- RMSE
- Bias (systematic offset between original and null)
- **Target**: r > 0.85
- **Key question**: Does original-trained model generalize to null distribution?

---

## Success Criteria

**Primary**:
- Test r > 0.85 (model generalizes to held-out original pairs)
- Validation r > 0.85 (original-trained model predicts null distribution)

**Secondary**:
- Low bias (<10% of mean)
- Residuals approximately normal
- Stable coefficients (no extreme values)

---

## Potential Outcomes

### Scenario A: Both test and validation r > 0.85
**Interpretation**: Original graph structure predicts null distribution
- **Next step**: Proceed to anomaly detection
- **Implication**: Can use 0 permutations at inference time

### Scenario B: Test r > 0.85, validation r < 0.85
**Interpretation**: Model works on original but not on permutations
- **Possible reason**: Biological signal in original differs from null
- **Next step**: Train correction model using permutation 0
- **Alternative**: Train directly on permutation averages (original Phase 2 approach)

### Scenario C: Both test and validation r < 0.85
**Interpretation**: Jaccard + degrees insufficient for pair-level prediction
- **Next step**: Add more link prediction features (Adamic-Adar, Resource Allocation)
- **Alternative**: Return to bin-level approach (already works with r > 0.99)

---

## Implementation Details

### Script
`test_src/run_pair_level_jaccard.py`

### Outputs
- `results/pair_level_jaccard/jaccard_results.csv` - Performance metrics
- `results/pair_level_jaccard/jaccard_analysis.png` - 6-panel visualization
- `docs/2025-11-03_PAIR_LEVEL_JACCARD_RESULTS.md` - Detailed results

### Visualization
**6-panel figure**:

Row 1: Predicted vs Observed
- Panel 1: Train set
- Panel 2: Test set
- Panel 3: Validation (predictions vs permutation means)

Row 2: Residual Analysis
- Panel 4: Train residuals
- Panel 5: Test residuals
- Panel 6: Validation residuals

---

## Comparison to Previous Approaches

### vs Compositional Null (r = 0.35)
- **Compositional**: Assumed edge independence (failed)
- **Jaccard**: Uses actual intermediate overlap (no independence assumption)

### vs Phase 2 with Data Leakage (r = 0.99)
- **Phase 2**: Trained and validated on same permutations (circular)
- **Jaccard**: Trains on original, validates on permutations (proper split)

### vs Bin-Level (r > 0.99)
- **Bin-Level**: Predicts average for degree bins (can't identify specific pairs)
- **Jaccard**: Predicts individual pairs (enables anomaly detection)

### vs Dynamic Programming (notebook 16, failed)
- **DP**: Analytical formula assuming independence (r unknown, likely ~0.35)
- **Jaccard**: ML learns empirical relationship (no independence assumption)

---

## Data Leakage Prevention

**Documented data sources**:
- Training features: Original Hetionet
- Training targets: Original Hetionet pathway counts
- Test features: Original Hetionet (different pairs)
- Test targets: Original Hetionet pathway counts (different pairs)
- Validation features: Original Hetionet (same pairs as test)
- Validation targets: Mean of permutations 1-20

**No circular dependencies**:
- Permutations never used for training
- Train and test use different pairs
- Model frozen after training, applied to validation without refitting

---

## Timeline

**Implementation**: 1 hour (script creation)
**Execution**: 10-15 minutes (sampling, training, validation)
**Documentation**: 30 minutes (results document)
**Total**: ~2 hours

---

## Next Steps After Completion

**If successful (r > 0.85)**:
1. Extend to other metapaths (CtDaG, CrCbG)
2. Implement variance estimation
3. Score all pairs for anomaly detection
4. Compare anomalies to DWPC results

**If unsuccessful (r < 0.85)**:
1. Add more link prediction features
2. Try ensemble models
3. Return to bin-level approach (already working)

---

## References

- **Link prediction**: Liben-Nowell & Kleinberg (2007) - Jaccard coefficient
- **DWPC**: Himmelstein et al. (2017) - Degree-weighted path count
- **Previous work**: docs/2025-11-03_CORRECTED_SUMMARY.md - Lessons learned
