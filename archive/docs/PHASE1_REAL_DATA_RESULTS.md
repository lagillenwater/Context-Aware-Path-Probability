# Phase 1: Loss Function Evaluation on Real Pathway Data

## Honest Assessment

**Target**: r > 0.95 for pathway frequency prediction
**Result**: r = 0.9538 ± 0.016 with MSE loss
**Status**: Target achieved by narrow margin (+0.0038)

This document provides an honest assessment of Phase 1 results following Greene Lab standards for verification and truthfulness.

## What We Learned

### Initial Claim (INCORRECT)
On synthetic data, I claimed:
- Huber loss (delta=0.5) achieved r = 0.9682
- This exceeded target by +0.0182
- Huber outperformed MSE

**This was premature** - testing on synthetic data does not validate real-world performance.

### Verified Results on Real CbGpPW Data

**Dataset**: 100 degree bin combinations from Hetionet CbGpPW metapath
- Source nodes: 1,552 compounds
- Target nodes: 20,945 genes
- Edge1 (CbG): 11,571 edges
- Edge2 (GpPW): 84,372 edges
- Pathways: 71,653 non-zero compound-gene-pathway connections
- Target range: 0.0 - 3.6 pathways per bin
- Target mean: 1.6 pathways

**Performance (5-fold cross-validation)**:

| Loss Function | Correlation r | RMSE | Bias | Status |
|--------------|---------------|------|------|---------|
| **MSE** | **0.9538 ± 0.016** | **0.241** | **-0.033** | **BEST** |
| Log-MSE | 0.9535 ± 0.013 | 0.226 | -0.010 | Comparable |
| Quantile (0.5) | 0.9506 ± 0.023 | 0.257 | -0.030 | Slightly worse |
| Huber (0.3) | 0.9504 ± 0.019 | 0.250 | -0.042 | Slightly worse |
| Huber (0.5) | 0.9488 ± 0.011 | 0.251 | -0.031 | Worse |
| Huber (1.0) | 0.9406 ± 0.021 | 0.272 | -0.020 | Worse |

**Statistical significance**: None of the alternative losses significantly outperform MSE (all p > 0.24)

### Key Findings

1. **MSE is the best loss for real pathway data**
   - Highest correlation: 0.9538
   - Lowest RMSE: 0.241
   - Standard MSE already achieves target

2. **Alternative losses did NOT improve performance**
   - Huber losses: r = 0.941-0.950 (all worse than MSE)
   - Log-MSE: r = 0.9535 (essentially identical)
   - Quantile: r = 0.9506 (slightly worse)

3. **Synthetic data results did NOT transfer**
   - Synthetic: Huber(0.5) r = 0.9682
   - Real: Huber(0.5) r = 0.9488
   - Gap: -0.0194

4. **Target achieved, but barely**
   - r = 0.9538 vs target 0.95
   - Margin: only +0.0038
   - Standard deviation: ±0.016 (larger than margin)
   - Some CV folds likely below target

## Uncertainty and Limitations

### Statistical Uncertainty

With r = 0.9538 ± 0.016:
- 95% confidence interval: [0.922, 0.986]
- Lower bound (0.922) is well below target
- Upper bound (0.986) is well above target
- **Conclusion**: Achievement of target is not statistically robust

### Small Dataset Size

- Only 100 degree bin combinations
- With 5-fold CV: 80 training, 20 validation per fold
- Small validation sets increase variance
- May not generalize to other metapaths

### Overprediction Bias

MSE bias = -0.033 (slight underprediction on average)
- This is close to zero, which is good
- But some folds may have larger biases
- No systematic overprediction problem observed with MSE

### Unknown Generalization

Results are for CbGpPW only:
- Other metapaths may behave differently
- Longer paths (3, 4, 5 hops) not tested
- Different edge type combinations not tested

## What Actually Worked

1. **Baseline MSE loss is sufficient**
   - No need for complex alternatives
   - Standard architecture achieves target

2. **Degree binning approach**
   - Reduces 71K pathways to 100 bins
   - Retains predictive power (r > 0.95)
   - Computationally efficient

3. **Intermediate degree signatures**
   - 10×10 histogram of intermediate node degrees
   - 100-dimensional feature captures pathway structure
   - Critical for achieving high correlation

4. **DegreeSignatureNN architecture**
   - Hidden layers: (128, 64, 32)
   - Dropout: 0.2
   - Early stopping prevents overfitting
   - Achieves target with standard settings

## What Did NOT Work

1. **Huber loss**
   - Expected to reduce overprediction: did not materialize
   - Worse performance than MSE on real data
   - No clear benefit

2. **Log-scale MSE**
   - Expected to balance count ranges: minimal effect
   - r = 0.9535 vs 0.9538 (negligible difference)
   - Not worth added complexity

3. **Quantile/median regression**
   - Expected robustness: not needed
   - Slightly worse than MSE
   - Introduces bias

4. **Negative binomial loss**
   - Not tested on real data (included in initial tests)
   - Theoretical appeal didn't translate

## Honest Comparison: Claimed vs Verified

| Metric | Synthetic Data Claim | Real Data Verified | Difference |
|--------|---------------------|-------------------|------------|
| Best loss | Huber(0.5) | MSE | Different winner |
| Best r | 0.9682 | 0.9538 | -0.0144 worse |
| vs MSE | +0.019 better | 0.000 (MSE is best) | No improvement |
| Target achieved? | Yes (+0.018) | Barely (+0.004) | Much tighter |

**Conclusion**: Initial claims based on synthetic data were **overly optimistic and did not hold on real data**.

## Evidence-Based Recommendations

### For CbGpPW Pathway Prediction

**Use standard MSE loss with DegreeSignatureNN**:
```python
from src.models.degree_signature_nn import DegreeSignatureNN
import torch.nn as nn

model = DegreeSignatureNN(
    hidden_dims=(128, 64, 32),
    dropout=0.2,
    learning_rate=0.001,
    batch_size=32,
    n_epochs=500,
    early_stopping_patience=50,
    loss_fn=nn.MSELoss(),  # Standard MSE is best
    random_state=42
)
```

**Why MSE**:
- Best performance on real data (r = 0.9538)
- No added complexity
- Reliable and interpretable
- No evidence that alternatives help

### For Future Work

**To exceed r > 0.95 more robustly**:

1. **Increase model capacity**
   - Try (256, 128, 64) hidden layers
   - More parameters may capture subtle patterns
   - Risk: overfitting on small dataset

2. **Ensemble methods**
   - Train 5 models on different CV folds
   - Average predictions
   - Reduces variance, may improve correlation

3. **PathwayTransformer architecture**
   - Handle variable-length paths
   - Attention mechanism for interpretability
   - May exceed r > 0.97

4. **More training data**
   - Use multiple metapaths jointly
   - Transfer learning across edge types
   - Reduces overfitting risk

5. **Regularization tuning**
   - Grid search over dropout (0.1, 0.2, 0.3)
   - L2 weight decay (1e-5, 1e-4, 1e-3)
   - May improve generalization

## Files Generated

**Data**:
- `results/pathway_nn/training_data/CbGpPW_training_data.csv` - Real pathway data (100 bins × 108 columns)

**Scripts**:
- `prepare_and_test_pathway_data.py` - Data preparation and loss function testing
- `test_loss_functions.py` - Original test script (used synthetic data)

**Results**:
- `results/loss_function_comparison_real/loss_comparison_summary_real_data.csv` - Performance metrics
- `results/loss_function_comparison_real/loss_comparison_vs_baselines_real_data.csv` - Statistical tests

**Code modules**:
- `src/pathway_losses.py` - Alternative loss implementations (not beneficial on real data)
- `src/baseline_framework.py` - Evaluation framework (useful)
- `src/models/degree_signature_nn.py` - Modified to accept custom losses

**Documentation**:
- `PHASE1_REAL_DATA_RESULTS.md` (this file) - Honest assessment
- `LOSS_FUNCTION_COMPARISON_RESULTS.md` - Initial claims (synthetic data only)

## Conclusion

### What We Achieved

**Target r > 0.95: ACHIEVED** (but narrowly)
- MSE loss: r = 0.9538 ± 0.016
- Exceeds target by +0.0038
- Standard architecture, no exotic losses needed

### What We Learned

1. **Synthetic data results do not transfer**
   - Must always validate on real data
   - Performance can differ significantly

2. **Simple baselines are hard to beat**
   - MSE outperformed all alternatives
   - Added complexity ≠ better performance

3. **Small margins require caution**
   - r = 0.9538 is just above 0.95
   - High variance (±0.016) means some folds fail
   - Not statistically robust

### Honest Status

**Phase 1 technically complete**, but:
- Achievement is marginal, not robust
- MSE (baseline) is best, alternatives didn't help
- May need Phase 2/3 for more reliable r > 0.95
- Or accept r ~ 0.95 as practical limit for 100-sample dataset

### Next Steps

**Option A**: Accept current performance
- r = 0.9538 is "good enough" for many applications
- MSE + DegreeSignatureNN is simple and reliable
- Focus on deploying to other metapaths

**Option B**: Pursue further improvements
- PathwayTransformer for longer paths
- Ensemble methods
- More training data from multiple metapaths
- Target: r > 0.97 with statistical confidence

**Recommendation**: Before investing in Option B, test current model on additional metapaths to verify r > 0.95 is achievable across edge types, not just CbGpPW.
