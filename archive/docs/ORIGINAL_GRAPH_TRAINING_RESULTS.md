# Original Graph Frequency Training: Results and Analysis

## Overview

This document summarizes the results of training neural network models on edge frequencies from the original Hetionet graph and validating against 200-permutation empirical frequencies.

## Methodology

### Training Approach

1. Compute edge frequencies from original Hetionet graph (data/edges/CbG.sparse.npz)
   - For each (source_degree, target_degree) pair: frequency = edges / possible_pairs
2. Split degree pairs into train (80%) and test (20%)
3. Train models on original graph frequencies (continuous regression task)
4. Validate predictions on 200-permutation empirical frequencies
5. Measure interpolation quality on unseen degree pairs

### Key Difference from Previous Approaches

Previous approach: Train on single-permutation binary labels, validate on empirical frequencies
New approach: Train on original graph frequencies, validate on empirical frequencies

Advantages:
- Training target matches validation target (both frequencies)
- Neural network naturally interpolates for unseen degree pairs
- Original graph frequencies are less noisy than single-permutation observations

## Results for CbG (Compound-binds-Gene)

### Data Characteristics

Original Hetionet graph:
- Shape: 1,552 compounds x 20,945 genes
- Total edges: 11,571
- Source degree range: 0-132
- Target degree range: 0-516
- Unique degree pairs: 2,746
- Frequency range: 0.000089 - 1.000000
- Mean frequency: 0.166

### Degree Pair Coverage

Coverage between original graph and 200-permutation data:
- Original graph degree pairs: 2,746
- 200-permutation degree pairs: 4,100
- Overlap (in both): 2,746 (67.0%)
- Unseen (only in 200-perm): 1,354 (33.0%)
- Original only: 0

Key insight: All degree pairs in the original graph also exist in the 200-permutation data (100% coverage). However, the permutations create 1,354 additional degree pairs (33% of total) that do not exist in the original graph.

### Model Performance

Train/test split: 2,196 train, 550 test degree pairs

Results on 200-permutation validation:

| Model | Features | Train r | Test r | Validation r | Interpolation r | Bias | RMSE | Training Time |
|-------|----------|---------|--------|--------------|-----------------|------|------|---------------|
| Analytical | 0 | N/A | 0.8260 | 0.9905 | N/A | -0.0007 | 0.0250 | 0s |
| SimpleNN | 2 | 0.8620 | 0.8619 | 0.9296 | 0.8953 | +0.0845 | 0.1255 | 0.67s |
| Minimal | 13 | 0.8758 | 0.8740 | 0.9158 | 0.9150 | +0.0865 | 0.1514 | 1.43s |

Definitions:
- Train r: Correlation on training split of original graph
- Test r: Correlation on test split of original graph
- Validation r: Correlation on all 4,100 degree pairs from 200-permutation data
- Interpolation r: Correlation on 1,354 unseen degree pairs only

### Key Findings

1. Analytical formula performs best
   - Validation r = 0.9905
   - Near-zero bias (-0.0007)
   - Lowest RMSE (0.0250)
   - No training required

2. SimpleNN achieves strong performance
   - Validation r = 0.9296
   - Successfully interpolates for unseen degree pairs (r = 0.8953)
   - Positive bias (+0.0845) indicates systematic overestimation
   - Fast training (0.67s)

3. Minimal features underperform SimpleNN
   - Validation r = 0.9158 (worse than SimpleNN despite 13 features)
   - Similar interpolation quality (r = 0.9150)
   - Higher bias (+0.0865) and RMSE (0.1514)
   - Longer training (1.43s)

4. Interpolation works well
   - Both neural network models successfully predict frequencies for 1,354 unseen degree pairs
   - Interpolation r approximately 0.90 for both models
   - This validates that neural networks learn continuous functions in degree space

### Unexpected Result: Minimal Model Underperforms

The minimal feature set includes P_L2_norm (analytical formula) as an input feature. We expected the model to achieve validation r greater than or equal to 0.9905 by learning to weight the analytical formula appropriately.

Actual result: Minimal model achieves validation r = 0.9158 (74 basis points worse than analytical)

Possible explanations:
1. Overfitting: Model overfit to training data (train r = 0.8758) and doesn't generalize
2. Feature noise: Additional features add noise rather than signal
3. Training issue: Model failed to learn to use P_L2_norm effectively
4. Task mismatch: Training on original graph frequencies (mean = 0.166) differs from validation on permutation frequencies

### Bias Analysis

Both neural network models show positive bias (approximately +0.085), indicating systematic overestimation of edge probabilities. This is likely due to:
- Original graph frequencies are higher than permutation frequencies
- Training data has different distribution than validation data
- Models learn the original graph distribution but are evaluated on null model distribution

Analytical formula shows near-zero bias because it is derived from the null model theory (XSwap stationary distribution).

## Comparison with Previous Approach

### Single-Permutation Binary Training (Previous)

Results from TRAINING_PIPELINE_FIX_SUMMARY.md:
- SimpleNN validation r: 0.7587
- Minimal features validation r: 0.8233
- Task: Binary classification on single permutation, validate on empirical frequencies

### Original Graph Frequency Training (Current)

Results:
- SimpleNN validation r: 0.9296 (15 percentage points better)
- Minimal features validation r: 0.9158 (9 percentage points better)
- Task: Frequency regression on original graph, validate on empirical frequencies

Improvement: Training on continuous frequencies from original graph yields significantly better validation performance than training on binary labels from single permutation.

## Conclusions

1. Training on original graph frequencies works well
   - SimpleNN achieves validation r = 0.9296
   - Significantly better than binary classification approach (r = 0.7587)

2. Interpolation is effective
   - Models successfully predict 33% unseen degree pairs with r approximately 0.90
   - Validates that neural networks learn continuous functions

3. Analytical formula remains best choice
   - Validation r = 0.9905
   - No training required
   - Near-zero bias and lowest RMSE

4. Minimal features do not improve over SimpleNN
   - Despite including analytical formula as input feature
   - May indicate overfitting or feature noise
   - Needs further investigation

5. Bias is a concern
   - Both ML models overestimate by approximately +0.085
   - Original graph has different frequency distribution than null model
   - May need calibration or different training approach

## Recommendations

### For CbG Edge Type

Use analytical formula (validation r = 0.9905). Neural network models provide minimal benefit and introduce positive bias.

### For Other Edge Types

Evaluate whether analytical formula underperforms on harder edge types (as found in FEATURE_REDUCTION_RECOMMENDATIONS.md):
- If analytical r greater than 0.99: Use analytical formula
- If analytical r less than 0.99: Consider SimpleNN with original graph training

### Future Work

1. Investigate why minimal features underperform
   - Check if P_L2_norm is being used by the model
   - Try simpler architectures
   - Test with different hyperparameters

2. Address positive bias
   - Investigate distribution shift between original and permutation frequencies
   - Consider calibration techniques
   - Try training on permutation data instead of original graph

3. Test on all edge types
   - Evaluate whether SimpleNN provides benefit for hard edge types (e.g., AeG)
   - Compare against previous binary training approach
   - Document edge type difficulty classification

4. Analyze interpolation quality
   - Characterize which unseen degree pairs are predicted accurately
   - Identify regions of degree space where interpolation fails
   - Consider feature engineering to improve interpolation

## Files Generated

Results for CbG:
- results/original_graph_training/CbG/original_frequencies.csv (2,746 degree pairs)
- results/original_graph_training/CbG/degree_pair_coverage.txt (coverage statistics)
- results/original_graph_training/CbG/analytical_residuals.png (4-panel residual plot)
- results/original_graph_training/CbG/simplenn_residuals.png (4-panel residual plot)
- results/original_graph_training/CbG/minimal_residuals.png (4-panel residual plot)
- results/original_graph_training/CbG/comparison_metrics.csv (performance comparison)

Code modules:
- src/original_graph_frequencies.py (frequency computation functions)
- src/evaluate_original_graph_approach.py (evaluation pipeline)
- test_original_graph_training.py (test script)

## Next Steps

1. Run evaluation on additional edge types (CtD, GpPW, AeG, DdG)
2. Compare results across edge types
3. Determine when ML provides benefit over analytical formula
4. Update FEATURE_REDUCTION_RECOMMENDATIONS.md with new findings
5. Consider hybrid approach: analytical formula + learned corrections
