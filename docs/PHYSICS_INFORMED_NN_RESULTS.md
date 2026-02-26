# Physics-Informed Neural Network Results

## Overview

We implemented a physics-informed neural network (PINN) trained using ONLY theoretical constraints from XSwap Markov chain theory. The goal was to learn edge probability predictions without using any empirical frequencies in training.

**Result**: The PINN performed significantly worse than both the analytical formula and theoretical corrections.

## Approach

### Architecture

Simple feedforward neural network:
- Input: (source_degree, target_degree)
- Hidden layers: [64, 32, 16] with ReLU activation and dropout
- Output: edge probability P(u,v) via sigmoid

### Loss Function

Weighted combination of four physics-based constraints:

```python
loss = w1 * conservation_loss +      # Total predicted edges = actual edge count
       w2 * marginal_loss +           # Marginal distributions match observed
       w3 * detailed_balance_loss +   # XSwap detailed balance
       w4 * monotonicity_loss         # Higher degree product → higher frequency
```

**Loss weights**: conservation=1.0, marginal=1.0, detailed_balance=0.5, monotonicity=0.5

### Training

- 1000 epochs
- Adam optimizer with learning rate 0.001
- Input normalization (z-score)
- No empirical frequencies used

## Results

### CbG (Compound-binds-Gene)

| Method | Correlation r | Bias | RMSE |
|--------|--------------|------|------|
| Analytical | 0.9905 | -0.0007 | 0.0250 |
| Corrected | 0.9833 | -0.0112 | 0.0327 |
| **PINN** | **0.5162** | **-0.0954** | **0.1735** |

**Result**: PINN failed catastrophically (Δr = -0.4743 vs analytical)

### AeG (Anatomy-expresses-Gene)

| Method | Correlation r | Bias | RMSE |
|--------|--------------|------|------|
| Analytical | 0.9598 | -0.0797 | 0.1534 |
| Corrected | 0.9628 | -0.0986 | 0.1795 |
| **PINN** | **0.7641** | **-0.1442** | **0.2833** |

**Result**: PINN much worse (Δr = -0.1957 vs analytical)

## Analysis: Why Did PINN Fail?

### 1. Marginal Loss is Too Strict

**Observation**: Marginal loss values remained very high throughout training:
- CbG: 14446 → 591 (epoch 999)
- AeG: 421073 → 6126 (epoch 999)

**Problem**: The marginal constraint requires that for each source node with degree u, the sum of predicted probabilities over all targets equals u. This is an extremely strict constraint that's hard to satisfy with a simple neural network.

**Mathematical issue**:
```
For source degree u:
  sum over all v of [n_targets(v) * P(u, v)] = u

For target degree v:
  sum over all u of [n_sources(u) * P(u, v)] = v
```

These constraints are **overdetermined** - there are more constraints than degrees of freedom. The analytical formula satisfies these in expectation but not exactly for each (u,v) pair.

### 2. Conflicting Constraints

The four loss components pull the model in different directions:

**Conservation vs Marginal**:
- Conservation: Total edges should sum to m
- Marginal: Each degree's edges should sum to its value
- These can conflict when trying to satisfy both exactly

**Detailed Balance vs Monotonicity**:
- Detailed Balance: P(u1,v1) * P(u2,v2) ≈ P(u1,v2) * P(u2,v1)
- Monotonicity: Higher degree product → higher probability
- These can conflict for certain degree combinations

### 3. Insufficient Degrees of Freedom

A simple feedforward NN with only (u, v) as inputs may not have enough capacity to satisfy all constraints simultaneously. The analytical formula P_L2_norm has a specific functional form derived from first principles - it's not clear an arbitrary NN can learn this.

### 4. Sampling in Marginal Loss

To make computation tractable, we sample only 20 unique degree values per epoch for marginal loss. This introduces:
- **Stochasticity**: Different samples each epoch
- **Incomplete coverage**: Many degree values never checked
- **Bias**: Sampling may not represent the full distribution

### 5. Wrong Objective

**Key insight**: The physics constraints don't directly optimize for matching the null distribution frequencies. They only constrain the solution space.

The analytical formula P_L2_norm is the **exact solution** to the XSwap stationary distribution under the configuration model. Our PINN is trying to learn this from scratch using only indirect constraints.

## What We Learned

### 1. Analytical Formula is Near-Optimal

The analytical formula (r = 0.9905 for CbG, r = 0.9598 for AeG) is already close to the best possible performance. It's derived from first principles and captures the essential physics of the system.

### 2. Simple Corrections Outperform Complex Models

Theoretical corrections (+0.003 improvement for AeG) are:
- Simpler to implement
- Faster to compute
- More interpretable
- More reliable than neural network approaches

### 3. Physics Constraints Alone Are Insufficient

The XSwap physics constraints (conservation, marginal, detailed balance) are necessary but not sufficient to uniquely determine edge probabilities. Additional information (like the specific functional form of P_L2_norm) is needed.

### 4. Conflicting Multi-Objective Optimization

Training with multiple conflicting objectives is challenging:
- Hard to balance loss weights
- Solution may not exist that satisfies all constraints
- Network may get stuck in poor local minima

### 5. Distribution Mismatch Problem Persists

Even though we're not training on original graph frequencies, we're still training on original graph **structure** (degrees, edge count). The constraints we impose reflect the original graph, not necessarily the null model.

## Comparison with Other Approaches

### Analytical Formula (Best)

**Pros**:
- Derived from first principles
- Excellent performance (r = 0.99)
- Fast to compute
- Interpretable

**Cons**:
- Small systematic bias for some edge types (AeG)
- Assumes random mixing (violated in real networks)

### Theoretical Corrections (Second Best)

**Pros**:
- Improves problematic edge types
- Uses only original graph features
- Fast to compute
- Interpretable

**Cons**:
- Modest improvement (+0.003)
- May hurt well-predicted edge types
- Limited to simple multiplicative factors

### Physics-Informed NN (Worst)

**Pros**:
- Theoretically appealing approach
- Could learn complex patterns

**Cons**:
- Much worse performance (r = 0.52 to 0.76)
- Slow to train
- Difficult to debug
- Many hyperparameters to tune
- Conflicting objectives

## Alternative Approaches That Might Work Better

### 1. Hybrid Approach: Analytical + NN Residual

Instead of learning from scratch, learn to correct the analytical formula:

```python
P_predicted = P_analytical + NN(u, v, graph_features)
```

Train NN to predict the residual using physics constraints as regularization.

### 2. Simpler Physics Constraints

Use only the most important constraint (conservation):

```python
loss = conservation_loss + regularization
```

This avoids conflicting objectives.

### 3. Meta-Learning Across Edge Types

Train on multiple edge types simultaneously to learn general correction patterns:

```python
P_corrected = P_analytical * correction_function(u, v, graph_features, edge_type_embedding)
```

But this violates the "no empirical data" constraint.

### 4. Accept Current Performance

The analytical formula with optional corrections achieves:
- CbG: r = 0.9905 (excellent, no corrections needed)
- AeG: r = 0.9628 (good, with corrections)

For most applications, this level of performance is sufficient.

## Recommendations

### Do NOT Use Physics-Informed NN

The PINN approach:
- Performs much worse than simple analytical formula
- Requires extensive tuning
- Is computationally expensive
- Is difficult to debug and interpret

### Use Analytical Formula + Theoretical Corrections

**Best practice**:
1. Start with analytical formula P_L2_norm
2. If r < 0.97, apply uniform theoretical corrections
3. Accept modest improvement as limit of simple approaches

### Code Availability

While the PINN approach failed, the code is preserved for reference:
- `src/physics_informed_nn.py`: PINN implementation
- `src/evaluate_physics_informed_nn.py`: Evaluation script
- `results/physics_informed_nn/`: Results and trained models

## Conclusion

The physics-informed neural network experiment demonstrates that:

1. **Theoretical constraints alone are insufficient** to learn edge probabilities
2. **The analytical formula is remarkably good** - hard to beat with data-driven methods
3. **Simple multiplicative corrections outperform complex neural networks**
4. **Multi-objective optimization with conflicting constraints is challenging**

**Final recommendation**: Use analytical formula with theoretical corrections for best results. The PINN approach is not viable for this problem.

## Files Generated

Results for CbG and AeG:
- results/physics_informed_nn/CbG/model.pt (trained model)
- results/physics_informed_nn/CbG/comparison.png (4-panel comparison)
- results/physics_informed_nn/CbG/comparison_metrics.csv (metrics)
- results/physics_informed_nn/CbG/predictions.csv (predictions)
- results/physics_informed_nn/AeG/ (same files)
- results/physics_informed_nn/summary.csv (cross-edge-type summary)

Code modules:
- src/physics_informed_nn.py (PINN implementation with loss functions)
- src/evaluate_physics_informed_nn.py (evaluation pipeline)

Documentation:
- PHYSICS_INFORMED_NN_RESULTS.md (this file)
