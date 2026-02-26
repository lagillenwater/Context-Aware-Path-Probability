# Experiments 2L and 2M: Non-Linear Models for Pathway Prediction
## Building on 2-Hop Success with Advanced Architectures

Date: 2025-11-05

---

## Motivation

### What We Know

**2-Hop models work (Exp 2D)**:
- CbGiG: r>0.95 from (deg_C, deg_G)
- GiGpPW: r>0.95 from (deg_G, deg_PW)

**Linear composition fails (Exp 2H, 2J)**:
- Composition + sparsity: r≈0.80 ceiling
- More permutations don't help

**Key insight**: If components work, the problem is in how we combine them.

### Why Non-Linear Models?

**Linear composition assumes**:
- Intermediates contribute independently
- Simple weighted sum captures all structure
- No interactions between intermediate contributions

**Reality may be**:
- Pathways through different intermediates are correlated
- High-degree intermediates dominate in non-linear ways
- Intermediate interactions matter (shared neighbors, clustering)

### Research Questions

**Primary**: Can non-linear models break the r=0.80 ceiling?

**Secondary**:
- Which non-linear architecture works best?
- How many training permutations are needed?
- Can we achieve r>0.95 cross-permutation prediction?

---

## Experiment 2L: Non-Linear Aggregation of 2-Hop Predictions

### Overview

Use 2-hop model predictions as features, learn non-linear aggregation with neural network or random forest.

### Hypothesis

**Linear aggregation** (current approach):
```
predicted = c1 × Σ(CbGiG × P_edge) + c2 × n_intermediates + c3
```

**Non-linear aggregation** can learn:
- Which intermediates contribute most
- Non-linear interactions between intermediate contributions
- Saturation effects (many intermediates don't scale linearly)
- Degree-dependent weighting schemes

### Detailed Method

#### Phase 1: Prepare 2-Hop Models

**Use existing trained models from Experiment 2D**:
- model_CbGiG: (deg_C, deg_G) → CbGiG count
- model_GiGpPW: (deg_G, deg_PW) → GiGpPW count

If not available, retrain:
```python
# Train on perm 1 (or mean of perms 1-K)
for each edge type:
    X = [(deg_source, deg_target) for all edges]
    y = [edge_count for all edges]
    model.fit(X, y)
```

#### Phase 2: Extract Per-Intermediate Features

**For each pair (C, PW)**:

1. **Identify intermediates** from perm 0 topology:
   ```python
   genes_to_PW = GpPW_0[:, PW].nonzero()[0]
   intermediates = [G2 for G2 in genes_to_PW if CbGiG_0[C, G2] > 0]
   ```

2. **For each intermediate G2**, compute:
   ```python
   deg_C = degree[C]
   deg_G2 = degree[G2]
   deg_PW = degree[PW]

   pred_CbGiG = model_CbGiG.predict([[deg_C, deg_G2]])[0]
   pred_GiGpPW = model_GiGpPW.predict([[deg_G2, deg_PW]])[0]

   # Store intermediate-level features
   intermediate_features = [
       pred_CbGiG,
       pred_GiGpPW,
       pred_CbGiG * pred_GiGpPW,  # Product (naive composition)
       deg_G2,  # Intermediate degree
       deg_G2 / mean(all_intermediate_degrees),  # Relative degree
   ]
   ```

3. **Aggregate across intermediates**:

   **Option A: Summary statistics** (for fixed-size input):
   ```python
   pair_features = [
       # Basic aggregations
       sum(pred_CbGiG for all intermediates),
       sum(pred_GiGpPW for all intermediates),
       sum(pred_CbGiG * pred_GiGpPW),  # Linear composition baseline

       # Count
       len(intermediates),

       # Degree statistics
       mean(deg_G2),
       std(deg_G2),
       min(deg_G2),
       max(deg_G2),
       median(deg_G2),

       # Prediction statistics
       mean(pred_CbGiG),
       max(pred_CbGiG),
       std(pred_CbGiG),
       mean(pred_GiGpPW),
       max(pred_GiGpPW),
       std(pred_GiGpPW),

       # Product statistics
       mean(pred_CbGiG * pred_GiGpPW),
       max(pred_CbGiG * pred_GiGpPW),
       sum((pred_CbGiG * pred_GiGpPW)^2),  # Sum of squares

       # Endpoint degrees
       deg_C,
       deg_PW,
       deg_C * deg_PW,
   ]
   ```

   Total: ~25 features per pair (fixed size)

   **Option B: Set-based representation** (for neural networks):
   - Keep all intermediate features (variable size per pair)
   - Use DeepSets or attention mechanism
   - More flexible but more complex

#### Phase 3: Train Non-Linear Models

**Model architectures to test**:

1. **Random Forest** (baseline non-linear)
   ```python
   from sklearn.ensemble import RandomForestRegressor

   model = RandomForestRegressor(
       n_estimators=100,
       max_depth=10,
       min_samples_leaf=5,
       random_state=42
   )
   model.fit(X_train, y_train)
   ```

2. **Gradient Boosting** (stronger non-linear learner)
   ```python
   from sklearn.ensemble import GradientBoostingRegressor

   model = GradientBoostingRegressor(
       n_estimators=100,
       learning_rate=0.1,
       max_depth=5,
       random_state=42
   )
   model.fit(X_train, y_train)
   ```

3. **Neural Network** (most flexible)
   ```python
   import torch.nn as nn

   class AggregationNet(nn.Module):
       def __init__(self, input_dim):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(input_dim, 64),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(64, 32),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(32, 16),
               nn.ReLU(),
               nn.Linear(16, 1)
           )

       def forward(self, x):
           return self.net(x).squeeze()
   ```

4. **DeepSets** (for variable-size intermediate sets)
   ```python
   class DeepSetsAggregation(nn.Module):
       def __init__(self, intermediate_dim):
           super().__init__()
           # Encode each intermediate
           self.encoder = nn.Sequential(
               nn.Linear(intermediate_dim, 32),
               nn.ReLU(),
               nn.Linear(32, 16)
           )
           # Aggregate (permutation invariant)
           self.aggregation = lambda x: torch.sum(x, dim=1)
           # Decode to prediction
           self.decoder = nn.Sequential(
               nn.Linear(16, 32),
               nn.ReLU(),
               nn.Linear(32, 1)
           )

       def forward(self, intermediates):
           # intermediates: (batch, n_intermediates, intermediate_dim)
           encoded = self.encoder(intermediates)  # (batch, n_intermediates, 16)
           aggregated = self.aggregation(encoded)  # (batch, 16)
           return self.decoder(aggregated).squeeze()
   ```

#### Phase 4: Training Strategy

**Configuration 1: Single Permutation (K=1)**

Training:
```python
# Extract features for all pairs from perm 0 topology
X_features = extract_aggregation_features(pairs, perm_0_topology)

# Compute training target from perm 1
y_train_target = compute_pathway_counts(perm_1, pairs)

# Compute validation target from perms 11-20
y_val_target = mean([compute_pathway_counts(p, pairs) for p in range(11, 21)])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_train_target, test_size=0.2, random_state=42
)

# Also split validation target
_, _, _, y_val_test = train_test_split(
    X_features, y_val_target, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r_within_perm = pearsonr(y_pred, y_test)[0]  # vs perm 1
r_cross_perm = pearsonr(y_pred, y_val_test)[0]  # vs mean(11-20)
```

**Configurations 2-6: Multiple Permutations (K=2,3,4,5,10)**

If K=1 doesn't achieve r>0.95:
```python
# Train target = mean of perms 1-K
train_counts = [compute_pathway_counts(p, pairs) for p in range(1, K+1)]
y_train_target_K = mean(train_counts, axis=0)

# Rest same as K=1
```

#### Phase 5: Evaluation and Analysis

**Primary metrics**:
- r_cross_perm: Correlation with mean(perms 11-20)
- MAE_cross_perm: Mean absolute error
- Bias: mean(predictions) / mean(validation)

**Secondary metrics**:
- r_within_perm: Correlation with training perm(s)
- Feature importance (for tree models)
- Residual analysis by n_intermediates bins

**Success criteria**:
- r_cross_perm > 0.95: SUCCESS
- r_cross_perm > 0.85: Improvement over linear (explore further)
- r_cross_perm < 0.85: No improvement (try Exp 2M)

**Comparisons**:
- Baseline: Linear composition (r=0.78)
- Best previous: Linear + sparsity (r=0.80)

---

## Experiment 2M: Graph Neural Network on Intermediate Subgraph

### Overview

Instead of aggregating intermediates independently, model their relationships using a graph neural network.

### Hypothesis

**Why linear aggregation fails**: Intermediates are not independent
- They share neighbors (common genes they interact with)
- They cluster together in some permutations
- High-degree intermediates may dominate through shared structure

**GNN can learn**: How intermediate connectivity affects pathway counts

### Detailed Method

#### Phase 1: Construct Intermediate Subgraphs

**For each pair (C, PW)**:

1. **Identify intermediates**:
   ```python
   intermediates = [G2 for G2 in genes_to_PW if CbGiG_0[C, G2] > 0]
   ```

2. **Build subgraph of intermediates**:
   ```python
   # Nodes: intermediate genes
   # Edges: GiG interactions between intermediates
   subgraph_edges = []
   for G1 in intermediates:
       for G2 in intermediates:
           if GiG_0[G1, G2] > 0:
               subgraph_edges.append((G1, G2, GiG_0[G1, G2]))
   ```

3. **Node features for each intermediate G2**:
   ```python
   node_features[G2] = [
       deg_G2,  # Degree in full graph
       degree_in_subgraph[G2],  # Connections to other intermediates
       pred_CbGiG(deg_C, deg_G2),  # Predicted incoming paths
       pred_GiGpPW(deg_G2, deg_PW),  # Predicted outgoing paths
       CbGiG_0[C, G2],  # Actual incoming paths from perm 0
   ]
   ```

4. **Global context features**:
   ```python
   context_features = [
       deg_C,
       deg_PW,
       len(intermediates),
       mean(node_features[:, 0]),  # Mean intermediate degree
   ]
   ```

#### Phase 2: GNN Architecture

**Message Passing GNN**:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class IntermediateGNN(nn.Module):
    def __init__(self, node_feature_dim, context_dim):
        super().__init__()

        # Node feature encoder
        self.node_encoder = nn.Linear(node_feature_dim, 32)

        # Graph convolution layers
        self.conv1 = GCNConv(32, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)

        # Context encoder
        self.context_encoder = nn.Linear(context_dim, 16)

        # Combine graph embedding + context
        self.predictor = nn.Sequential(
            nn.Linear(16 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index, batch, context):
        # x: node features (n_nodes, node_feature_dim)
        # edge_index: graph connectivity
        # batch: which graph each node belongs to
        # context: global features per graph

        # Encode nodes
        x = self.node_encoder(x)
        x = torch.relu(x)

        # Message passing
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)

        # Aggregate to graph-level representation
        graph_embedding = global_mean_pool(x, batch)

        # Encode context
        context_embedding = self.context_encoder(context)
        context_embedding = torch.relu(context_embedding)

        # Combine and predict
        combined = torch.cat([graph_embedding, context_embedding], dim=1)
        return self.predictor(combined).squeeze()
```

**Key components**:
- **Node encoder**: Projects intermediate features to embedding space
- **Graph convolutions**: Learn from intermediate connectivity
- **Pooling**: Aggregate intermediate representations
- **Context**: Include endpoint information
- **Predictor**: Map to pathway count

#### Phase 3: Data Preparation

**Create PyTorch Geometric dataset**:

```python
from torch_geometric.data import Data, DataLoader

graphs = []
for (C, PW) in pairs:
    # Extract intermediate subgraph
    intermediates, edges, node_features = build_subgraph(C, PW, perm_0)

    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    x = torch.tensor(node_features, dtype=torch.float)
    context = torch.tensor([deg_C, deg_PW, len(intermediates), ...],
                          dtype=torch.float)

    # Target (from perm 1 or mean of perms 1-K)
    y = torch.tensor([pathway_count], dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, context=context, y=y)
    graphs.append(graph)

# Create data loader
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
```

#### Phase 4: Training

```python
model = IntermediateGNN(node_feature_dim=5, context_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index, batch.batch, batch.context)
        loss = criterion(pred, batch.y)

        loss.backward()
        optimizer.step()
```

#### Phase 5: Evaluation

Same as Experiment 2L:
- Train on perm 1 (K=1)
- Evaluate on mean(perms 11-20)
- If r<0.95, try K=2,3,4,5,10

---

## Experimental Workflow

### Stage 1: Single Permutation (K=1)

**For both Exp 2L and 2M**:

1. **Extract features** from perm 0 topology and perm 1 targets
2. **Train models**:
   - Exp 2L: Random Forest, Gradient Boosting, Neural Net, DeepSets
   - Exp 2M: GNN
3. **Evaluate** on mean(perms 11-20)
4. **Compare** to baseline (r=0.80)

**Decision point**:
- If ANY model achieves r>0.95: SUCCESS, document and stop
- If best model r>0.85: Promising, proceed to Stage 2
- If all models r<0.85: No improvement, analyze failure modes

### Stage 2: Multiple Permutations (If Needed)

**Test K in {2, 3, 4, 5, 10}**:

For each K:
1. Compute training target = mean(perms 1-K)
2. Retrain best model(s) from Stage 1
3. Evaluate on mean(perms 11-20)
4. Track improvement vs K

**Early stopping**:
- If r>0.95 achieved at any K, stop
- If no improvement from K=5 to K=10, stop

### Stage 3: Analysis and Interpretation

**If successful (r>0.95)**:
- Document minimum K required
- Analyze feature importance / learned representations
- Test on additional metapaths
- Compare computational cost vs enumeration

**If unsuccessful (r<0.95)**:
- Analyze predictions vs residuals
- Compare to linear baseline
- Identify systematic failures
- Propose next experiments or accept enumeration

---

## Success Criteria

### Primary Success
- r > 0.95 on mean(perms 11-20)
- MAE < 2.0
- Generalizes across K values tested

### Secondary Success
- r > 0.85 (improvement over r=0.80 baseline)
- Provides insights into non-linear pathway structure
- Demonstrates value of 2-hop model predictions

### Failure Indicators
- r < 0.85 (no improvement)
- High computational cost with minimal gains
- Overfits to training permutations

---

## Computational Considerations

### Experiment 2L Costs

**Feature extraction**: ~5-10 seconds
- Reuse 2-hop models
- Compute aggregation features once

**Model training**:
- Random Forest: ~10-30 seconds
- Neural Network: ~1-5 minutes (depends on architecture)

**Total per configuration**: ~5-10 minutes

### Experiment 2M Costs

**Subgraph construction**: ~30-60 seconds
- Build graph for each pair
- Extract node features

**GNN training**: ~5-15 minutes
- Depends on graph sizes
- May need GPU for efficiency

**Total per configuration**: ~10-20 minutes

### Comparison to Enumeration

**Enumeration (baseline)**:
- 10 permutations: ~1-5 seconds
- 100% accurate

**Non-linear models**:
- Training: 10-20 minutes once
- Inference: <1 second for new pairs
- Amortized if used repeatedly

---

## Expected Outcomes

### Optimistic Scenario
- Non-linear models capture intermediate interactions
- Achieve r>0.90 or even r>0.95
- K=1 or K=2 sufficient
- Breakthrough for compositional null models

### Realistic Scenario
- Modest improvement to r=0.82-0.87
- Requires K=5-10 permutations
- Better than linear but not r>0.95
- Useful for approximate screening, not precise nulls

### Pessimistic Scenario
- No improvement over r=0.80
- Non-linearity doesn't help
- Fundamental limitation confirmed
- Accept enumeration necessity

---

## Implementation Order

1. **Experiment 2L with Random Forest** (fastest to implement)
   - Test K=1 first
   - If promising, try K=2,3,4,5,10

2. **Experiment 2L with Neural Network** (if RF shows promise)
   - More flexible architecture
   - May capture subtler patterns

3. **Experiment 2M with GNN** (if 2L doesn't reach r>0.95)
   - More sophisticated but slower
   - Last attempt before accepting enumeration

---

## Files to Create

1. `test_src/test_nonlinear_aggregation_exp2l.py` - Exp 2L implementation
2. `test_src/test_gnn_intermediates_exp2m.py` - Exp 2M implementation
3. `docs/2025-11-05_EXPERIMENT_2L_RESULTS.md` - Results documentation
4. `docs/2025-11-05_EXPERIMENT_2M_RESULTS.md` - Results documentation

---

## Next Steps

Ready to implement Experiment 2L first (Random Forest with aggregation features), starting with K=1 and evaluating on mean(perms 11-20)?
