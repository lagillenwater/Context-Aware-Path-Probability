# Phase 2: Feature Summary

## Training Data Structure

Each CSV file in `results/phase2_training_data/` contains training data for one metapath and one feature set.

### Column Types

**Metadata Columns (12 total):**
1. `source_bin` - Degree bin index for source nodes (0-9)
2. `target_bin` - Degree bin index for target nodes (0-9)
3. `n_source_nodes` - Number of source nodes in this bin
4. `n_target_nodes` - Number of target nodes in this bin
5. `n_pairs` - Total (source, target) pairs in this bin combination
6. `pathway_count_mean` - **TARGET VARIABLE** - Mean pathway count for pairs in this bin
7. `pathway_count_std` - Standard deviation of pathway counts
8. `pathway_count_median` - Median pathway count
9. `pathway_count_p25` - 25th percentile
10. `pathway_count_p75` - 75th percentile
11. `pathway_count_min` - Minimum pathway count
12. `pathway_count_max` - Maximum pathway count

**Feature Columns (102-116 total, depending on set):**
All columns prefixed with `feat_` are input features for the model.

---

## Feature Set Breakdown

### Set A: 102 Features (Baseline)

**Degree Bin Features (2):**
- `feat_source_bin` - Quantile bin index for source degree (0-9)
- `feat_target_bin` - Quantile bin index for target degree (0-9)

**Intermediate Node Degree Signature (100):**
A flattened 10×10 histogram of (in_degree, out_degree) for intermediate nodes:
- `feat_hist_in0_out0` to `feat_hist_in9_out9` (100 features)
- Example: `feat_hist_in2_out5` = count of intermediate nodes with in_degree bin 2, out_degree bin 5

This signature captures the degree distribution of nodes connecting source to target.

---

### Set B: 104 Features (+ Log Transforms)

**Set A features (102) PLUS:**

**Log Degree Features (2):**
- `feat_log_source_deg` - log(1 + source_degree)
- `feat_log_target_deg` - log(1 + target_degree)

**Rationale:** Log transforms handle heavy-tailed degree distributions common in biological networks.

---

### Set C: 109 Features (+ Summary Statistics)

**Set B features (104) PLUS:**

**Intermediate Degree Summary Statistics (5):**
- `feat_mean_int_deg` - Mean degree of intermediate nodes
- `feat_std_int_deg` - Std of intermediate node degrees
- `feat_min_int_deg` - Min intermediate degree
- `feat_max_int_deg` - Max intermediate degree
- `feat_median_int_deg` - Median intermediate degree

**Rationale:** Compress 100-dim histogram into interpretable statistics.

---

### Set D: 111 Features (+ Neighbor Context)

**Set C features (109) PLUS:**

**2nd-Order Degree Statistics (2):**
- `feat_neighbor_deg_mean` - Mean degree of neighbors' neighbors
- `feat_neighbor_deg_std` - Std of neighbors' neighbors' degrees

**Rationale:** Capture local network topology beyond direct connections.

---

### Set E: 113 Features (+ Polynomial Terms)

**Set D features (111) PLUS:**

**Polynomial Degree Features (2):**
- `feat_source_deg_sq` - (source_degree)²
- `feat_target_deg_sq` - (target_degree)²

**Rationale:** Capture non-linear degree effects (e.g., high-degree hubs).

---

### Set F: 116 Features (+ Interaction Terms)

**Set E features (113) PLUS:**

**Degree Interaction Features (3):**
- `feat_source_x_target` - source_degree × target_degree
- `feat_source_x_mean_int` - source_degree × mean_intermediate_degree
- `feat_target_x_mean_int` - target_degree × mean_intermediate_degree

**Rationale:** Capture synergistic effects between node degrees.

---

## Example Training Row (CbGpPW, Bin 0×0)

```
Metadata:
  source_bin = 0           (lowest degree compounds)
  target_bin = 0           (lowest degree pathways)
  n_source_nodes = 332     (332 compounds in this degree bin)
  n_target_nodes = 254     (254 pathways in this degree bin)
  n_pairs = 84,328         (332 × 254 = 84,328 possible pairs)

Target:
  pathway_count_mean = 0.000984   (on average, 0.001 pathways per pair)
  pathway_count_std = 0.0314      (high variance relative to mean)
  pathway_count_median = 0.0      (most pairs have 0 pathways)

Features (Set A example):
  feat_source_bin = 0.0
  feat_target_bin = 0.0
  feat_hist_in0_out0 = 795.0     (795 genes with lowest in/out degree)
  feat_hist_in0_out1 = 27.0
  feat_hist_in0_out2 = 2.0
  ... (97 more histogram bins)
```

**Interpretation:** This bin represents low-degree compounds and low-degree pathways. The intermediate gene signature shows most genes also have low degree (hist_in0_out0 = 795).

---

## What the Model Learns

**Training Task:**
Given the degree bins and intermediate node signature, predict the mean pathway count for that bin combination.

**Example:**
```
Input:  source_bin=0, target_bin=0, hist=[795, 27, 2, ...]
Output: pathway_count_mean = 0.000984
```

The model learns: "For low-degree compounds and low-degree pathways connected through low-degree genes, expect ~0.001 pathways on average."

---

## Testing Task (Hybrid Evaluation)

**Training:** Model learns on ~10-90 bin combinations (aggregated statistics)

**Testing:** Model predicts for millions of individual (source, target) pairs
1. For each node pair (compound_i, pathway_j):
   - Look up degree bin for compound_i → bin_s
   - Look up degree bin for pathway_j → bin_t
   - Extract intermediate node signature
   - Use trained model to predict pathway count
2. Compare predictions to actual counts (averaged over 20 permutations)
3. Compute Pearson r

**Success criterion:** r ≥ 0.95

---

## Key Insights

**Why so few training samples?**
- We're predicting BIN-LEVEL statistics, not individual pairs
- 10 source bins × 10 target bins = max 100 combinations
- Some metapaths have fewer due to duplicated degrees → 6-90 samples

**Why does this work?**
- Assumes pathway counts are determined by degree bins (degree-deterministic)
- Oracle analysis showed r~0.995, confirming strong degree signal
- Challenge: Extract maximum signal with minimal training data

**What are we testing?**
- Does adding features beyond baseline (Set A: 102) improve predictions?
- Which feature types help most: log transforms, summary stats, interactions?
- Can we reach r≥0.95 (vs current r=0.94)?
