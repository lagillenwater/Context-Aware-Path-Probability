# Comprehensive Repository Overhaul Plan

This plan combines repository restructuring with pipeline methodology fixes to achieve both clean code organization and correct scientific implementation.

## Part A: Repository Restructuring (Foundation)

### Phase A1: Directory Setup
**Create new directory structure:**
```bash
mkdir -p notebooks/exploratory/{compositionality,formula_testing,degree_testing}
mkdir -p src/helpers
mkdir -p scripts/utils
mkdir -p tests
```

### Phase A2: File Organization

**Move exploratory notebooks:**
```
notebooks/10_metapath_compositionality_analysis.ipynb → notebooks/exploratory/compositionality/
notebooks/11_degree_conditioned_compositionality.ipynb → notebooks/exploratory/compositionality/
notebooks/12_degree_aware_compositional_model.ipynb → notebooks/exploratory/compositionality/
notebooks/8b_formula_testing.ipynb → notebooks/exploratory/formula_testing/
notebooks/8c_learned_formula_diagnostics.ipynb → notebooks/exploratory/formula_testing/
notebooks/8d_fair_learned_analytical_comparison.ipynb → notebooks/exploratory/formula_testing/
notebooks/8d_fair_executed.ipynb → notebooks/exploratory/formula_testing/
notebooks/test_degree_analysis_small_graphs.ipynb → notebooks/exploratory/degree_testing/
```

**Move helper scripts:**
```
fix_analytical_correlation.py → src/helpers/
notebook_5_correlation_fix.py → src/helpers/
extract_all_correlations.py → src/helpers/
notebook_5_correlation_additions.py → src/helpers/
degree_analysis_empirical_additions.py → src/helpers/
```

**Move test files:**
```
test_notebook5_fix.py → tests/
test_updated_notebook.py → tests/
```

**Move utility scripts:**
```
scripts/monitor_jobs.sh → scripts/utils/
scripts/submit_all.sh → scripts/utils/
scripts/run_all_degree_analysis.sh → scripts/utils/
```

### Phase A3: Rename Core Pipeline Files

**Rename notebooks (add zero-padding):**
```
notebooks/0_create-hetmat.ipynb → notebooks/00_create_hetmat.ipynb
notebooks/1_generate-permutations.ipynb → notebooks/01_generate_permutations.ipynb
notebooks/2_download_null_graphs.ipynb → notebooks/02_download_null_graphs.ipynb
notebooks/3_edge_frequency_by_degree.ipynb → notebooks/03_edge_frequency_by_degree.ipynb
notebooks/4_model_testing_reorganized.ipynb → notebooks/04_model_testing.ipynb
notebooks/5_model_testing_summary_with_degree_analysis.ipynb → notebooks/05_model_testing_summary.ipynb
notebooks/6_minimum_permutations_analysis.ipynb → notebooks/06_minimum_permutations_analysis.ipynb
notebooks/7_minimum_permutations_summary.ipynb → notebooks/07_minimum_permutations_summary.ipynb
notebooks/8_learned_analytical_formula_with_degree_analysis.ipynb → notebooks/08_learned_analytical_formula.ipynb
notebooks/9_learned_analytical_summary.ipynb → notebooks/09_learned_analytical_summary.ipynb
notebooks/14_fast_compositional_null_optimized.ipynb → notebooks/14_fast_compositional_null.ipynb
```

**Rename scripts:**
```
scripts/0_create_hetmat.sh → scripts/00_create_hetmat.sh
scripts/1_create_permutations.sh → scripts/01_create_permutations.sh
scripts/6_minimum_permutations_ml.sh → scripts/06_minimum_permutations_analysis.sh
scripts/6_metapath_analysis.sh → scripts/06_metapath_analysis.sh (if different from above)
scripts/013_null_model_training.sh → scripts/13_null_model_training.sh
scripts/014_fast_compositional_null.sh → scripts/14_fast_compositional_null.sh
scripts/015_metapath_null_distributions.sh → scripts/15_metapath_null_distributions.sh
scripts/016_dynamic_programming_dwpc.sh → scripts/16_dynamic_programming_dwpc.sh
```

### Phase A4: Delete Obsolete Files

**Remove duplicate/old notebooks:**
```
notebooks/4_model_testing.ipynb
notebooks/4_model_testing_executed.ipynb
notebooks/4_model_testing_reorganized-Lucas's MacBook Pro.ipynb
notebooks/5_cross_edge_type_summary.ipynb
notebooks/5_model_testing_summary.ipynb
notebooks/6_metapath_probability_analysis.ipynb
notebooks/8_learned_analytical_formula.ipynb
notebooks/14_fast_compositional_null.ipynb (non-optimized)
```

**Remove obsolete scripts:**
```
scripts/004.1_degree_analysis.sh
scripts/compositional_null_old.sh
scripts/compositional_null_extended.sh
scripts/enhanced_model_testing.sh
scripts/learned_formula_enhanced.sh
scripts/metapath_array.sh
scripts/notebook5_fix.sh
scripts/full_pipeline.sh
```

**Remove executed notebook duplicates:**
```
notebooks/outputs/5_model_testing_summary_executed.ipynb  # Superseded
notebooks/outputs/10_metapath_compositionality_analysis_executed.ipynb  # Move to exploratory first
```

**After organizing, remove outputs directory:**
```bash
rm -rf notebooks/outputs/
```

---

### Phase A5: Organize Executed Notebooks

**Purpose**: Consolidate and organize executed notebooks with clear structure for archival, active executions, and pipeline stages.

**Current Issues:**
- Executed notebooks scattered across `notebooks/executed/` and `notebooks/outputs/`
- No clear organization by pipeline stage or date
- Mix of successful, failed, and superseded executions
- Difficult to find latest successful run for each notebook
- `notebooks/outputs/` is deprecated but still contains files

**Proposed Structure:**
```bash
notebooks/executed/
├── pipeline/              # Active pipeline executions (keep latest)
│   ├── 00_create_hetmat_executed.ipynb
│   ├── 01_generate_permutations_executed.ipynb
│   ├── 03_edge_frequency_by_degree/
│   │   ├── 03_edge_frequency_by_degree_AdG_executed.ipynb
│   │   ├── 03_edge_frequency_by_degree_CbG_executed.ipynb
│   │   └── ...
│   ├── 04_model_testing/
│   │   ├── 04_model_testing_AdG_executed.ipynb
│   │   ├── 04_model_testing_CbG_executed.ipynb
│   │   └── ...
│   ├── 05_model_testing_summary_executed.ipynb
│   ├── 13_null_model_training_executed.ipynb
│   ├── 14_fast_compositional_null_executed.ipynb
│   ├── 15_metapath_null_distributions/
│   │   ├── 15_metapath_CbGpPW_executed.ipynb
│   │   ├── 15_metapath_CtDaG_executed.ipynb
│   │   └── ...
│   ├── 16_dynamic_programming_dwpc_executed.ipynb
│   ├── 17_compositional_validation_executed.ipynb
│   └── 17b_compositional_failure_analysis_executed.ipynb
│
├── exploratory/           # Exploratory analysis executions
│   ├── 10_metapath_compositionality_analysis_executed.ipynb
│   ├── 11_degree_conditioned_compositionality_executed.ipynb
│   └── 12_degree_aware_compositional_model_executed.ipynb
│
├── archive/               # Old/superseded executions (by date)
│   ├── 2024-10/
│   │   ├── 04_model_testing_CbG_executed_20241003.ipynb
│   │   ├── 05_model_testing_summary_executed_20241003.ipynb
│   │   └── ...
│   └── 2024-09/
│       └── ...
│
└── failed/                # Failed executions for debugging
    ├── 17b_compositional_failure_analysis_oom_20241010.ipynb
    └── ...
```

**Implementation Steps:**

**Step 1: Create directory structure**
```bash
cd notebooks/executed/
mkdir -p pipeline/{03_edge_frequency_by_degree,04_model_testing,15_metapath_null_distributions}
mkdir -p exploratory
mkdir -p archive/2024-{09,10}
mkdir -p failed
```

**Step 2: Move active pipeline executions**
```bash
# Move single-execution notebooks
mv 13_null_model_training_executed.ipynb pipeline/
mv 14_fast_compositional_null_executed.ipynb pipeline/
mv 17_compositional_validation_executed.ipynb pipeline/

# Move array job outputs to subdirectories
mv 15_metapath_*_executed.ipynb pipeline/15_metapath_null_distributions/

# When 04 and 03 execute, organize similarly:
# mv 04_model_testing_*_executed.ipynb pipeline/04_model_testing/
# mv 03_edge_frequency_by_degree_*_executed.ipynb pipeline/03_edge_frequency_by_degree/
```

**Step 3: Move exploratory notebook executions**
```bash
cd ../
# Move exploratory executions to executed/exploratory/
mv outputs/10_metapath_compositionality_analysis_executed.ipynb executed/exploratory/
```

**Step 4: Archive old executions from outputs/**
```bash
# Identify date from file metadata or filename
# Move to archive with date suffix
mv outputs/5_model_testing_summary_executed.ipynb \
   executed/archive/2024-10/5_model_testing_summary_executed_20241003.ipynb
```

**Step 5: Clean up deprecated outputs/ directory**
```bash
# After moving all valuable executions:
cd notebooks/
rm -rf outputs/  # Only after confirming all important files moved
```

**Step 6: Update .gitignore**
```bash
# Add to .gitignore (if not already present):
notebooks/executed/archive/
notebooks/executed/failed/
notebooks/executed/*_executed.ipynb  # Exclude individual executed notebooks from git
!notebooks/executed/pipeline/  # But include the structure
```

**Naming Conventions:**

**Active pipeline executions:**
- Format: `{NN}_{name}_executed.ipynb` or `{NN}_{name}_{parameter}_executed.ipynb`
- Examples:
  - `05_model_testing_summary_executed.ipynb`
  - `15_metapath_CbGpPW_executed.ipynb`
  - `04_model_testing_CbG_executed.ipynb`

**Archived executions:**
- Format: `{NN}_{name}_executed_YYYYMMDD.ipynb`
- Examples:
  - `05_model_testing_summary_executed_20241003.ipynb`
  - `17_compositional_validation_executed_20241009.ipynb`

**Failed executions:**
- Format: `{NN}_{name}_{failure_reason}_YYYYMMDD.ipynb`
- Examples:
  - `17b_compositional_failure_analysis_oom_20241010.ipynb`
  - `04_model_testing_CbG_timeout_20241008.ipynb`

**Maintenance Strategy:**

**When new execution completes:**
1. If successful → Move old version to `archive/YYYY-MM/` with date suffix
2. If failed → Move to `failed/` with failure reason
3. Place new execution in appropriate `pipeline/` or `exploratory/` location

**Quarterly cleanup:**
1. Review `archive/` and delete executions older than 6 months
2. Review `failed/` and delete after issues resolved
3. Compress old archives: `tar -czf archive_2024-Q3.tar.gz archive/2024-07/ archive/2024-08/ archive/2024-09/`

**Benefits:**
- ✅ Clear separation: active vs archived vs failed
- ✅ Easy to find latest successful execution
- ✅ Organized by pipeline stage (array jobs in subdirectories)
- ✅ Historical record with dates
- ✅ Debugging aid (failed executions preserved)
- ✅ Reduced clutter in main executed/ directory

**Script Integration:**

Update all shell scripts to output to organized structure:
```bash
# In scripts/15_metapath_null_distributions.sh:
OUTPUT_NOTEBOOK="${notebooks_path}/executed/pipeline/15_metapath_null_distributions/15_metapath_${METAPATH}_executed.ipynb"

# In scripts/04_model_comparison_analysis.sh:
OUTPUT_NOTEBOOK="${notebooks_path}/executed/pipeline/04_model_testing/04_model_testing_${EDGE_TYPE}_executed.ipynb"
```

**Documentation Updates:**

Update `scripts/README.md` to document the new structure:
```markdown
## Executed Notebooks Location

All executed notebooks are organized in `notebooks/executed/`:

- `pipeline/`: Latest successful executions of pipeline notebooks
  - Array job outputs in subdirectories (04, 15, etc.)
- `exploratory/`: Exploratory analysis executions
- `archive/YYYY-MM/`: Historical executions (dated)
- `failed/`: Failed executions for debugging

To find latest successful execution of a notebook:
```bash
# Single-execution notebooks
ls notebooks/executed/pipeline/05_model_testing_summary_executed.ipynb

# Array job notebooks
ls notebooks/executed/pipeline/04_model_testing/
```
```

**Priority:** Medium (Part A to be executed after scientific methodology validated)

**Estimated Time:** 1 hour for initial organization + 10 minutes per future execution to maintain

---

### Phase A6: Update All Script Paths

**For each script in scripts/, update:**
1. Notebook input paths (use notebooks/XX_ prefix)
2. Notebook output paths (all to notebooks/executed/XX_..._executed.ipynb)
3. Use relative paths from repo root
4. Remove hardcoded absolute paths

**Example pattern for scripts/04_model_comparison_analysis.sh:**
```bash
# Before
INPUT_NOTEBOOK="notebooks/4_model_testing_reorganized.ipynb"
OUTPUT_NOTEBOOK="notebooks/outputs/${EDGE_TYPE}_model_testing.ipynb"

# After
INPUT_NOTEBOOK="notebooks/04_model_testing.ipynb"
OUTPUT_NOTEBOOK="notebooks/executed/04_model_testing_${EDGE_TYPE}_executed.ipynb"
```

### Phase A7: Create Documentation

**Create notebooks/exploratory/compositionality/README.md:**
```markdown
# Compositionality Analysis

## Key Findings
- Metapath PMI in Hetionet: mean = 7.11 (strongly conditional)
- Hetionet vs Null: No significant difference (p=0.715)
- **Conclusion**: Conditional structure is degree-driven, not biology-driven

## Notebooks
- **10**: PMI analysis, tests compositional assumption
- **11**: Degree-stratified PMI comparison
- **12**: Continuous degree-aware compositional model

## Status
Research complete. Core finding: degree-aware models capture conditional structure.
```

**Create notebooks/exploratory/formula_testing/README.md:**
```markdown
# Learned Formula Testing

## Formula Variants Tested
- Original (9 params): Standard parameterized formula
- Extended (11 params): Adds log-degree terms
- Polynomial (9 params): Polynomial ratio form

## Notebooks
- **8b**: Compare formula variants on sparse/dense graphs
- **8c**: Diagnostics and convergence analysis
- **8d**: Fair comparison methodology

## Status
Exploratory. Main formula implementation in notebook 08.
```

**Create notebooks/exploratory/degree_testing/README.md:**
```markdown
# Degree Analysis Testing

Small graph testing utilities for validating degree-stratified analyses.

## Status
Testing utilities.
```

### Phase A8: Update CLAUDE.md

Update core pipeline section and add exploratory section with new notebook numbering and organization.

---

## Part B: Pipeline Methodology Fixes (REVISED - Validation First!)

### Critical Finding

**Notebooks 10-12 showed**: Compositional assumption is violated (PMI ≈ 7, edges NOT independent)
**Notebooks 13-16**: Train models and compute DWPC, but never validated compositional calculation for pathway nulls
**Notebook 14**: Validation failed (NaN) due to methodology issues

**WE MUST VALIDATE BEFORE DEPLOYING!**

### Decision Tree Architecture

```
Step 1: Validate Compositional Calculation
   ├─→ Works (r > 0.95)? → Proceed to Step 2
   └─→ Fails? → Must use direct empirical approach (expensive)

Step 2: Find Minimum N for Approximation
   ├─→ Analytical (0 perms): Accurate? → Deploy immediately!
   ├─→ Learned Formula (N perms): Min N? → Deploy with N perms
   ├─→ ML Models (N perms): Min N? → Deploy with N perms
   └─→ None work? → Must use full 200 permutations

Step 3: Variance Estimation
   └─→ Compute Var from held-out permutations

Step 4: Anomaly Detection
   └─→ z = (observed - expected) / sqrt(var)
```

---

### Phase B0: Minimum Permutations Analysis (INDEPENDENT - Can Run Now)

**Status**: Notebooks 6 and 7 exist and are functional. Independent of validation results.

**Purpose**: Determine minimum number of permutations needed for ML models to accurately approximate 200-permutation empirical edge frequencies. **This analysis runs in parallel with compositional validation.**

#### Notebook 6: Progressive Training Analysis

**File**: `notebooks/6_minimum_permutations_analysis.ipynb`

**Method**:
- Train models from notebook 4 with progressively more permutations: N = 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50
- Compare predictions against 200-perm gold standard empirical frequencies (from notebook 3)
- Find minimum N where each model achieves target performance (e.g., correlation > 0.90)
- Detect convergence (when additional perms provide < 2% improvement)

**Models tested** (from notebook 4):
- Simple Neural Network
- Random Forest
- Logistic Regression
- Polynomial Logistic Regression

**Outputs**:
- `results/minimum_permutations_ml/{edge_type}_results/{edge_type}_convergence_data.csv`
- `results/minimum_permutations_ml/{edge_type}_results/{edge_type}_summary.json`
- Convergence curves showing N vs. correlation/MAE/RMSE
- N_min comparison across models

**Shell script**: `scripts/6_minimum_permutations_ml.sh` (exists)

**Updates needed**:
1. Rename script: `6_minimum_permutations_ml.sh` → `06_minimum_permutations_analysis.sh`
2. Update output paths to use `notebooks/executed/06_minimum_permutations_analysis_${EDGE_TYPE}_executed.ipynb`
3. Update input notebook reference to `06_minimum_permutations_analysis.ipynb`

**Execution**: SLURM array job (1-24) for all edge types in parallel

---

#### Notebook 7: Cross-Edge-Type Summary

**File**: `notebooks/7_minimum_permutations_summary.ipynb`

**Method**:
- Aggregate N_min results from all 24 edge types (from notebook 6)
- Load graph characteristics (density, size, degree distributions)
- Analyze relationship between N_min and graph properties
- Compare data efficiency across models
- Generate deployment recommendations

**Key analyses**:
1. N_min statistics by model (mean, median, range across edge types)
2. N_min heatmap (edge types × models)
3. N_min distribution box plots
4. Correlation between N_min and graph density
5. Best model by density category (very sparse, sparse, medium, dense)
6. Model efficiency ranking

**Outputs**:
- `results/minimum_permutations_ml_summary/N_min_by_edge_type.csv`
- `results/minimum_permutations_ml_summary/model_statistics.csv`
- Multiple visualizations (heatmap, box plots, scatter plots, bar charts)

**Shell script**: MISSING - must create `scripts/07_minimum_permutations_summary.sh`

**Script template**:
```bash
#!/bin/bash
#SBATCH --job-name=min_perm_summary
#SBATCH --output=../logs/07_minimum_permutations_summary_%j.out
#SBATCH --error=../logs/07_minimum_permutations_summary_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Must run AFTER all notebook 6 jobs complete

module load anaconda
conda activate CAPP

INPUT_NOTEBOOK="notebooks/07_minimum_permutations_summary.ipynb"
OUTPUT_NOTEBOOK="notebooks/executed/07_minimum_permutations_summary_executed.ipynb"

papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK"
```

**Updates needed**:
1. Create shell script `scripts/07_minimum_permutations_summary.sh`
2. Rename notebook: `7_minimum_permutations_summary.ipynb` → `07_minimum_permutations_summary.ipynb`

---

#### Integration and Dependencies

**Dependencies**:
- Notebook 6 requires: Notebook 3 (empirical frequencies), Notebook 4 (ML models)
- Notebook 7 requires: All notebook 6 jobs completing

**Relationship to compositional validation**:
- **Completely independent** of notebooks 17-20
- Can run **in parallel** while notebook 17/17b execute
- Results inform notebook 18 (if compositional validation passes)
- Provides minimum N recommendations for ML-based approximations

**Execution order**:
```bash
# Run in parallel with notebook 17
sbatch scripts/06_minimum_permutations_analysis.sh  # Array job
# After all complete:
sbatch scripts/07_minimum_permutations_summary.sh
```

---

### Phase B1: Validate Compositional Calculation (PRIORITY 0)

**Create notebooks/17_compositional_validation.ipynb**

**Purpose**: Test whether compositional calculation accurately predicts pathway counts in held-out permutations.

**Research Question**: "Can we predict pathway nulls using E[path] = Σ P(edge1) × P(edge2)?"

**Method:**
```python
# Step 1: Use permutations 1-20 to compute edge probabilities
edge_probs_CbG = empirical_frequencies(perms_1_20, 'CbG')
edge_probs_GpPW = empirical_frequencies(perms_1_20, 'GpPW')

# Step 2: Predict pathway counts using compositional calculation (DP)
predicted_pathways = edge_probs_CbG @ edge_probs_GpPW  # Matrix mult = DP

# Step 3: Compare to ACTUAL pathway counts in held-out perms 21-30
for perm_id in range(21, 31):
    # Compute actual metapath matrix for this permutation
    actual = compute_metapath_matrix(perm_id, 'CbG', 'GpPW')

    # Compare predicted vs actual
    correlation = pearsonr(predicted_pathways.flatten(),
                          actual.flatten())
    mae = mean_absolute_error(predicted_pathways.flatten(),
                              actual.flatten())

    results.append({
        'perm_id': perm_id,
        'metapath': 'CbGpPW',
        'correlation': correlation,
        'mae': mae,
        'source': 'empirical_1_20'
    })

# Step 4: Test multiple metapaths
for metapath in ['CbGpPW', 'CtDaG', 'CrCbG', ...]:
    # Repeat validation
    ...

# Decision criteria
mean_correlation = results['correlation'].mean()
if mean_correlation > 0.95:
    print("✓ Compositional calculation is accurate!")
    print("  → Proceed with approximation methods")
elif mean_correlation > 0.85:
    print("→ Compositional calculation has bias")
    print("  → Consider correction factors")
else:
    print("✗ Compositional calculation fails")
    print("  → Must use direct empirical pathway counts")
```

**Output:**
- `results/compositional_validation/accuracy_by_metapath.csv`
- `results/compositional_validation/validation_summary.json`
- Decision: Proceed with compositional methods? Y/N

**Create scripts/17_compositional_validation.sh**

---

### Phase B2: Direct Pathway Neural Network (PRIORITY 1)

**CRITICAL**: Phase B1 (notebook 17) showed compositional calculation **FAILED** (r=0.35 vs required r>0.85). All compositional methods (analytical formulas, learned formulas, ML on edges) are invalidated.

**New Approach**: Train neural network **directly on pathway counts**, not edge probabilities.

**Create notebooks/18_pathway_neural_network.ipynb**

**Purpose**: Develop and validate direct pathway prediction model that bypasses compositional assumption.

**Research Question**: "Can a neural network learn pathway null distributions from node degrees without assuming edge independence?"

**Method:**
```python
# Architecture: PathwayNullPredictor
class PathwayNullPredictor(nn.Module):
    """
    Direct pathway count prediction from node degrees.

    NO compositional assumption - learns pathway formation directly.
    """
    def __init__(self, n_intermediate_bins=20):
        super().__init__()
        # Encode source/target degrees
        self.degree_encoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(0.2)
        )
        # Encode intermediate degree distribution
        self.intermediate_encoder = nn.Sequential(
            nn.Linear(n_intermediate_bins, 64), nn.ReLU(), nn.Dropout(0.2)
        )
        # Combine and predict pathway counts
        self.predictor = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Softplus()  # Positive counts
        )

    def forward(self, source_deg, target_deg, inter_deg_dist):
        deg_features = self.degree_encoder(
            torch.cat([source_deg, target_deg], dim=1)
        )
        inter_features = self.intermediate_encoder(inter_deg_dist)
        combined = torch.cat([deg_features, inter_features], dim=1)
        return self.predictor(combined)

# Training data generation from permutations 1-20
training_data = []

for metapath in ['CbGaD', 'CbGdD', 'CbGiGpPW', 'CbGpBP', 'CbGpCC',
                 'CbGpPW', 'CpDaG', 'CrCbG', 'CtDaG', 'CtDuG']:

    for perm_id in range(1, 21):  # Training on perms 1-20
        # Load permuted graph
        graph = load_permutation(perm_id)

        # Compute ACTUAL pathway matrix directly
        pathway_matrix = compute_metapath_matrix(graph, metapath)

        # Extract training examples
        for i, j in get_pathway_locations(pathway_matrix):
            source_deg = graph.degree(i)
            target_deg = graph.degree(j)
            inter_deg_dist = get_intermediate_degree_distribution(
                graph, metapath, n_bins=20
            )
            pathway_count = pathway_matrix[i, j]

            training_data.append({
                'source_deg': source_deg,
                'target_deg': target_deg,
                'inter_deg_dist': inter_deg_dist,
                'pathway_count': pathway_count,
                'metapath': metapath,
                'perm_id': perm_id
            })

# Train one model per metapath
models = {}
for metapath in metapaths:
    X_train, y_train = prepare_training_data(training_data, metapath)

    model = PathwayNullPredictor(n_intermediate_bins=20)
    train_pathway_nn(model, X_train, y_train, epochs=100)

    models[metapath] = model

# Validation on permutations 21-30
validation_results = []

for metapath in metapaths:
    model = models[metapath]

    for perm_id in range(21, 31):  # Held-out validation
        # Actual pathway counts
        graph = load_permutation(perm_id)
        actual_pathways = compute_metapath_matrix(graph, metapath)

        # Predicted pathway counts
        predicted_pathways = model.predict(
            source_degrees=graph.source_degrees,
            target_degrees=graph.target_degrees,
            inter_deg_dist=graph.inter_deg_dist
        )

        # Compare
        corr = pearsonr(predicted_pathways.flatten(),
                       actual_pathways.flatten())
        mae = mean_absolute_error(predicted_pathways.flatten(),
                                  actual_pathways.flatten())

        validation_results.append({
            'metapath': metapath,
            'perm_id': perm_id,
            'correlation': corr,
            'mae': mae
        })

# Decision criteria
mean_r = validation_results.groupby('metapath')['correlation'].mean()

print("\nPathway NN Validation Results:")
print("="*60)
for metapath, r in mean_r.items():
    status = "✓ PASS" if r > 0.85 else "✗ FAIL"
    print(f"{status} {metapath}: r = {r:.3f}")

overall_mean_r = mean_r.mean()
print(f"\nOverall mean r = {overall_mean_r:.3f}")

if overall_mean_r > 0.85:
    print("\n✓ PATHWAY NN VALIDATED - Proceed to Phase B3")
    decision = "VALIDATED"
elif overall_mean_r > 0.70:
    print("\n→ PATHWAY NN PARTIAL - Consider stratified sampling (Approach 4)")
    decision = "PARTIAL"
else:
    print("\n✗ PATHWAY NN FAILED - Must use direct empirical (200 perms)")
    decision = "FAILED"

# Save results
save_json({
    'overall_mean_r': overall_mean_r,
    'decision': decision,
    'by_metapath': mean_r.to_dict()
}, 'results/pathway_nn/validation_summary.json')
```

**Key Advantages**:
1. **No independence assumption** - learns correlations directly
2. **Degree-aware** - conditions on source/target/intermediate degrees
3. **Validated on real pathways** - trains on actual counts, not edge products
4. **Efficient** - 20 perms for training vs. 200 for full empirical

**Output:**
- `results/pathway_nn/trained_models/{metapath}_pathway_nn.pkl`
- `results/pathway_nn/validation_results.csv`
- `results/pathway_nn/validation_summary.json`
- Decision: VALIDATED / PARTIAL / FAILED

**Create scripts/18_pathway_neural_network.sh**

**Fallback Strategy**:
If validation r < 0.85, implement **Approach 4** (stratified sampling + NN hybrid):
- Use 30 strategically sampled permutations instead of 200
- Train NN on degree-stratified sample
- Expected improvement: r > 0.85 with 30 perms

---

### Phase B3: Variance Estimation (PRIORITY 2)

**Only execute if Phase B2 shows Pathway NN validation r > 0.85**

**Create notebooks/19_variance_estimation.ipynb**

**Purpose**: Estimate variance of pathway null distributions for z-score calculation using Pathway NN predictions.

**Method:**
```python
# Load trained Pathway NN models from Phase B2
pathway_nn_models = {}
for metapath in metapaths:
    pathway_nn_models[metapath] = load_model(
        f'results/pathway_nn/trained_models/{metapath}_pathway_nn.pkl'
    )

# Two sources of variance:
# 1. Empirical variance from held-out permutations 21-30
# 2. Model uncertainty via bootstrap

variance_estimates = {}

for metapath in metapaths:
    model = pathway_nn_models[metapath]

    # SOURCE 1: Empirical variance from permutations 21-30
    pathway_samples_empirical = []
    for perm_id in range(21, 31):
        # Compute actual pathway counts in this permutation
        graph = load_permutation(perm_id)
        pathways = compute_metapath_matrix(graph, metapath)
        pathway_samples_empirical.append(pathways)

    # Stack and compute variance
    pathway_samples_empirical = np.stack(pathway_samples_empirical)  # (10, n_sources, n_targets)
    empirical_variance = np.var(pathway_samples_empirical, axis=0)

    # SOURCE 2: Model uncertainty via bootstrap
    # Generate predictions with input perturbation
    hetionet = load_hetionet()
    source_degrees = get_source_degrees(hetionet, metapath)
    target_degrees = get_target_degrees(hetionet, metapath)
    inter_deg_dist = get_intermediate_degree_distribution(hetionet, metapath)

    bootstrap_predictions = []
    for _ in range(1000):  # Bootstrap samples
        # Add small noise to degree inputs to estimate uncertainty
        noisy_source_deg = source_degrees + np.random.normal(0, 0.1, source_degrees.shape)
        noisy_target_deg = target_degrees + np.random.normal(0, 0.1, target_degrees.shape)

        # Predict with noisy inputs
        bootstrap_pred = model.predict(
            noisy_source_deg, noisy_target_deg, inter_deg_dist
        )
        bootstrap_predictions.append(bootstrap_pred)

    # Compute model uncertainty
    bootstrap_predictions = np.stack(bootstrap_predictions)
    model_variance = np.var(bootstrap_predictions, axis=0)

    # COMBINED VARIANCE: Empirical + Model Uncertainty
    total_variance = empirical_variance + model_variance

    variance_estimates[metapath] = {
        'empirical_variance': empirical_variance,
        'model_variance': model_variance,
        'total_variance': total_variance
    }

    # Save variance matrices
    np.savez_compressed(
        f'results/variance_estimates/{metapath}_variance.npz',
        empirical=empirical_variance,
        model=model_variance,
        total=total_variance
    )

# Adaptive degree binning for sparse variance estimates
# If specific (source, target) pair has no observations in perms 21-30,
# use pooled variance from degree bin

from src.degree_binning import create_adaptive_variance_bins

for metapath in metapaths:
    # Get all degrees observed
    source_degrees = get_source_degrees(metapath)
    target_degrees = get_target_degrees(metapath)

    # Create adaptive bins (edge-type specific!)
    source_bins = create_adaptive_variance_bins(source_degrees, min_samples=30)
    target_bins = create_adaptive_variance_bins(target_degrees, min_samples=30)

    # Compute pooled variance within bins
    for src_bin in source_bins:
        for tgt_bin in target_bins:
            pairs_in_bin = get_pairs_in_bin(src_bin, tgt_bin)

            # Use total variance (empirical + model)
            pooled_var = np.mean(variance_estimates[metapath]['total_variance'][pairs_in_bin])

            variance_lookup[(metapath, src_bin, tgt_bin)] = pooled_var

    # Save lookup table
    save_pickle(variance_lookup, f'results/variance_estimates/{metapath}_variance_lookup.pkl')
```

**Output:**
- `results/variance_estimates/{metapath}_variance.npz` (empirical, model, total variance)
- `results/variance_estimates/{metapath}_variance_lookup.pkl` (degree bin pooling)

**Create scripts/19_variance_estimation.sh**

---

### Phase B4: Anomaly Detection (PRIORITY 3)

**Only execute if Phase B2 and B3 complete successfully**

**Create notebooks/20_anomaly_detection.ipynb**

**Purpose**: Identify metapaths with biological signal beyond degree structure using z-scores from Pathway NN null distributions.

**Method:**
```python
# Load trained Pathway NN models from Phase B2
pathway_nn_models = {}
for metapath in metapaths:
    pathway_nn_models[metapath] = load_model(
        f'results/pathway_nn/trained_models/{metapath}_pathway_nn.pkl'
    )

# Load variance estimates from Phase B3
variance_estimates = {}
for metapath in metapaths:
    var_data = np.load(f'results/variance_estimates/{metapath}_variance.npz')
    variance_estimates[metapath] = var_data['total']  # Use total variance (empirical + model)

# Compute OBSERVED pathway counts in Hetionet
hetionet = load_hetionet()
observed = {}
for metapath in metapaths:
    observed[metapath] = compute_metapath_matrix(hetionet, metapath)

# Compute EXPECTED pathway counts using Pathway NN
expected = {}
for metapath in metapaths:
    model = pathway_nn_models[metapath]

    # Get node degrees from Hetionet
    source_degrees = get_source_degrees(hetionet, metapath)
    target_degrees = get_target_degrees(hetionet, metapath)
    inter_deg_dist = get_intermediate_degree_distribution(hetionet, metapath)

    # Predict expected pathway counts
    expected[metapath] = model.predict(
        source_degrees, target_degrees, inter_deg_dist
    )

# Compute z-scores
results = []
for metapath in metapaths:
    obs = observed[metapath]
    exp = expected[metapath]
    var = variance_estimates[metapath]

    # Element-wise z-score
    # z = (observed - expected) / sqrt(variance)
    z = (obs - exp) / np.sqrt(var + 1e-10)  # Add small constant to avoid division by zero

    # Convert to DataFrame
    sources, targets = np.nonzero(obs + exp)  # Include both observed and predicted locations
    for i, j in zip(sources, targets):
        results.append({
            'metapath': metapath,
            'source_id': i,
            'target_id': j,
            'source_degree': get_node_degree(hetionet, metapath, 'source', i),
            'target_degree': get_node_degree(hetionet, metapath, 'target', j),
            'observed': obs[i, j],
            'expected': exp[i, j],
            'variance': var[i, j],
            'z_score': z[i, j],
            'p_value': 2 * (1 - norm.cdf(abs(z[i, j])))
        })

results_df = pd.DataFrame(results)

# FDR correction (Benjamini-Hochberg)
from statsmodels.stats.multitest import multipletests
results_df['p_adj_BH'] = multipletests(results_df['p_value'], method='fdr_bh')[1]

# Identify significant anomalies
significant = results_df[results_df['p_adj_BH'] < 0.05]
significant = significant.sort_values('z_score', key=abs, ascending=False)

# Save all results
results_df.to_csv('results/anomaly_detection/all_z_scores.csv', index=False)
significant.to_csv('results/anomaly_detection/significant_FDR05.csv', index=False)

# Display top anomalies
print(f"\nMost Anomalous Metapaths (FDR < 0.05):")
print(f"{'='*80}")
print(f"Total significant: {len(significant):,} / {len(results_df):,} ({100*len(significant)/len(results_df):.2f}%)")
print(f"\nTop 20 Enriched:")
enriched = significant[significant['z_score'] > 0].head(20)
for _, row in enriched.iterrows():
    print(f"z={row['z_score']:+6.2f} (FDR={row['p_adj_BH']:.2e}): "
          f"{row['metapath']} | obs={row['observed']:.0f} exp={row['expected']:.1f}")

print(f"\nTop 20 Depleted:")
depleted = significant[significant['z_score'] < 0].head(20)
for _, row in depleted.iterrows():
    print(f"z={row['z_score']:+6.2f} (FDR={row['p_adj_BH']:.2e}): "
          f"{row['metapath']} | obs={row['observed']:.0f} exp={row['expected']:.1f}")

# Visualizations
import matplotlib.pyplot as plt

# Volcano plot
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(results_df['z_score'], -np.log10(results_df['p_adj_BH']),
                     c=results_df['metapath'].astype('category').cat.codes,
                     alpha=0.5, s=10)
ax.axhline(-np.log10(0.05), color='red', linestyle='--', label='FDR=0.05')
ax.set_xlabel('Z-score')
ax.set_ylabel('-log10(FDR-adjusted p-value)')
ax.set_title('Pathway Anomaly Detection - Volcano Plot')
ax.legend()
plt.savefig('results/anomaly_detection/volcano_plot.png', dpi=300, bbox_inches='tight')

# Degree stratification of anomalies
degree_bins = pd.cut(results_df['source_degree'], bins=10)
degree_summary = results_df.groupby(degree_bins).agg({
    'z_score': ['mean', 'std', 'count'],
    'p_adj_BH': lambda x: (x < 0.05).sum()
})
print("\nAnomalies by Source Degree Bin:")
print(degree_summary)
```

**Output:**
- `results/anomaly_detection/all_z_scores.csv` (all pathway z-scores)
- `results/anomaly_detection/significant_FDR05.csv` (FDR-corrected significant anomalies)
- `results/anomaly_detection/volcano_plot.png`
- `results/anomaly_detection/degree_stratification.png`

**Create scripts/20_anomaly_detection.sh**

---

### Phase B5: Update Existing Notebooks (PRIORITY 4)

**Only after Phase B2 validates Pathway NN (r > 0.85)**

#### Update Notebook 15: Metapath Null Distributions
**Modify notebooks/15_metapath_null_distributions.ipynb**

**Current Issue**: Uses compositional calculation (ML models → edge probs → multiply)

**Required Changes**:
```python
# BEFORE (compositional - INVALIDATED):
def compute_metapath_null_2edge(source_degrees, target_degrees,
                                edge1_type, edge2_type, model_type='rf'):
    edge1_models = load_null_models(edge1_type, model_type)
    edge2_models = load_null_models(edge2_type, model_type)

    for source_deg, target_deg in zip(source_degrees, target_degrees):
        total_prob = 0.0
        for inter_deg, freq in intermediate_degree_freq.items():
            p1 = predict_edge_probability(source_deg, inter_deg, edge1_models)
            p2 = predict_edge_probability(inter_deg, target_deg, edge2_models)
            total_prob += p1 * p2 * freq  # COMPOSITIONAL MULTIPLICATION

# AFTER (Pathway NN - VALIDATED):
def compute_metapath_null_pathway_nn(source_degrees, target_degrees,
                                     metapath):
    # Load trained Pathway NN
    pathway_nn = load_model(f'results/pathway_nn/trained_models/{metapath}_pathway_nn.pkl')

    # Get intermediate degree distribution
    inter_deg_dist = get_intermediate_degree_distribution(hetionet, metapath)

    # Direct pathway prediction (no compositional assumption)
    pathway_counts = pathway_nn.predict(source_degrees, target_degrees, inter_deg_dist)

    return pathway_counts
```

**Status**: BLOCKED until Phase B2 completes

#### Update Notebook 16: Dynamic Programming DWPC
**Modify notebooks/16_dynamic_programming_dwpc.ipynb**

**Current Issue**: Uses analytical DP formula assuming independence

**Required Changes**:
```python
# BEFORE (analytical DP - assumes independence):
def compute_exact_expectation(self, source_idx, target_idx, edge_data):
    """
    E[DWPC_{s→t}] = deg_s^{1-w} * deg_t^{1-w} / (E1 * E2) * S
    where S = Σ_m deg1_m^{1-w} * deg2_m^{1-w}

    ASSUMES INDEPENDENCE between edges!
    """

# AFTER (Pathway NN - learns dependencies):
def compute_exact_expectation_pathway_nn(self, source_idx, target_idx, metapath):
    """
    Use trained Pathway NN to compute expected pathway counts.

    NO independence assumption - NN captures edge correlations.
    """
    pathway_nn = load_model(f'results/pathway_nn/trained_models/{metapath}_pathway_nn.pkl')

    source_deg = self.graph.degree(source_idx)
    target_deg = self.graph.degree(target_idx)
    inter_deg_dist = get_intermediate_degree_distribution(self.graph, metapath)

    expected_pathways = pathway_nn.predict(
        np.array([source_deg]),
        np.array([target_deg]),
        inter_deg_dist
    )

    return expected_pathways[0]
```

**Status**: BLOCKED until Phase B2 completes

#### Fix Notebook 14: Compositional Null Validation
**Status**: Already executed. Showed compositional FAILED (r=0.35). No fixes needed - results documented.

#### Review Notebook 04: Model Testing
**Status**: Keep as-is. Notebook 04 trains ML models on **edge probabilities**, which is still valid for edge-level analysis. Only pathway-level calculations need Pathway NN.

---

### Phase B6: Future Optimizations (DEFERRED)

**Only pursue if Phase B2 Pathway NN validates successfully (r > 0.85)**

These optimizations are **tabled** until Pathway NN approach is validated. If Pathway NN works, we can explore efficiency improvements.

#### Optimization 1: Stratified Sampling (Approach 4)

**Goal**: Reduce training permutations from 20 → 30 total while maintaining accuracy

**Method**:
- Sample 30 permutations strategically:
  - 10 random
  - 10 enriched for low-degree nodes
  - 10 enriched for high-degree nodes
- Train Pathway NN on stratified sample
- Validate that r > 0.85 still holds

**Expected Benefits**:
- Reduce computational cost by 33%
- Maintain or improve accuracy through better degree coverage

#### Optimization 2: Minimum Permutations Analysis for Pathway NN

**Goal**: Determine minimum N permutations needed for Pathway NN to converge

**Method** (analogous to notebooks 6-7 for edge models):
- Train Pathway NN with N = 5, 10, 15, 20, 25, 30 permutations
- Validate against held-out perms
- Find minimum N where r > 0.85

**Expected Benefits**:
- Further reduce computational requirements
- Inform optimal training set size

**Status**: DEFERRED - pursue only after Phase B2 validation

---

## Part B Summary: New Notebook Architecture

**REVISED: Compositional Failed → Pathway Neural Network Approach**

### Current Status (as of notebook 17 execution):
- ✅ **Notebook 17**: Compositional validation **FAILED** (r=0.35 vs required r>0.85)
- ❌ **Compositional methods invalidated**: Analytical formulas, learned formulas, ML on edges
- 🔄 **Pivot to Pathway NN**: Direct pathway modeling without independence assumption

### New Pipeline Architecture:

| Notebook | Purpose | Depends On | Status | Validates |
|----------|---------|------------|--------|-----------|
| **17** | Compositional validation | 3, perms 21-30 | ✅ **COMPLETED** | Compositional fails (r=0.35) |
| **17b** | Failure analysis | 17 | ⏳ Running | Why compositional fails |
| **18** | **Pathway Neural Network** | 17, perms 1-20, 21-30 | 📋 **TO CREATE** | Can NN learn pathways directly? (target: r>0.85) |
| **19** | Variance estimation (NN-based) | 18 validates | 📋 TO CREATE | Compute variance from NN + empirical |
| **20** | Anomaly detection (NN-based) | 18, 19 | 📋 TO CREATE | Final z-score pipeline with Pathway NN |

### Decision Gates:

**Gate 1: Compositional Validation (Notebook 17)**
- ❌ **FAILED** (r=0.35) → Cannot use compositional methods
- ✅ Proceed to **Pathway NN approach** (notebook 18)

**Gate 2: Pathway NN Validation (Notebook 18)**
- ✅ If r > 0.85 → Proceed to notebooks 19-20, update notebooks 15-16
- 🔄 If 0.70 < r < 0.85 → Implement Approach 4 (stratified sampling)
- ❌ If r < 0.70 → Must use direct empirical (compute all 200 perms, expensive)

**Gate 3: Deployment (Notebook 20)**
- Use validated Pathway NN for anomaly detection
- Future: Explore optimizations (Phase B6)

### Updated Dependencies:

**Notebooks 15-16: BLOCKED until Pathway NN validates**
- Both currently use compositional calculation (invalidated)
- Must be rewritten to use Pathway NN from notebook 18
- Cannot proceed until notebook 18 completes and validates

**Archive Strategy:**
- Keep executed notebooks showing compositional failure (17, 17b) for documentation
- Archive compositional-based versions of notebooks 15-16 when rewritten
- Maintain clear history of methodological pivot

## Part C: Integration and Testing

### Phase C1: Update All Notebooks with Consistent Structure

**Standard cell structure for all pipeline notebooks:**

1. **Metadata cell** (markdown):
   ```markdown
   # Notebook XX: Title

   ## Purpose
   Brief description

   ## Inputs
   - data/...
   - results/...

   ## Outputs
   - results/...

   ## Dependencies
   - Notebook YY must be run first
   ```

2. **Papermill parameters cell**:
   ```python
   # Papermill parameters
   edge_type = "CtD"
   random_seed = 42
   ```

3. **Imports and setup**:
   ```python
   import sys
   from pathlib import Path

   repo_dir = Path.cwd().parent
   sys.path.append(str(repo_dir / 'src'))

   from degree_binning import get_degree_bins
   ```

### Phase C2: Create Master Pipeline Script

**Create scripts/utils/run_full_pipeline.sh:**
```bash
#!/bin/bash
# Run complete pipeline with dependency management

set -e  # Exit on error

echo "Starting full pipeline..."

# Phase 1: Setup
bash scripts/00_create_hetmat.sh
bash scripts/01_create_permutations.sh
bash scripts/02_download_null_graphs.sh

# Phase 2: Edge frequency analysis
sbatch --wait scripts/03_edge_frequency_analysis.sh

# Phase 3: Model training
sbatch --wait scripts/04_model_comparison_analysis.sh
bash scripts/05_model_testing_summary.sh

# Phase 4: Minimum permutations
sbatch --wait scripts/06_minimum_permutations_analysis.sh
bash scripts/07_minimum_permutations_summary.sh

# Phase 5: Learned formulas
sbatch --wait scripts/08_learned_analytical.sh
bash scripts/09_learned_analytical_summary.sh

# Phase 6: Null models and validation
sbatch --wait scripts/13_null_model_training.sh
sbatch --wait scripts/17_variance_estimation.sh
bash scripts/14_fast_compositional_null.sh

# Phase 7: Metapath analysis
sbatch --wait scripts/15_metapath_null_distributions.sh
bash scripts/16_dynamic_programming_dwpc.sh

# Phase 8: Anomaly detection
bash scripts/18_anomaly_detection.sh

echo "Pipeline complete!"
```

### Phase C3: Testing Protocol

**Test each phase sequentially:**

1. **Test restructuring:**
   ```bash
   # Verify notebooks renamed
   ls notebooks/0*.ipynb

   # Verify exploratory moved
   ls notebooks/exploratory/compositionality/

   # Verify helpers moved
   python -c "from src.helpers.fix_analytical_correlation import *"
   ```

2. **Test notebook 17 (variance estimation):**
   ```bash
   papermill notebooks/17_variance_estimation.ipynb \
     notebooks/executed/17_variance_estimation_CtD_executed.ipynb \
     -p edge_type "CtD"

   # Verify output exists
   ls results/variance_estimates/CtD_variance.pkl
   ```

3. **Test notebook 14 (fixed validation):**
   ```bash
   papermill notebooks/14_fast_compositional_null.ipynb \
     notebooks/executed/14_fast_compositional_null_executed.ipynb \
     -p metapath "CbGpPW"

   # Verify no NaN correlations
   grep "correlation" notebooks/executed/14_*.ipynb
   ```

4. **Test notebook 18 (anomaly detection):**
   ```bash
   papermill notebooks/18_anomaly_detection.ipynb \
     notebooks/executed/18_anomaly_detection_executed.ipynb

   # Verify z-scores computed
   head results/anomaly_detection/significant_anomalies_FDR05.csv
   ```

### Phase C4: Update Documentation

**Update CLAUDE.md with:**
1. New pipeline structure (00-18)
2. Exploratory analyses organization
3. Key concept: anomaly detection via z-scores
4. Updated workflow examples

**Update scripts/README.md with:**
1. New script numbering
2. Dependency graph
3. Full pipeline execution instructions

---

## Summary

### Files Created/Updated

**COMPLETED:**
1. ✅ notebooks/17_compositional_validation.ipynb (executed, showed compositional FAILED)
2. ✅ scripts/17_compositional_validation.sh
3. ✅ notebooks/17b_compositional_failure_analysis.ipynb (created, running)
4. ✅ scripts/17b_compositional_failure_analysis.sh

**TO CREATE (Pathway NN Approach):**
5. 📋 notebooks/18_pathway_neural_network.ipynb (direct pathway modeling)
6. 📋 scripts/18_pathway_neural_network.sh
7. 📋 notebooks/19_variance_estimation.ipynb (NN-based variance)
8. 📋 scripts/19_variance_estimation.sh
9. 📋 notebooks/20_anomaly_detection.ipynb (NN-based z-scores)
10. 📋 scripts/20_anomaly_detection.sh
11. 📋 src/pathway_nn.py (PathwayNullPredictor class and training utilities)
12. 📋 src/degree_binning.py (adaptive binning utilities)
13. 📋 notebooks/exploratory/{compositionality,formula_testing,degree_testing}/README.md (3 files)

**TO UPDATE (After Pathway NN validates):**
14. 📋 notebooks/15_metapath_null_distributions.ipynb (replace compositional with Pathway NN)
15. 📋 notebooks/16_dynamic_programming_dwpc.ipynb (replace analytical DP with Pathway NN)

### Files Renamed (Part A - Deferred)
- 16 notebooks (00-09, 13-16 with zero-padding)
- 10 scripts (matching notebooks)
- Best versions: 04, 05, 08, 14

### Files Moved (Part A - Deferred)
- 8 exploratory notebooks → notebooks/exploratory/
- 5 helper scripts → src/helpers/
- 2 test files → tests/
- 3 utility scripts → scripts/utils/

### Files Deleted (Part A - Deferred)
- 7 duplicate/old notebooks
- 5 obsolete scripts

### Files Modified
- ✅ IMPLEMENTATION_PLAN.md (this file - added Pathway NN approach)
- 📋 notebooks/05_model_testing_summary.ipynb (already fixed .sparse.npz loading)
- 📋 CLAUDE.md (update with Pathway NN methodology)
- 📋 scripts/README.md (update with new pipeline)

### Execution Order (REVISED)

**IMMEDIATE PRIORITY - Part B (Pathway NN Development):**
1. ⏳ **Wait for notebook 17b** to complete (understand why compositional fails)
2. 📋 **Phase B2**: Create notebook 18 (Pathway NN) - ~2-3 days development + 6-8 hours training/validation
3. 📋 **Phase B3**: Create notebook 19 (variance estimation) - ~1 day
4. 📋 **Phase B4**: Create notebook 20 (anomaly detection) - ~1 day
5. 📋 **Phase B5**: Update notebooks 15-16 - ~2 days

**DEFERRED - Part A (Restructuring):**
- Phases A1-A7: File organization and documentation (~2 hours)
- Execute AFTER scientific methodology is validated

**DEFERRED - Part B6 (Optimizations):**
- Approach 4 (stratified sampling)
- Minimum permutations analysis for Pathway NN
- Execute ONLY IF Pathway NN validates successfully

**DEFERRED - Part C (Integration):**
- Phases C1-C4: Testing and final updates (~2 hours)
- Execute AFTER Part B completes

**Total estimated time: ~1-2 weeks for Pathway NN development and validation**

### Benefits of Pathway NN Approach

**Scientific:**
- ✅ No compositional assumption required (avoids r=0.35 failure)
- ✅ Learns edge correlations directly from data
- ✅ Degree-aware pathway modeling
- ✅ Validated on real pathway counts, not edge products

**Computational:**
- ✅ Efficient: 20 training perms vs. 200 full empirical
- ✅ Scalable: Can reduce to 30 perms with stratified sampling (Phase B6)
- ✅ Flexible: Can handle any metapath length

**Implementation:**
- ✅ Clean integration with existing pipeline
- ✅ Greene Lab standards throughout
- ✅ HPC-ready with proper resource allocation
- ✅ Clear validation methodology (r > 0.85 threshold)

**Achieves scientific goal:** Identify biological signals beyond degree structure using validated null distributions

### Next Steps

1. **Review notebook 17b results** when complete (ETA: 3-7 hours from now)
2. **Create src/pathway_nn.py** with PathwayNullPredictor architecture
3. **Create notebook 18** to train and validate Pathway NN
4. **Decision point**: If Pathway NN validates (r > 0.85), proceed to notebooks 19-20
5. **Update notebooks 15-16** to use Pathway NN instead of compositional
6. **Execute Part A** (restructuring) once scientific pipeline is stable