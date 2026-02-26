# Implementation Status: Degree Signature Pipeline

**Last Updated:** 2025-10-13
**Status:** Foundation Complete, Ready for Component Implementation

## Summary

A comprehensive plan for fair benchmarking of 5 pathway count prediction models using degree signatures has been implemented. The master orchestration script is ready to submit the entire pipeline to Alpine HPC once individual component scripts and notebooks are created.

---

## ✅ Completed Components

### 1. Updated Script: `scripts/17b_compositional_failure_analysis.sh`
**Purpose:** Baseline compositional failure analysis (for comparison)

**Features:**
- Standardized SLURM directives (4 cores, 32GB, 6 hours)
- **Robust CSV cleanup:** Deletes ALL corrupted CSV files before running
- Simplified Alpine HPC conventions
- **Fixes the CSV corruption issue immediately!**

**Usage:**
```bash
cd /projects/$USER/repositories/Context-Aware-Path-Probability/scripts
sbatch 17b_compositional_failure_analysis.sh
```

### 2. Created Module: `src/benchmarking.py`
**Purpose:** Fair model comparison utilities

**Features:**
- `BenchmarkResult` dataclass for standardized results
- `Timer` context manager (excludes I/O from timing)
- `MemoryTracker` for peak RSS monitoring
- `ModelBenchmarker` class for comprehensive benchmarking
- `compute_normalized_cost()` for fair comparison across models
- All models benchmarked with identical methodology

**Usage Example:**
```python
from src.benchmarking import ModelBenchmarker

benchmarker = ModelBenchmarker(
    model_name='Random_Forest',
    metapath='CbGpPW',
    n_training_samples=800
)

# Training (timed automatically)
with benchmarker.time_training():
    model.fit(X_train, y_train)

# Prediction (timed automatically)
with benchmarker.time_prediction():
    predictions = model.predict(X_test)

# Record validation metrics
benchmarker.record_validation(predictions, y_test)

# Save results
result = benchmarker.finalize()
result.save_json('results/pathway_nn/benchmarks/CbGpPW_Random_Forest_benchmark.json')
```

### 3. Created Script: `scripts/18_submit_all.sh`
**Purpose:** Master orchestration script with fair benchmarking

**Features:**
- Complete pipeline submission with SLURM dependencies
- 5 separate training array jobs (one per model) for fair comparison
- Standardized resources across all tasks
- Comprehensive monitoring and logging
- Decision tree for conditional Tier 2 validation
- Detailed resource budget tracking

**Pipeline Structure:**
```
17b Baseline (6h)
  ↓
18a Data Prep (8 tasks × 6h = 48 node-hours @ 64GB)
  ↓
18b-f Training (5 parallel arrays)
  - 18b Random (8 tasks × 0.5h = 2 node-hours @ 16GB)
  - 18c Degree Product (8 tasks × 0.5h = 2 node-hours @ 16GB)
  - 18d NegBin GLM (8 tasks × 1h = 4 node-hours @ 16GB)
  - 18e Random Forest (8 tasks × 2h = 16 node-hours @ 32GB)
  - 18f Degree Sig NN (8 tasks × 2h = 16 node-hours @ 32GB)
  ↓
18g CV Validation (8 tasks × 1h = 8 node-hours @ 16GB)
  ↓
18i Benchmark Summary (0.5h @ 16GB)

TOTAL: 102.5 node-hours
Wall clock: ~15-17 hours (phases run in parallel)
```

**Usage:**
```bash
cd /projects/$USER/repositories/Context-Aware-Path-Probability/scripts
bash 18_submit_all.sh
```

---

## 📋 Pending Components

### Phase 1: Core Modules (High Priority)

#### A. `src/intermediate_signatures.py`
**Purpose:** Precompute intermediate node degree signatures

**Required Functions:**
```python
def precompute_intermediate_signatures_fast(
    edge1, edge2,
    source_degrees, target_degrees,
    n_source_bins=10, n_target_bins=10, n_inter_bins=10,
    samples_per_bin=100, random_state=42
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute joint (in_degree, out_degree) histogram for intermediate nodes.
    Returns: signature_table[(src_bin, tgt_bin)] → 100-dim array
    """
    pass

def create_quantile_bins(degrees, n_bins):
    """Create quantile-based degree bins."""
    pass

def assign_to_bins(degrees, bin_edges):
    """Assign degrees to bins."""
    pass
```

**Design Notes:**
- Vectorized operations where possible
- Progress reporting for SLURM logs
- ~2 hours computation per metapath on HPC

#### B. `src/models/` Module Structure
**Purpose:** Separate implementation for each model

**Required Files:**
```
src/models/
├── __init__.py
├── base.py                      # Base model interface
├── random_baseline.py           # Model 1: Random sampling
├── degree_product.py            # Model 2: Degree product baseline
├── negbin_glm.py                # Model 3: Negative binomial GLM
├── random_forest.py             # Model 4: Random forest
└── degree_signature_nn.py       # Model 5: Neural network
```

**Base Interface:**
```python
class BaseModel:
    def fit(self, training_data: pd.DataFrame):
        """Train on degree-binned data."""
        raise NotImplementedError

    def predict(self, source_deg, target_deg, inter_signature=None):
        """Predict pathway count."""
        raise NotImplementedError

    def save(self, filepath):
        """Save trained model."""
        raise NotImplementedError

    @classmethod
    def load(cls, filepath):
        """Load trained model."""
        raise NotImplementedError
```

### Phase 2: Notebooks (Medium Priority)

#### A. `notebooks/18a_data_preparation.ipynb`
**Purpose:** Generate training data from original graph

**Parameters (papermill):**
- `metapath` (e.g., 'CbGpPW')
- `edge1_type` (e.g., 'CbG')
- `edge2_type` (e.g., 'GpPW')
- `n_degree_bins = 10`
- `n_inter_bins = 10`
- `samples_per_bin = 100`

**Outputs:**
- `results/pathway_nn/training_data/{metapath}_training_data.csv` (~100 rows)

**Key Steps:**
1. Load original graph edges (not permutations!)
2. Precompute intermediate signatures
3. Compute pathway counts per degree bin
4. Aggregate statistics (mean, std, quantiles)
5. Save CSV with columns: `source_deg_bin`, `target_deg_bin`, `inter_sig_0`...`inter_sig_99`, `pathway_count_mean`, `pathway_count_std`, etc.

#### B. `notebooks/18b-18f_train_*.ipynb` (5 notebooks, one per model)
**Purpose:** Train individual models

**Parameters (papermill):**
- `metapath` (e.g., 'CbGpPW')
- `record_benchmarks = true`

**Outputs:**
- `results/pathway_nn/trained_models/{metapath}_{model_name}.pkl`
- `results/pathway_nn/benchmarks/{metapath}_{model_name}_benchmark.json`

**Template Structure:**
```python
# Load training data
training_data = pd.read_csv(f'results/pathway_nn/training_data/{metapath}_training_data.csv')

# Initialize benchmarker
benchmarker = ModelBenchmarker(
    model_name='Random_Forest',
    metapath=metapath,
    n_training_samples=len(training_data)
)

# Train with timing
with benchmarker.time_training():
    model = RandomForestModel()
    model.fit(training_data)

# Validation (load test data)
test_data = ...  # Load from validation set

with benchmarker.time_prediction():
    predictions = model.predict(test_data)

# Record metrics
benchmarker.record_validation(predictions, test_data['pathway_count_mean'])

# Save
model.save(f'results/pathway_nn/trained_models/{metapath}_Random_Forest.pkl')
result = benchmarker.finalize()
result.save_json(f'results/pathway_nn/benchmarks/{metapath}_Random_Forest_benchmark.json')
```

#### C. `notebooks/18g_validate_cv.ipynb`
**Purpose:** K-fold CV validation on original graph (Tier 1)

**Parameters:**
- `metapath`
- `n_folds = 5`

**Outputs:**
- `results/pathway_nn/validation/cv_results_{metapath}.csv`

**Key Steps:**
1. Load all 5 trained models for this metapath
2. Load training data
3. For each fold:
   - Split by degree bins (stratified)
   - Test each model on held-out bins
   - Record correlation, MAE, RMSE
4. Aggregate results across folds
5. Save per-model CV scores

#### D. `notebooks/18i_benchmark_summary.ipynb`
**Purpose:** Aggregate all results and generate figures

**Outputs:**
- `results/pathway_nn/benchmarks/comparison_table.csv`
- `results/pathway_nn/figures/efficiency_frontier.png`
- `results/pathway_nn/figures/model_comparison_heatmap.png`
- `results/pathway_nn/figures/cv_performance_by_metapath.png`

**Key Analyses:**
1. Load all benchmark JSON files
2. Compute normalized costs
3. Compute efficiency (r / cost)
4. Identify Pareto frontier
5. Generate comparison table
6. Create efficiency frontier plot
7. Identify best model per metapath

### Phase 3: SLURM Scripts (Medium Priority)

All scripts follow this template:

```bash
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4                  # STANDARD
#SBATCH --mem={memory}GB            # 16/32/64 by task
#SBATCH --time={time}:00:00         # 0.5-6 hours
#SBATCH --array=1-8                 # 8 metapaths
#SBATCH --output=logs/{name}_%A_%a.out
#SBATCH --error=logs/{name}_%A_%a.err

# Load environment
module purge
module load anaconda
conda activate CAPP

# Metapath mapping
METAPATHS=(CbGpPW CtDaG CbGaD CrCbG CbGiG CpDaG CbGpBP CbGpCC)
METAPATH=${METAPATHS[$SLURM_ARRAY_TASK_ID-1]}

# Edge type mapping (for data prep)
declare -A EDGE1_MAP=(...)
declare -A EDGE2_MAP=(...)

# Set paths
REPO_DIR=/projects/$USER/repositories/Context-Aware-Path-Probability
cd $REPO_DIR

# Run notebook
papermill \
    notebooks/{notebook}.ipynb \
    notebooks/executed/{notebook}_${METAPATH}_executed.ipynb \
    -p metapath "$METAPATH" \
    --log-output

EXIT_CODE=$?
echo "Job completed at $(date)"
exit $EXIT_CODE
```

**Required Scripts:**
- `18a_data_preparation.sh` (64GB, 6h)
- `18b_train_random.sh` (16GB, 0.5h)
- `18c_train_degree_product.sh` (16GB, 0.5h)
- `18d_train_negbin_glm.sh` (16GB, 1h)
- `18e_train_random_forest.sh` (32GB, 2h)
- `18f_train_degree_signature_nn.sh` (32GB, 2h)
- `18g_validate_cv_all_models.sh` (16GB, 1h)
- `18i_benchmark_summary.sh` (16GB, 0.5h, single job not array)

---

## 🎯 Implementation Roadmap

### Week 1: Core Infrastructure
- [x] Day 1: Fix 17b script (COMPLETE)
- [x] Day 1: Create benchmarking module (COMPLETE)
- [x] Day 1: Create master submission script (COMPLETE)
- [ ] Day 2: Create `src/intermediate_signatures.py`
- [ ] Day 3: Create `src/models/` structure with base class
- [ ] Day 4: Implement Models 1-2 (Random, Degree Product - simple)
- [ ] Day 5: Test locally on one metapath

### Week 2: Model Implementation
- [ ] Day 6: Implement Model 3 (NegBin GLM)
- [ ] Day 7: Implement Model 4 (Random Forest)
- [ ] Day 8: Implement Model 5 (Degree Signature NN)
- [ ] Day 9: Create notebook 18a (data prep) + script
- [ ] Day 10: Test data prep locally

### Week 3: Training Pipeline
- [ ] Day 11: Create notebooks 18b-18f (training) + scripts
- [ ] Day 12: Create notebook 18g (CV validation) + script
- [ ] Day 13: Create notebook 18i (summary) + script
- [ ] Day 14: Local testing of full pipeline (1 metapath)
- [ ] Day 15: Submit to Alpine HPC

### Week 4: Validation & Analysis
- [ ] Day 16-17: Monitor HPC execution
- [ ] Day 18: Analyze results, adjust if needed
- [ ] Day 19: Create Tier 2 validation (if needed)
- [ ] Day 20: Generate final figures and tables

---

## 🚀 Quick Start (Once Components Ready)

### On Alpine HPC:

```bash
# Navigate to repo
cd /projects/$USER/repositories/Context-Aware-Path-Probability/scripts

# Submit entire pipeline
bash 18_submit_all.sh

# Monitor progress
squeue -u $USER

# Check results (after ~15 hours)
cat ../results/pathway_nn/validation/cv_results.csv
cat ../results/pathway_nn/benchmarks/comparison_table.csv

# View figures
open ../results/pathway_nn/figures/efficiency_frontier.png
```

### Local Testing:

```bash
# Test individual notebook
cd notebooks
papermill 18a_data_preparation.ipynb \
    test_output.ipynb \
    -p metapath "CbGpPW" \
    -p edge1_type "CbG" \
    -p edge2_type "GpPW"

# Test benchmarking module
python -c "from src.benchmarking import ModelBenchmarker; print('OK')"
```

---

## 📊 Expected Results

### Training Time (per metapath)

| Model | Time | Memory | Cost (node-hrs) |
|-------|------|--------|-----------------|
| Random | 30s | 4GB | 0.01 |
| Degree Product | 1min | 4GB | 0.02 |
| NegBin GLM | 5min | 8GB | 0.05 |
| Random Forest | 1h | 28GB | 0.88 |
| Degree Sig NN | 1.5h | 30GB | 1.41 |

### Validation Accuracy (expected)

| Model | Expected r | Range |
|-------|------------|-------|
| Random | 0.15 | 0.10-0.20 |
| Degree Product | 0.60 | 0.55-0.65 |
| NegBin GLM | 0.72 | 0.68-0.76 |
| Random Forest | 0.81 | 0.77-0.85 |
| Degree Sig NN | 0.85 | 0.82-0.88 |

### Decision Criteria

- **If Degree Sig NN achieves r > 0.85**: ✅ Use for anomaly detection
- **If Random Forest achieves r > 0.80**: Consider as faster alternative
- **If all models r < 0.75**: Need Tier 2 validation or re-evaluate approach

---

## 🔍 Key Design Decisions

### 1. Always Use Array Format
**Rationale:** Fair comparison requires identical resource allocation. Each model gets its own array job with standardized resources.

### 2. Train on Original Graph Only
**Rationale:** Faster, captures true degree structure, sufficient for initial models.

### 3. Separate Jobs Per Model
**Rationale:** Enables fair benchmarking - each model timed independently with same resources.

### 4. Degree-Binned Training Data
**Rationale:** 100 rows vs millions, captures essential structure, 1000× memory reduction.

### 5. Tiered Validation
**Rationale:** CV first (fast), permutations only if needed (expensive).

---

## 📝 Notes

- **17b Fix:** The CSV corruption issue in notebook 17b is NOW RESOLVED with the updated script. The script deletes all old CSV files before running.

- **Benchmarking:** All timing excludes I/O. Only computational time is measured for fair comparison.

- **Normalized Cost:** Standard node-hour = 1h × 4 cores × 32GB RAM. All costs normalized to this.

- **Reproducibility:** All notebooks are parameterized, all scripts use standard resources, all results are timestamped.

- **HPC Optimization:** Job arrays parallelize across metapaths, dependencies ensure correct execution order.

---

## 🆘 Troubleshooting

### If 17b fails again:
```bash
# Manually delete CSV
rm /projects/$USER/repositories/Context-Aware-Path-Probability/results/compositional_validation/failure_analysis*.csv

# Re-submit
sbatch scripts/17b_compositional_failure_analysis.sh
```

### If data prep fails:
Check that notebook 17 completed:
```bash
ls -lh results/compositional_validation/validation_summary.json
```

### If training fails:
Check that training data exists:
```bash
ls -lh results/pathway_nn/training_data/*.csv
```

### If CV validation fails:
Check that all 5 models are trained:
```bash
ls -lh results/pathway_nn/trained_models/*.pkl
ls -lh results/pathway_nn/trained_models/*.pt
```

---

## 📚 References

- Alpine HPC Docs: https://curc.readthedocs.io/en/latest/clusters/alpine/
- Greene Lab Standards: https://github.com/greenelab/onboarding
- Fair Benchmarking Plan: See conversation history
