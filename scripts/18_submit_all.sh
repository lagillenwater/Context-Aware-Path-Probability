#!/bin/bash
# Master submission script for Degree Signature Pipeline
# WITH FAIR BENCHMARKING AND STANDARDIZED RESOURCES
#
# All tasks use standardized resources for fair comparison:
# - Standard: 4 cores, 32GB RAM
# - Time limits vary by task complexity
#
# Usage:
#   cd /projects/$USER/repositories/Context-Aware-Path-Probability/scripts
#   bash 18_submit_all.sh                    # Production: all 8 metapaths
#   bash 18_submit_all.sh --debug CbGpPW     # Debug: single metapath

# Parse arguments
DEBUG_MODE=false
DEBUG_METAPATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      DEBUG_MODE=true
      DEBUG_METAPATH="$2"
      if [ -z "$DEBUG_METAPATH" ]; then
        echo "ERROR: --debug requires a metapath argument"
        echo "Usage: bash 18_submit_all.sh --debug METAPATH"
        echo "Valid metapaths: CbGpPW, CtDaG, CbGaD, CrCbG, CbGiG, CpDaG, CbGpBP, CbGpCC"
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "ERROR: Unknown option: $1"
      echo "Usage: bash 18_submit_all.sh [--debug METAPATH]"
      exit 1
      ;;
  esac
done

SCRIPTS_DIR=$(dirname "$0")
cd $SCRIPTS_DIR

# Create necessary directories
mkdir -p ../logs
mkdir -p ../results/pathway_nn/{training_data,trained_models,benchmarks,validation,figures}

echo "=========================================="
echo "DEGREE SIGNATURE PIPELINE SUBMISSION"
echo "Fair Benchmarking with Standardized Resources"
echo "=========================================="
echo ""

if [ "$DEBUG_MODE" = true ]; then
  echo "MODE: DEBUG (single metapath)"
  echo "Metapath: $DEBUG_METAPATH"
  echo "Array spec: --array=1 (1 task per job)"
  ARRAY_SPEC="--array=1"
  EXPORT_VARS="--export=ALL,DEBUG_METAPATH=$DEBUG_METAPATH"
else
  echo "MODE: PRODUCTION (all metapaths)"
  echo "Array spec: --array=1-8 (8 tasks per job)"
  ARRAY_SPEC="--array=1-8"
  EXPORT_VARS=""
fi

echo ""
echo "Repository: $(pwd)/.."
echo "Submission time: $(date)"
echo ""


# ============================================================================
# STEP 1: Data Preparation (Array Job)
# ============================================================================
echo "=========================================="
echo "STEP 1: Data Preparation"
echo "=========================================="
echo ""
echo "Submitting 18a (data preparation)..."
echo "  Purpose: Generate training data from original graph"
echo "  Resources per task: 4 cores, 64GB, 6h"
if [ "$DEBUG_MODE" = true ]; then
  echo "  Tasks: 1 (metapath: $DEBUG_METAPATH)"
  echo "  Total: ~6 node-hours @ 64GB"
else
  echo "  Tasks: 8 (one per metapath)"
  echo "  Total: 8 tasks × 6h = 48 node-hours @ 64GB"
fi
echo ""

JOB_DATA=$(sbatch --parsable $ARRAY_SPEC $EXPORT_VARS 18a_data_preparation.sh)

if [ -z "$JOB_DATA" ]; then
    echo "ERROR: Failed to submit 18a"
    exit 1
fi

echo "  Job ID: $JOB_DATA"
echo "  Status: Queued"
echo ""

# ============================================================================
# STEP 2: Model Training (5 Separate Array Jobs for Fair Comparison)
# ============================================================================
echo "=========================================="
echo "STEP 2: Model Training (Fair Benchmarking)"
echo "=========================================="
echo ""
echo "Submitting 5 model training arrays (parallel execution)..."
echo "Each array: 1-8 tasks (one per metapath)"
echo "Dependency: After data prep completes"
echo ""

# Model 1: Random Baseline
echo "[2a] Random Baseline"
echo "     Resources: 8 tasks × (4 cores, 16GB, 0.5h) = 2 node-hours"
JOB_TRAIN_RAND=$(sbatch --parsable --dependency=afterok:$JOB_DATA --kill-on-invalid-dep=yes $ARRAY_SPEC $EXPORT_VARS 18b_train_random.sh)
echo "     Job ID: $JOB_TRAIN_RAND"
echo ""

# Model 2: Degree Product Baseline
echo "[2b] Degree Product Baseline"
echo "     Resources: 8 tasks × (4 cores, 16GB, 0.5h) = 2 node-hours"
JOB_TRAIN_DEG=$(sbatch --parsable --dependency=afterok:$JOB_DATA --kill-on-invalid-dep=yes $ARRAY_SPEC $EXPORT_VARS 18c_train_degree_product.sh)
echo "     Job ID: $JOB_TRAIN_DEG"
echo ""

# Model 3: Negative Binomial GLM
echo "[2c] Negative Binomial GLM"
echo "     Resources: 8 tasks × (4 cores, 16GB, 1h) = 4 node-hours"
JOB_TRAIN_GLM=$(sbatch --parsable --dependency=afterok:$JOB_DATA --kill-on-invalid-dep=yes $ARRAY_SPEC $EXPORT_VARS 18d_train_negbin_glm.sh)
echo "     Job ID: $JOB_TRAIN_GLM"
echo ""

# Model 4: Random Forest
echo "[2d] Random Forest"
echo "     Resources: 8 tasks × (4 cores, 32GB, 2h) = 16 node-hours"
JOB_TRAIN_RF=$(sbatch --parsable --dependency=afterok:$JOB_DATA --kill-on-invalid-dep=yes $ARRAY_SPEC $EXPORT_VARS 18e_train_random_forest.sh)
echo "     Job ID: $JOB_TRAIN_RF"
echo ""

# Model 5: Degree Signature Neural Network
echo "[2e] Degree Signature NN"
echo "     Resources: 8 tasks × (4 cores, 32GB, 2h) = 16 node-hours"
JOB_TRAIN_NN=$(sbatch --parsable --dependency=afterok:$JOB_DATA --kill-on-invalid-dep=yes $ARRAY_SPEC $EXPORT_VARS 18f_train_degree_signature_nn.sh)
echo "     Job ID: $JOB_TRAIN_NN"
echo ""

# Wait for all training jobs
TRAIN_JOBS="$JOB_TRAIN_RAND:$JOB_TRAIN_DEG:$JOB_TRAIN_GLM:$JOB_TRAIN_RF:$JOB_TRAIN_NN"

# ============================================================================
# STEP 3: Cross-Validation (Tests All 5 Models)
# ============================================================================
echo "=========================================="
echo "STEP 3: Cross-Validation (Tier 1)"
echo "=========================================="
echo ""
echo "Submitting 18g (CV validation array)..."
echo "  Purpose: Test all 5 models with K-fold CV on original graph"
echo "  Array: 1-8 (one task per metapath)"
echo "  Resources per task: 4 cores, 16GB, 1h"
echo "  Total: 8 tasks × 1h = 8 node-hours @ 16GB"
echo "  Tests: All 5 models per metapath"
echo "  Dependency: After all training completes"
echo ""

JOB_CV=$(sbatch --parsable --dependency=afterok:$TRAIN_JOBS --kill-on-invalid-dep=yes $ARRAY_SPEC $EXPORT_VARS 18g_validate_cv_all_models.sh)

if [ -z "$JOB_CV" ]; then
    echo "ERROR: Failed to submit 18g"
    exit 1
fi

echo "  Job ID: $JOB_CV (array 1-8)"
echo "  Status: Queued (waiting for training)"
echo ""

# ============================================================================
# STEP 4: Benchmark Summary and Analysis
# ============================================================================
echo "=========================================="
echo "STEP 4: Benchmark Summary"
echo "=========================================="
echo ""
echo "Submitting 18i (benchmark summary)..."
echo "  Purpose: Aggregate results, generate comparison figures"
echo "  Resources: 4 cores, 16GB, 0.5h"
echo "  Dependency: After CV completes"
echo ""

JOB_SUMMARY=$(sbatch --parsable --dependency=afterok:$JOB_CV --kill-on-invalid-dep=yes 18i_benchmark_summary.sh)

if [ -z "$JOB_SUMMARY" ]; then
    echo "ERROR: Failed to submit 18i"
    exit 1
fi

echo "  Job ID: $JOB_SUMMARY"
echo "  Status: Queued (waiting for CV)"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=========================================="
echo "PIPELINE SUBMITTED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Job Dependency Chain:"
echo "  18a Data Prep ($JOB_DATA)"
echo "    ↓"
echo "  18b-f Training (5 parallel arrays):"
echo "    - Random ($JOB_TRAIN_RAND)"
echo "    - Degree Product ($JOB_TRAIN_DEG)"
echo "    - NegBin GLM ($JOB_TRAIN_GLM)"
echo "    - Random Forest ($JOB_TRAIN_RF)"
echo "    - Degree Sig NN ($JOB_TRAIN_NN)"
echo "    ↓"
echo "  18g CV Validation ($JOB_CV array 1-8)"
echo "    ↓"
echo "  18i Benchmark Summary ($JOB_SUMMARY)"
echo ""

# ============================================================================
# RESOURCE SUMMARY
# ============================================================================
echo "=========================================="
echo "TOTAL RESOURCE BUDGET"
echo "=========================================="
echo ""
echo "Resource standardization:"
echo "  - All tasks use: 4 cores (STANDARD)"
echo "  - Memory varies: 16GB (light), 32GB (medium), 64GB (heavy)"
echo "  - Time varies: 0.5h-6h by task"
echo ""
echo "Estimated costs:"
echo "  17b Baseline:      6 node-hours @ 32GB"
echo "  Data Prep (18a):  48 node-hours @ 64GB"
echo "  Training (18b-f): 40 node-hours @ 16-32GB (combined)"
echo "  CV Validation:     8 node-hours @ 16GB"
echo "  Summary (18i):   0.5 node-hours @ 16GB"
echo "  ─────────────────────────────────────"
echo "  TOTAL:         102.5 node-hours"
echo ""
echo "Wall clock time (approximate):"
echo "  Sequential phases: ~15-17 hours"
echo "  (Parallel within each phase)"
echo ""

# ============================================================================
# MONITORING COMMANDS
# ============================================================================
echo "=========================================="
echo "MONITORING COMMANDS"
echo "=========================================="
echo ""
echo "Check job status:"
echo "  squeue -u \$USER"
echo ""
echo "Monitor specific job:"
echo "  squeue -j $JOB_DATA  # Data prep"
echo "  squeue -j $JOB_CV    # CV validation"
echo "  squeue -j $JOB_SUMMARY  # Summary"
echo ""
echo "View logs (as jobs run):"
echo "  tail -f ../logs/18a_data_prep_*.out"
echo "  tail -f ../logs/18g_validate_cv_*.out"
echo "  tail -f ../logs/18i_summary_*.out"
echo ""
echo "View all recent logs:"
echo "  ls -lt ../logs/*.out | head -20"
echo ""

# ============================================================================
# DECISION POINT: TIER 2 VALIDATION
# ============================================================================
echo "=========================================="
echo "TIER 2 VALIDATION (CONDITIONAL)"
echo "=========================================="
echo ""
echo "After CV validation completes, check results:"
echo "  cat ../results/pathway_nn/validation/cv_results.csv"
echo ""
echo "If best model achieves r > 0.85:"
echo "  → SUCCESS! Use that model for anomaly detection"
echo "  → Pipeline complete"
echo ""
echo "If best model achieves 0.75 < r < 0.85:"
echo "  → Consider Tier 2: Permutation validation"
echo "  → Submit manually:"
echo "      sbatch --dependency=afterok:$JOB_SUMMARY 18h_validate_permutations_all_models.sh"
echo "  → Resources: 8 tasks × (4 cores, 32GB, 4h) = 32 node-hours"
echo ""
echo "If all models r < 0.75:"
echo "  → Re-evaluate approach"
echo "  → May need to add permutation data to training"
echo ""

# ============================================================================
# EXPECTED OUTPUTS
# ============================================================================
echo "=========================================="
echo "EXPECTED OUTPUTS"
echo "=========================================="
echo ""
echo "Training data:"
echo "  ../results/pathway_nn/training_data/"
echo "    - CbGpPW_training_data.csv (~100 rows)"
echo "    - CtDaG_training_data.csv"
echo "    - ... (8 metapaths total)"
echo ""
echo "Trained models:"
echo "  ../results/pathway_nn/trained_models/"
echo "    - CbGpPW_Random.pkl"
echo "    - CbGpPW_Degree_Product.pkl"
echo "    - CbGpPW_NegBin_GLM.pkl"
echo "    - CbGpPW_Random_Forest.pkl"
echo "    - CbGpPW_Degree_Signature_NN.pt"
echo "    - ... (40 models total: 8 metapaths × 5 models)"
echo ""
echo "Benchmarks:"
echo "  ../results/pathway_nn/benchmarks/"
echo "    - CbGpPW_Random_benchmark.json"
echo "    - ... (40 benchmark files)"
echo "    - comparison_table.csv (aggregated results)"
echo ""
echo "Validation results:"
echo "  ../results/pathway_nn/validation/"
echo "    - cv_results.csv (CV performance by model/metapath)"
echo ""
echo "Figures:"
echo "  ../results/pathway_nn/figures/"
echo "    - efficiency_frontier.png"
echo "    - model_comparison_heatmap.png"
echo "    - cv_performance_by_metapath.png"
echo ""

# ============================================================================
# SAVE JOB IDS
# ============================================================================
cat > job_ids.txt <<EOF
# Pipeline Job IDs - $(date)
# Use these IDs for monitoring and dependencies

JOB_DATA=$JOB_DATA
JOB_TRAIN_RAND=$JOB_TRAIN_RAND
JOB_TRAIN_DEG=$JOB_TRAIN_DEG
JOB_TRAIN_GLM=$JOB_TRAIN_GLM
JOB_TRAIN_RF=$JOB_TRAIN_RF
JOB_TRAIN_NN=$JOB_TRAIN_NN
JOB_CV=$JOB_CV
JOB_SUMMARY=$JOB_SUMMARY

# Combined training jobs (for dependencies)
TRAIN_JOBS=$TRAIN_JOBS

# Monitoring commands
# squeue -j $JOB_DATA,$JOB_CV,$JOB_SUMMARY
EOF

echo "Job IDs saved to: job_ids.txt"
echo ""

# ============================================================================
# MANUAL CANCELLATION (if needed)
# ============================================================================
echo "=========================================="
echo "MANUAL CANCELLATION (if needed)"
echo "=========================================="
echo ""
echo "To cancel the entire pipeline:"
echo "  scancel $JOB_DATA $JOB_TRAIN_RAND $JOB_TRAIN_DEG $JOB_TRAIN_GLM $JOB_TRAIN_RF $JOB_TRAIN_NN $JOB_CV $JOB_SUMMARY"
echo ""
echo "Or cancel all your pending jobs:"
echo "  scancel -u \$USER --state=PENDING"
echo ""

# ============================================================================
# HELPFUL ALIASES
# ============================================================================
echo "=========================================="
echo "HELPFUL ALIASES (copy to terminal)"
echo "=========================================="
echo ""
echo "alias watch-jobs='watch -n 5 squeue -u \$USER'"
echo "alias check-cv='cat ../results/pathway_nn/validation/cv_results.csv'"
echo "alias show-benchmarks='cat ../results/pathway_nn/benchmarks/comparison_table.csv | column -t -s,'"
echo ""

echo "=========================================="
echo "SUBMISSION COMPLETE"
echo "=========================================="
echo ""
echo "Next: Monitor job progress with 'squeue -u \$USER'"
echo "Expected completion: ~15-17 hours from now"
echo ""
