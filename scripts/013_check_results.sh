#!/bin/bash
# Check which edge types completed successfully

EDGE_TYPES=(
    "AdG" "AeG" "AuG" "CbG" "CcSE" "CdG" "CpD" "CrC" "CtD" "CuG"
    "DaG" "DdG" "DlA" "DpS" "DrD" "DuG" "GcG" "GiG" "GpBP" "GpCC"
    "GpMF" "GpPW" "Gr>G" "PCiC"
)

echo "========================================================================"
echo "Checking Null Model Training Results"
echo "========================================================================"
echo ""

COMPLETED=0
FAILED=0
MISSING_FILES=""

for EDGE_TYPE in "${EDGE_TYPES[@]}"; do
    RF_MODEL="results/null_models/${EDGE_TYPE}_rf_null.pkl"
    POLY_MODEL="results/null_models/${EDGE_TYPE}_poly_null.pkl"
    POLY_FEATURES="results/null_models/${EDGE_TYPE}_poly_features.pkl"

    if [ -f "$RF_MODEL" ] && [ -f "$POLY_MODEL" ] && [ -f "$POLY_FEATURES" ]; then
        echo "✓ ${EDGE_TYPE}: All models found"
        ((COMPLETED++))
    else
        echo "✗ ${EDGE_TYPE}: Models MISSING"
        MISSING_FILES="${MISSING_FILES}${EDGE_TYPE} "
        ((FAILED++))
    fi
done

echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "Completed: ${COMPLETED}/24"
echo "Failed:    ${FAILED}/24"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All edge types completed successfully!"
    echo ""
    echo "Checking empirical validation results..."

    if [ -f "results/null_models/empirical_validation_results.csv" ]; then
        echo "✓ Empirical validation results found"
        echo ""
        echo "Top 5 lines of empirical_validation_results.csv:"
        head -n 6 results/null_models/empirical_validation_results.csv | column -t -s,
        echo ""
        echo "To see full results:"
        echo "  cat results/null_models/empirical_validation_results.csv"
    else
        echo "⚠ Empirical validation results not found (may need to aggregate from individual runs)"
    fi

    exit 0
else
    echo "✗ ${FAILED} edge types failed:"
    echo "  ${MISSING_FILES}"
    echo ""
    echo "Check logs in logs/model_training/null_models/ for details:"
    echo "  ls -lht logs/model_training/null_models/ | head -20"
    echo ""
    echo "To rerun failed edge types, use:"
    echo "  sbatch --array=<task_ids> scripts/013_null_model_training.sh"
    echo ""
    echo "Failed edge type task IDs:"
    TASK_ID=1
    for EDGE_TYPE in "${EDGE_TYPES[@]}"; do
        if [[ " ${MISSING_FILES} " =~ " ${EDGE_TYPE} " ]]; then
            echo "  ${EDGE_TYPE}: task ${TASK_ID}"
        fi
        ((TASK_ID++))
    done

    exit 1
fi
