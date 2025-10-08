#!/bin/bash

# Monitor running jobs
echo "=========================================="
echo "HPC Job Status"
echo "=========================================="
echo ""

# Check queue
echo "Active jobs:"
squeue -u $USER --format="%.18i %.12j %.8T %.10M %.10l %.6D %R" | head -20

echo ""
echo "Recent completed jobs:"
sacct -u $USER --starttime=today --format=JobID,JobName,State,Elapsed,MaxRSS | tail -20

echo ""
echo "=========================================="
echo "Log Files (recent errors):"
echo "=========================================="

# Check for recent errors
if ls logs/*err 1> /dev/null 2>&1; then
    echo ""
    echo "Recent error logs with content:"
    for errfile in logs/*err; do
        if [ -s "$errfile" ]; then  # Check if file is not empty
            echo "  - $errfile ($(wc -l < "$errfile") lines)"
        fi
    done
else
    echo "No error logs found"
fi

echo ""
echo "=========================================="
echo "Output Status:"
echo "=========================================="

# Check output directories
echo ""
echo "Null models: $(ls results/null_models/*.pkl 2>/dev/null | wc -l) files"
echo "Compositional nulls: $(ls results/compositional_null/*.csv 2>/dev/null | wc -l) files"
echo "Metapath nulls: $(ls results/metapath_nulls/*.csv 2>/dev/null | wc -l) files"
echo "Executed notebooks: $(ls notebooks/executed/*.ipynb 2>/dev/null | wc -l) files"
