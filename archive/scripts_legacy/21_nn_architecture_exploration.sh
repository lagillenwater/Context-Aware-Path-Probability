#!/bin/bash

# Script: 21_nn_architecture_exploration.sh
# Description: Execute neural network architecture exploration notebook
# Author: Claude Code Assistant
# Date: 2025-10-16

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NOTEBOOK_DIR="$PROJECT_ROOT/notebooks"
RESULTS_DIR="$PROJECT_ROOT/results"
EXECUTED_DIR="$NOTEBOOK_DIR/executed"

# Input and output files
INPUT_NOTEBOOK="$NOTEBOOK_DIR/21_nn_architecture_exploration.ipynb"
OUTPUT_NOTEBOOK="$EXECUTED_DIR/21_nn_architecture_exploration_executed.ipynb"

# Create directories if they don't exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$EXECUTED_DIR"

echo "=================================================================="
echo "Neural Network Architecture Exploration"
echo "=================================================================="
echo "Input notebook: $INPUT_NOTEBOOK"
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if input notebook exists
if [[ ! -f "$INPUT_NOTEBOOK" ]]; then
    echo "❌ Error: Input notebook not found: $INPUT_NOTEBOOK"
    exit 1
fi

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Warning: No conda environment detected."
    echo "   Recommend running: conda activate CAPP"
fi

# Check for required Python packages
echo "🔍 Checking dependencies..."
python -c "
import sys
import subprocess

required_packages = [
    'numpy', 'pandas', 'matplotlib', 'seaborn',
    'torch', 'sklearn', 'scipy'
]

missing = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print(f'❌ Missing packages: {missing}')
    print('   Install with: pip install {}'.format(' '.join(missing)))
    sys.exit(1)
else:
    print('✅ All required packages available')
"

if [[ $? -ne 0 ]]; then
    echo "❌ Dependency check failed"
    exit 1
fi

# Execute notebook with papermill
echo ""
echo "🚀 Executing notebook..."
echo "   This may take 10-20 minutes depending on your hardware"
echo ""

# Set parameters for reproducibility
papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK" \
    -p edge_type "CbG" \
    -p random_seed 42 \
    -p max_epochs 30 \
    --log-output \
    --progress-bar

# Check if execution was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Notebook executed successfully!"
    echo "📊 Results saved to:"
    echo "   - Executed notebook: $OUTPUT_NOTEBOOK"
    echo "   - Architecture comparison: $RESULTS_DIR/architecture_comparison.csv"
    echo "   - Path dependency analysis: $RESULTS_DIR/path_dependency_decay.png"
    echo "   - Calibration plots: $RESULTS_DIR/sanity_check_calibration.png"
    echo "   - Complete analysis: $RESULTS_DIR/nn_architecture_analysis_complete.md"
    echo ""

    # Display file sizes for verification
    echo "📁 Output file sizes:"
    if [[ -f "$OUTPUT_NOTEBOOK" ]]; then
        echo "   $(ls -lh "$OUTPUT_NOTEBOOK" | awk '{print $5, $9}')"
    fi
    if [[ -f "$RESULTS_DIR/architecture_comparison.csv" ]]; then
        echo "   $(ls -lh "$RESULTS_DIR/architecture_comparison.csv" | awk '{print $5, $9}')"
    fi

    # Quick summary from results if available
    if [[ -f "$RESULTS_DIR/architecture_comparison.csv" ]]; then
        echo ""
        echo "🎯 Quick Results Summary:"
        echo "   Best performing architecture:"
        python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$RESULTS_DIR/architecture_comparison.csv')
    valid_df = df.dropna(subset=['Final AUC'])
    if len(valid_df) > 0:
        best = valid_df.loc[valid_df['Final AUC'].idxmax()]
        print(f'   {best[\"Architecture\"]}: AUC = {best[\"Final AUC\"]:.4f}')
        print(f'   Correlation = {best[\"Correlation\"]:.4f}')
    else:
        print('   No valid results found')
except Exception as e:
    print(f'   Could not parse results: {e}')
"
    fi

else
    echo ""
    echo "❌ Notebook execution failed!"
    echo "Check the error messages above for details."
    exit 1
fi

echo ""
echo "🎉 Neural Network Architecture Exploration Complete!"
echo "=================================================================="