# Degree-Based Error Analysis Framework

This document provides a comprehensive guide to the enhanced degree-based error analysis framework for understanding model performance patterns across different node degree combinations.

## Overview

The degree-based error analysis framework extends the original model testing and learned analytical formula analysis with systematic error decomposition by node degree ranges. This provides critical insights into where models succeed or fail based on graph topology characteristics.

## Framework Components

### 1. Core Analysis Module (`src/degree_analysis.py`)

**DegreeAnalyzer Class Features:**
- Configurable degree binning for different graph sizes
- Comprehensive error metrics by degree combination
- Visualization tools for degree-based analysis
- Scalable design for both local and HPC execution

**Key Functions:**
- `categorize_degrees()`: Flexible degree binning
- `analyze_predictions_by_degree()`: Stratified prediction analysis
- `compute_degree_error_metrics()`: Comprehensive error decomposition
- `run_degree_analysis_pipeline()`: End-to-end analysis pipeline

### 2. Enhanced Model Testing

**Enhanced Features:**
- Degree-stratified performance metrics across all models
- Bias-variance decomposition by degree ranges
- Model recommendations based on graph characteristics
- Automated degree-aware best model selection

**Files:**
- `notebooks/5_model_testing_summary_with_degree_analysis.ipynb`
- `notebooks/test_degree_analysis_small_graphs.ipynb` (for local testing)

### 3. Enhanced Learned Analytical Formula

**Degree-Based Enhancements:**
- Residual analysis with degree decomposition
- Parameter sensitivity by degree range
- Enhanced convergence analysis with degree stratification
- Formula performance validation across degree combinations

**Files:**
- `notebooks/8_learned_analytical_formula_with_degree_analysis.ipynb`
- Enhanced `src/learned_analytical.py` with degree-aware validation

### 4. HPC Deployment Scripts

**Scalable Execution:**
- Job array scripts for parallel processing
- Automated dependency management
- Resource-optimized configurations
- Summary analysis and aggregation

**Scripts:**
- `scripts/degree_analysis_hpc.sh`: Core degree analysis
- `scripts/enhanced_model_testing_hpc.sh`: Enhanced model testing
- `scripts/learned_formula_enhanced_hpc.sh`: Enhanced formula analysis
- `scripts/run_all_degree_analysis.sh`: Master coordination script

## Quick Start

### Local Testing (Small Graphs)

```bash
# 1. Test the framework on small graphs
jupyter notebook notebooks/test_degree_analysis_small_graphs.ipynb

# 2. Run enhanced model testing (small graphs)
jupyter notebook notebooks/5_model_testing_summary_with_degree_analysis.ipynb

# 3. Test enhanced learned formula analysis
jupyter notebook notebooks/8_learned_analytical_formula_with_degree_analysis.ipynb
```

### HPC Deployment (All Edge Types)

```bash
# 1. Configure HPC account and paths in scripts
# Edit scripts/*.sh to set YOUR_ACCOUNT and environment paths

# 2. Run complete analysis pipeline
bash scripts/run_all_degree_analysis.sh

# 3. Monitor progress
squeue -u $USER
tail -f logs/degree_analysis_*.out
```

## Degree Binning Strategy

### Small Graph Mode (Local Testing)
- **Very Low (1-4)**: Minimal connectivity nodes
- **Low (5-19)**: Low-degree nodes
- **Medium (20-99)**: Medium-degree nodes
- **High (100+)**: High-degree nodes

### Full Scale Mode (HPC Analysis)
- **Low (1-9)**: Low-degree nodes
- **Medium (10-99)**: Medium-degree nodes
- **High (100-999)**: High-degree nodes
- **Hub (1000+)**: Hub nodes

### Degree Combinations
- Creates pairwise combinations: Low-Low, Low-Medium, Medium-High, etc.
- Analyzes error patterns for each source-target degree combination
- Identifies systematic biases and performance patterns

## Error Metrics by Degree

### Comprehensive Metrics
- **Bias**: Mean(Predicted - Empirical) by degree combination
- **Variance**: Var(Predicted - Empirical) by degree combination
- **RMSE**: Root mean squared error by degree combination
- **MAE**: Mean absolute error by degree combination
- **Correlation**: Pearson correlation by degree combination
- **Relative Error**: (Predicted - Empirical) / Empirical by degree combination

### Analysis Outputs
- **Error heatmaps**: RMSE/MAE by source × target degree categories
- **Bias analysis**: Systematic over/under-prediction patterns
- **Sample size analysis**: Statistical power by degree combination
- **Model comparison**: Best performing models by degree range

## Key Insights and Applications

### Model Selection
- **Random Forest**: Best for high-degree nodes and dense graphs
- **Polynomial Logistic Regression**: Best for sparse graphs and low-degree nodes
- **Neural Networks**: Strong overall performance but less consistent
- **Logistic Regression**: Most consistent across degree ranges

### Learned Formula Performance
- **Parameter sensitivity**: Different parameters matter for different degree ranges
- **Convergence patterns**: Some degree combinations require more training data
- **Formula variants**: Extended formulas help with high-degree nodes

### Graph Characteristics Impact
- **Density effects**: Dense graphs benefit from different approaches than sparse graphs
- **Degree distribution**: Heavy-tailed distributions require specialized handling
- **Hub nodes**: Special consideration needed for very high-degree nodes

## Generated Outputs

### Local Testing Outputs
```
results/degree_analysis/
├── {edge_type}_{model}_degree_analysis.csv     # Detailed analysis
├── {edge_type}_{model}_degree_metrics.csv      # Error metrics summary
├── {edge_type}_{model}_mae_by_degree.png       # MAE visualization
├── {edge_type}_{model}_correlation_by_degree.png # Correlation visualization
└── {edge_type}_{model}_empirical_heatmap.png   # Empirical frequency heatmap
```

### HPC Deployment Outputs
```
results/
├── degree_analysis_hpc/           # Core degree analysis results
├── enhanced_model_testing/        # Enhanced model testing notebooks
├── learned_analytical_enhanced_hpc/ # Enhanced formula analysis
│   └── {edge_type}_{formula_type}/ # Results by edge type and formula
└── degree_analysis_summary.json   # Overall summary statistics
```

## Performance and Scalability

### Local Testing
- **Target**: 3-5 smallest edge types (<10k edges)
- **Runtime**: 10-30 minutes per edge type
- **Memory**: 4-8 GB recommended
- **Purpose**: Framework validation and methodology testing

### HPC Deployment
- **Target**: All 24 edge types
- **Runtime**: 6-12 hours total (parallel execution)
- **Resources**: 144-192 core-hours across all jobs
- **Memory**: 16-32 GB per job
- **Purpose**: Production analysis and publication-ready results

## Advanced Configuration

### Custom Degree Bins
```python
# Custom degree binning
analyzer = DegreeAnalyzer(
    degree_bins=[1, 5, 25, 100, 500, np.inf],
    small_graph_mode=False
)
```

### Formula Type Selection
```python
# Test different formula variants
learner = LearnedAnalyticalFormula(
    formula_type='extended',  # 'original', 'extended', 'polynomial'
    n_random_starts=10,
    regularization_lambda=0.001
)
```

### Degree-Stratified Convergence
```python
# Enable degree-stratified minimum permutations analysis
results = learner.find_minimum_permutations(
    graph_name=edge_type,
    degree_stratified=True,
    small_graph_mode=False
)
```

## Troubleshooting

### Common Issues

1. **Missing empirical data**: Ensure empirical frequency files exist
2. **Memory issues**: Reduce degree bins or use smaller graphs for testing
3. **HPC permissions**: Check account settings and module availability
4. **Missing dependencies**: Ensure all required Python packages are installed

### Performance Optimization

1. **Small graphs first**: Always validate on small graphs before HPC deployment
2. **Parallel execution**: Use job arrays for maximum efficiency
3. **Resource allocation**: Match memory/CPU requirements to graph size
4. **Checkpoint results**: Save intermediate results for fault tolerance

## Citation and Usage

When using this framework, please cite the original research and acknowledge the enhanced degree-based analysis extensions. The framework is designed to be publication-ready and provides comprehensive documentation for reproducibility.

## Support and Development

This framework is designed to be extensible and maintainable. For questions, issues, or contributions:

1. Check the troubleshooting section above
2. Review the example notebooks for usage patterns
3. Examine the HPC scripts for deployment guidance
4. Consider the performance characteristics for your specific use case

The degree-based error analysis framework provides unprecedented insight into model performance patterns and enables more informed model selection and parameter tuning for biological network analysis.