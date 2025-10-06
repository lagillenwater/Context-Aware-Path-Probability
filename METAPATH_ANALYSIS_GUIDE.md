# Metapath Probability Analysis and Anomaly Detection

## Overview

This framework analyzes the compositionality of metapath probabilities in hetionet and implements anomaly detection for identifying unusual biological pathways. The analysis addresses the fundamental question: **Are metapath probabilities compositional (independent edges) or conditional (dependent edges)?**

## Key Research Questions

### 1. **Compositionality vs Conditionality**
- **Compositional**: P(path) = P(edge₁) × P(edge₂) × P(edge₃)
- **Conditional**: P(path) = P(edge₁) × P(edge₂|edge₁) × P(edge₃|edge₁,edge₂)

### 2. **Degree Dependency**
How do source/target node degrees affect path probabilities across different models?

### 3. **Anomaly Detection**
Can we identify biologically interesting or unusual metapaths based on probability discrepancies?

## Framework Components

### 📊 **1. Metapath Extraction** (`MetapathExtractor`)
- Extracts actual 3-edge metapaths from hetionet
- Supports patterns like: C→b→G→p→PW→p←G (CbGpPWpG)
- Handles large-scale graph traversal efficiently
- Returns node indices and path identifiers

### 🧮 **2. Probability Calculation** (`ProbabilityCalculator`)
- **Edge Probabilities**: Uses pre-trained model predictions from notebook 4
- **Compositional**: Simple multiplication of edge probabilities
- **Conditional**: P(edge₂|edge₁) and P(edge₃|edge₁,edge₂) using binned conditioning
- **Empirical**: Observed frequency-based probabilities

### 🔬 **3. Compositionality Testing** (`CompositionalityTester`)
- **Independence Tests**: Spearman correlation between edge probabilities
- **Model Comparison**: Compositional vs Conditional vs Empirical
- **Validation Metrics**: R², correlation coefficients, p-values

### 🎯 **4. Anomaly Detection** (`AnomalyDetector`)
- **Multi-component Scoring**:
  - **Residual Score**: |Empirical - Predicted| / Predicted
  - **Z-score**: Deviation from compositional probability distribution
  - **Consistency Score**: Variance in edge probabilities
- **Combined Anomaly Score**: Weighted combination of components
- **Statistical Validation**: Mann-Whitney U tests

## Example Metapath: CbGpPWpG

```
Compound → binds → Gene → participates → Pathway → participates ← Gene
    C    →   b   →  G  →      p       →   PW    →      p       ←  G
```

### Edge Types:
1. **CbG**: Compound binds Gene
2. **GpPW**: Gene participates in Pathway
3. **GpPW**: Gene participates in Pathway (reverse direction)

### Analysis Flow:
1. Extract all CbGpPWpG paths from hetionet
2. Calculate edge probabilities using Random Forest model
3. Test: P(CbGpPWpG) = P(CbG) × P(GpPW) × P(GpPW)?
4. Compare with conditional: P(CbG) × P(GpPW|CbG) × P(GpPW|CbG,GpPW)
5. Identify anomalous compound-gene-pathway-gene relationships

## Files Structure

### **Core Analysis**
- `notebooks/6_metapath_probability_analysis.ipynb`: Main analysis notebook
- `src/metapath_analysis.py`: Supporting classes and functions
- `scripts/6_metapath_analysis.sh`: HPC job array script

### **Input Dependencies**
- Model predictions from notebook 4: `{EDGE_TYPE}_all_model_predictions.csv`
- Edge matrices: `data/permutations/000.hetmat/edges/{EDGE_TYPE}.sparse.npz`

### **Output Files**
- `{METAPATH}_metapath_analysis.csv`: Complete metapath data with probabilities and scores
- `{METAPATH}_anomalies.csv`: Detected anomalous metapaths
- `{METAPATH}_analysis_summary.json`: Statistical summary and validation results
- Visualization plots: `{METAPATH}_compositionality_analysis.png`, `{METAPATH}_anomaly_detection.png`

## Interpretation Guide

### **Compositionality Results**
- **High correlation (r > 0.7)**: Metapath probabilities are approximately compositional
- **Low correlation (r < 0.3)**: Strong conditional dependencies exist
- **R² score**: Proportion of variance explained by compositional model

### **Anomaly Types**
1. **High Residual**: Large difference between predicted and empirical probabilities
2. **Extreme Z-score**: Unusually high or low compositional probabilities
3. **Inconsistent Edges**: High variance in individual edge probabilities

### **Biological Interpretation**
- **High-probability anomalies**: Well-known or highly conserved pathways
- **Low-probability anomalies**: Novel or disease-associated relationships
- **Edge inconsistencies**: Potential data quality issues or interesting biology

## Usage Examples

### **Interactive Analysis**
```python
from src.metapath_analysis import run_metapath_analysis

results = run_metapath_analysis(
    metapath_pattern="CbGpPWpG",
    edge_types=["CbG", "GpPW", "GpPW"],
    data_dir="data",
    prediction_dir="results/model_comparison",
    max_paths=50000,
    anomaly_threshold=0.05
)

# Access results
metapaths_df = results['metapaths_df']
anomalies = results['anomalies']
summary = results['summary_stats']
```

### **HPC Batch Processing**
```bash
# Submit metapath analysis for multiple patterns
cd scripts
sbatch 6_metapath_analysis.sh

# Monitor progress
squeue -u $USER
tail -f ../logs/output_metapath_analysis_*.log
```

## Supported Metapath Patterns

The framework currently supports these biologically relevant 3-edge metapaths:

1. **CbGpPWpG**: Compound-Gene-Pathway-Gene relationships
2. **CtDaGdG**: Compound treats Disease associated with Gene that interacts with Gene
3. **CbGiGdG**: Compound binds Gene that interacts with Gene that is associated with Disease
4. **CuGcGdG**: Compound upregulates Gene that co-occurs with Gene that is associated with Disease
5. **CdGpPWpG**: Compound downregulates Gene that participates in Pathway with Gene

## Statistical Validation

### **Independence Testing**
- Spearman correlation between edge probabilities
- Null hypothesis: Edge probabilities are independent
- Significance threshold: p < 0.05

### **Model Comparison**
- Pearson and Spearman correlations between probability models
- R² for explained variance
- Cross-validation metrics

### **Anomaly Validation**
- Mann-Whitney U tests comparing anomalies vs normal metapaths
- Effect size calculations
- False discovery rate control

## Applications

### **Drug Discovery**
- Identify novel compound-target-pathway relationships
- Detect unexpected drug mechanisms of action
- Prioritize compounds for specific disease pathways

### **Network Analysis**
- Understand information flow patterns in biological networks
- Validate network reconstruction methods
- Identify key regulatory motifs

### **Quality Control**
- Detect potential errors in biological databases
- Identify outlier experimental results
- Validate computational predictions

### **Biomarker Discovery**
- Find unusual gene-pathway-disease associations
- Identify potential therapeutic targets
- Discover novel disease mechanisms

## Technical Notes

### **Computational Complexity**
- Time complexity: O(|E₁| × |E₂| × |E₃|) for metapath extraction
- Memory usage: Scales with number of extracted metapaths
- Recommended: Use sampling for very large networks

### **Statistical Considerations**
- **Multiple Testing**: Apply Bonferroni or FDR correction for multiple metapaths
- **Sample Size**: Ensure adequate metapath counts for statistical power
- **Normalization**: Log-transform highly skewed probability distributions

### **Model Selection**
- Random Forest generally provides best probability estimates
- Consider ensemble averaging across multiple models
- Validate on held-out test metapaths

## Future Extensions

### **N-edge Metapaths**
- Extend to longer metapaths (4+ edges)
- Implement hierarchical conditioning
- Model path-length-specific effects

### **Temporal Analysis**
- Incorporate time-varying edge probabilities
- Model dynamic metapath patterns
- Predict future pathway activation

### **Multi-scale Integration**
- Combine metapath analysis with gene expression
- Integrate protein interaction data
- Include metabolic pathway information

---

## Quick Start

1. **Run model predictions** (notebook 4) for all edge types
2. **Execute metapath analysis**: `sbatch scripts/6_metapath_analysis.sh`
3. **Review results** in `results/metapath_analysis/` directory
4. **Investigate anomalies** for biological insights

This framework provides a comprehensive approach to understanding metapath probability patterns and identifying biologically relevant anomalies in complex biological networks.