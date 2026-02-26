Archive of experimental or deprecated modules
=============================================

This directory stores code that the null-distribution draft identified as non-essential or lower-value for the current pipelines. These modules were moved out of the active src/ root to keep imports and maintenance focused on the working baselines (analytical prior, edge nulls, bin-level and pair-level null prediction).

Archived files (moved from src/):
- enhanced_experiments.py
- enhanced_features.py
- optimized_model.py
- pathway_sequence_data.py
- pathway_transformer.py
- physics_informed_nn.py
- evaluate_physics_informed_nn.py
- theoretical_correction.py
- theoretical_corrections.py
- evaluate_theoretical_approach.py
- evaluate_theory_guided_models.py
- theory_guided_model.py
- model_comparison-Lucas’s MacBook Pro.py

Context: These experiments (transformers, GNN/physics-inspired, theory-guided tweaks, host-specific duplicates) did not outperform the simpler analytical prior or the degree-based linear/RF baselines described in the null-distribution draft v1. Keep them here for provenance; do not import from this folder in production code.