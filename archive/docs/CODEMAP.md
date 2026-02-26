CODEMAP: Active vs. Archived Code
=================================

Purpose
-------
Map the active null-prediction code paths and note which experiments were archived after the null-distribution draft v1 review.

Active modules (keep in src/)
- Edge/null priors: src/model_comparison.py, src/data_processing.py, src/degree_analysis.py, src/learned_analytical.py, src/permutation_validation.py
- Pathway/bin-level pipeline: src/pathway_features_v2.py, src/pathway_training_v2.py, src/pathway_evaluation_v2.py, src/pathway_models_v2.py, src/pathway_losses.py, src/pathway_nn.py (DegreeSignatureNN variant used in v2), src/graph_features.py
- Pair-level nulls: src/pair_level_features.py, src/pair_level_sampling.py
- Utilities: src/sampling.py, src/validation_utils.py (and other helpers these depend on)

Archived experiments (moved to src/archive/)
- Transformers / sequence variants: pathway_transformer.py, pathway_sequence_data.py
- Physics-inspired and theory-guided attempts: physics_informed_nn.py, evaluate_physics_informed_nn.py, theoretical_correction.py, theoretical_corrections.py, evaluate_theoretical_approach.py, evaluate_theory_guided_models.py, theory_guided_model.py
- Enhanced/optimized variants that did not beat baselines: enhanced_experiments.py, enhanced_features.py, optimized_model.py
- Host-specific duplicate: model_comparison-Lucas’s MacBook Pro.py

Notes
- The archived modules remain for provenance; do not import them in production.
- If any downstream script still references archived modules, update imports to use the active set above.