# Legacy Shell Wrappers (Archived)

This directory stores shell/SLURM wrappers that were superseded during Track A
(`papermill` -> script-first `poe` migration).

These files were moved out of `scripts/` after equivalent Poe tasks were validated
with non-executing checks (`poe -d`).

## Direct replacements
- `0_create_hetmat.sh` -> `poe fetch-hetmat`
- `1_create_permutations.sh` -> `poe generate-permutations`
- `2_download_null_graphs.sh` -> `poe download-permutations`
- `3_edge_frequency_analysis.sh` -> `poe compute-edge-frequencies`
- `013_null_model_training.sh` -> `poe train-null-models`
- `014_fast_compositional_null.sh` -> `poe compose-null`
- `015_metapath_null_distributions.sh` -> `poe build-metapath-nulls`
- `17_compositional_validation.sh` -> `poe validate-composition`
- `17b_compositional_failure_analysis.sh` -> `poe analyze-composition-failures`
- `04_model_comparison_analysis.sh` -> `poe model-comparison-analysis`
- `05_model_testing_summary.sh` -> `poe model-testing-summary`
- `008_learned_analytical.sh` -> `poe learned-analytical`
- `18a_data_preparation.sh` -> `poe pathway-data-preparation`
- `18b_train_random.sh` -> `poe pathway-train-random`
- `18c_train_degree_product.sh` -> `poe pathway-train-degree-product`
- `18d_train_negbin_glm.sh` -> `poe pathway-train-negbin-glm`
- `18e_train_random_forest.sh` -> `poe pathway-train-random-forest`
- `18f_train_degree_signature_nn.sh` -> `poe pathway-train-degree-signature-nn`
- `18g_variance_estimation.sh` -> `poe pathway-variance-estimation`
- `18h_anomaly_detection.sh` -> `poe pathway-anomaly-detection`
- `19_variance_estimation.sh` -> `poe pathway-variance-estimation`
- `20_anomaly_detection.sh` -> `poe pathway-anomaly-detection`
- `21_nn_architecture_exploration.sh` -> `poe nn-architecture-exploration`
