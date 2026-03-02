# Context-Aware-Path-Probability

Reproducible pipelines for null-distribution and path-probability experiments on biomedical knowledge graphs (Hetionet permutations and related metapaths). This README is the single source for running everything end-to-end via Poe tasks.

## Environment (Canonical)

```bash
# from repo root
cd environments
conda env create -f environment.yml    # first time only
conda activate CAPP
cd ..                                   # back to repo root
```

## Figure Reproduction Commands (Start Here)
```bash
# Quick end-to-end validation (smoke-scale figures + model/comparison suite)
poe quick-test-end-to-end

# Full figure/diagnostic reproduction suite
poe reproduce-figures-full

# CI-oriented smoke command (figures + one model/comparison smoke target)
poe ci-smoke
```

If using conda directly:
```bash
conda run -n CAPP poe quick-test-end-to-end
conda run -n CAPP poe reproduce-figures-full
conda run -n CAPP poe ci-smoke
```


Run tasks with either:
- `poe <task-name>` (inside active `CAPP`)
- `conda run -n CAPP poe <task-name>` (without activation)

Poetry remains supported for dependency management, but canonical task execution is `conda + poe`.

## Data prerequisites
- Hetionet hetmat and generated permutations expected under `data/`:
  - Base graph: `data/edges/*.sparse.npz`
  - Generated perms: `data/permutations/###.hetmat/edges/*.npz`
- Downloaded prebuilt permutations land under:
  - `data/downloads/hetionet-permutations/permutations/*.hetmat`
- Empirical edge-frequency CSVs (produced by `compute-edge-frequencies`) land in `results/empirical_edge_frequencies/`.

### Permutation sources
- `generate-permutations` builds local degree-preserved permutations in `data/permutations/`.
  - Supports `--count/--seed/--start`.
- `download-permutations` fetches the full prebuilt Hetionet bundle (about 200 permutations) into `data/downloads/hetionet-permutations/`.
  - This task does **not** support `--count`.
  - Current source ZIP size is ~`863 MB` (about `823 MiB`) before extraction.
  - Typical wall-clock estimate (download + extract):
    - fast links: ~`4-12` minutes
    - slower links: ~`15-35` minutes

Use-cases:
- Core null/compositional pipeline and most figure scripts read from `data/permutations/`.
- `compute-edge-frequencies` (notebook 3 workflow) reads downloaded permutations from `data/downloads/hetionet-permutations/permutations/`.

## Task runner
All pipelines are encoded as Poethepoet tasks. Run from repo root:
```bash
poe --help          # list tasks
poe <task-name>     # run a task
```

### Core data + null pipeline
1. `fetch-hetmat` – uses the active environment's `python` by default. If `data/` already contains a hetmat, it validates metanode/metaedge counts; if not, it downloads Hetionet v1.0 JSON (https://github.com/dhimmel/hetionet/raw/76550e6c93fbe92124edc71725e8c7dd4ca8b1f5/hetnet/json/hetionet-v1.0.json.bz2) and builds hetmat into `data/`. If needed, override by setting `CAPP_PY` before running poe tasks.
2. `generate-permutations` **or** `download-permutations` – generate missing degree-preserved permutations (default target 50; skips existing) or fetch prebuilt ones  
   - Parameters: `--count N` (total permutations desired, incl. existing; default 50), `--seed S` (base seed; default 42), `--start K` (force starting index; default = next unused).  
   - Examples: `poe generate-permutations --count 10`, `poe generate-permutations --count 60 --seed 123`, `poe generate-permutations --start 20 --count 25`.
3. `compute-edge-frequencies` – empirical edge frequencies (feeds compositional notebooks)
4. `train-null-models` – null model training (`results/null_models/`)
  - Defaults: train on permutations `1-20`, validate on `21-30`.
  - Uses `data/permutations/###.hetmat/edges/*.sparse.npz`.
  - Quick sample: `poe train-null-models --edge-type CbG --training-perm-end 2 --validation-perm-start 3 --validation-perm-end 4 --skip-empirical-validation`.
5. `compose-null` – compositional null fitting
  - Defaults: metapath `CbGpPW` (`CbG -> GpPW`) using validation permutations `21-30`.
  - Outputs to `results/compositional_null/` (validation CSVs, summary, optional plot/checkpoint).
  - Quick sample: `poe compose-null --validation-perm-start 1 --validation-perm-end 2 --max-pairs 5000 --skip-plot`.
6. `build-metapath-nulls` – metapath null distributions
  - Computes observed 2-edge metapath path probabilities and compares to compositional null predictions.
  - Defaults: metapaths `CbGpPW`, `CtDaG`, `CrCbG`, `CbGaD`; model types `rf` and `poly`.
  - Outputs to `results/metapath_nulls/`.
  - Quick sample: `poe build-metapath-nulls --metapath CbGpPW --model-type rf --max-pairs 5000 --skip-plot`.
7. `validate-composition` – compositional validation
  - Validates compositional predictions across held-out permutations for default 2-hop metapaths.
  - Outputs to `results/compositional_validation/` (`accuracy_by_metapath.csv`, `per_permutation_metrics.csv`, `validation_summary.json`, optional plots).
  - Quick sample: `poe validate-composition --metapath CbGpPW --train-perms-end 2 --valid-perms-start 3 --valid-perms-end 4 --max-compared-pairs 20000 --skip-plot`.
8. `analyze-composition-failures` – failure analysis
  - Runs stratified residual analysis by degree bins and reports where compositional predictions fail.
  - Outputs to `results/compositional_validation/` (`failure_analysis.csv`, `degree_stratified_correlations.csv`, `correction_analysis.csv`, optional plots).
  - Quick sample: `poe analyze-composition-failures --metapath CbGpPW --train-perms-end 2 --valid-perms-start 3 --valid-perms-end 4 --n-degree-bins 4 --samples-per-bin 20 --max-locations 10000 --skip-plot`.

### Manuscript model-comparison workflow 

1. `model-comparison-analysis` 
  - Main outputs under `results/model_comparison/<EDGE_TYPE>_results/`:
    - `model_comparison.csv`
    - `models_vs_analytical_comparison.csv`
    - `test_vs_empirical_comparison.csv` (if empirical frequencies exist)
    - `raw_logit_comparison.csv`
    - `probability_vs_raw_logit_comparison.csv`
    - optional all-pairs exports:
      - `<EDGE_TYPE>_all_model_predictions.csv(.gz)`
      - `<EDGE_TYPE>_predictions_by_degree.csv`
      - `<EDGE_TYPE>_predictions_metadata.json`
  - Quick sample (terminal smoke test):
    - `poe model-comparison-analysis --edge-type CtD --skip-plots --max-all-pairs 300000`
  - Notes:
    - All-pairs prediction export is guarded by `--max-all-pairs` (default `2,000,000`) to avoid huge files on dense edge types.
    - For dense edge types, leave defaults (auto-skip) or disable with `--no-generate-all-predictions`.

2. `model-testing-summary` 
  - Aggregates per-edge outputs from `results/model_comparison/*_results/`.
  - Writes summaries to `results/model_comparison_summary_with_degree/`, including:
    - `model_comparison_all_edges.csv`
    - `analytical_comparison_all_edges.csv`
    - `empirical_comparison_all_edges.csv`
    - `model_performance_summary.csv`
    - `graph_characteristics.csv`
    - `degree_analysis_summary.json`
  - Optional degree-analysis aggregation output:
    - `aggregate_degree_metrics.csv` (if degree metrics exist or `--run-degree-analysis` is enabled)
  - Quick sample:
    - `poe model-testing-summary --edge-type CtD --skip-plots`

### script-first notebook workflows

1. `pathway-data-preparation` (notebook 18a replacement)
  - Builds degree-binned training data for pathway-NN benchmarks.
  - Default metapath: `CbGpPW` (or run all defaults with `--all-metapaths`).
  - Outputs under `results/pathway_nn/training_data/`:
    - `<METAPATH>_training_data.csv`
    - `<METAPATH>_training_data_summary.json`
    - `pathway_data_preparation_run_summary.json`
  - Quick samples:
    - `poe pathway-data-preparation --metapath CbGpPW`
    - `poe pathway-data-preparation --all-metapaths`

2. `pathway-train-random` (notebook 18b replacement)
  - Trains random baseline on degree-binned pathway data and writes benchmark metadata.
  - Outputs:
    - `results/pathway_nn/trained_models/<METAPATH>_Random.pkl`
    - `results/pathway_nn/benchmarks/<METAPATH>_Random_benchmark.json`
    - `results/pathway_nn/benchmarks/pathway_train_random_run_summary.json`
  - Quick sample:
    - `poe pathway-train-random --metapath CbGpPW`

3. `pathway-train-degree-product` (notebook 18c replacement)
  - Trains degree-product baseline on degree-binned pathway data.
  - Uses notebook-parity synthetic edge-probability proxy columns for current compatibility.
  - Outputs:
    - `results/pathway_nn/trained_models/<METAPATH>_Degree_Product.pkl`
    - `results/pathway_nn/benchmarks/<METAPATH>_Degree_Product_benchmark.json`
    - `results/pathway_nn/benchmarks/pathway_train_degree_product_run_summary.json`
  - Quick sample:
    - `poe pathway-train-degree-product --metapath CbGpPW`

4. `pathway-train-negbin-glm` (notebook 18d replacement)
  - Trains Negative Binomial GLM baseline on degree-binned pathway data.
  - Uses statsmodels NegBin GLM when available; falls back to deterministic log-linear fit if needed.
  - Outputs:
    - `results/pathway_nn/trained_models/<METAPATH>_NegBin_GLM.pkl`
    - `results/pathway_nn/benchmarks/<METAPATH>_NegBin_GLM_benchmark.json`
    - `results/pathway_nn/benchmarks/pathway_train_negbin_glm_run_summary.json`
  - Quick sample:
    - `poe pathway-train-negbin-glm --metapath CbGpPW`

5. `pathway-train-random-forest` (notebook 18e replacement)
  - Trains random forest baseline on degree-binned pathway data.
  - Outputs:
    - `results/pathway_nn/trained_models/<METAPATH>_Random_Forest.pkl`
    - `results/pathway_nn/benchmarks/<METAPATH>_Random_Forest_benchmark.json`
    - `results/pathway_nn/benchmarks/pathway_train_random_forest_run_summary.json`
  - Quick sample:
    - `poe pathway-train-random-forest --metapath CbGpPW`

6. `pathway-train-degree-signature-nn` (notebook 18f replacement)
  - Trains the Degree Signature neural network on degree-bin pathway features.
  - Outputs:
    - `results/pathway_nn/trained_models/<METAPATH>_Degree_Sig_NN.pt`
    - `results/pathway_nn/benchmarks/<METAPATH>_Degree_Sig_NN_benchmark.json`
    - `results/pathway_nn/visualizations/<METAPATH>_Degree_Sig_NN_validation.png`
    - `results/pathway_nn/intermediate/<METAPATH>_test_*.npy`
    - `results/pathway_nn/benchmarks/pathway_train_degree_signature_nn_run_summary.json`
  - Quick sample:
    - `poe pathway-train-degree-signature-nn --metapath CbGpPW --skip-plots`

7. `pathway-variance-estimation` (notebook 18g replacement)
  - Validates the trained Degree Sig NN against permutation graphs and estimates per-bin variance.
  - If `--n-inter-bins` does not match the checkpoint input size, the task infers bins from the checkpoint and logs the adjustment.
  - Outputs:
    - `results/pathway_nn/variance_analysis/<METAPATH>_variance_estimates.csv`
    - `results/pathway_nn/variance_analysis/<METAPATH>_permutation_metrics.csv`
    - `results/pathway_nn/variance_analysis/<METAPATH>_validation_summary.json`
    - `results/pathway_nn/variance_analysis/permutation_<ID>_predictions.npy`
    - `results/pathway_nn/variance_analysis/all_permutations_results.npz`
    - `results/pathway_nn/variance_analysis/<METAPATH>_permutation_validation.png` (unless `--skip-plots`)
  - Quick sample:
    - `poe pathway-variance-estimation --metapath CbGpPW --n-permutations 2 --skip-plots`

8. `pathway-anomaly-detection` (notebook 18h replacement)
  - Computes positive anomalies (enrichment only), compares to DWPC, and exports ranked discoveries.
  - If `--n-inter-bins` does not match the checkpoint input size, the task infers bins from the checkpoint and logs the adjustment.
  - Outputs:
    - `results/pathway_nn/anomaly_detection/<METAPATH>_all_anomalies.csv`
    - `results/pathway_nn/anomaly_detection/<METAPATH>_significant_anomalies.csv`
    - `results/pathway_nn/anomaly_detection/<METAPATH>_novel_discoveries.csv`
    - `results/pathway_nn/anomaly_detection/<METAPATH>_anomaly_summary.json`
    - `results/pathway_nn/anomaly_detection/<METAPATH>_volcano_plot.png` (unless `--skip-plots`)
    - `results/pathway_nn/anomaly_detection/<METAPATH>_dwpc_comparison.png` (unless `--skip-plots`)
    - `results/pathway_nn/anomaly_detection/<METAPATH>_anomaly_distributions.png` (unless `--skip-plots`)
  - Quick sample:
    - `poe pathway-anomaly-detection --metapath CbGpPW --max-pairs 5000 --skip-plots`

9. `learned-analytical` (notebook 8 replacement)
  - Trains/evaluates learned analytical formula variants and degree diagnostics.
  - Outputs under `results/learned_formula/`.
  - Quick sample:
    - `poe learned-analytical --edge-type CtD --n-candidates 2 3 5 --skip-comparison-plot`

10. `degree-conditioned-compositionality` (notebook 11.x replacement)
  - Degree-conditioned compositional analysis (Option A style) for a selected 2-hop metapath.
  - Outputs under `results/compositionality/`.
  - Quick sample:
    - `poe degree-conditioned-compositionality --metapath CbGpPW --perm-start 1 --perm-end 2 --max-pairs 5000 --skip-plots`

11. `degree-aware-compositional-model` (notebook 12 replacement)
  - Compares naive vs continuous degree-aware compositional predictions.
  - Outputs under `results/compositionality/degree_aware/`.
  - Quick sample:
    - `poe degree-aware-compositional-model --metapath CbGpPW --perm-start 1 --perm-end 2 --max-pairs 5000 --skip-plots`

12. `nn-architecture-exploration` (notebook 21 replacement)
  - Runs script-first optimizer/loss/architecture tests and saves analysis artifacts.
  - Outputs under `results/nn_optimizer_comparison/`.
  - Quick sample:
    - `poe nn-architecture-exploration --tests 1,2 --max-samples 5000 --max-epochs-linear 5 --patience-linear 3 --skip-plots --no-save-pkl`

### Figures & diagnostics
- `make-pathcount-heatmaps` – path-count variance figures (`results/path_count_visualization/`)
- `assess-length-effects` – length degradation analysis (`results/length_degradation/`)
- `assess-sparsity` – sparsity effects analysis (`results/hierarchical_prediction/`)
- `assess-topology` – topology/outlier diagnostics (`results/topology_specific_outliers/`)
- All four scripts support `--help` and `--smoke`, and automatically use available local permutations when higher IDs are missing.

### Model and comparison analyses
- `baseline-pair-level` – baseline pair-level
- `pair-level-degree-correction` – degree-aware correction
- `feature-comparison` – feature/binning comparisons
- `control-experiments` – control experiments
- `linear-model-cv` – linear CV
- `degree-aware-correction-eval`
- `bias-diagnostics`
- `theoretical-correction-eval`
- `regularization-study`
- `gnn-variant-comparison` – GNN/multitask tests (defaults to `CbGpPW`)
- `focused-composition-tests` – focused composition tests
- `perm000-vs-permuted-comparison` – perm000 vs perms comparison (supports `--smoke`)

### Optional validation
- `validate-dwpc` – DWPC p-value validation suite
- `smoke-figures` – sequence task that runs smoke checks for all four figure/diagnostic scripts
- `validate-figures` – runs `smoke-figures` plus `perm000-vs-permuted-comparison --smoke --skip-plots`
- `ci-smoke` – CI-oriented smoke sequence (`validate-figures` + `quick-baseline-pair-level`)
- `quick-test-end-to-end` – smoke-scale end-to-end run across figures + model/comparison suite + focused composition
- `reproduce-figures-full` – full figure/diagnostic/model-comparison reproduction suite

## Suggested end-to-end run
```bash
poe fetch-hetmat
poe generate-permutations   # or download-permutations
poe compute-edge-frequencies
poe train-null-models
poe compose-null
poe build-metapath-nulls
poe validate-composition
poe analyze-composition-failures
poe model-comparison-analysis --edge-type CtD --skip-plots --max-all-pairs 300000
poe model-testing-summary --edge-type CtD --skip-plots
poe pathway-data-preparation --metapath CbGpPW
poe pathway-train-random --metapath CbGpPW
poe pathway-train-degree-product --metapath CbGpPW
poe pathway-train-negbin-glm --metapath CbGpPW
poe pathway-train-random-forest --metapath CbGpPW
poe pathway-train-degree-signature-nn --metapath CbGpPW --skip-plots
poe pathway-variance-estimation --metapath CbGpPW --n-permutations 2 --skip-plots
poe pathway-anomaly-detection --metapath CbGpPW --max-pairs 5000 --skip-plots
poe learned-analytical --edge-type CtD --n-candidates 2 3 5 --skip-comparison-plot
poe degree-conditioned-compositionality --metapath CbGpPW --perm-start 1 --perm-end 2 --max-pairs 5000 --skip-plots
poe degree-aware-compositional-model --metapath CbGpPW --perm-start 1 --perm-end 2 --max-pairs 5000 --skip-plots
poe nn-architecture-exploration --tests 1,2 --max-samples 5000 --max-epochs-linear 5 --patience-linear 3 --skip-plots --no-save-pkl
# Figures / model-comparison analyses as needed
```


## Manuscript Figure Reproduction 
This is the canonical, script-backed set currently implemented for manuscript draft reproduction.

Prerequisites:
- Run from repo root with `CAPP` active.
- Ensure permutations exist under `data/permutations/` (for manuscript-faithful runs, keep `000-020` available).

Run all currently implemented manuscript repro tasks:
```bash
poe repro-script-backed
```

Run in manuscript order (script-backed subset):
```bash
poe repro-manuscript-v1
```

Individual commands and outputs:
- Figure 2 (`CbGpPWpG` heatmap):
  - Command: `poe repro-fig2-pathcount-heatmap`
  - Output: `results/path_count_visualization/CbGpPWpG_path_count_heatmap.png`
- Figure 3 (model failures):
  - Command: `poe repro-fig3-model-failures`
  - Output: `results/model_failures/model_failure_analysis.png`
- Figure 4 (permutation similarity):
  - Command: `poe repro-fig4-permutation-similarity`
  - Output: `results/permuations_similarlity/AeG_permutation_similarity.png`
- Figure 13 (variance vs PMI):
  - Command: `poe repro-fig13-variance-pmi`
  - Output dir: `results/variance_pmi/`
- Figure 14 (perm0 vs perm-mean topology outliers):
  - Command: `poe repro-fig14-topology-outliers`
  - Output dir: `results/topology_specific_outliers/`
- Table 1 (count prediction performance):
  - Command: `poe repro-table1-count-prediction`
  - Output: `results/model_comparison/table1_count_prediction.csv`
- Figure 15:
  - Command: `poe repro-fig3-model-failures` (same backend/output as Figure 3)
  - Output: `results/model_failures/model_failure_analysis.png`
- Figure 16 (z-score + QQ calibration):
  - Command: `poe repro-fig16-zscore-qq`
  - Output dir: `results/model_comparison/qq_and_zscore/`

## Notes
- Heavy tasks may require HPC resources; adjust scripts accordingly.
- Superseded shell wrappers are archived under `archive/scripts_legacy/`.
- Legacy docs have been moved to `archive/docs/` and will be deleted once reproducibility is confirmed.
- PYTHONPATH is set by poe tasks to include repo root for `src/` imports.

# AI Assistance
This project utilized the AI assistants Claude and ChatGPT, developed by Anthropic and OpenAI, during the development process. Its assistance included generating initial code snippets and improving documentation. All AI-generated content was reviewed, tested, and validated by human developers.
