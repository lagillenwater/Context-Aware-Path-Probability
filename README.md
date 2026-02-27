# Context-Aware-Path-Probability

Reproducible pipelines for null-distribution and path-probability experiments on biomedical knowledge graphs (Hetionet permutations and related metapaths). This README is the single source for running everything end-to-end via Poetry + Poethepoet.

## Environments

You can use either Poetry or the existing conda environment. Pick one and stick with it for a session.

- **Conda (recommended on this repo)**
  ```bash
  # from repo root
  cd environments
  conda env create -f environment.yml    # first time only
  conda activate CAPP
  cd ..                                   # back to repo root
  ```

- **Poetry (alternative; useful for poe tasks)**
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -   # if poetry not installed
  poetry install
  # activate a shell with the venv
  poetry shell
  ```

> Tip: If you use conda, the `poe` runner is available via `conda run -n CAPP poe ...` or after `conda activate CAPP` if poethepoet is installed in that env.

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
poetry run poe --help          # list tasks
poetry run poe <task-name>     # run a task
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

### Figures & diagnostics
- `make-pathcount-heatmaps` – path-count variance figures
- `assess-length-effects` – length degradation scripts
- `assess-sparsity` – sparsity effects
- `assess-topology` – topology/outlier diagnostics

### Phase experiments
- `run-phase1` – baseline pair-level
- `run-phase2` – degree-aware corrections
- `run-phase3` – feature/binning comparisons
- `run-phase4` – control experiments
- `run-phase5` – linear CV; variants:
  - `run-phase5b-degree-aware`
  - `run-phase5b-bias`
  - `run-phase5b-theoretical`
  - `run-phase5c-regularization`
- `run-gnn-variants` – GNN/multitask tests
- `test-composition-focused` – focused composition experiments
- `compare-perm0` – perm000 vs perms comparison

### Optional validation
- `validate-dwpc` – DWPC p-value validation suite
- `smoke-figures` – quick wiring check (small subsets)

## Suggested end-to-end run
```bash
poetry run poe fetch-hetmat
poetry run poe generate-permutations   # or download-permutations
poetry run poe compute-edge-frequencies
poetry run poe train-null-models
poetry run poe compose-null
poetry run poe build-metapath-nulls
poetry run poe validate-composition
poetry run poe analyze-composition-failures
# A2 model-comparison (notebook 4/5 replacement)
poetry run poe model-comparison-analysis --edge-type CtD --skip-plots --max-all-pairs 300000
poetry run poe model-testing-summary --edge-type CtD --skip-plots
# Figures / phases as needed
```

## Notes
- Heavy tasks may require HPC resources; adjust scripts accordingly.
- Legacy docs have been moved to `archive/docs/` and will be deleted once reproducibility is confirmed.
- PYTHONPATH is set by poe tasks to include repo root for `src/` imports.

## Checklist (top-level)
- Data present (hetmat + permutations)
- Empirical edge frequencies generated
- Null/compositional models trained
- A2 model-comparison outputs generated (`model-comparison-analysis` + `model-testing-summary`)
- Figures/phase scripts executed as needed
