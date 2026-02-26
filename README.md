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
- Hetionet hetmat and permutations expected under `data/`:
  - Base graph: `data/edges/*.sparse.npz`
  - Perms: `data/permutations/###.hetmat/edges/*.npz`
- Empirical edge-frequency CSVs (produced by `compute-edge-frequencies`) land in `results/empirical_edge_frequencies/`.

## Task runner
All pipelines are encoded as Poethepoet tasks. Run from repo root:
```bash
poetry run poe --help          # list tasks
poetry run poe <task-name>     # run a task
```

### Core data + null pipeline
1. `fetch-hetmat` – uses the CAPP env python (default path in pyproject). If `data/` already contains a hetmat, it validates metanode/metaedge counts; if not, it downloads Hetionet v1.0 JSON (https://github.com/dhimmel/hetionet/raw/76550e6c93fbe92124edc71725e8c7dd4ca8b1f5/hetnet/json/hetionet-v1.0.json.bz2) and builds hetmat into `data/`. If your env lives elsewhere, set `CAPP_PY` to your python path before running poe tasks.
2. `generate-permutations` **or** `download-permutations` – generate missing degree-preserved permutations (default target 50; skip existing) or fetch prebuilt ones
3. `compute-edge-frequencies` – empirical edge frequencies (feeds compositional notebooks)
4. `train-null-models` – null model training
5. `compose-null` – compositional null fitting
6. `build-metapath-nulls` – metapath null distributions
7. `validate-composition` – compositional validation
8. `analyze-composition-failures` – failure analysis

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
- Figures/phase scripts executed as needed
