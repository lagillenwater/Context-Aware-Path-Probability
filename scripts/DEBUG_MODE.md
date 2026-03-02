# Debug Mode for Pipeline 18

## Overview

The degree signature pipeline (scripts 18a-18i) supports a debug mode that allows you to test the entire pipeline on a single metapath instead of all 8 metapaths. This is useful for:

- Quick testing of changes before full pipeline runs
- Debugging errors in specific metapaths
- Faster iteration during development
- Resource conservation during testing

## Usage

### Production Mode (All 8 Metapaths)

```bash
cd /projects/$USER/repositories/Context-Aware-Path-Probability/scripts
bash 18_submit_all.sh
```

This will submit 8 array jobs (one task per metapath) for each pipeline step.

**Resource usage:**
- Data prep: 8 tasks × 6h = 48 node-hours @ 64GB
- Training: 40 node-hours @ 16-32GB (5 models × 8 tasks)
- Validation: 8 node-hours @ 16GB
- **Total: ~102.5 node-hours**

**Wall clock time:** ~15-17 hours (parallel execution within each step)

### Debug Mode (Single Metapath)

```bash
cd /projects/$USER/repositories/Context-Aware-Path-Probability/scripts
bash 18_submit_all.sh --debug CbGpPW
```

This will submit single-task jobs for each pipeline step, all processing only the specified metapath.

**Resource usage:**
- Data prep: 1 task × 6h = 6 node-hours @ 64GB
- Training: 5 node-hours @ 16-32GB (5 models × 1 task)
- Validation: 1 node-hour @ 16GB
- **Total: ~12.5 node-hours**

**Wall clock time:** ~8-10 hours (sequential steps)

## Valid Metapaths

Use any of these 8 metapaths with `--debug`:

- `CbGpPW` - Compound-binds-Gene-participates-Pathway
- `CtDaG` - Compound-treats-Disease-associates-Gene
- `CbGaD` - Compound-binds-Gene-associates-Disease
- `CrCbG` - Compound-regulates-Compound-binds-Gene
- `CbGiG` - Compound-binds-Gene-interacts-Gene
- `CpDaG` - Compound-palliates-Disease-associates-Gene
- `CbGpBP` - Compound-binds-Gene-participates-BiologicalProcess
- `CbGpCC` - Compound-binds-Gene-participates-CellularComponent

## How It Works

When you run `bash 18_submit_all.sh --debug METAPATH`:

1. **Master script** sets debug mode variables:
   - `ARRAY_SPEC="--array=1"` (single task instead of 8)
   - `EXPORT_VARS="--export=ALL,DEBUG_METAPATH=$METAPATH"` (passes metapath to jobs)

2. **Individual scripts** detect `DEBUG_METAPATH` environment variable:
   - If set: Use the specified metapath (ignoring array task ID)
   - If not set: Use array task ID to select metapath from list

3. **Job dependencies** remain the same:
   ```
   18a Data Prep
     ↓
   18b-f Training (5 parallel jobs)
     ↓
   18g CV Validation
     ↓
   18i Benchmark Summary
   ```

## Example: Testing Changes

If you modified the DegreeSignatureNN model and want to test it quickly:

```bash
# Run debug mode with CbGpPW (a well-behaved metapath)
bash 18_submit_all.sh --debug CbGpPW

# Monitor progress
squeue -u $USER

# Check logs
tail -f ../logs/18a_data_prep_*.out
tail -f ../logs/18f_train_nn_*.out

# View results (after completion)
ls -lh ../results/pathway_nn/trained_models/CbGpPW_*
cat ../results/pathway_nn/benchmarks/CbGpPW_Degree_Signature_NN_benchmark.json
```

## Comparison: Debug vs Production

| Metric | Production | Debug (CbGpPW) | Speedup |
|--------|-----------|----------------|---------|
| **Array tasks** | 8 per job | 1 per job | 8× |
| **Node-hours** | ~102.5 | ~12.5 | 8× |
| **Wall clock** | ~15-17h | ~8-10h | ~2× |
| **Output files** | 40 models + 8 validations | 5 models + 1 validation | 8× |

Debug mode gives you ~8× resource savings and ~2× faster wall clock time (because steps are parallelized in production but sequential in debug).

## Troubleshooting

### Error: Unknown metapath

```
ERROR: --debug requires a metapath argument
Valid metapaths: CbGpPW, CtDaG, CbGaD, CrCbG, CbGiG, CpDaG, CbGpBP, CbGpCC
```

**Fix:** Provide a valid metapath after `--debug` flag.

### Jobs still running for all metapaths

Check that you used `--debug` (not `--test` or other flag):

```bash
bash 18_submit_all.sh --debug CbGpPW  # Correct
bash 18_submit_all.sh -d CbGpPW       # Wrong
```

### Array task ID errors in logs

This usually means `DEBUG_METAPATH` wasn't properly exported. Verify:

```bash
# In any job's .out file, you should see:
# DEBUG MODE: Using metapath CbGpPW
```

If you don't see this message, the environment variable wasn't passed. Check that you're using the latest `18_submit_all.sh` script.

## Implementation Details

### Modified Scripts

All of these scripts support debug mode:

- `18_submit_all.sh` - Master orchestration (parses `--debug` flag)
- `18a_data_preparation.sh` - Data prep (detects `DEBUG_METAPATH`)
- `18b_train_random.sh` - Random baseline training
- `18c_train_degree_product.sh` - Degree product training
- `18d_train_negbin_glm.sh` - NegBin GLM training
- `18e_train_random_forest.sh` - Random forest training
- `18f_train_degree_signature_nn.sh` - Degree signature NN training
- `18g_validate_cv_all_models.sh` - Cross-validation (when created)

### Code Pattern

Each script checks for `DEBUG_METAPATH` at the top:

```bash
declare -a METAPATHS=("CbGpPW" "CtDaG" ...)

# DEBUG MODE: Override metapath selection if DEBUG_METAPATH is set
if [ -n "$DEBUG_METAPATH" ]; then
    echo "DEBUG MODE: Using metapath $DEBUG_METAPATH"
    METAPATH="$DEBUG_METAPATH"
else
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    METAPATH=${METAPATHS[$IDX]}
fi
```

This pattern ensures backward compatibility: existing production runs continue to work without changes.

## Notes

- **17b not included:** The baseline failure analysis (17b) is run separately and is not part of the 18 pipeline
- **Summary job (18i):** Always runs as a single job (no array), even in production mode
- **Job IDs:** Saved to `job_ids.txt` for monitoring
- **Logs:** Located in `../logs/` with job ID and array task ID in filename

## References

- Main pipeline documentation: [README_18_PIPELINE.md](README_18_PIPELINE.md)
- SLURM job arrays: https://slurm.schedmd.com/job_array.html
- Alpine HPC docs: https://curc.readthedocs.io/
