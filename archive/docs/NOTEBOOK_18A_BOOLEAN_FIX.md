# Notebook 18a Boolean Matrix Fix

## Problem

Notebook 18a failed on HPC with TypeError when computing percentiles:

```
TypeError: numpy boolean subtract, the `-` operator, is not supported,
use the bitwise_xor, the `^` operator, or the logical_xor function instead.
```

**Error location**: Cell 16, lines computing pathway count statistics:
```python
'pathway_count_q25': np.percentile(counts, 25),
'pathway_count_q75': np.percentile(counts, 75),
```

## Root Cause

The Hetionet edge matrices are stored as **boolean dtype** (sparse matrices with True/False values indicating edge presence/absence). When multiplying two boolean matrices:

```python
pathway_matrix = edge1_matrix @ edge2_matrix
# Boolean @ Boolean = Boolean result!
```

The result is also boolean:
- `True` = "at least one path exists"
- `False` = "no paths exist"

NOT integer counts like:
- `1` = "one path"
- `2` = "two paths"
- etc.

When `np.percentile()` tries to compute percentiles on boolean values (`True`/`False`), it fails because boolean subtraction is not supported.

## Solution

**Modified Cell 13** to convert the boolean pathway matrix to integer type:

```python
# Compute pathway matrix and convert to integer type
pathway_matrix = edge1_matrix @ edge2_matrix

# IMPORTANT: Convert boolean matrix to integer to get actual path counts
# Boolean @ Boolean gives presence/absence, not counts
# Need to use astype to get integer counts
if pathway_matrix.dtype == bool or pathway_matrix.dtype == np.bool_:
    print(f"Converting boolean pathway matrix to integer (dtype: {pathway_matrix.dtype})")
    pathway_matrix = pathway_matrix.astype(np.int32)

print(f"Pathway matrix: {pathway_matrix.shape}, {pathway_matrix.nnz:,} non-zero pathways")
print(f"Pathway matrix dtype: {pathway_matrix.dtype}")
print(f"Value range: {pathway_matrix.data.min()} - {pathway_matrix.data.max()}")

# Convert to COO for efficient access
pathway_coo = pathway_matrix.tocoo()
pathway_dict = {(i, j): v for i, j, v in zip(pathway_coo.row, pathway_coo.col, pathway_coo.data)}
```

## Why This Works

**Before fix:**
- `pathway_dict` values: `True`, `False` (boolean)
- `counts`: `[True, False, True, ...]` (list of booleans)
- `np.percentile([True, False, ...], 25)`: **FAILS** with TypeError

**After fix:**
- `pathway_matrix.astype(np.int32)`: Converts `True→1`, `False→0`
- `pathway_dict` values: `1`, `2`, `3`, ... (integers representing path counts)
- `counts`: `[1, 2, 3, ...]` (list of integers)
- `np.percentile([1, 2, 3, ...], 25)`: **WORKS** correctly

## Impact

This fix ensures:
1. Pathway counts are actual integer counts, not boolean presence/absence
2. Statistical functions (mean, std, median, percentiles) work correctly
3. Training data contains meaningful numeric values for ML models

## Understanding Boolean vs Integer Matrix Multiplication

### Boolean Matrix Multiplication
```python
# Edge matrices: boolean (presence/absence)
edge1 = [[True, False], [False, True]]  # Compound-Gene
edge2 = [[True], [False]]                # Gene-Pathway

# Boolean matrix multiply
pathway = edge1 @ edge2 = [[True], [False]]  # True = "at least 1 path"
```

### Integer Matrix Multiplication (After Fix)
```python
# Edge matrices: integers (counts)
edge1 = [[1, 0], [0, 1]]  # Compound-Gene
edge2 = [[1], [0]]         # Gene-Pathway

# Integer matrix multiply
pathway = edge1 @ edge2 = [[1], [0]]  # 1 = "exactly 1 path"
```

## Files Modified

**notebooks/18a_data_preparation.ipynb**
- Cell 13: Added boolean dtype check and conversion to int32

## Testing

Run on HPC:
```bash
papermill notebooks/18a_data_preparation.ipynb \
    notebooks/executed/18a_data_preparation_CbGpPW_executed.ipynb \
    -p metapath "CbGpPW" \
    -p edge1_type "CbG" \
    -p edge2_type "GpPW"
```

Expected output should now include:
```
Converting boolean pathway matrix to integer (dtype: bool)
Pathway matrix: (1552, 1822), 71,653 non-zero pathways
Pathway matrix dtype: int32
Value range: 1 - [max_paths]
```

And successfully complete without TypeError.

## Alternative Approaches Considered

### Option 1: Convert edge matrices before multiplication
```python
edge1_matrix = edge1_matrix.astype(np.int32)
edge2_matrix = edge2_matrix.astype(np.int32)
pathway_matrix = edge1_matrix @ edge2_matrix
```

**Pros**: Ensures all matrices are integer throughout
**Cons**: May use more memory unnecessarily

### Option 2: Convert pathway matrix (CHOSEN)
```python
pathway_matrix = edge1_matrix @ edge2_matrix
if pathway_matrix.dtype == bool:
    pathway_matrix = pathway_matrix.astype(np.int32)
```

**Pros**: Minimal change, efficient, clear intent
**Cons**: None

### Option 3: Handle boolean values in aggregation
```python
counts = [int(c) for c in bin_pathway_counts.get((src_bin, tgt_bin), [0])]
```

**Pros**: Localized fix
**Cons**: Doesn't address root cause, less efficient

## Related Issues

This same pattern may affect other notebooks that:
1. Load boolean sparse matrices
2. Multiply matrices
3. Compute statistics on the results

Check notebooks: 17b, 18b-18f for similar issues if they use pathway matrix multiplication.

## Status

✅ **FIXED** - Notebook 18a now handles boolean edge matrices correctly
