"""
Benchmark different approaches for finding intermediates.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import time

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'

def load_edge_matrix(edge_abbrev, perm_num):
    edge_file = data_dir / 'permutations' / f'{perm_num:03d}.hetmat' / 'edges' / f'{edge_abbrev}.sparse.npz'
    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)
    return None

print("Loading perm 0...")
CbG_0 = load_edge_matrix('CbG', 0)
GiG_0 = load_edge_matrix('GiG', 0)
GpPW_0 = load_edge_matrix('GpPW', 0)

CbGiG_0 = CbG_0 @ GiG_0
CbGiGpPW_0 = CbGiG_0 @ GpPW_0

# Sample pairs
CbGiGpPW_0_coo = CbGiGpPW_0.tocoo()
n_sample = 1000
sample_idx = np.random.choice(len(CbGiGpPW_0_coo.data), n_sample, replace=False)
pairs = [(CbGiGpPW_0_coo.row[i], CbGiGpPW_0_coo.col[i]) for i in sample_idx]

print(f"\nBenchmarking on {n_sample} pairs...\n")

# Approach 1: Original (list comprehension)
print("Approach 1: Original list comprehension")
CbGiG_0_lil = CbGiG_0.tolil()
GpPW_0_csr = GpPW_0.tocsr()

start = time.time()
intermediates_1 = []
for C, PW in pairs:
    genes_to_PW = GpPW_0_csr[:, PW].nonzero()[0]
    intermediates = [G for G in genes_to_PW if CbGiG_0_lil[C, G] > 0]
    intermediates_1.append(intermediates)
elapsed_1 = time.time() - start
print(f"  Time: {elapsed_1:.3f} seconds")
print(f"  Per pair: {1000*elapsed_1/n_sample:.2f} ms")

# Approach 2: Sparse row-column multiplication
print("\nApproach 2: Sparse row-column multiplication")
CbGiG_0_csr = CbGiG_0.tocsr()

start = time.time()
intermediates_2 = []
for C, PW in pairs:
    row_C = CbGiG_0_csr[C, :]
    col_PW = GpPW_0_csr[:, PW]
    intersection = row_C.multiply(col_PW.T)
    intermediates = intersection.nonzero()[1]
    intermediates_2.append(intermediates)
elapsed_2 = time.time() - start
print(f"  Time: {elapsed_2:.3f} seconds")
print(f"  Per pair: {1000*elapsed_2/n_sample:.2f} ms")
print(f"  Speedup: {elapsed_1/elapsed_2:.2f}x")

# Approach 3: Boolean masking with dense conversion (for small intermediates)
print("\nApproach 3: Boolean masking (dense)")
CbGiG_0_bool = CbGiG_0.astype(bool).tocsr()
GpPW_0_bool = GpPW_0.astype(bool).tocsr()

start = time.time()
intermediates_3 = []
for C, PW in pairs:
    mask_C = CbGiG_0_bool[C, :].toarray().flatten()
    mask_PW = GpPW_0_bool[:, PW].toarray().flatten()
    intermediate_mask = mask_C & mask_PW
    intermediates = np.where(intermediate_mask)[0]
    intermediates_3.append(intermediates)
elapsed_3 = time.time() - start
print(f"  Time: {elapsed_3:.3f} seconds")
print(f"  Per pair: {1000*elapsed_3/n_sample:.2f} ms")
print(f"  Speedup: {elapsed_1/elapsed_3:.2f}x")

# Approach 4: Using sets (for sparse intermediates)
print("\nApproach 4: Set intersection")
start = time.time()
intermediates_4 = []
for C, PW in pairs:
    genes_from_C = set(CbGiG_0_csr[C, :].nonzero()[1])
    genes_to_PW = set(GpPW_0_csr[:, PW].nonzero()[0])
    intermediates = list(genes_from_C & genes_to_PW)
    intermediates_4.append(intermediates)
elapsed_4 = time.time() - start
print(f"  Time: {elapsed_4:.3f} seconds")
print(f"  Per pair: {1000*elapsed_4/n_sample:.2f} ms")
print(f"  Speedup: {elapsed_1/elapsed_4:.2f}x")

# Verify all methods give same results
print("\n" + "="*60)
print("Verification: All methods produce identical results?")
for i in range(min(10, n_sample)):
    s1 = set(intermediates_1[i])
    s2 = set(intermediates_2[i])
    s3 = set(intermediates_3[i])
    s4 = set(intermediates_4[i])
    if not (s1 == s2 == s3 == s4):
        print(f"  MISMATCH at pair {i}: {s1} vs {s2} vs {s3} vs {s4}")
        break
else:
    print("  All methods match (checked first 10 pairs)")

# Summary
print("\n" + "="*60)
print("Summary:")
print(f"  Original:            {elapsed_1:.3f}s  (1.00x)")
print(f"  Sparse multiply:     {elapsed_2:.3f}s  ({elapsed_1/elapsed_2:.2f}x)")
print(f"  Boolean masking:     {elapsed_3:.3f}s  ({elapsed_1/elapsed_3:.2f}x)")
print(f"  Set intersection:    {elapsed_4:.3f}s  ({elapsed_1/elapsed_4:.2f}x)")

best_time = min(elapsed_1, elapsed_2, elapsed_3, elapsed_4)
best_method = ['Original', 'Sparse multiply', 'Boolean masking', 'Set intersection'][
    [elapsed_1, elapsed_2, elapsed_3, elapsed_4].index(best_time)
]
print(f"\nBest: {best_method} ({best_time:.3f}s)")

# Estimate for full 10k pairs
print(f"\nEstimated time for 10,000 pairs:")
print(f"  Original:        {10 * elapsed_1:.1f}s = {10*elapsed_1/60:.1f} min")
print(f"  Best method:     {10 * best_time:.1f}s = {10*best_time/60:.1f} min")
print(f"  Savings:         {10*(elapsed_1 - best_time):.1f}s = {10*(elapsed_1-best_time)/60:.1f} min")
