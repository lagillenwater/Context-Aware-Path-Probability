#!/usr/bin/env python3
"""
Test the updated optimized notebook path structure.
"""

import sys
from pathlib import Path
import os

# Test the path structure that will be used in the notebook
print("Testing notebook path structure...")

# Current working directory (should be repo root when run from HPC)
repo_dir = Path.cwd()
data_dir = repo_dir / 'data'
null_models_dir = repo_dir / 'results' / 'null_models'
results_dir = repo_dir / 'results' / 'compositional_null'

print(f"Repository directory: {repo_dir}")
print(f"Data directory: {data_dir}")
print(f"Null models directory: {null_models_dir}")
print(f"Results directory: {results_dir}")

# Check if key directories exist
print("\nDirectory existence check:")
print(f"  Data: {data_dir.exists()} ✓" if data_dir.exists() else f"  Data: {data_dir.exists()} ❌")
print(f"  Null models: {null_models_dir.exists()} ✓" if null_models_dir.exists() else f"  Null models: {null_models_dir.exists()} ❌")

# Check for key model files
if null_models_dir.exists():
    cbg_model = null_models_dir / 'CbG_rf_null.pkl'
    gppw_model = null_models_dir / 'GpPW_rf_null.pkl'
    print(f"  CbG model: {cbg_model.exists()} ✓" if cbg_model.exists() else f"  CbG model: {cbg_model.exists()} ❌")
    print(f"  GpPW model: {gppw_model.exists()} ✓" if gppw_model.exists() else f"  GpPW model: {gppw_model.exists()} ❌")

print("\n✅ Path structure test complete!")
print("\nNote: This matches the cluster path structure where HPC jobs")
print("run from the repository root directory.")