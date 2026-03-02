#!/usr/bin/env bash
set -euo pipefail

# Script-backed subset for manuscript v1 reproducibility.
# Pending wrappers (Figures 5-12) should be added as they are implemented.

poe repro-fig2-pathcount-heatmap
poe repro-fig3-model-failures
poe repro-fig4-permutation-similarity
poe repro-fig13-variance-pmi
poe repro-fig14-topology-outliers
poe repro-table1-count-prediction
poe repro-fig16-zscore-qq
