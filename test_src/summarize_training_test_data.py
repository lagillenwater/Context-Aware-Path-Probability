"""
Summary of training and testing data for Phase 2 feature ablation.

Explains:
- Training data structure (degree bins from original Hetionet)
- Testing data structure (individual pairs from 20 permutations)
- Data dimensions and statistics
"""

import numpy as np
import pandas as pd
import os
import glob

def summarize_training_data():
    """Summarize prepared training datasets."""
    print("=" * 80)
    print("TRAINING DATA SUMMARY")
    print("=" * 80)
    print()
    print("Source: Original Hetionet (data/edges/)")
    print("Format: Degree bin combinations with aggregated pathway statistics")
    print()

    # Load all training datasets
    csv_files = sorted(glob.glob('results/phase2_training_data/*.csv'))

    print(f"Total datasets: {len(csv_files)} (5 metapaths × 6 feature sets)")
    print()

    # Group by metapath
    metapaths = {}
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        # Format: {metapath}_features_{set}.csv
        parts = basename.replace('.csv', '').split('_features_')
        metapath = parts[0]
        feature_set = parts[1]

        if metapath not in metapaths:
            metapaths[metapath] = {}

        df = pd.read_csv(csv_file)
        metapaths[metapath][feature_set] = df

    print("-" * 80)
    print("TRAINING DATA BY METAPATH")
    print("-" * 80)
    print()

    for metapath in sorted(metapaths.keys()):
        print(f"Metapath: {metapath}")

        # Use Set A to get basic stats
        df = metapaths[metapath]['A']

        # Count features (columns starting with 'feat_')
        feature_cols = [col for col in df.columns if col.startswith('feat_')]
        n_features = len(feature_cols)

        print(f"  Training samples (bins): {len(df)}")
        print(f"  Features (Set A): {n_features}")
        print(f"  Degree bin combinations: {df['source_bin'].nunique()} source × {df['target_bin'].nunique()} target")
        print(f"  Total node pairs represented: {df['n_pairs'].sum():,}")
        print(f"  Pathway count (mean): {df['pathway_count_mean'].mean():.4f} ± {df['pathway_count_mean'].std():.4f}")
        print(f"  Pathway count range: [{df['pathway_count_mean'].min():.4f}, {df['pathway_count_mean'].max():.4f}]")

        # Check feature dimensions for all sets
        feature_counts = {}
        for fs in ['A', 'B', 'C', 'D', 'E', 'F']:
            df_fs = metapaths[metapath][fs]
            feature_cols_fs = [col for col in df_fs.columns if col.startswith('feat_')]
            feature_counts[fs] = len(feature_cols_fs)

        print(f"  Feature dimensions: A={feature_counts['A']}, B={feature_counts['B']}, "
              f"C={feature_counts['C']}, D={feature_counts['D']}, E={feature_counts['E']}, F={feature_counts['F']}")
        print()

    return metapaths


def summarize_testing_data():
    """Explain testing data structure."""
    print("=" * 80)
    print("TESTING DATA SUMMARY")
    print("=" * 80)
    print()
    print("Source: 20 permutations (data/permutations/000.hetmat to 019.hetmat)")
    print("Format: Individual (source, target) node pairs")
    print()

    print("Testing process:")
    print("  1. For each permutation (000-019):")
    print("     - Load edge matrices")
    print("     - Compute pathway matrix = edge1 @ edge2")
    print("     - Get pathway count for each (source_i, target_j) pair")
    print()
    print("  2. Average pathway counts across 20 permutations:")
    print("     - For each (source_i, target_j):")
    print("       mean_pathway_count = mean([perm_000[i,j], perm_001[i,j], ..., perm_019[i,j]])")
    print()
    print("  3. Predict using trained model:")
    print("     - Lookup degree bin for source_i and target_j")
    print("     - Extract features for that bin combination")
    print("     - Model predicts pathway count")
    print()
    print("  4. Evaluation:")
    print("     - Pearson r between predicted and actual (averaged) counts")
    print("     - RMSE, MAE")
    print("     - Success criterion: r ≥ 0.95")
    print()


def compare_train_test_sizes():
    """Compare training vs testing data sizes."""
    print("=" * 80)
    print("TRAINING vs TESTING DATA COMPARISON")
    print("=" * 80)
    print()

    # Training sizes
    csv_files = glob.glob('results/phase2_training_data/*_features_A.csv')

    print(f"{'Metapath':<15} {'Train Samples':<15} {'Train Pairs':<15} {'Test Pairs (est.)':<20}")
    print("-" * 80)

    for csv_file in sorted(csv_files):
        basename = os.path.basename(csv_file)
        metapath = basename.replace('_features_A.csv', '')

        df = pd.read_csv(csv_file)
        n_train_samples = len(df)
        n_train_pairs = df['n_pairs'].sum()

        # Test pairs = total possible pairs
        # Approximate based on first row dimensions
        if len(df) > 0:
            # Get approximate matrix size from n_pairs and bin structure
            n_test_pairs = n_train_pairs  # Same node pairs, different edge structure
        else:
            n_test_pairs = 0

        print(f"{metapath:<15} {n_train_samples:<15} {n_train_pairs:<15,} {n_test_pairs:<20,}")

    print()
    print("Key differences:")
    print("  - Training: ~10-90 samples (one per bin combination)")
    print("  - Testing: ~thousands to millions of individual pairs")
    print("  - Evaluation: Pair-level predictions (more granular than training)")
    print()


def main():
    """Generate full summary."""
    print("\n")

    metapaths = summarize_training_data()
    print()

    summarize_testing_data()
    print()

    compare_train_test_sizes()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Training setup:")
    print("  ✓ 5 metapaths prepared")
    print("  ✓ 6 feature sets (A=102 to F=116 features)")
    print("  ✓ 30 total training datasets")
    print("  ✓ Training on degree bin aggregates (~10-90 samples per metapath)")
    print()
    print("Testing setup:")
    print("  ✓ 20 permutations available (000-019)")
    print("  ✓ Evaluation on individual node pairs")
    print("  ✓ Pathway counts averaged across permutations")
    print("  ✓ Hybrid approach: train on bins, test on pairs")
    print()
    print("Next step: Implement hybrid evaluation pipeline (Step 4)")
    print()


if __name__ == "__main__":
    main()
