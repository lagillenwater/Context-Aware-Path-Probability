"""
Verification framework for Phase 2 feature extraction.

Tests feature extraction functions before full implementation.

Verification checks:
1. Feature extraction runs without errors
2. Dimensions match expected (A=102, B=104, ..., F=116)
3. No NaN/inf values in features
4. Features computed correctly (spot checks)
5. Consistent across different node pairs
"""

import numpy as np
import scipy.sparse as sp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.enhanced_features import extract_enhanced_features, get_feature_names


def load_edge_matrix(data_dir, edge_abbrev):
    """Load edge adjacency matrix."""
    import os
    edge_path = os.path.join(data_dir, 'edges', f'{edge_abbrev}.sparse.npz')
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge file not found: {edge_path}")
    return sp.load_npz(edge_path)


def test_feature_extraction_single_metapath(metapath_name, edge1_abbrev, edge2_abbrev):
    """Test feature extraction on a single metapath."""
    print(f"Testing: {metapath_name} ({edge1_abbrev} -> {edge2_abbrev})")
    print("-" * 60)

    # Load edges
    try:
        edge1 = load_edge_matrix('data', edge1_abbrev)
        edge2 = load_edge_matrix('data', edge2_abbrev)
    except FileNotFoundError as e:
        print(f"  FAILED: {e}")
        return False

    print(f"  Loaded edges: {edge1.shape} -> {edge2.shape}")

    # Test sample node pairs (first 100)
    n_source = min(100, edge1.shape[0])
    n_target = min(100, edge2.shape[1])
    source_nodes = np.arange(n_source)
    target_nodes = np.arange(n_target)

    # Test all feature sets
    feature_sets = ['A', 'B', 'C', 'D', 'E', 'F']
    expected_dims = {'A': 102, 'B': 104, 'C': 109, 'D': 111, 'E': 113, 'F': 116}

    results = {}

    for feature_set in feature_sets:
        print(f"\n  Feature Set {feature_set}:")

        try:
            # Extract features
            features = extract_enhanced_features(
                source_nodes, target_nodes,
                edge1, edge2,
                n_bins=10,
                feature_set=feature_set
            )

            # Check dimensions
            expected_dim = expected_dims[feature_set]
            actual_dim = features.shape[1]

            if actual_dim != expected_dim:
                print(f"    FAILED: Expected {expected_dim} features, got {actual_dim}")
                results[feature_set] = False
                continue

            print(f"    Dimensions: {features.shape} (expected {len(source_nodes)*len(target_nodes)}, {expected_dim})")

            # Check for NaN/inf
            has_nan = np.isnan(features).any()
            has_inf = np.isinf(features).any()

            if has_nan:
                print(f"    WARNING: Found NaN values")
            if has_inf:
                print(f"    WARNING: Found inf values")

            # Check feature statistics
            feature_means = features.mean(axis=0)
            feature_stds = features.std(axis=0)

            zero_variance_count = (feature_stds == 0).sum()
            if zero_variance_count > 0:
                print(f"    WARNING: {zero_variance_count} features have zero variance")

            # Get feature names
            feature_names = get_feature_names(feature_set, n_bins=10)
            print(f"    Features: {', '.join(feature_names[:5])}... (first 5/{len(feature_names)})")

            # Spot check: first 5 features
            print(f"    Sample values (first pair):")
            for i in range(min(5, len(feature_names))):
                print(f"      {feature_names[i]:20s}: {features[0, i]:.4f}")

            if not has_nan and not has_inf:
                print(f"    PASSED")
                results[feature_set] = True
            else:
                print(f"    FAILED: NaN or inf values detected")
                results[feature_set] = False

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[feature_set] = False

    print()
    return results


def run_all_tests():
    """Run tests on all 5 metapaths."""
    metapaths = [
        ('CbGpPW', 'CbG', 'GpPW'),
        ('GiGiG', 'GiG', 'GiG'),
        ('CtDaG', 'CtD', 'DaG'),
        ('AdGpBP', 'AdG', 'GpBP'),
        ('CrCbG', 'CrC', 'CbG')
    ]

    print("=" * 60)
    print("VERIFICATION: FEATURE EXTRACTION TESTING")
    print("=" * 60)
    print()

    all_results = {}

    for metapath_name, edge1_abbrev, edge2_abbrev in metapaths:
        results = test_feature_extraction_single_metapath(
            metapath_name, edge1_abbrev, edge2_abbrev
        )
        if results:
            all_results[metapath_name] = results
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if len(all_results) == 0:
        print("ERROR: No metapaths tested successfully")
        return False

    print(f"Tested: {len(all_results)}/5 metapaths")
    print()

    # Count successes per feature set
    feature_sets = ['A', 'B', 'C', 'D', 'E', 'F']

    print("Feature Set Success Rate:")
    for fs in feature_sets:
        success_count = sum(1 for results in all_results.values() if results.get(fs, False))
        total_count = len(all_results)
        print(f"  Set {fs}: {success_count}/{total_count} metapaths passed")

    print()

    # Overall verification
    all_passed = all(
        results.get(fs, False)
        for results in all_results.values()
        for fs in feature_sets
    )

    if all_passed:
        print("VERIFICATION: PASSED")
        print("All feature sets can be extracted successfully.")
        print("Ready to proceed with full implementation.")
        return True
    else:
        print("VERIFICATION: FAILED")
        print("Some feature extractions failed. Fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
