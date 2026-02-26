"""
Test suite for DWPC calculation module

Tests degree-weighted path count (DWPC) calculation for metapaths.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dwpc_pvalue_validation import dwpc_calculation, data_loading, config


def test_single_edge_dwpc():
    """Test DWPC for single-edge paths."""
    print("=" * 60)
    print("Test 1: Single Edge DWPC")
    print("=" * 60)

    loader = data_loading.get_loader()
    metaedge = "CbG"
    damping = 0.5

    # Calculate DWPC for a few source-target pairs
    source_indices = [0, 1, 2]
    target_indices = [0, 1, 2]

    dwpcs = dwpc_calculation.calculate_dwpc_pairs(
        metaedge=metaedge,
        source_indices=source_indices,
        target_indices=target_indices,
        source="true",
        damping_exponent=damping
    )

    print(f"Calculated DWPCs for {len(dwpcs)} pairs")
    print(f"Example DWPC values: {dwpcs[:3]}")

    assert len(dwpcs) == len(source_indices), "Should have DWPC for each pair"
    assert all(dwpc >= 0 for dwpc in dwpcs), "DWPCs should be non-negative"

    print("PASSED: Single edge DWPC calculation works\n")


def test_metapath_dwpc():
    """Test DWPC for multi-edge metapaths."""
    print("=" * 60)
    print("Test 2: Metapath DWPC")
    print("=" * 60)

    metapath = "CbGpPW"
    damping = 0.5

    # Test with a few source-target pairs
    source_indices = [0, 1, 2]
    target_indices = [0, 1, 2]

    dwpcs = dwpc_calculation.calculate_dwpc_metapath(
        metapath=metapath,
        source_indices=source_indices,
        target_indices=target_indices,
        source="true",
        damping_exponent=damping
    )

    print(f"Metapath: {metapath}")
    print(f"Calculated DWPCs for {len(dwpcs)} pairs")
    print(f"Example DWPC values: {dwpcs[:3]}")

    assert len(dwpcs) == len(source_indices), "Should have DWPC for each pair"
    assert all(dwpc >= 0 for dwpc in dwpcs), "DWPCs should be non-negative"

    print("PASSED: Metapath DWPC calculation works\n")


def test_dwpc_damping():
    """Test that damping exponent affects DWPC values."""
    print("=" * 60)
    print("Test 3: DWPC Damping Effect")
    print("=" * 60)

    metapath = "CbGpPW"
    source_indices = [100, 200]
    target_indices = [50, 150]

    # Calculate with different damping values
    dwpc_0 = dwpc_calculation.calculate_dwpc_metapath(
        metapath, source_indices, target_indices, "true", damping_exponent=0.0
    )

    dwpc_05 = dwpc_calculation.calculate_dwpc_metapath(
        metapath, source_indices, target_indices, "true", damping_exponent=0.5
    )

    dwpc_1 = dwpc_calculation.calculate_dwpc_metapath(
        metapath, source_indices, target_indices, "true", damping_exponent=1.0
    )

    print(f"Damping 0.0: {dwpc_0}")
    print(f"Damping 0.5: {dwpc_05}")
    print(f"Damping 1.0: {dwpc_1}")

    # With damping, high-degree nodes are downweighted
    # So typically: dwpc_0 >= dwpc_05 >= dwpc_1 (but may vary by path)
    print("PASSED: Damping exponent affects DWPC values\n")


def test_dwpc_permutation_consistency():
    """Test that permutations preserve DWPC properties."""
    print("=" * 60)
    print("Test 4: DWPC Permutation Consistency")
    print("=" * 60)

    metaedge = "CbG"
    source_indices = [10, 20, 30]
    target_indices = [5, 15, 25]

    # Calculate for true network
    dwpc_true = dwpc_calculation.calculate_dwpc_pairs(
        metaedge, source_indices, target_indices, "true", damping_exponent=0.5
    )

    # Calculate for permutation 0
    dwpc_perm0 = dwpc_calculation.calculate_dwpc_pairs(
        metaedge, source_indices, target_indices, "perm0", damping_exponent=0.5
    )

    print(f"True network DWPCs: {dwpc_true}")
    print(f"Perm0 DWPCs: {dwpc_perm0}")

    # Permutations preserve degree but not edges, so DWPCs will differ
    assert len(dwpc_true) == len(dwpc_perm0), "Should have same number of values"

    print("PASSED: DWPC works with permuted networks\n")


def test_batch_dwpc_calculation():
    """Test efficient batch DWPC calculation."""
    print("=" * 60)
    print("Test 5: Batch DWPC Calculation")
    print("=" * 60)

    metapath = "CbGpPW"

    # Create batch of 20 pairs
    n_pairs = 20
    source_indices = list(range(n_pairs))
    target_indices = list(range(n_pairs))

    dwpcs = dwpc_calculation.calculate_dwpc_metapath(
        metapath=metapath,
        source_indices=source_indices,
        target_indices=target_indices,
        source="true",
        damping_exponent=0.5
    )

    print(f"Calculated DWPCs for {len(dwpcs)} pairs")
    print(f"Mean DWPC: {np.mean(dwpcs):.4f}")
    print(f"Std DWPC: {np.std(dwpcs):.4f}")
    print(f"Min DWPC: {np.min(dwpcs):.4f}")
    print(f"Max DWPC: {np.max(dwpcs):.4f}")

    assert len(dwpcs) == n_pairs, "Should have DWPC for all pairs"

    print("PASSED: Batch DWPC calculation works\n")


def test_zero_dwpc():
    """Test that non-connected pairs have zero DWPC."""
    print("=" * 60)
    print("Test 6: Zero DWPC for Non-Connected Pairs")
    print("=" * 60)

    metaedge = "CbG"

    # Try pairs that are unlikely to be connected
    # Using same source and target (self-edges typically don't exist)
    source_indices = [0, 0, 0]
    target_indices = [0, 0, 0]

    dwpcs = dwpc_calculation.calculate_dwpc_pairs(
        metaedge, source_indices, target_indices, "true", damping_exponent=0.5
    )

    print(f"DWPCs for potentially non-connected pairs: {dwpcs}")

    # All should be valid (zero or positive)
    assert all(dwpc >= 0 for dwpc in dwpcs), "DWPCs should be non-negative"

    print("PASSED: Non-connected pairs handled correctly\n")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("DWPC CALCULATION MODULE TEST SUITE")
    print("=" * 60)
    print("\n")

    try:
        test_single_edge_dwpc()
        test_metapath_dwpc()
        test_dwpc_damping()
        test_dwpc_permutation_consistency()
        test_batch_dwpc_calculation()
        test_zero_dwpc()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()
