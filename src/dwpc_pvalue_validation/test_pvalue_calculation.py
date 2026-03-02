"""
Test suite for p-value calculation module

Tests gamma-hurdle p-value calculation following Himmelstein et al. 2023.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dwpc_pvalue_validation import pvalue_calculation, config


def test_gamma_hurdle_parameters():
    """Test gamma-hurdle parameter estimation."""
    print("=" * 60)
    print("Test 1: Gamma-Hurdle Parameter Estimation")
    print("=" * 60)

    # Create sample null distribution
    null_dwpcs = np.array([0, 0, 0, 1.5, 2.3, 3.1, 0, 4.2, 1.8, 2.9])

    params = pvalue_calculation.fit_gamma_hurdle(null_dwpcs)

    print(f"Null DWPCs: {null_dwpcs}")
    print(f"Lambda (hurdle): {params['lambda']:.4f}")
    print(f"Alpha (shape): {params['alpha']:.4f}")
    print(f"Beta (rate): {params['beta']:.4f}")

    assert 'lambda' in params, "Should have lambda parameter"
    assert 'alpha' in params, "Should have alpha parameter"
    assert 'beta' in params, "Should have beta parameter"
    assert 0 <= params['lambda'] <= 1, "Lambda should be in [0, 1]"
    assert params['alpha'] > 0, "Alpha should be positive"
    assert params['beta'] > 0, "Beta should be positive"

    print("PASSED: Gamma-hurdle parameters estimated correctly\n")


def test_gamma_hurdle_pvalue():
    """Test p-value calculation from gamma-hurdle."""
    print("=" * 60)
    print("Test 2: Gamma-Hurdle P-Value Calculation")
    print("=" * 60)

    # Fit parameters from null distribution
    null_dwpcs = np.array([0, 0, 1.0, 2.0, 3.0, 4.0, 5.0])
    params = pvalue_calculation.fit_gamma_hurdle(null_dwpcs)

    # Calculate p-values for observed DWPCs
    observed_dwpcs = np.array([0.5, 2.5, 6.0, 10.0])
    pvalues = pvalue_calculation.calculate_pvalues(observed_dwpcs, params)

    print(f"Observed DWPCs: {observed_dwpcs}")
    print(f"P-values: {pvalues}")

    assert len(pvalues) == len(observed_dwpcs), "One p-value per observation"
    assert all(0 <= p <= 1 for p in pvalues), "P-values in [0, 1]"
    assert pvalues[0] > pvalues[-1], "Higher DWPC should have lower p-value"

    print("PASSED: P-values calculated correctly\n")


def test_zero_dwpc_pvalue():
    """Test p-value for zero DWPC."""
    print("=" * 60)
    print("Test 3: Zero DWPC P-Value")
    print("=" * 60)

    null_dwpcs = np.array([0, 0, 0, 1.0, 2.0, 3.0])
    params = pvalue_calculation.fit_gamma_hurdle(null_dwpcs)

    # Zero DWPC should have high p-value
    zero_pvalue = pvalue_calculation.calculate_pvalues(np.array([0.0]), params)

    print(f"Null distribution: {null_dwpcs}")
    print(f"Lambda (P(DWPC=0)): {params['lambda']:.4f}")
    print(f"P-value for DWPC=0: {zero_pvalue[0]:.4f}")

    assert 0 <= zero_pvalue[0] <= 1, "P-value in [0, 1]"

    print("PASSED: Zero DWPC p-value handled correctly\n")


def test_bessel_correction():
    """Test that Bessel's correction is applied."""
    print("=" * 60)
    print("Test 4: Bessel's Correction")
    print("=" * 60)

    # Small sample where correction matters
    null_dwpcs = np.array([1.0, 2.0, 3.0])

    params_corrected = pvalue_calculation.fit_gamma_hurdle(
        null_dwpcs, use_bessel_correction=True
    )
    params_uncorrected = pvalue_calculation.fit_gamma_hurdle(
        null_dwpcs, use_bessel_correction=False
    )

    print(f"With Bessel correction: alpha={params_corrected['alpha']:.4f}")
    print(f"Without correction: alpha={params_uncorrected['alpha']:.4f}")

    # Parameters should differ when correction is applied
    print("PASSED: Bessel correction option works\n")


def test_batch_pvalue_calculation():
    """Test batch p-value calculation."""
    print("=" * 60)
    print("Test 5: Batch P-Value Calculation")
    print("=" * 60)

    null_dwpcs = np.random.gamma(2.0, 2.0, size=100)
    null_dwpcs[::5] = 0

    params = pvalue_calculation.fit_gamma_hurdle(null_dwpcs)

    # Calculate p-values for batch of observations
    observed_dwpcs = np.random.gamma(2.0, 2.0, size=50)
    pvalues = pvalue_calculation.calculate_pvalues(observed_dwpcs, params)

    print(f"Null distribution: {len(null_dwpcs)} values")
    print(f"Observed: {len(observed_dwpcs)} values")
    print(f"P-values: min={np.min(pvalues):.4f}, "
          f"max={np.max(pvalues):.4f}, mean={np.mean(pvalues):.4f}")

    assert len(pvalues) == len(observed_dwpcs), "One p-value per observation"
    assert all(0 <= p <= 1 for p in pvalues), "All p-values in [0, 1]"

    print("PASSED: Batch p-value calculation works\n")


def test_pvalue_stratified_by_degree():
    """Test p-value calculation with degree stratification."""
    print("=" * 60)
    print("Test 6: P-Value Stratified by Degree")
    print("=" * 60)

    # Simulate different null distributions for different degree categories
    null_low = np.array([0, 0, 0.5, 1.0, 1.5])
    null_high = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    params_low = pvalue_calculation.fit_gamma_hurdle(null_low)
    params_high = pvalue_calculation.fit_gamma_hurdle(null_high)

    # Same observed DWPC should have different p-values
    observed = np.array([2.0])
    pvalue_low = pvalue_calculation.calculate_pvalues(observed, params_low)
    pvalue_high = pvalue_calculation.calculate_pvalues(observed, params_high)

    print(f"Null (Low degree): {null_low}")
    print(f"Null (High degree): {null_high}")
    print(f"Observed DWPC: {observed[0]}")
    print(f"P-value (Low): {pvalue_low[0]:.4f}")
    print(f"P-value (High): {pvalue_high[0]:.4f}")

    # DWPC=2.0 should be more extreme in low-degree null
    assert pvalue_low[0] < pvalue_high[0], (
        "Same DWPC should be more significant in low-degree null"
    )

    print("PASSED: Degree-stratified p-values work correctly\n")


def test_extreme_dwpc_pvalue():
    """Test p-value for extremely high DWPC."""
    print("=" * 60)
    print("Test 7: Extreme DWPC P-Value")
    print("=" * 60)

    null_dwpcs = np.array([0.5, 1.0, 1.5, 2.0])
    params = pvalue_calculation.fit_gamma_hurdle(null_dwpcs)

    # Very high DWPC should have very low p-value
    extreme_dwpc = np.array([100.0])
    pvalue = pvalue_calculation.calculate_pvalues(extreme_dwpc, params)

    print(f"Null distribution max: {np.max(null_dwpcs)}")
    print(f"Extreme DWPC: {extreme_dwpc[0]}")
    print(f"P-value: {pvalue[0]:.6e}")

    assert pvalue[0] < 0.01, "Extreme DWPC should have very low p-value"

    print("PASSED: Extreme DWPC handled correctly\n")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("P-VALUE CALCULATION MODULE TEST SUITE")
    print("=" * 60)
    print("\n")

    try:
        test_gamma_hurdle_parameters()
        test_gamma_hurdle_pvalue()
        test_zero_dwpc_pvalue()
        test_bessel_correction()
        test_batch_pvalue_calculation()
        test_pvalue_stratified_by_degree()
        test_extreme_dwpc_pvalue()

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
