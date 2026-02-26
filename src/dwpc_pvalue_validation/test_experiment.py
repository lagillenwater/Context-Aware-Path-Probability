"""
Test suite for experiment module

Tests end-to-end experiment workflow for DWPC p-value validation.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dwpc_pvalue_validation import experiment, config


def test_scenario_a_null_test():
    """Test Scenario A: permutation 0 vs permutations 1-3 as null."""
    print("=" * 60)
    print("Test 1: Scenario A - Null Test")
    print("=" * 60)

    metapath = "CbGpPW"
    results = experiment.run_scenario_a(
        metapath=metapath,
        n_samples_per_category=5,
        observed_perm=0,
        null_perms=[1, 2, 3],
        damping_exponent=0.5,
        random_state=42
    )

    print(f"Metapath: {metapath}")
    print(f"Observed perm: 0")
    print(f"Null perms: [1, 2, 3]")
    print(f"Number of categories: {len(results['pvalues_by_category'])}")

    assert 'observed_dwpcs' in results, "Should have observed DWPCs"
    assert 'null_distributions' in results, "Should have null distributions"
    assert 'pvalues_by_category' in results, "Should have p-values by category"
    assert 'gamma_hurdle_params' in results, "Should have gamma-hurdle params"

    total_pvalues = sum(len(pvals) for pvals in results['pvalues_by_category'].values())
    print(f"Total p-values calculated: {total_pvalues}")

    assert total_pvalues > 0, "Should have calculated p-values"

    print("PASSED: Scenario A null test works\n")


def test_scenario_b_positive_control():
    """Test Scenario B: true Hetionet vs permutations as positive control."""
    print("=" * 60)
    print("Test 2: Scenario B - Positive Control")
    print("=" * 60)

    metapath = "CbGpPW"
    results = experiment.run_scenario_b(
        metapath=metapath,
        n_samples_per_category=5,
        null_perms=[1, 2, 3],
        damping_exponent=0.5,
        random_state=42
    )

    print(f"Metapath: {metapath}")
    print(f"Observed: true Hetionet")
    print(f"Null perms: [1, 2, 3]")
    print(f"Number of categories: {len(results['pvalues_by_category'])}")

    assert 'observed_dwpcs' in results, "Should have observed DWPCs"
    assert 'null_distributions' in results, "Should have null distributions"
    assert 'pvalues_by_category' in results, "Should have p-values"

    total_pvalues = sum(len(pvals) for pvals in results['pvalues_by_category'].values())
    print(f"Total p-values calculated: {total_pvalues}")

    print("PASSED: Scenario B positive control works\n")


def test_calibration_analysis():
    """Test p-value calibration analysis."""
    print("=" * 60)
    print("Test 3: P-Value Calibration Analysis")
    print("=" * 60)

    metapath = "CbGpPW"

    # Run scenario A (null test - should have uniform p-values)
    results_a = experiment.run_scenario_a(
        metapath, 5, 0, [1, 2], 0.5, random_state=42
    )

    calibration = experiment.analyze_calibration(results_a['pvalues_by_category'])

    print(f"Calibration metrics:")
    print(f"  Total p-values: {calibration['n_total']}")
    print(f"  p < 0.05: {calibration['n_sig_005']}")
    print(f"  p < 0.01: {calibration['n_sig_001']}")
    print(f"  Mean p-value: {calibration['mean_pvalue']:.4f}")

    assert calibration['n_total'] > 0, "Should have p-values"
    assert 'mean_pvalue' in calibration, "Should have mean p-value"
    assert 'ks_statistic' in calibration, "Should have KS test statistic"

    print("PASSED: Calibration analysis works\n")


def test_compare_scenarios():
    """Test comparison of scenarios A and B."""
    print("=" * 60)
    print("Test 4: Compare Scenarios")
    print("=" * 60)

    metapath = "CbGpPW"

    results_a = experiment.run_scenario_a(
        metapath, 3, 0, [1, 2], 0.5, random_state=42
    )

    results_b = experiment.run_scenario_b(
        metapath, 3, [1, 2], 0.5, random_state=42
    )

    comparison = experiment.compare_scenarios(results_a, results_b)

    print(f"Scenario A (null):")
    print(f"  Mean p-value: {comparison['scenario_a']['mean_pvalue']:.4f}")
    print(f"  p < 0.05: {comparison['scenario_a']['n_sig_005']}")

    print(f"Scenario B (positive):")
    print(f"  Mean p-value: {comparison['scenario_b']['mean_pvalue']:.4f}")
    print(f"  p < 0.05: {comparison['scenario_b']['n_sig_005']}")

    assert 'scenario_a' in comparison, "Should have scenario A results"
    assert 'scenario_b' in comparison, "Should have scenario B results"

    print("PASSED: Scenario comparison works\n")


def test_degree_stratified_results():
    """Test that results are properly stratified by degree."""
    print("=" * 60)
    print("Test 5: Degree-Stratified Results")
    print("=" * 60)

    metapath = "CbGpPW"
    results = experiment.run_scenario_a(
        metapath, 5, 0, [1, 2], 0.5, random_state=42
    )

    categories = list(results['pvalues_by_category'].keys())
    print(f"Number of degree categories: {len(categories)}")
    print(f"Categories: {categories[:3]}...")

    assert len(categories) > 1, "Should have multiple degree categories"

    for cat in categories:
        assert isinstance(cat, tuple), "Category should be tuple"
        assert len(cat) == 2, "Category should be (src_cat, tgt_cat)"

    print("PASSED: Degree stratification works\n")


def test_save_load_results():
    """Test saving and loading experiment results."""
    print("=" * 60)
    print("Test 6: Save and Load Results")
    print("=" * 60)

    metapath = "CbGpPW"
    results = experiment.run_scenario_a(
        metapath, 3, 0, [1, 2], 0.5, random_state=42
    )

    output_path = config.RESULTS_DIR / "test_experiment_results.npz"
    experiment.save_results(results, output_path)
    print(f"Saved to: {output_path}")

    loaded = experiment.load_results(output_path)
    print(f"Loaded results")

    assert 'observed_dwpcs' in loaded, "Should have observed DWPCs"
    assert 'pvalues_by_category' in loaded, "Should have p-values"

    output_path.unlink()
    print("PASSED: Save and load works\n")


def test_multiple_metapaths():
    """Test running experiment for multiple metapaths."""
    print("=" * 60)
    print("Test 7: Multiple Metapaths")
    print("=" * 60)

    metapaths = ["CbGpPW", "CtDaG"]

    all_results = {}
    for metapath in metapaths:
        results = experiment.run_scenario_a(
            metapath, 3, 0, [1, 2], 0.5, random_state=42
        )
        all_results[metapath] = results
        print(f"  {metapath}: {sum(len(p) for p in results['pvalues_by_category'].values())} p-values")

    assert len(all_results) == len(metapaths), "Should have results for all metapaths"

    print("PASSED: Multiple metapaths work\n")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("EXPERIMENT MODULE TEST SUITE")
    print("=" * 60)
    print("\n")

    try:
        test_scenario_a_null_test()
        test_scenario_b_positive_control()
        test_calibration_analysis()
        test_compare_scenarios()
        test_degree_stratified_results()
        test_save_load_results()
        test_multiple_metapaths()

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
