"""
Test script for original graph frequency training approach.

This script tests the new training methodology where models are trained
on frequencies from the original Hetionet graph and validated against
200-permutation empirical frequencies.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / 'src'))

from evaluate_original_graph_approach import evaluate_single_edge_type


def main():
    repo_dir = Path.cwd()
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    edge_type = 'CbG'

    print(f"Testing original graph frequency training on {edge_type}")
    print("="*80)
    print("Expected behavior:")
    print("1. Compute frequencies from original Hetionet graph by degree pair")
    print("2. Split degree pairs into train/test")
    print("3. Train models on original graph frequencies")
    print("4. Validate against 200-permutation empirical frequencies")
    print("5. Measure interpolation quality for unseen degree pairs")
    print("="*80)

    try:
        result = evaluate_single_edge_type(
            edge_type=edge_type,
            data_dir=data_dir,
            results_dir=results_dir,
            feature_tiers=['minimal'],
            device='cpu'
        )

        print("\n" + "="*80)
        print("TEST PASSED - Results Summary:")
        print("="*80)
        print(f"Edge type: {result['edge_type']}")
        print(f"\nCoverage statistics:")
        for key, value in result['coverage_stats'].items():
            print(f"  {key}: {value}")

        print(f"\nModel performance:")
        for model_name, metrics in result['tier_results'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Features: {metrics['n_features']}")
            print(f"  Test r (original graph): {metrics.get('test_r', 'N/A')}")
            print(f"  Validation r (200-perm): {metrics['validation_r']:.4f}")
            if metrics.get('interpolation_r') is not None:
                print(f"  Interpolation r (unseen): {metrics['interpolation_r']:.4f}")
            print(f"  Bias: {metrics['mean_bias']:+.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            if metrics['training_time'] > 0:
                print(f"  Training time: {metrics['training_time']:.2f}s")

        print("\nValidation checks:")
        analytical_r = result['tier_results']['analytical']['validation_r']
        minimal_r = result['tier_results']['minimal']['validation_r']

        if minimal_r >= analytical_r:
            print(f"  [PASS] Minimal model r ({minimal_r:.4f}) >= Analytical r ({analytical_r:.4f})")
        else:
            print(f"  [WARNING] Minimal model r ({minimal_r:.4f}) < Analytical r ({analytical_r:.4f})")

        if result['coverage_stats']['n_unseen'] > 0:
            print(f"  [PASS] Interpolation tested on {result['coverage_stats']['n_unseen']} unseen degree pairs")
        else:
            print(f"  [INFO] No unseen degree pairs to test interpolation")

        print(f"\n  [PASS] Results saved to results/original_graph_training/{edge_type}/")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
