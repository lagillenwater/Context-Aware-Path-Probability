"""
Test corrected training pipeline on single edge type.

This script tests the fix where models are trained on single-permutation
binary labels and validated against 200-permutation empirical frequencies.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from evaluate_feature_reduction import evaluate_single_edge_type

def main():
    repo_dir = Path.cwd()
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    edge_type = 'CbG'

    print(f"Testing corrected training pipeline on {edge_type}")
    print("="*80)
    print("Expected behavior:")
    print("1. Load binary labels from single permutation (001.hetmat)")
    print("2. Train models on binary labels")
    print("3. Validate predictions against empirical frequencies")
    print("4. Residual plots show grouped predictions vs empirical frequencies")
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
        print("TEST PASSED - Results:")
        print("="*80)
        print(f"Edge type: {result['edge_type']}")
        print(f"Difficulty: {result['difficulty']}")
        print(f"Recommended tier: {result['recommended_tier']}")
        print(f"\nAnalytical r: {result['tier_results']['analytical']['pearson_r']:.4f}")
        print(f"SimpleNN r: {result['tier_results']['simplenn']['pearson_r']:.4f}")
        print(f"Minimal tier r: {result['tier_results']['minimal']['pearson_r']:.4f}")
        print(f"\nMatched degree pairs (analytical): {result['tier_results']['analytical']['n_matched']}")
        print(f"Matched degree pairs (SimpleNN): {result['tier_results']['simplenn']['n_matched']}")
        print(f"Matched degree pairs (minimal): {result['tier_results']['minimal']['n_matched']}")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
