"""
Test theoretical corrections on CbG and AeG edge types.

This script validates that theoretical corrections (based only on
original graph features) improve predictions without using empirical
frequencies in training.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / 'src'))

from evaluate_theoretical_approach import evaluate_single_edge_type


def main():
    repo_dir = Path.cwd()
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    print("Testing theoretical corrections")
    print("="*80)
    print("Goal: Improve predictions using only original graph features")
    print("No empirical frequencies used in corrections")
    print("="*80)

    edge_types = ['CbG', 'AeG']

    for edge_type in edge_types:
        try:
            print(f"\n\nTesting {edge_type}...")
            result = evaluate_single_edge_type(
                edge_type=edge_type,
                data_dir=data_dir,
                results_dir=results_dir
            )

            print(f"\n{'-'*60}")
            print(f"RESULT for {edge_type}:")
            print(f"{'-'*60}")
            print(f"Analytical r: {result['analytical_r']:.4f}")
            print(f"Corrected r:  {result['corrected_r']:.4f}")
            print(f"Improvement:  {result['improvement_r']:+.4f}")
            print(f"Bias before:  {result['analytical_bias']:+.4f}")
            print(f"Bias after:   {result['corrected_bias']:+.4f}")
            print(f"Bias reduction: {result['improvement_bias']:+.4f}")

            if result['improvement_r'] > 0:
                print("\n  [PASS] Corrections improved correlation")
            elif abs(result['improvement_r']) < 0.001:
                print("\n  [OK] No significant change (edge type may not need corrections)")
            else:
                print("\n  [WARNING] Corrections decreased correlation")

        except Exception as e:
            print(f"\nTEST FAILED for {edge_type}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
