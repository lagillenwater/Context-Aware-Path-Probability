"""
Test feature reduction evaluation on a single edge type.
Quick validation before running full pipeline.
"""

import sys
from pathlib import Path
import torch

# Add src to path
repo_dir = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
sys.path.insert(0, str(repo_dir / 'src'))

from evaluate_feature_reduction import evaluate_single_edge_type


def main():
    """Test on CbG edge type."""
    data_dir = repo_dir / 'data'
    results_dir = repo_dir / 'results'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing feature reduction evaluation on CbG")
    print(f"Using device: {device}\n")

    try:
        result = evaluate_single_edge_type(
            edge_type='CbG',
            data_dir=data_dir,
            results_dir=results_dir,
            feature_tiers=['minimal', 'standard', 'extended'],
            device=device
        )

        print("\n" + "="*80)
        print("TEST PASSED")
        print("="*80)
        print(f"\nEdge type: {result['edge_type']}")
        print(f"Difficulty: {result['difficulty']}")
        print(f"Recommended tier: {result['recommended_tier']}")

        print("\nResults:")
        for tier, metrics in result['tier_results'].items():
            print(f"  {tier}: r={metrics['pearson_r']:.4f}, "
                  f"bias={metrics['mean_bias']:+.4f}, "
                  f"n_features={metrics['n_features']}")

    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
