#!/usr/bin/env python
"""
DWPC P-Value Validation Experiment - Main Execution Script

Run end-to-end DWPC p-value validation experiments following Himmelstein et al. 2023.

Usage:
    python scripts/23_dwpc_pvalue_validation.py --metapaths CbGpPW CtDaG --n-samples 100
    python scripts/23_dwpc_pvalue_validation.py --all-metapaths --n-samples 50
    python scripts/23_dwpc_pvalue_validation.py --quick-test  # Run with minimal samples
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dwpc_pvalue_validation import (
    config, experiment, utils
)

logger = utils.setup_logging(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run DWPC p-value validation experiments"
    )

    # Metapath selection
    metapath_group = parser.add_mutually_exclusive_group(required=True)
    metapath_group.add_argument(
        '--metapaths',
        nargs='+',
        help='Specific metapaths to analyze (e.g., CbGpPW CtDaG)'
    )
    metapath_group.add_argument(
        '--all-metapaths',
        action='store_true',
        help='Run all configured metapaths'
    )
    metapath_group.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with 2 metapaths, 3 permutations, 10 samples'
    )

    # Experiment parameters
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of samples per degree category (default: 100)'
    )
    parser.add_argument(
        '--null-perms',
        nargs='+',
        type=int,
        help='Null permutation indices (default: 1-20)'
    )
    parser.add_argument(
        '--observed-perm',
        type=int,
        default=0,
        help='Observed permutation for Scenario A (default: 0)'
    )
    parser.add_argument(
        '--damping',
        type=float,
        default=0.5,
        help='DWPC damping exponent (default: 0.5)'
    )

    # Scenario selection
    parser.add_argument(
        '--scenario',
        choices=['A', 'B', 'both'],
        default='both',
        help='Which scenario to run: A (null test), B (positive control), or both'
    )

    # P-value calculation method
    parser.add_argument(
        '--method',
        choices=['gamma_hurdle', 'empirical'],
        default='gamma_hurdle',
        help='P-value calculation method: gamma_hurdle (parametric) or empirical (non-parametric)'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=config.RESULTS_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        default=True,
        help='Save results to disk'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Execution options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    return parser.parse_args()


def select_metapaths(args):
    """Select metapaths based on arguments."""
    if args.quick_test:
        # Quick test: first 2 metapaths
        selected = [config.METAPATHS[0], config.METAPATHS[1]]
        logger.info("Quick test mode: using first 2 metapaths")
    elif args.all_metapaths:
        selected = config.METAPATHS
        logger.info(f"Running all {len(selected)} metapaths")
    else:
        # Filter by abbreviation
        selected = [
            mp for mp in config.METAPATHS
            if mp[0] in args.metapaths
        ]
        if len(selected) != len(args.metapaths):
            found = [mp[0] for mp in selected]
            logger.warning(
                f"Requested {args.metapaths}, found {found}"
            )

    return selected


def determine_null_perms(args):
    """Determine null permutation indices."""
    if args.quick_test:
        return [1, 2, 3]
    elif args.null_perms:
        return args.null_perms
    else:
        return list(range(config.PERMUTATION_START, config.PERMUTATION_END + 1))


def run_scenario_a(metapath, args, null_perms):
    """Run Scenario A for a metapath."""
    logger.info(f"Scenario A: {metapath} (method={args.method})")

    n_samples = 10 if args.quick_test else args.n_samples

    start_time = time.time()
    results = experiment.run_scenario_a(
        metapath=metapath,
        n_samples_per_category=n_samples,
        observed_perm=args.observed_perm,
        null_perms=null_perms,
        damping_exponent=args.damping,
        random_state=args.random_seed,
        method=args.method
    )
    elapsed = time.time() - start_time

    # Calculate calibration
    calibration = experiment.analyze_calibration(results['pvalues_by_category'])

    logger.info(
        f"Scenario A complete: {elapsed:.1f}s, "
        f"{calibration['n_total']} p-values, mean={calibration['mean_pvalue']:.3f}"
    )

    return results, calibration


def run_scenario_b(metapath, args, null_perms):
    """Run Scenario B for a metapath."""
    logger.info(f"Scenario B: {metapath} (method={args.method})")

    n_samples = 10 if args.quick_test else args.n_samples

    start_time = time.time()
    results = experiment.run_scenario_b(
        metapath=metapath,
        n_samples_per_category=n_samples,
        null_perms=null_perms,
        damping_exponent=args.damping,
        random_state=args.random_seed,
        method=args.method
    )
    elapsed = time.time() - start_time

    # Calculate calibration
    calibration = experiment.analyze_calibration(results['pvalues_by_category'])

    logger.info(
        f"Scenario B complete: {elapsed:.1f}s, "
        f"{calibration['n_total']} p-values, mean={calibration['mean_pvalue']:.3f}"
    )

    return results, calibration


def save_experiment_results(metapath, results_a, results_b,
                            calibration_a, calibration_b,
                            comparison, output_dir):
    """Save results for a metapath."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Scenario A results
    if results_a:
        output_path_a = output_dir / f"{metapath}_scenario_a.npz"
        experiment.save_results(results_a, output_path_a)

    # Save Scenario B results
    if results_b:
        output_path_b = output_dir / f"{metapath}_scenario_b.npz"
        experiment.save_results(results_b, output_path_b)

    # Save calibration and comparison as JSON
    summary = {
        'metapath': metapath,
        'timestamp': datetime.now().isoformat(),
        'scenario_a_calibration': calibration_a if calibration_a else None,
        'scenario_b_calibration': calibration_b if calibration_b else None,
        'comparison': comparison if comparison else None
    }

    summary_path = output_dir / f"{metapath}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved results to {output_dir}")


def print_summary(all_results):
    """Print summary of all experiments."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    for metapath, data in all_results.items():
        print(f"\n{metapath}:")

        if data['calibration_a']:
            cal_a = data['calibration_a']
            print(f"  Scenario A (null test): "
                  f"n={cal_a['n_total']}, "
                  f"mean_p={cal_a['mean_pvalue']:.3f}, "
                  f"p<0.05={cal_a['proportion_sig_005']:.3f}, "
                  f"KS_p={cal_a['ks_pvalue']:.3f}")

        if data['calibration_b']:
            cal_b = data['calibration_b']
            print(f"  Scenario B (positive control): "
                  f"n={cal_b['n_total']}, "
                  f"mean_p={cal_b['mean_pvalue']:.3f}, "
                  f"p<0.05={cal_b['proportion_sig_005']:.3f}, "
                  f"KS_p={cal_b['ks_pvalue']:.3f}")

        if data['comparison']:
            comp = data['comparison']
            print(f"  Comparison:")
            print(f"    Delta mean_p: {comp['difference_mean_pvalue']:.3f}")
            print(f"    Delta n_sig: {comp['difference_sig_005']}")

    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    args = parse_args()

    # Setup logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting DWPC P-Value Validation Experiment")
    logger.info(f"Output directory: {args.output_dir}")

    # Select metapaths
    metapaths = select_metapaths(args)
    null_perms = determine_null_perms(args)

    logger.info(f"Metapaths: {[mp[0] for mp in metapaths]}")
    logger.info(f"Null permutations: {null_perms}")
    logger.info(f"Samples per category: {10 if args.quick_test else args.n_samples}")

    # Run experiments
    all_results = {}
    total_start = time.time()

    for metapath_abbrev, metapath_name, path_length in metapaths:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing: {metapath_abbrev} ({metapath_name})")
        logger.info(f"Path length: {path_length}")
        logger.info(f"{'=' * 80}")

        results_a = None
        results_b = None
        calibration_a = None
        calibration_b = None
        comparison = None

        try:
            # Run Scenario A
            if args.scenario in ['A', 'both']:
                results_a, calibration_a = run_scenario_a(
                    metapath_abbrev, args, null_perms
                )

            # Run Scenario B
            if args.scenario in ['B', 'both']:
                results_b, calibration_b = run_scenario_b(
                    metapath_abbrev, args, null_perms
                )

            # Compare scenarios
            if results_a and results_b:
                comparison = experiment.compare_scenarios(results_a, results_b)
                logger.info(
                    f"Comparison: delta_mean_p={comparison['difference_mean_pvalue']:.3f}"
                )

            # Save results
            if args.save_results:
                save_experiment_results(
                    metapath_abbrev, results_a, results_b,
                    calibration_a, calibration_b,
                    comparison, args.output_dir
                )

            # Store for summary
            all_results[metapath_abbrev] = {
                'results_a': results_a,
                'results_b': results_b,
                'calibration_a': calibration_a,
                'calibration_b': calibration_b,
                'comparison': comparison
            }

        except Exception as e:
            logger.error(f"Error processing {metapath_abbrev}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Print summary
    print_summary(all_results)

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
