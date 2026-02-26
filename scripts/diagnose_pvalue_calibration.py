"""
Diagnostic script to analyze DWPC p-value calibration issues.

This script loads existing validation results and diagnoses why p-values
are over-conservative (mean ~0.9 instead of ~0.5).
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir, metapath, scenario):
    """Load results for a metapath and scenario."""
    results_file = Path(results_dir) / f"{metapath}_scenario_{scenario.lower()}.npz"

    if not results_file.exists():
        print(f"File not found: {results_file}")
        return None

    data = np.load(results_file, allow_pickle=True)

    results = {
        'metapath': str(data['metapath']),
        'scenario': str(data['scenario']),
        'pvalues_by_category': {},
        'observed_dwpcs': {}
    }

    for key in data.keys():
        if key.startswith('pvalues_'):
            parts = key.replace('pvalues_', '').split('_', 1)
            if len(parts) == 2:
                cat = tuple(parts)
                results['pvalues_by_category'][cat] = data[key]
        elif key.startswith('observed_'):
            parts = key.replace('observed_', '').split('_', 1)
            if len(parts) == 2:
                cat = tuple(parts)
                results['observed_dwpcs'][cat] = data[key]

    return results


def analyze_metapath(results_dir, metapath):
    """Analyze a single metapath's results."""
    print(f"\n{'=' * 80}")
    print(f"Analyzing: {metapath}")
    print(f"{'=' * 80}\n")

    # Load results
    results_a = load_results(results_dir, metapath, 'A')
    results_b = load_results(results_dir, metapath, 'B')

    if not results_a or not results_b:
        print(f"Skipping {metapath} - missing results")
        return

    # Analyze Scenario A (null test)
    print("SCENARIO A (Null Test)")
    print("-" * 40)

    all_pvals = []
    all_dwpcs = []

    for category in results_a['pvalues_by_category'].keys():
        pvals = results_a['pvalues_by_category'][category]
        dwpcs = results_a['observed_dwpcs'][category]

        n_zero = np.sum(dwpcs == 0)
        n_nonzero = np.sum(dwpcs > 0)
        pct_zero = 100 * n_zero / len(dwpcs)

        pvals_zero = pvals[dwpcs == 0]
        pvals_nonzero = pvals[dwpcs > 0]

        print(f"\nCategory {category}:")
        print(f"  Total samples: {len(dwpcs)}")
        print(f"  Zero DWPCs: {n_zero} ({pct_zero:.1f}%)")
        print(f"  Non-zero DWPCs: {n_nonzero} ({100-pct_zero:.1f}%)")

        if n_zero > 0:
            print(f"  P-values for zero DWPCs: mean={np.mean(pvals_zero):.3f}, "
                  f"median={np.median(pvals_zero):.3f}")

        if n_nonzero > 0:
            print(f"  P-values for non-zero DWPCs: mean={np.mean(pvals_nonzero):.3f}, "
                  f"median={np.median(pvals_nonzero):.3f}")
            print(f"    DWPC range: [{np.min(dwpcs[dwpcs > 0]):.6f}, "
                  f"{np.max(dwpcs):.6f}]")

        all_pvals.extend(pvals)
        all_dwpcs.extend(dwpcs)

    all_pvals = np.array(all_pvals)
    all_dwpcs = np.array(all_dwpcs)

    n_zero_total = np.sum(all_dwpcs == 0)
    n_total = len(all_dwpcs)
    pct_zero_total = 100 * n_zero_total / n_total

    print(f"\n{'=' * 40}")
    print("OVERALL SUMMARY:")
    print(f"  Total samples: {n_total}")
    print(f"  Zero DWPCs: {n_zero_total} ({pct_zero_total:.1f}%)")
    print(f"  Non-zero DWPCs: {n_total - n_zero_total} ({100-pct_zero_total:.1f}%)")
    print(f"  Mean p-value (all): {np.mean(all_pvals):.3f}")
    print(f"  Median p-value (all): {np.median(all_pvals):.3f}")

    pvals_zero = all_pvals[all_dwpcs == 0]
    pvals_nonzero = all_pvals[all_dwpcs > 0]

    if len(pvals_zero) > 0:
        print(f"  Mean p-value (zero DWPCs): {np.mean(pvals_zero):.3f}")
    if len(pvals_nonzero) > 0:
        print(f"  Mean p-value (non-zero DWPCs): {np.mean(pvals_nonzero):.3f}")
        print(f"  Proportion p < 0.05 (non-zero): {np.mean(pvals_nonzero < 0.05):.3f}")

    return {
        'metapath': metapath,
        'n_total': n_total,
        'n_zero': n_zero_total,
        'pct_zero': pct_zero_total,
        'mean_pval_all': np.mean(all_pvals),
        'mean_pval_nonzero': np.mean(pvals_nonzero) if len(pvals_nonzero) > 0 else None,
        'median_pval_all': np.median(all_pvals),
        'all_dwpcs': all_dwpcs,
        'all_pvals': all_pvals
    }


def create_diagnostic_plots(analysis_results, output_dir):
    """Create diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in analysis_results:
        metapath = result['metapath']
        dwpcs = result['all_dwpcs']
        pvals = result['all_pvals']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Histogram of DWPCs
        ax = axes[0, 0]
        dwpcs_nonzero = dwpcs[dwpcs > 0]
        if len(dwpcs_nonzero) > 0:
            ax.hist(dwpcs_nonzero, bins=50, edgecolor='black')
            ax.set_xlabel('DWPC (non-zero only)')
            ax.set_ylabel('Count')
            ax.set_title(f'{metapath}: DWPC Distribution (Non-zero)')
        else:
            ax.text(0.5, 0.5, 'All DWPCs are zero',
                   ha='center', va='center', transform=ax.transAxes)

        # Histogram of p-values (all)
        ax = axes[0, 1]
        ax.hist(pvals, bins=50, edgecolor='black')
        ax.axvline(x=0.5, color='r', linestyle='--', label='Expected mean')
        ax.set_xlabel('P-value')
        ax.set_ylabel('Count')
        ax.set_title(f'{metapath}: P-value Distribution (All)')
        ax.legend()

        # Histogram of p-values (non-zero only)
        ax = axes[1, 0]
        pvals_nonzero = pvals[dwpcs > 0]
        if len(pvals_nonzero) > 0:
            ax.hist(pvals_nonzero, bins=50, edgecolor='black')
            ax.axvline(x=0.5, color='r', linestyle='--', label='Expected mean')
            ax.set_xlabel('P-value')
            ax.set_ylabel('Count')
            ax.set_title(f'{metapath}: P-value Distribution (Non-zero DWPCs)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No non-zero DWPCs',
                   ha='center', va='center', transform=ax.transAxes)

        # Q-Q plot for p-values (non-zero)
        ax = axes[1, 1]
        if len(pvals_nonzero) > 0:
            sorted_pvals = np.sort(pvals_nonzero)
            theoretical_quantiles = np.linspace(0, 1, len(sorted_pvals))
            ax.scatter(theoretical_quantiles, sorted_pvals, alpha=0.5)
            ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
            ax.set_xlabel('Theoretical quantiles (uniform)')
            ax.set_ylabel('Observed p-values')
            ax.set_title(f'{metapath}: Q-Q Plot (Non-zero DWPCs)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No non-zero DWPCs',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        output_file = output_dir / f"{metapath}_diagnostic.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved diagnostic plot: {output_file}")


def main():
    """Main execution."""
    results_dir = Path("results/dwpc_pvalue_validation")

    metapaths = ['CbGpPW', 'CtDaG', 'GiGaD', 'CbGpPWpG', 'CtDaGiG', 'CbGpPWpGaD']

    analysis_results = []
    for metapath in metapaths:
        result = analyze_metapath(results_dir, metapath)
        if result:
            analysis_results.append(result)

    # Create summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}\n")
    print(f"{'Metapath':<15} {'N':<8} {'% Zero':<10} {'Mean P (all)':<15} {'Mean P (nonzero)':<15}")
    print("-" * 80)

    for result in analysis_results:
        mean_nz = result['mean_pval_nonzero']
        mean_nz_str = f"{mean_nz:.3f}" if mean_nz is not None else "N/A"
        print(f"{result['metapath']:<15} {result['n_total']:<8} "
              f"{result['pct_zero']:<10.1f} {result['mean_pval_all']:<15.3f} "
              f"{mean_nz_str:<15}")

    # Create diagnostic plots
    output_dir = results_dir / "diagnostics"
    create_diagnostic_plots(analysis_results, output_dir)

    print(f"\nDiagnostic plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
