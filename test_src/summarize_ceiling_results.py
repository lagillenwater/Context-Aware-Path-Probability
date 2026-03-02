"""
Summarize ceiling analysis results across all 2-hop metapaths.

Aggregates results from results/ceiling_analysis/{metapath}/summary.txt
into a single table.

Usage:
    python summarize_ceiling_results.py

Outputs:
    results/ceiling_analysis_summary.csv
    results/ceiling_analysis_summary.md
"""

import os
import pandas as pd
import numpy as np
from typing import Dict


def parse_summary_file(summary_path: str) -> Dict:
    """
    Parse summary.txt file for a metapath.

    Returns dict with oracle_exact_r, oracle_binned_r, best_model_r, gap, conclusion.
    """
    if not os.path.exists(summary_path):
        return None

    result = {}

    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Metapath:'):
                result['metapath'] = line.split(': ')[1]
            elif line.startswith('Oracle (exact):'):
                val_str = line.split('= ')[1].split()[0]
                if val_str == 'NaN':
                    result['oracle_exact_r'] = np.nan
                else:
                    result['oracle_exact_r'] = float(val_str)
            elif line.startswith('Oracle (binned):'):
                val_str = line.split('= ')[1].split()[0]
                if val_str == 'NaN':
                    result['oracle_binned_r'] = np.nan
                else:
                    result['oracle_binned_r'] = float(val_str)
            elif line.startswith('Gap (exact - binned):'):
                result['gap'] = float(line.split(': ')[1])
            elif line.startswith('Conclusion:'):
                result['conclusion'] = line.split(': ')[1]

    return result if result else None


def main():
    """Summarize all ceiling analysis results."""
    results_dir = 'results/ceiling_analysis'

    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        print("Run ceiling analysis first.")
        return

    all_results = []

    metapath_dirs = [d for d in os.listdir(results_dir)
                     if os.path.isdir(os.path.join(results_dir, d))]

    print(f"Found {len(metapath_dirs)} metapath result directories")

    for metapath_dir in sorted(metapath_dirs):
        summary_path = os.path.join(results_dir, metapath_dir, 'summary.txt')

        result = parse_summary_file(summary_path)

        if result:
            all_results.append(result)
        else:
            print(f"Warning: Could not parse {summary_path}")

    if len(all_results) == 0:
        print("ERROR: No valid results found!")
        return

    df = pd.DataFrame(all_results)

    df = df[[
        'metapath', 'oracle_exact_r', 'oracle_binned_r', 'gap', 'conclusion'
    ]]

    df = df.sort_values('oracle_binned_r', ascending=False)

    csv_path = 'results/ceiling_analysis_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")

    md_path = 'results/ceiling_analysis_summary.md'
    with open(md_path, 'w') as f:
        f.write("# Ceiling Analysis Summary\n\n")
        f.write(f"Analysis of {len(df)} 2-hop metapaths\n\n")

        f.write("## Overall Statistics\n\n")
        f.write(f"- Mean oracle (exact): {df['oracle_exact_r'].mean():.4f} ± "
                f"{df['oracle_exact_r'].std():.4f}\n")
        f.write(f"- Mean oracle (binned): {df['oracle_binned_r'].mean():.4f} ± "
                f"{df['oracle_binned_r'].std():.4f}\n")
        f.write(f"- Mean gap (exact - binned): {df['gap'].mean():.4f} ± "
                f"{df['gap'].std():.4f}\n")
        f.write(f"- Metapaths with valid r: {(~df['oracle_binned_r'].isna()).sum()} / {len(df)}\n\n")

        f.write("## Conclusion Distribution\n\n")
        conclusion_counts = df['conclusion'].value_counts()
        for conclusion, count in conclusion_counts.items():
            pct = 100 * count / len(df)
            f.write(f"- {conclusion}: {count} ({pct:.1f}%)\n")

        f.write("\n## Top 10 Metapaths (by oracle binned r)\n\n")
        f.write(df.head(10).to_markdown(index=False))

        f.write("\n\n## Bottom 10 Metapaths (by oracle binned r)\n\n")
        f.write(df.tail(10).to_markdown(index=False))

        f.write("\n\n## Full Results\n\n")
        f.write(df.to_markdown(index=False))

    print(f"Saved markdown summary to: {md_path}")

    print("\n" + "=" * 70)
    print("CEILING ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nAnalyzed {len(df)} metapaths")
    print(f"\nMean oracle (exact): {df['oracle_exact_r'].mean():.4f}")
    print(f"Mean oracle (binned): {df['oracle_binned_r'].mean():.4f}")
    print(f"Mean gap (exact - binned): {df['gap'].mean():.4f}")
    print(f"Metapaths with valid r: {(~df['oracle_binned_r'].isna()).sum()} / {len(df)}")

    print("\nConclusion distribution:")
    for conclusion, count in conclusion_counts.items():
        pct = 100 * count / len(df)
        print(f"  {conclusion}: {count} ({pct:.1f}%)")

    print("\nTop 5 metapaths (by oracle binned r):")
    for i, row in df.head(5).iterrows():
        if np.isnan(row['oracle_binned_r']):
            print(f"  {row['metapath']}: r = NaN")
        else:
            print(f"  {row['metapath']}: r = {row['oracle_binned_r']:.4f} "
                  f"(gap = {row['gap']:.4f})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
