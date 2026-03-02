"""
Utilities for loading node labels from Hetionet data files.

This module provides functions to map node indices to human-readable labels
(compound names, pathway names, etc.) for interpretation of analysis results.
"""

import pandas as pd
from pathlib import Path
from typing import List


def load_node_labels(data_dir: Path, node_type: str) -> List[str]:
    """
    Load node labels from TSV files.

    Parameters
    ----------
    data_dir : Path
        Path to data directory containing nodes/ subdirectory
    node_type : str
        Node type (e.g., 'Compound', 'Pathway', 'Gene', 'Disease')

    Returns
    -------
    list of str
        List of node labels indexed by node position

    Raises
    ------
    FileNotFoundError
        If node file does not exist
    ValueError
        If node file is malformed or missing required columns

    Examples
    --------
    >>> data_dir = Path('data')
    >>> compounds = load_node_labels(data_dir, 'Compound')
    >>> compounds[0]
    'Goserelin'

    >>> pathways = load_node_labels(data_dir, 'Pathway')
    >>> len(pathways)
    1822
    """
    node_file = data_dir / 'nodes' / f'{node_type}.tsv'

    if not node_file.exists():
        raise FileNotFoundError(
            f"Node file not found: {node_file}\n"
            f"Expected file: data/nodes/{node_type}.tsv"
        )

    # Load TSV file
    try:
        df = pd.read_csv(node_file, sep='\t')
    except Exception as e:
        raise ValueError(f"Error reading {node_file}: {e}")

    # Verify required columns
    if 'name' not in df.columns:
        raise ValueError(
            f"Node file {node_file} missing 'name' column. "
            f"Found columns: {list(df.columns)}"
        )

    # Return names as list (indexed by position)
    return df['name'].tolist()


def add_labels_to_anomaly_df(
    anomaly_df: pd.DataFrame,
    data_dir: Path,
    source_type: str,
    target_type: str
) -> pd.DataFrame:
    """
    Add human-readable node labels to anomaly detection dataframe.

    Parameters
    ----------
    anomaly_df : DataFrame
        Anomaly detection results with source_idx and target_idx columns
    data_dir : Path
        Path to data directory
    source_type : str
        Source node type (e.g., 'Compound')
    target_type : str
        Target node type (e.g., 'Pathway')

    Returns
    -------
    DataFrame
        Copy of anomaly_df with added source_name and target_name columns

    Examples
    --------
    >>> anomaly_df = pd.DataFrame({
    ...     'source_idx': [0, 1, 2],
    ...     'target_idx': [10, 20, 30],
    ...     'z_score': [5.2, 4.8, 4.1]
    ... })
    >>> labeled_df = add_labels_to_anomaly_df(
    ...     anomaly_df, Path('data'), 'Compound', 'Pathway'
    ... )
    >>> 'source_name' in labeled_df.columns
    True
    """
    # Create copy to avoid modifying original
    df = anomaly_df.copy()

    # Load labels
    try:
        source_labels = load_node_labels(data_dir, source_type)
        target_labels = load_node_labels(data_dir, target_type)

        # Map indices to names
        df['source_name'] = df['source_idx'].apply(
            lambda idx: source_labels[int(idx)]
            if idx < len(source_labels) else f'Unknown_{idx}'
        )
        df['target_name'] = df['target_idx'].apply(
            lambda idx: target_labels[int(idx)]
            if idx < len(target_labels) else f'Unknown_{idx}'
        )

        print(f"✓ Mapped {len(source_labels)} {source_type}s "
              f"and {len(target_labels)} {target_type}s")

    except Exception as e:
        print(f"⚠️  Warning: Could not load node labels: {e}")
        print("   Continuing with indices only")
        df['source_name'] = df['source_idx'].astype(str)
        df['target_name'] = df['target_idx'].astype(str)

    return df
