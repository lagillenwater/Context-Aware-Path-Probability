"""
Utility functions for DWPC P-Value Validation

Helper functions for file I/O, logging, and common operations.
"""

import logging
from pathlib import Path
from typing import Any, Dict
import json
import pandas as pd


def setup_logging(name: str = "dwpc_pvalue_validation", level: int = logging.INFO):
    """
    Set up logging configuration.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def save_dataframe(df: pd.DataFrame, filepath: Path, **kwargs):
    """
    Save DataFrame to file with proper formatting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : Path
        Output file path.
    **kwargs
        Additional arguments passed to to_csv.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, sep="\t", index=False, **kwargs)


def load_dataframe(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from file.

    Parameters
    ----------
    filepath : Path
        Input file path.
    **kwargs
        Additional arguments passed to read_csv.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_csv(filepath, sep="\t", **kwargs)


def save_json(data: Dict[str, Any], filepath: Path):
    """
    Save dictionary to JSON file.

    Parameters
    ----------
    data : dict
        Data to save.
    filepath : Path
        Output file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Parameters
    ----------
    filepath : Path
        Input file path.

    Returns
    -------
    dict
        Loaded data.
    """
    with open(filepath) as f:
        return json.load(f)
