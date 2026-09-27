"""
AIForge ML Dataset Analysis & Inspection Module
===============================================

Provides deterministic statistics and column detection for datasets prior to training.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Inspect a pandas DataFrame and extract dataset summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.

    Returns
    -------
    dict[str, Any]
        Summary metrics including row/col counts, missing values, duplicates,
        detected text and label columns, class count, and class distribution.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    total_rows = len(df)
    columns = [str(c) for c in df.columns]

    missing_values = {str(col): int(df[col].isnull().sum()) for col in df.columns}
    duplicate_rows = int(df.duplicated().sum())

    # Detect text column
    text_column = None
    if "text" in df.columns:
        text_column = "text"
    else:
        # Fallback to first string/object column
        string_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype == "string"]
        if string_cols:
            text_column = str(string_cols[0])
        elif len(df.columns) > 0:
            text_column = str(df.columns[0])

    # Detect label column
    label_column = None
    if "label" in df.columns:
        label_column = "label"
    elif "category" in df.columns:
        label_column = "category"
    elif "target" in df.columns:
        label_column = "target"
    else:
        # Fallback to second string/object column or second column overall
        other_cols = [c for c in df.columns if str(c) != text_column]
        if other_cols:
            label_column = str(other_cols[0])

    classes = 0
    class_distribution: dict[str, int] = {}

    if label_column and label_column in df.columns:
        label_series = df[label_column].dropna().astype(str)
        classes = int(label_series.nunique())
        val_counts = label_series.value_counts().to_dict()
        class_distribution = {str(k): int(v) for k, v in val_counts.items()}

    return {
        "rows": total_rows,
        "columns": columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "text_column": text_column,
        "label_column": label_column,
        "classes": classes,
        "class_distribution": class_distribution,
    }
