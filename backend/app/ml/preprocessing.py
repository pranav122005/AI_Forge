"""
AIForge ML Data Preprocessing Module
====================================
"""
from __future__ import annotations

import pandas as pd


def validate_and_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean CSV dataset for ML training.

    Ensures 'text' and 'label' columns exist, removes NaNs, and checks
    minimum sample and class count requirements.
    """
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV dataset must contain 'text' and 'label' columns.")

    cleaned_df = df[["text", "label"]].dropna().copy()
    cleaned_df["text"] = cleaned_df["text"].astype(str).str.strip()
    cleaned_df["label"] = cleaned_df["label"].astype(str).str.strip()

    # Filter out empty strings
    cleaned_df = cleaned_df[cleaned_df["text"].str.len() > 0]
    cleaned_df = cleaned_df[cleaned_df["label"].str.len() > 0]

    if len(cleaned_df) < 6:
        raise ValueError("Dataset needs at least 6 valid samples to train.")

    if cleaned_df["label"].nunique() < 2:
        raise ValueError("Dataset requires at least 2 distinct class labels.")

    return cleaned_df
