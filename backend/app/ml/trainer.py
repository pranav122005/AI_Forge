"""
AIForge ML Training Orchestrator
================================
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml.evaluator import evaluate_model
from app.ml.model import build_classifier_pipeline
from app.ml.preprocessing import validate_and_clean_dataframe


def train_text_classifier(
    df: pd.DataFrame,
) -> tuple[Pipeline, dict[str, Any], list[str]]:
    """
    Train and evaluate a text classification model from a pandas DataFrame.

    Returns (fitted_pipeline, metrics_dict, sorted_class_labels).
    """
    cleaned_df = validate_and_clean_dataframe(df)

    num_classes = int(cleaned_df["label"].nunique())
    total_samples = len(cleaned_df)
    labels = sorted(cleaned_df["label"].unique().tolist())

    test_count = max(num_classes, int(round(total_samples * 0.25)))
    if test_count >= total_samples:
        test_count = num_classes

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_df["text"],
        cleaned_df["label"],
        test_size=test_count,
        random_state=42,
        stratify=cleaned_df["label"],
    )

    pipeline = build_classifier_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        total_samples=total_samples,
        num_classes=num_classes,
    )

    return pipeline, metrics, labels
