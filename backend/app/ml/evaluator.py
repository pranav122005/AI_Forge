"""
AIForge ML Model Evaluator
==========================
"""
from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test,
    y_test,
    total_samples: int,
    num_classes: int,
) -> dict[str, Any]:
    """
    Evaluate fitted pipeline against test set and compute metrics.
    """
    pred = pipeline.predict(X_test)

    accuracy = round(float(accuracy_score(y_test, pred)), 4)
    precision = round(
        float(precision_score(y_test, pred, average="weighted", zero_division=0)), 4
    )
    recall = round(
        float(recall_score(y_test, pred, average="weighted", zero_division=0)), 4
    )
    f1 = round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "samples": total_samples,
        "classes": num_classes,
        "test_samples": len(y_test),
    }
