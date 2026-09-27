"""
Tests for ML Dataset Inspection Module
"""
from __future__ import annotations

import pandas as pd
import pytest
from app.ml.dataset import inspect_dataset


def test_inspect_dataset_basic():
    df = pd.DataFrame({
        "text": ["Doc A", "Doc B", "Doc C", "Doc D"],
        "label": ["Cat1", "Cat2", "Cat1", "Cat2"],
    })
    res = inspect_dataset(df)

    assert res["rows"] == 4
    assert res["columns"] == ["text", "label"]
    assert res["text_column"] == "text"
    assert res["label_column"] == "label"
    assert res["classes"] == 2
    assert res["class_distribution"] == {"Cat1": 2, "Cat2": 2}
    assert res["duplicate_rows"] == 0


def test_inspect_dataset_detects_duplicates_and_missing():
    df = pd.DataFrame({
        "text": ["Doc A", "Doc A", None, "Doc B"],
        "category": ["X", "X", "Y", "Y"],
    })
    res = inspect_dataset(df)

    assert res["rows"] == 4
    assert res["text_column"] == "text"
    assert res["label_column"] == "category"
    assert res["duplicate_rows"] == 1
    assert res["missing_values"]["text"] == 1


def test_inspect_dataset_invalid_input():
    with pytest.raises(ValueError):
        inspect_dataset("not a dataframe")  # type: ignore
