"""
Tests for ML Training Pipeline, Evaluation, Persistence & Prediction
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.ml.evaluator import evaluate_model
from app.ml.model import build_classifier_pipeline
from app.ml.persistence import get_model_metadata, load_model, model_exists, save_model
from app.ml.preprocessing import validate_and_clean_dataframe
from app.ml.trainer import train_text_classifier


def test_validate_and_clean_dataframe_success():
    data = {
        "text": ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5", "Doc 6"],
        "label": ["A", "B", "A", "B", "A", "B"],
    }
    df = pd.DataFrame(data)
    cleaned = validate_and_clean_dataframe(df)

    assert len(cleaned) == 6
    assert list(cleaned.columns) == ["text", "label"]


def test_validate_and_clean_dataframe_invalid_columns():
    df = pd.DataFrame({"title": ["Doc 1"], "category": ["A"]})
    with pytest.raises(ValueError) as exc:
        validate_and_clean_dataframe(df)
    assert "must contain 'text' and 'label'" in str(exc.value)


def test_validate_and_clean_dataframe_insufficient_samples():
    df = pd.DataFrame({
        "text": ["Doc 1", "Doc 2"],
        "label": ["A", "B"],
    })
    with pytest.raises(ValueError) as exc:
        validate_and_clean_dataframe(df)
    assert "at least 6 valid samples" in str(exc.value)


def test_validate_and_clean_dataframe_single_class():
    df = pd.DataFrame({
        "text": [f"Doc {i}" for i in range(10)],
        "label": ["A"] * 10,
    })
    with pytest.raises(ValueError) as exc:
        validate_and_clean_dataframe(df)
    assert "at least 2 distinct class labels" in str(exc.value)


def test_train_text_classifier_and_persistence(tmp_path, monkeypatch):
    data = {
        "text": [
            "Legal petition filed in high court for relief",
            "Urgent affidavit submitted by legal counsel",
            "Official court petition regarding property dispute",
            "Sworn affidavit confirming identity and address",
            "Formal petition seeking injunction order",
            "Signed affidavit of witness testimony",
            "Petition for writ of habeas corpus",
            "Notarized affidavit of financial status",
        ],
        "label": ["Petition", "Affidavit", "Petition", "Affidavit", "Petition", "Affidavit", "Petition", "Affidavit"],
    }
    df = pd.DataFrame(data)

    pipeline, metrics, labels = train_text_classifier(df)

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert metrics["samples"] == 8
    assert metrics["classes"] == 2
    assert labels == ["Affidavit", "Petition"]

    # Test persistence
    project_id = "test-ml-proj-123"
    save_model(project_id, pipeline, metrics, labels)

    assert model_exists(project_id) is True

    loaded_pipeline = load_model(project_id)
    pred = loaded_pipeline.predict(["Legal petition in court"])[0]
    assert pred in ["Petition", "Affidavit"]

    metadata = get_model_metadata(project_id)
    assert metadata["project_id"] == project_id
    assert metadata["labels"] == ["Affidavit", "Petition"]
