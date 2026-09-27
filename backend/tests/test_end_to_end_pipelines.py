"""
End-to-End Integration Tests for AIForge Pipelines
"""
from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from fastapi.testclient import TestClient

from app.llm.models import RequirementSpec
from app.main import app
from app.ocr_component import is_available as ocr_available

client = TestClient(app)


def test_end_to_end_text_classification_pipeline():
    """
    Complete end-to-end text classification pipeline test:
    Requirement -> LLM -> Planner -> Builder -> Scaffolding -> Training -> Model -> Prediction
    """
    # 1. Submit requirement to POST /api/generate
    mock_spec = RequirementSpec(
        goal="Classify legal documents into categories.",
        requires_classification=True,
        requires_api=True,
        domain="legal",
    )

    with patch("app.llm.service.parse_requirement") as mock_parse:
        mock_parse.return_value = mock_spec

        gen_resp = client.post(
            "/api/generate",
            json={"requirement": "Build a text classifier for legal documents."},
        )
        assert gen_resp.status_code == 200
        gen_data = gen_resp.json()

        project_id = gen_data["project_id"]
        assert "text_classifier" in gen_data["selected_components"]
        assert gen_data["lifecycle_status"] == "READY_FOR_TRAINING"

    # 2. Train model via POST /api/train with dataset
    df = pd.DataFrame({
        "text": [
            "Court petition for stay order",
            "Signed affidavit of legal representative",
            "Petition filed in high court for injunction",
            "Sworn affidavit confirming statements",
            "Writ petition for legal remedy",
            "Notarized affidavit of witness",
        ],
        "label": ["Petition", "Affidavit", "Petition", "Affidavit", "Petition", "Affidavit"],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    train_resp = client.post(
        f"/api/train?project_id={project_id}",
        files={"dataset": ("dataset.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["status"] == "trained"
    assert "inspection" in train_data
    assert train_data["inspection"]["classes"] == 2

    # 3. Predict via POST /api/predict
    pred_resp = client.post(
        "/api/predict",
        json={"project_id": project_id, "text": "Urgent court petition for stay order"},
    )
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["prediction"] in ["Petition", "Affidavit"]
    assert pred_data["confidence"] > 0

    # 4. Execute pipeline via POST /api/agent/run
    agent_resp = client.post(
        "/api/agent/run",
        json={"project_id": project_id, "input": "Urgent court petition for stay order"},
    )
    assert agent_resp.status_code == 200
    agent_data = agent_resp.json()
    assert agent_data["status"] == "completed"
    assert "text_classifier" in [s["component"] for s in agent_data["steps"]]


def test_end_to_end_image_pipeline():
    """
    Complete end-to-end image pipeline test:
    Image -> OpenCV Preprocessing -> OCR -> Text Classifier -> Result
    """
    # 1. Build a project scaffold
    gen_resp = client.post(
        "/api/generate",
        json={"requirement": "Build a system that reads legal documents from images and classifies them."},
    )
    assert gen_resp.status_code == 200
    project_id = gen_resp.json()["project_id"]

    # 2. Train model first
    df = pd.DataFrame({
        "text": [
            "Legal petition filed in court",
            "Sworn affidavit of identity",
            "Petition for injunction order",
            "Affidavit of witness statement",
            "Court petition for remedy",
            "Notarized affidavit document",
        ],
        "label": ["Petition", "Affidavit", "Petition", "Affidavit", "Petition", "Affidavit"],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    train_resp = client.post(
        f"/api/train?project_id={project_id}",
        files={"dataset": ("dataset.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert train_resp.status_code == 200

    # 3. Generate sample image bytes
    img = Image.new("RGB", (100, 50), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # If tesseract system binary is available, test predict-image directly
    # If not, test with mocked pytesseract.image_to_string
    if ocr_available():
        res = client.post(
            f"/api/predict-image?project_id={project_id}",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert res.status_code in [200, 422]
    else:
        with patch("app.ocr_component.is_available", return_value=True):
            with patch("app.ocr_component.extract_text", return_value="Legal petition filed in court"):
                res = client.post(
                    f"/api/predict-image?project_id={project_id}",
                    files={"file": ("test.png", img_bytes, "image/png")},
                )
                assert res.status_code == 200
                data = res.json()
                assert data["extracted_text"] == "Legal petition filed in court"
                assert data["prediction"] in ["Petition", "Affidavit"]

