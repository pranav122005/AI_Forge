"""
Integration tests for AIForge APIs: /api/train, /api/predict, /api/agent/run, /api/predict-image
"""
from __future__ import annotations

import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_build_train_predict_agent_run_flow():
    # 1. Build a project scaffold
    build_resp = client.post(
        "/api/build",
        json={
            "project_name": "agent-test-proj",
            "spec": "Classify legal documents and expose a REST API.",
        },
    )
    assert build_resp.status_code == 200
    project_id = build_resp.json()["project_id"]

    # 2. Try predicting before training -> 404
    pred_404 = client.post(
        "/api/predict",
        json={"project_id": project_id, "text": "Some legal document text"},
    )
    assert pred_404.status_code == 404

    # 3. Train the model with valid CSV
    df = pd.DataFrame({
        "text": [
            "Court petition for stay order",
            "Signed affidavit of legal representative",
            "Petition filed in high court",
            "Affidavit of sworn statement",
            "Writ petition for legal remedy",
            "Affidavit verifying facts",
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
    assert "metrics" in train_data
    assert "accuracy" in train_data["metrics"]
    assert set(train_data["labels"]) == {"Affidavit", "Petition"}

    # 4. Predict via /api/predict
    pred_resp = client.post(
        "/api/predict",
        json={"project_id": project_id, "text": "Court petition for stay order"},
    )
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["project_id"] == project_id
    assert pred_data["prediction"] in ["Petition", "Affidavit"]
    assert "confidence" in pred_data
    assert len(pred_data["top_predictions"]) > 0

    # 5. Execute agent via /api/agent/run
    agent_resp = client.post(
        "/api/agent/run",
        json={"project_id": project_id, "input": "Court petition for stay order"},
    )
    assert agent_resp.status_code == 200
    agent_data = agent_resp.json()
    assert agent_data["status"] == "completed"
    assert "result" in agent_data
    assert "steps" in agent_data


def test_train_endpoint_invalid_csv():
    # Invalid CSV missing required columns
    df_bad = pd.DataFrame({"title": ["A"], "val": [1]})
    csv_bytes = df_bad.to_csv(index=False).encode("utf-8")

    resp = client.post(
        "/api/train?project_id=nonexistent-proj",
        files={"dataset": ("bad.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 404 or resp.status_code == 400


def test_agent_run_nonexistent_project():
    resp = client.post(
        "/api/agent/run",
        json={"project_id": "nonexistent_proj_999", "input": "test"},
    )
    assert resp.status_code == 404
