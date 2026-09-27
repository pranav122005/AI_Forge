"""
Tests for /api/health and /api/capabilities Endpoints
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "components" in data
    assert "opencv" in data["components"]
    assert "ocr" in data["components"]
    assert "ml" in data["components"]
    assert "llm" in data["components"]


def test_capabilities_endpoint():
    res = client.get("/api/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert "llm" in data
    assert "vision" in data
    assert "ml" in data
    assert "agent" in data
    assert data["vision"]["opencv"] is True
    assert data["ml"]["text_classification"] is True
    assert data["agent"]["runtime"] is True
