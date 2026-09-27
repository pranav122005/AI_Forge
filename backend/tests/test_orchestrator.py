"""
Tests for AIForge Single Orchestration Service & /api/generate
"""
from __future__ import annotations

import asyncio
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.orchestrator.models import ProjectLifecycle
from app.orchestrator.service import AIForgeOrchestrator, default_orchestrator

client = TestClient(app)


def test_orchestrator_generate_project_flow():
    orchestrator = AIForgeOrchestrator()
    requirement = "Classify legal documents and expose a REST API."

    result = asyncio.run(orchestrator.generate_project(requirement, project_name="orch-test-proj"))

    assert "project_id" in result
    assert result["project_name"] == "orch-test-proj"
    assert "requirement" in result
    assert "plan" in result
    assert result["lifecycle_status"] == ProjectLifecycle.READY_FOR_TRAINING
    assert "text_classifier" in result["selected_components"]
    assert "agent/runtime.py" in result["generated_files"]
    assert result["training"]["enabled"] is True


def test_api_generate_endpoint():
    resp = client.post(
        "/api/generate",
        json={"requirement": "Build a legal document classifier using OCR"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "project_id" in data
    assert "requirement" in data
    assert "plan" in data
    assert "execution_status" in data
    assert "selected_components" in data
    assert "generated_files" in data
    assert "agent/runtime.py" in data["generated_files"]
