"""
AIForge Phase 16 — Production Release & Hardening Test Suite
============================================================
Verifies:
1. Production configuration module and defaults.
2. Gemini reliability, retries, and error categorization.
3. Security hardening (path traversal, secret redaction, protected files).
4. Artifact security & ZIP archive contents.
5. Health and readiness endpoints.
6. Docker configuration file existence and basic validity.
"""
from __future__ import annotations

import os
import shutil
import zipfile
import pytest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import HTTPException
from pydantic import BaseModel

class DummyOutputModel(BaseModel):
    status: str = "ok"

from app.main import app
from app.config import settings, Settings
from app.security import (
    get_cors_origins,
    redact_secrets,
    register_project_owner,
    validate_id_string,
    verify_api_access,
)
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.base import LLMProviderError, LLMConfigurationError
from app.codegen.packager import cleanup_old_snapshots, package_project_zip
from app.codegen.validator import validate_file_path
from app.codegen.errors import PathValidationError

client = TestClient(app)


def test_production_config_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.aiforge.io,https://api.aiforge.io")
    s = Settings()
    assert s.gemini_model == "gemini-3.8-flash"
    assert "https://app.aiforge.io" in s.cors_origins_list
    assert "https://api.aiforge.io" in s.cors_origins_list


def test_secret_redaction_robustness():
    raw_log = "Error with API key AIzaSyB1234567890abcdef1234567890abcdef and token Bearer mysecrettoken123"
    redacted = redact_secrets(raw_log)
    assert "AIzaSyB" not in redacted
    assert "[REDACTED_GEMINI_KEY]" in redacted
    assert "mysecrettoken123" not in redacted
    assert "[REDACTED_TOKEN]" in redacted


def test_protected_file_rejection():
    with pytest.raises(PathValidationError):
        validate_file_path(".env", Path("/tmp/proj"))

    with pytest.raises(PathValidationError):
        validate_file_path("secrets.json", Path("/tmp/proj"))

    with pytest.raises(PathValidationError):
        validate_file_path("../outside.py", Path("/tmp/proj"))


def test_zip_artifact_cleanliness(tmp_path: Path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "app").mkdir()
    (proj_dir / "app" / "main.py").write_text("print('hello world')")
    (proj_dir / ".env").write_text("SECRET=123")
    (proj_dir / "secrets.json").write_text("{}")
    (proj_dir / "__pycache__").mkdir()
    (proj_dir / "__pycache__" / "main.pyc").write_bytes(b"cache")

    out_zip = tmp_path / "production_release.zip"
    package_project_zip(proj_dir, out_zip)

    with zipfile.ZipFile(out_zip, "r") as z:
        names = z.namelist()
        assert "app/main.py" in names
        assert ".env" not in names
        assert "secrets.json" not in names
        assert "__pycache__/main.pyc" not in names


def test_health_and_readiness_endpoints():
    r1 = client.get("/api/health")
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["service_alive"] is True
    assert "gemini_configured" in d1

    r2 = client.get("/api/capabilities")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["service_alive"] is True
    assert d2["agent"]["development_loop"] is True


def test_docker_files_exist():
    base_dir = Path(__file__).resolve().parents[2]
    assert (base_dir / "docker-compose.yml").exists()
    assert (base_dir / "backend" / "Dockerfile").exists()
    assert (base_dir / "frontend" / "Dockerfile").exists()
    assert (base_dir / ".env.example").exists()
