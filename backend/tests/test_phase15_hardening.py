import os
import shutil
import pytest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import HTTPException
from pydantic import BaseModel

class DummyOutputModel(BaseModel):
    status: str = "ok"

from app.main import app
from app.security import (
    get_cors_origins,
    redact_secrets,
    register_project_owner,
    validate_id_string,
    verify_api_access,
)
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.base import LLMProviderError, LLMConfigurationError
from app.codegen.agent.models import validate_state_transition
from app.codegen.packager import cleanup_old_snapshots, package_project_zip
from app.codegen.validator import validate_file_path
from app.codegen.errors import PathValidationError

client = TestClient(app)


def test_secret_redaction():
    raw_log = "API_KEY=AIzaSyA1234567890abcdef1234567890abc and token Bearer mysecrettoken123"
    redacted = redact_secrets(raw_log)
    assert "AIzaSyA" not in redacted
    assert "[REDACTED_GEMINI_KEY]" in redacted
    assert "mysecrettoken123" not in redacted
    assert "[REDACTED_TOKEN]" in redacted


def test_id_string_validation():
    # Valid IDs
    assert validate_id_string("proj-123_abc") == "proj-123_abc"
    assert validate_id_string("run_456") == "run_456"

    # Insecure path traversal attempts
    with pytest.raises(HTTPException) as exc1:
        validate_id_string("../outside")
    assert exc1.value.status_code == 400

    with pytest.raises(HTTPException) as exc2:
        validate_id_string("proj/../../etc/passwd")
    assert exc2.value.status_code == 400

    with pytest.raises(HTTPException) as exc3:
        validate_id_string("proj\\win\\path")
    assert exc3.value.status_code == 400

    with pytest.raises(HTTPException) as exc4:
        validate_id_string("proj@special$id")
    assert exc4.value.status_code == 400


def test_cors_origins_config(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://aiforge.example.com, https://app.aiforge.io")
    origins = get_cors_origins()
    assert "https://aiforge.example.com" in origins
    assert "https://app.aiforge.io" in origins


def test_tenant_isolation_and_auth(monkeypatch):
    register_project_owner("proj-user1", "user-1")

    # Allowed for correct user
    assert verify_api_access("proj-user1", x_user_id="user-1") is True

    # Forbidden for different user
    with pytest.raises(HTTPException) as exc:
        verify_api_access("proj-user1", x_user_id="user-2")
    assert exc.value.status_code == 403

    # API key enforcement when required
    monkeypatch.setenv("AIFORGE_API_KEY", "secret-key-123")
    with pytest.raises(HTTPException) as exc_auth:
        verify_api_access("proj-user1", x_api_key="wrong-key")
    assert exc_auth.value.status_code == 401

    assert verify_api_access("proj-user1", x_api_key="secret-key-123", x_user_id="user-1") is True


def test_state_transitions():
    assert validate_state_transition("IDLE", "ANALYZING") is True
    assert validate_state_transition("ANALYZING", "PLANNING") is True
    assert validate_state_transition("PLANNING", "EXECUTING") is True
    assert validate_state_transition("EXECUTING", "TESTING") is True
    assert validate_state_transition("TESTING", "REPAIRING") is True
    assert validate_state_transition("REPAIRING", "TESTING") is True
    assert validate_state_transition("TESTING", "VERIFYING") is True
    assert validate_state_transition("VERIFYING", "PACKAGING") is True
    assert validate_state_transition("PACKAGING", "COMPLETED") is True

    # Invalid transitions
    assert validate_state_transition("COMPLETED", "EXECUTING") is False
    assert validate_state_transition("ROLLED_BACK", "PLANNING") is False
    assert validate_state_transition("IDLE", "COMPLETED") is False


def test_snapshot_cleanup(tmp_path: Path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()

    for i in range(5):
        s_dir = snapshots_dir / f"snapshot_{100 + i}"
        s_dir.mkdir()
        (s_dir / "file.txt").write_text(f"content {i}")

    removed = cleanup_old_snapshots(snapshots_dir, keep_latest=3)
    assert removed == 2
    remaining = [d.name for d in snapshots_dir.iterdir() if d.is_dir()]
    assert len(remaining) == 3


def test_zip_packaging_security(tmp_path: Path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "app").mkdir()
    (proj_dir / "app" / "main.py").write_text("print('hello')")
    (proj_dir / ".env").write_text("SECRET=123")
    (proj_dir / "secrets.json").write_text("{}")
    (proj_dir / "__pycache__").mkdir()
    (proj_dir / "__pycache__" / "main.pyc").write_bytes(b"cache")

    out_zip = tmp_path / "out.zip"
    package_project_zip(proj_dir, out_zip)

    import zipfile
    with zipfile.ZipFile(out_zip, "r") as z:
        names = z.namelist()
        assert "app/main.py" in names
        assert ".env" not in names
        assert "secrets.json" not in names
        assert "__pycache__/main.pyc" not in names


def test_health_and_capabilities_readiness():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service_alive"] is True
    assert "gemini_configured" in data
    assert "agent_available" in data

    cap_resp = client.get("/api/capabilities")
    assert cap_resp.status_code == 200
    cap_data = cap_resp.json()
    assert cap_data["service_alive"] is True
    assert cap_data["agent"]["development_loop"] is True


def test_gemini_provider_error_handling(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError):
        GeminiProvider(api_key=None)

    # Test error redaction
    provider = GeminiProvider(api_key="AIzaSyA_TEST_KEY_12345", model="gemini-3.8-flash")

    async def _test_fail():
        with pytest.raises(LLMProviderError) as exc_info:
            await provider.generate_structured("test", response_model=DummyOutputModel)
        assert "AIzaSyA_TEST_KEY_12345" not in str(exc_info.value)
        assert "[REDACTED_GEMINI_KEY]" in str(exc_info.value) or "[REDACTED_API_KEY]" in str(exc_info.value) or "failed" in str(exc_info.value).lower()

    asyncio.run(_test_fail())
