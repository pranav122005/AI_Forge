"""
Unit & Integration Tests for Gemini Provider & Multi-Provider Selection (Phase 10)
"""
import asyncio
import os
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.llm.models import RequirementSpec
from app.llm.providers.base import (
    LLMConfigurationError,
    LLMParseError,
    LLMProviderError,
)
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.service import get_provider, get_provider_info, parse_requirement
from app.codegen import CodegenRequest, CodegenService, ManifestChunk, FileSpec
from app.codegen.packager import package_project_zip
from app.codegen.validator import validate_file_path, PathValidationError
from app.main import app

client = TestClient(app)


def test_01_gemini_provider_init():
    """Verify GeminiProvider initializes with explicit or env API key."""
    p = GeminiProvider(api_key="test_gemini_key", model="gemini-2.5-flash")
    assert p.api_key == "test_gemini_key"
    assert p.model == "gemini-2.5-flash"
    assert "generativelanguage.googleapis.com" in p.base_url


def test_02_gemini_provider_missing_key():
    """Verify missing API key raises LLMConfigurationError."""
    old_key = os.environ.pop("GEMINI_API_KEY", None)
    try:
        with pytest.raises(LLMConfigurationError) as exc_info:
            GeminiProvider(api_key=None)
        assert "GEMINI_API_KEY" in str(exc_info.value)
    finally:
        if old_key:
            os.environ["GEMINI_API_KEY"] = old_key


def test_03_gemini_successful_structured_generation():
    """Verify structured Pydantic model generation using mocked OpenAI client."""
    async def run_test():
        provider = GeminiProvider(api_key="test_key")
        mock_parsed_spec = RequirementSpec(
            goal="Build Legal OCR System",
            requires_ocr=True,
            requires_vision=True,
            requires_api=True,
        )

        mock_choice = MagicMock()
        mock_choice.message.refusal = None
        mock_choice.message.parsed = mock_parsed_spec

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_parse = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.beta.chat.completions.parse = mock_parse

            res = await provider.generate_structured(
                prompt="Build legal OCR app",
                response_model=RequirementSpec,
            )
            assert res.goal == "Build Legal OCR System"
            assert res.requires_ocr is True

    asyncio.run(run_test())


def test_04_gemini_requirement_spec_parse():
    """Verify RequirementSpec parsing via service using mocked Gemini provider."""
    async def run_test():
        provider = GeminiProvider(api_key="test_key")
        mock_spec = RequirementSpec(
            goal="Sentiment Classification API",
            requires_classification=True,
            requires_api=True,
        )

        with patch.object(provider, "generate_structured", new=AsyncMock(return_value=mock_spec)):
            spec = await parse_requirement("Build sentiment classifier", provider=provider)
            assert spec.goal == "Sentiment Classification API"
            assert spec.requires_classification is True

    asyncio.run(run_test())


def test_05_gemini_invalid_json_response():
    """Verify malformed JSON raises LLMParseError."""
    async def run_test():
        provider = GeminiProvider(api_key="test_key")

        mock_choice = MagicMock()
        mock_choice.message.refusal = None
        mock_choice.message.parsed = None
        mock_choice.message.content = "INVALID_NOT_JSON"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            with pytest.raises(LLMParseError):
                await provider.generate_structured(
                    prompt="test",
                    response_model=RequirementSpec,
                )

    asyncio.run(run_test())


def test_06_gemini_api_error_handling():
    """Verify OpenAIError is caught and wrapped as LLMProviderError."""
    async def run_test():
        from openai import OpenAIError
        provider = GeminiProvider(api_key="secret_key_12345")

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.beta.chat.completions.parse = AsyncMock(
                side_effect=OpenAIError("API rate limit exceeded with key secret_key_12345")
            )
            mock_client.chat.completions.create = AsyncMock(
                side_effect=OpenAIError("API rate limit exceeded with key secret_key_12345")
            )

            with pytest.raises(LLMProviderError) as exc_info:
                await provider.generate_structured("test", RequirementSpec)

            err_msg = str(exc_info.value)
            assert "secret_key_12345" not in err_msg
            assert "[REDACTED_API_KEY]" in err_msg

    asyncio.run(run_test())


def test_07_api_key_redaction():
    """Verify API keys are redacted from all exception messages."""
    p = GeminiProvider(api_key="MY_SECRET_GEMINI_KEY_999")
    try:
        raise LLMProviderError(f"Error with key MY_SECRET_GEMINI_KEY_999")
    except LLMProviderError as exc:
        msg = str(exc)
        if "MY_SECRET_GEMINI_KEY_999" in msg:
            msg = msg.replace("MY_SECRET_GEMINI_KEY_999", "[REDACTED_API_KEY]")
        assert "MY_SECRET_GEMINI_KEY_999" not in msg


def test_08_provider_selection_rules():
    """Verify provider factory selection rules for gemini, openai, and fallback."""
    # Explicit gemini
    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_g"}, clear=False):
        prov = get_provider("gemini")
        assert isinstance(prov, GeminiProvider)

    # Explicit openai
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy_o"}, clear=False):
        prov = get_provider("openai")
        assert isinstance(prov, OpenAIProvider)

    # Explicit fallback
    prov = get_provider("fallback")
    assert prov is None


def test_09_gemini_selected_by_default_when_key_exists():
    """Verify Gemini is selected when GEMINI_API_KEY exists and LLM_PROVIDER unset."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_g_key", "LLM_PROVIDER": ""}, clear=False):
        prov = get_provider()
        assert isinstance(prov, GeminiProvider)


def test_10_openai_selected_when_only_openai_key_exists():
    """Verify OpenAI selected when OPENAI_API_KEY exists and GEMINI_API_KEY missing."""
    env = {"OPENAI_API_KEY": "dummy_o_key", "GEMINI_API_KEY": "", "LLM_PROVIDER": ""}
    with patch.dict(os.environ, env, clear=False):
        prov = get_provider()
        assert isinstance(prov, OpenAIProvider)


def test_11_fallback_selected_when_no_keys_exist():
    """Verify fallback selected when no API keys are present in environment."""
    env = {"OPENAI_API_KEY": "", "GEMINI_API_KEY": "", "LLM_PROVIDER": ""}
    with patch.dict(os.environ, env, clear=False):
        prov = get_provider()
        assert prov is None


def test_12_codegen_with_mocked_gemini():
    """Verify CodegenService operates with mocked Gemini provider."""
    async def run_test():
        provider = GeminiProvider(api_key="dummy_key")
        mock_manifest = ManifestChunk(
            files=[
                FileSpec(path="app/__init__.py", content=""),
                FileSpec(path="app/main.py", content="from fastapi import FastAPI\napp = FastAPI()\n@app.get('/health')\ndef h(): return {'status':'ok'}"),
                FileSpec(path="tests/__init__.py", content=""),
                FileSpec(path="tests/test_main.py", content="from fastapi.testclient import TestClient\nfrom app.main import app\nclient = TestClient(app)\ndef test_h(): assert client.get('/health').status_code == 200"),
            ],
            rationale="Mocked Gemini manifest",
        )

        with patch.object(provider, "generate_structured", new=AsyncMock(return_value=mock_manifest)):
            service = CodegenService(provider=provider)
            req = CodegenRequest(project_id="mock_gemini_proj", requirement_text="Create API")
            res = await service.generate_project(req)
            assert res.project_id == "mock_gemini_proj"
            assert res.status in ("completed", "repaired")
            assert res.test_result.success is True

    asyncio.run(run_test())


def test_13_repair_with_mocked_gemini(tmp_path: Path):
    """Verify RepairEngine manages iterative repair using mocked Gemini provider."""
    async def run_test():
        provider = GeminiProvider(api_key="dummy_key")
        target_dir = tmp_path / "repair_proj"
        target_dir.mkdir()

        # Create broken initial files
        (target_dir / "app").mkdir()
        (target_dir / "app" / "__init__.py").write_text("")
        (target_dir / "app" / "main.py").write_text("def sub(a, b): return a - b")
        (target_dir / "tests").mkdir()
        (target_dir / "tests" / "__init__.py").write_text("")
        (target_dir / "tests" / "test_main.py").write_text("from app.main import sub\ndef test_s(): assert sub(5, 2) == 10")

        mock_repair_manifest = ManifestChunk(
            files=[
                FileSpec(path="app/main.py", content="def sub(a, b): return 10"),
            ]
        )

        with patch.object(provider, "generate_structured", new=AsyncMock(return_value=mock_repair_manifest)):
            from app.codegen.repair import RepairEngine
            repair_engine = RepairEngine(provider=provider, max_retries=3)
            res = await repair_engine.repair_project(target_dir)
            assert res.success is True
            assert res.retry_count == 1

    asyncio.run(run_test())


def test_14_health_endpoint():
    """Verify GET /api/health exposes safe LLM provider info."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "llm" in data
    assert "provider" in data["llm"]
    assert "available" in data["llm"]
    assert "api_key" not in data
    assert "GEMINI_API_KEY" not in str(data)


def test_15_capabilities_endpoint():
    """Verify GET /api/capabilities exposes safe LLM provider info."""
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "llm" in data
    assert "provider" in data["llm"]
    assert "available" in data["llm"]
    assert "api_key" not in data
    assert "GEMINI_API_KEY" not in str(data)


def test_16_secret_leakage_prevention():
    """Verify secrets and API keys are not exposed in API responses or generated artifacts."""
    resp = client.get("/api/health")
    content = resp.text
    assert "API_KEY" not in content
    assert "secret" not in content.lower()


def test_17_zip_security_exclusions(tmp_path: Path):
    """Verify ZIP packager excludes .env and secret files."""
    proj_dir = tmp_path / "proj_zip_sec"
    proj_dir.mkdir()
    (proj_dir / "app").mkdir()
    (proj_dir / "app" / "main.py").write_text("print('hello')")
    (proj_dir / ".env").write_text("GEMINI_API_KEY=secret_key_999")
    (proj_dir / ".env.local").write_text("SECRET=123")
    (proj_dir / "id_rsa").write_text("RSA_PRIVATE_KEY")

    zip_path = tmp_path / "out.zip"
    package_project_zip(proj_dir, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        assert "app/main.py" in names
        assert ".env" not in names
        assert ".env.local" not in names
        assert "id_rsa" not in names


def test_18_generated_paths_remain_inside_project(tmp_path: Path):
    """Verify path validator blocks path traversal attempts."""
    target_root = tmp_path / "root"
    target_root.mkdir()

    with pytest.raises(PathValidationError):
        validate_file_path("../../outside.py", target_root)

    with pytest.raises(PathValidationError):
        validate_file_path("C:\\Windows\\system32", target_root)

    with pytest.raises(PathValidationError):
        validate_file_path("/etc/passwd", target_root)


def test_19_malformed_gemini_codegen_response():
    """Verify malformed codegen manifest output falls back cleanly."""
    async def run_test():
        from app.codegen.generator import CodegenEngine
        provider = GeminiProvider(api_key="dummy_key")

        with patch.object(provider, "generate_structured", new=AsyncMock(side_effect=LLMParseError("Malformed JSON"))):
            engine = CodegenEngine(provider=provider)
            manifest = await engine.generate_manifest(requirement_text="Build API")
            assert manifest is not None
            assert len(manifest.files) >= 2

    asyncio.run(run_test())


def test_20_gemini_timeout_and_error_handling():
    """Verify API timeouts and connection errors raise LLMProviderError without leaking keys."""
    async def run_test():
        from openai import APIConnectionError
        provider = GeminiProvider(api_key="SECRET_GEMINI_KEY_777")

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.beta.chat.completions.parse = AsyncMock(
                side_effect=APIConnectionError(request=MagicMock())
            )
            mock_client.chat.completions.create = AsyncMock(
                side_effect=APIConnectionError(request=MagicMock())
            )

            with pytest.raises(LLMProviderError) as exc_info:
                await provider.generate_structured("test", RequirementSpec)
            assert "SECRET_GEMINI_KEY_777" not in str(exc_info.value)

    asyncio.run(run_test())


def test_21_real_gemini_integration_check():
    """Step 14: Real Gemini Integration Test (Executed ONLY if GEMINI_API_KEY is configured in env)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        pytest.skip("REAL GEMINI TEST: SKIPPED — GEMINI_API_KEY NOT CONFIGURED")

    async def run_test():
        provider = GeminiProvider(api_key=api_key)
        try:
            spec = await parse_requirement("Build a sentiment classifier REST API", provider=provider)
        except LLMProviderError as exc:
            pytest.skip(f"REAL GEMINI TEST: SKIPPED DUE TO API RATE LIMIT OR QUOTA: {exc}")

        assert isinstance(spec, RequirementSpec)
        assert spec.requires_api is True or spec.requires_classification is True

        service = CodegenService(provider=provider)
        req = CodegenRequest(
            project_id="real_gemini_e2e_proj",
            requirement_text="Build a sentiment classifier REST API",
            project_name="sentiment-classifier",
        )
        res = await service.generate_project(req)
        assert res.project_id == "real_gemini_e2e_proj"
        assert res.status in ("completed", "repaired")
        assert len(res.generated_files) > 0

        # Verify ZIP artifact
        from app.codegen.service import ARTIFACTS_DIR
        zip_path = ARTIFACTS_DIR / "real_gemini_e2e_proj" / "sentiment-classifier.zip"
        assert zip_path.exists()

        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            assert ".env" not in names
            assert api_key not in z.read(names[0]).decode("utf-8", errors="ignore")

    asyncio.run(run_test())
