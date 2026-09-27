import os
import shutil
import pytest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.codegen.service import CodegenService, resolve_project_dir
from app.codegen.models import (
    CodegenRequest,
    CodegenModifyRequest,
    FileChangeSpec,
    ProjectModificationManifest,
)
from app.codegen.modifier import ProjectModifierEngine, build_project_context
from app.codegen.validator import validate_file_path
from app.codegen.errors import PathValidationError
from app.llm.providers.base import BaseLLMProvider, T

client = TestClient(app)

class MockModifyLLMProvider(BaseLLMProvider):
    def __init__(self, manifest: ProjectModificationManifest):
        self.manifest = manifest

    async def generate_structured(self, prompt: str, response_model: type[T], system_prompt: str | None = None) -> T:
        return self.manifest


def test_build_project_context(tmp_path: Path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "main.py").write_text("print('hello')", encoding="utf-8")
    (proj_dir / ".env").write_text("SECRET=123", encoding="utf-8")
    (proj_dir / "__pycache__").mkdir()
    (proj_dir / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"bin")

    ctx = build_project_context(proj_dir)
    assert "main.py" in ctx
    assert ".env" not in ctx
    assert "__pycache__" not in ctx
    assert "print('hello')" in ctx


def test_modifier_engine_fallback(tmp_path: Path):
    async def _run():
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        (proj_dir / "app").mkdir()
        (proj_dir / "app" / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")

        engine = ProjectModifierEngine(provider=None)
        manifest = await engine.generate_modification(proj_dir, instruction="Add health check endpoint")

        assert "health" in manifest.summary.lower()
        assert len(manifest.changes) >= 1

    asyncio.run(_run())


def test_modifier_path_traversal_rejection(tmp_path: Path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()

    with pytest.raises(PathValidationError):
        validate_file_path("../outside.txt", proj_dir)


def test_codegen_service_modify_flow():
    async def _run():
        service = CodegenService()

        # First, generate a base project
        base_res = await service.generate_project(CodegenRequest(requirement_text="Build a simple calculator python module with test_calc.py"))
        assert base_res.status in ("completed", "repaired")
        project_id = base_res.project_id

        # Mock provider for modification
        mock_manifest = ProjectModificationManifest(
            summary="Add multiply feature",
            changes=[
                FileChangeSpec(
                    action="create",
                    path="calculator.py",
                    content="def add(a, b): return a + b\ndef multiply(a, b): return a * b\n",
                    description="Add multiply function"
                ),
                FileChangeSpec(
                    action="create",
                    path="tests/test_calc.py",
                    content="from calculator import add, multiply\ndef test_add(): assert add(2, 3) == 5\ndef test_mult(): assert multiply(2, 3) == 6\n",
                    description="Add calc tests"
                )
            ]
        )
        mock_provider = MockModifyLLMProvider(mock_manifest)
        service.provider = mock_provider

        mod_res = await service.modify_project(
            project_id=project_id,
            instruction="Add multiply feature to calculator"
        )

        assert mod_res.status in ("SUCCESS", "COMPLETED", "REPAIRED")
        assert mod_res.project_id == project_id
        assert mod_res.summary == "Add multiply feature"
        assert mod_res.history is not None
        assert len(mod_res.history) >= 1

        # Verify zip path exists
        zip_path = service.get_zip_file_path(project_id)
        assert os.path.exists(zip_path)

    asyncio.run(_run())


def test_codegen_service_modify_rollback_on_failure():
    async def _run():
        service = CodegenService()

        # Base project
        base_res = await service.generate_project(CodegenRequest(requirement_text="Build a python math script with passing pytest tests/test_math.py"))
        assert base_res.status in ("completed", "repaired")
        project_id = base_res.project_id

        # Mock broken code that fails pytest and cannot be repaired
        broken_manifest = ProjectModificationManifest(
            summary="Introduce broken test",
            changes=[
                FileChangeSpec(
                    action="create",
                    path="tests/test_broken.py",
                    content="def test_fail(): assert False, 'forced failure'",
                    description="Broken test"
                )
            ]
        )
        mock_provider = MockModifyLLMProvider(broken_manifest)
        service.provider = mock_provider

        # Patch repair_engine max_retries to 0 so it fails fast
        original_attempts = service.repair_engine.max_retries
        service.repair_engine.max_retries = 0

        try:
            mod_res = await service.modify_project(
                project_id=project_id,
                instruction="Break the tests"
            )

            assert mod_res.status == "ROLLED_BACK"
            # Verify broken file was rolled back
            project_dir = resolve_project_dir(project_id)
            assert not (project_dir / "tests" / "test_broken.py").exists()
        finally:
            service.repair_engine.max_retries = original_attempts

    asyncio.run(_run())


def test_codegen_modify_api_endpoint():
    # Generate a project via API client
    gen_resp = client.post("/api/codegen/generate", json={"requirement_text": "Build a python string helper with passing tests/test_str.py"})
    assert gen_resp.status_code == 200
    gen_data = gen_resp.json()
    project_id = gen_data["project_id"]

    # Now call modify endpoint with prompt
    mod_resp = client.post(
        f"/api/codegen/{project_id}/modify",
        json={"instruction": "Add health check endpoint"}
    )
    assert mod_resp.status_code in (200, 429)
    if mod_resp.status_code == 200:
        res_data = mod_resp.json()
        assert res_data["project_id"] == project_id
        assert "status" in res_data
        assert "summary" in res_data


def test_real_gemini_project_modification():
    """Optional real Gemini integration test if GEMINI_API_KEY is available."""
    async def _run():
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY not configured in environment")

        service = CodegenService()
        # Base generation
        gen_res = await service.generate_project(CodegenRequest(requirement_text="Build a Python math utility with add(a, b) and tests/test_math.py with passing pytest"))
        assert gen_res.status in ("completed", "repaired")

        project_id = gen_res.project_id

        from app.llm.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key=api_key, model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
        service.provider = provider

        try:
            mod_res = await service.modify_project(
                project_id=project_id,
                instruction="Add a subtract(a, b) function to math utility and add pytest in tests/test_math.py for subtract"
            )
            assert mod_res.status in ("SUCCESS", "COMPLETED", "REPAIRED", "ROLLED_BACK")
            assert mod_res.project_id == project_id
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e) or "quota" in str(e).lower():
                pytest.skip(f"Gemini API rate limit / quota exceeded: {e}")
                return
            raise

    asyncio.run(_run())
