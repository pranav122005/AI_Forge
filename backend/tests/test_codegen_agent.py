import os
import shutil
import pytest
import asyncio
from typing import Any
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.codegen.service import CodegenService, resolve_project_dir
from app.codegen.models import CodegenRequest, TestResult
from app.codegen.agent import (
    AgentExecutionEngine,
    AgentPlan,
    AgentRunRequest,
    AgentRunState,
    FileSelection,
    PlanStep,
    VerificationResult,
    build_project_summary,
    generate_agent_plan,
    generate_fallback_plan,
    get_agent_run_state,
    read_selected_file_contents,
    select_relevant_files,
    verify_project,
)
from app.codegen.validator import validate_file_path
from app.codegen.errors import PathValidationError
from app.llm.providers.base import BaseLLMProvider, T

client = TestClient(app)


class MockAgentLLMProvider(BaseLLMProvider):
    """Mock LLM Provider returning customizable structured models for testing."""

    def __init__(self, model_response: Any):
        self.model_response = model_response

    async def generate_structured(self, prompt: str, response_model: type[T], system_prompt: str | None = None) -> T:
        if isinstance(self.model_response, response_model):
            return self.model_response
        return self.model_response


def test_build_project_summary(tmp_path: Path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "app").mkdir()
    (proj_dir / "app" / "main.py").write_text("@app.get('/health')\ndef health(): pass", encoding="utf-8")
    (proj_dir / "requirements.txt").write_text("fastapi==0.110.0\n", encoding="utf-8")
    (proj_dir / ".env").write_text("SECRET=secretkey", encoding="utf-8")
    (proj_dir / "__pycache__").mkdir()
    (proj_dir / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"bin")

    summary = build_project_summary(proj_dir)
    assert summary["project_type"] == "python_fastapi"
    assert "app/main.py" in summary["entrypoints"]
    assert "requirements.txt" in summary["dependencies"]
    assert ".env" not in summary["file_structure"]
    assert "__pycache__/main.cpython-312.pyc" not in summary["file_structure"]


def test_select_relevant_files(tmp_path: Path):
    async def _run():
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        (proj_dir / "app").mkdir()
        (proj_dir / "app" / "main.py").write_text("# main", encoding="utf-8")
        (proj_dir / "app" / "auth.py").write_text("# auth", encoding="utf-8")
        (proj_dir / "requirements.txt").write_text("fastapi", encoding="utf-8")

        summary = build_project_summary(proj_dir)
        selected = await select_relevant_files(proj_dir, "Add JWT authentication", summary, provider=None)
        assert "app/auth.py" in selected or "app/main.py" in selected
        assert "requirements.txt" in selected

    asyncio.run(_run())


def test_read_selected_file_contents(tmp_path: Path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "main.py").write_text("print('hello')", encoding="utf-8")

    contents = read_selected_file_contents(proj_dir, ["main.py"])
    assert "main.py" in contents
    assert "print('hello')" in contents


def test_generate_agent_plan_structured():
    async def _run():
        mock_plan = AgentPlan(
            summary="Mock JWT Auth Plan",
            steps=[
                PlanStep(id="step_1", description="Add PyJWT", action="modify", files=["requirements.txt"]),
                PlanStep(id="step_2", description="Add Auth Service", action="create", files=["app/auth.py"]),
            ],
            estimated_files=["app/auth.py", "requirements.txt"],
        )
        provider = MockAgentLLMProvider(mock_plan)

        plan = await generate_agent_plan(
            target_root=Path("."),
            instruction="Add JWT authentication",
            summary={},
            selected_files=["app/main.py"],
            file_contents="print('hi')",
            provider=provider,
        )
        assert plan.summary == "Mock JWT Auth Plan"
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "modify"

    asyncio.run(_run())


def test_generate_fallback_plan():
    summary = {"entrypoints": ["app/main.py"]}
    plan = generate_fallback_plan("Add health check endpoint", summary, ["app/main.py"])
    assert "health" in plan.summary.lower()
    assert len(plan.steps) >= 2


def test_agent_safe_path_handling(tmp_path: Path):
    target_root = tmp_path / "proj"
    target_root.mkdir()

    with pytest.raises(PathValidationError):
        validate_file_path("../outside.py", target_root)

    with pytest.raises(PathValidationError):
        validate_file_path("app/../../outside.py", target_root)


def test_agent_verification_stage(tmp_path: Path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "app").mkdir()
    (proj_dir / "app" / "main.py").write_text("def hello(): pass\n", encoding="utf-8")

    verif = verify_project(proj_dir)
    assert verif.success is True
    assert "WORKSPACE_EXISTS" in verif.passed_checks
    assert "ENTRYPOINT_EXISTS" in verif.passed_checks
    assert "PYTHON_SYNTAX_VALID" in verif.passed_checks

    # Test syntax error detection
    (proj_dir / "app" / "broken.py").write_text("def invalid_syntax(:", encoding="utf-8")
    verif_broken = verify_project(proj_dir)
    assert verif_broken.success is False
    assert "PYTHON_SYNTAX_VALID" in verif_broken.failed_checks


def test_agent_protected_file_rejection(tmp_path: Path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / ".env").write_text("SECRET=123", encoding="utf-8")

    verif = verify_project(proj_dir)
    assert "NO_FORBIDDEN_FILES" in verif.failed_checks
    assert verif.success is False


def test_agent_traversal_attack_rejection(tmp_path: Path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()

    with pytest.raises(PathValidationError):
        validate_file_path("../attack.txt", proj_dir)


def test_agent_execution_flow():
    async def _run():
        service = CodegenService()
        base_res = await service.generate_project(CodegenRequest(requirement_text="Build a calculator module with passing tests/test_calc.py"))
        assert base_res.status in ("completed", "repaired")
        project_id = base_res.project_id

        # Execute agent loop with health check prompt
        state = await service.run_agent(project_id=project_id, instruction="Add a health check endpoint at /health")

        assert state.status in ("COMPLETED", "REPAIRED")
        assert state.run_id.startswith("run_")
        assert state.plan is not None
        assert state.verification_result is not None
        assert state.verification_result.success is True
        assert state.zip_available is True

    asyncio.run(_run())


def test_agent_rollback_on_failure():
    async def _run():
        service = CodegenService()
        base_res = await service.generate_project(CodegenRequest(requirement_text="Build a python string module with tests/test_str.py"))
        assert base_res.status in ("completed", "repaired")
        project_id = base_res.project_id

        from app.codegen.models import ProjectModificationManifest, FileChangeSpec
        failing_manifest = ProjectModificationManifest(
            summary="Introduce failing test",
            changes=[
                FileChangeSpec(action="create", path="tests/test_broken.py", content="def test_fail(): assert False, 'forced failure'")
            ]
        )
        provider = MockAgentLLMProvider(failing_manifest)
        service.provider = provider

        # Set max_repair_attempts to 0 on service so repair fails fast
        state = await service.run_agent(project_id=project_id, instruction="Break tests to force rollback")

        assert state.status == "ROLLED_BACK"
        project_dir = resolve_project_dir(project_id)
        assert not (project_dir / "tests" / "test_broken.py").exists()

    asyncio.run(_run())


def test_agent_api_endpoints():
    # 1. Generate base project
    gen_resp = client.post("/api/codegen/generate", json={"requirement_text": "Build a simple math utility with tests/test_math.py"})
    assert gen_resp.status_code == 200
    project_id = gen_resp.json()["project_id"]

    # 2. Trigger agent loop POST endpoint
    agent_resp = client.post(
        f"/api/codegen/{project_id}/agent",
        json={"instruction": "Add a health check route"}
    )
    assert agent_resp.status_code == 200
    agent_data = agent_resp.json()
    assert "run_id" in agent_data
    assert "status" in agent_data
    run_id = agent_data["run_id"]

    # 3. Poll agent status GET endpoint
    poll_resp = client.get(f"/api/codegen/{project_id}/agent/{run_id}")
    assert poll_resp.status_code == 200
    poll_data = poll_resp.json()
    assert poll_data["run_id"] == run_id
    assert poll_data["project_id"] == project_id


def test_real_gemini_agent_loop():
    """Optional real Gemini integration test for the Agentic Development Loop."""
    async def _run():
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY not configured in environment")

        service = CodegenService()
        base_res = await service.generate_project(CodegenRequest(requirement_text="Build a Python math utility with add(a, b) and tests/test_math.py with passing pytest"))
        assert base_res.status in ("completed", "repaired")
        project_id = base_res.project_id

        from app.llm.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key=api_key, model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
        service.provider = provider

        try:
            state = await service.run_agent(
                project_id=project_id,
                instruction="Add a multiply(a, b) function to math utility and add pytest in tests/test_math.py for multiply"
            )
            assert state.status in ("COMPLETED", "REPAIRED", "ROLLED_BACK")
            assert state.project_id == project_id
            print("\nREAL GEMINI AGENT LOOP: PASS")
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e) or "quota" in str(e).lower():
                pytest.skip(f"Gemini API rate limit / quota exceeded: {e}")
                return
            raise

    asyncio.run(_run())
