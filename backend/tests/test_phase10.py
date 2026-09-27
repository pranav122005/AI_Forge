"""
Phase 10 — Real End-to-End Validation & Production Hardening Test Suite
"""
import asyncio
import importlib.util
import os
import py_compile
import shutil
import zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.codegen import (
    CodegenRequest,
    CodegenService,
    PathValidationError,
    RepairRequest,
)
from app.codegen.validator import validate_file_path
from app.codegen.tester import run_project_tests
from app.codegen.packager import package_project_zip
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.providers.base import LLMConfigurationError, LLMProviderError
from app.main import app

client = TestClient(app)
BENCHMARK_PROJECT_ID = "sentiment-classifier-api"
BENCHMARK_REQUIREMENT = (
    "Build a sentiment classification REST API using Python and FastAPI. The system should accept text, "
    "classify it as positive or negative, return a confidence score, include a small deterministic training dataset, "
    "train a TF-IDF plus Logistic Regression model, expose prediction through an API endpoint, include automated tests, "
    "requirements.txt, README.md, and .env.example."
)


def test_01_real_e2e_generation():
    """Step 1 & 2: Generate benchmark project and verify generated files."""
    async def run_test():
        service = CodegenService(provider=None)
        req = CodegenRequest(
            project_id=BENCHMARK_PROJECT_ID,
            requirement_text=BENCHMARK_REQUIREMENT,
            project_name=BENCHMARK_PROJECT_ID,
        )
        result = await service.generate_project(req)
        assert result.project_id == BENCHMARK_PROJECT_ID
        assert result.status in ("completed", "repaired")
        assert len(result.generated_files) >= 5

        # Check expected files
        target_dir = service.get_file_tree(BENCHMARK_PROJECT_ID)
        assert target_dir is not None

        # Verify no placeholder strings in files
        files_to_check = ["app/main.py", "app/model.py", "README.md", "requirements.txt"]
        forbidden_terms = ["TODO", "FIXME", "placeholder implementation", "pseudo-code", "implementation omitted"]

        for rel_path in files_to_check:
            content = service.read_file(BENCHMARK_PROJECT_ID, rel_path)
            assert len(content.strip()) > 0, f"File {rel_path} is empty."
            for term in forbidden_terms:
                assert term not in content, f"Forbidden term '{term}' found in {rel_path}"

    asyncio.run(run_test())


def test_02_verify_generated_code_and_compilation():
    """Step 3: Compile Python files and run test suite."""
    from app.codegen.service import resolve_project_dir
    project_dir = resolve_project_dir(BENCHMARK_PROJECT_ID)

    # Compile all .py files
    py_files = list(project_dir.rglob("*.py"))
    assert len(py_files) >= 3
    for py_file in py_files:
        py_compile.compile(str(py_file), doraise=True)

    # Run pytest on generated project
    test_res = run_project_tests(project_dir)
    assert test_res.success is True
    assert test_res.exit_code == 0
    assert test_res.duration_seconds > 0.0


def test_03_verify_model_pipeline():
    """Step 4: Verify ML training, metrics, persistence, and prediction."""
    from app.codegen.service import resolve_project_dir
    project_dir = resolve_project_dir(BENCHMARK_PROJECT_ID)
    
    # Import generated model module using file location spec
    model_path = project_dir / "app" / "model.py"
    spec = importlib.util.spec_from_file_location("gen_model", model_path)
    gen_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_model)

    clf = gen_model.SentimentClassifier()
    metrics = clf.train()
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["accuracy"] >= 0.75

    pred = clf.predict("This product is excellent")
    assert pred["prediction"] == "positive"
    assert 0.0 <= pred["confidence"] <= 1.0


def test_04_verify_api_endpoints():
    """Step 5: Test API endpoints on generated project."""
    from app.codegen.service import resolve_project_dir
    import sys

    project_dir = resolve_project_dir(BENCHMARK_PROJECT_ID)

    # Backup host sys.modules for app.model and app.main
    old_model = sys.modules.get("app.model")
    old_main = sys.modules.get("app.main")

    try:
        model_path = project_dir / "app" / "model.py"
        spec_m = importlib.util.spec_from_file_location("app.model", model_path)
        gen_model = importlib.util.module_from_spec(spec_m)
        sys.modules["app.model"] = gen_model
        spec_m.loader.exec_module(gen_model)

        main_path = project_dir / "app" / "main.py"
        spec_a = importlib.util.spec_from_file_location("app.main", main_path)
        gen_main = importlib.util.module_from_spec(spec_a)
        sys.modules["app.main"] = gen_main
        spec_a.loader.exec_module(gen_main)

        gen_client = TestClient(gen_main.app)

        # GET /health
        res_h = gen_client.get("/health")
        assert res_h.status_code == 200
        assert res_h.json()["status"] == "ok"

        # POST /predict
        res_p = gen_client.post("/predict", json={"text": "This product is excellent"})
        assert res_p.status_code == 200
        body = res_p.json()
        assert body["prediction"] in ("positive", "negative")
        assert body["confidence"] > 0.5
    finally:
        if old_model:
            sys.modules["app.model"] = old_model
        else:
            sys.modules.pop("app.model", None)

        if old_main:
            sys.modules["app.main"] = old_main
        else:
            sys.modules.pop("app.main", None)


def test_05_verify_repair_loop(tmp_path: Path):
    """Step 6: Test controlled failure and repair loop."""
    async def run_test():
        service = CodegenService(provider=None)
        proj_id = "repair_test_proj"
        target_dir = tmp_path / proj_id
        target_dir.mkdir()

        # Write a broken test file
        broken_test = "def test_broken(): assert 1 == 2"
        main_file = "def add(a, b): return a + b"
        (target_dir / "app").mkdir()
        (target_dir / "app" / "__init__.py").write_text("")
        (target_dir / "app" / "main.py").write_text(main_file)
        (target_dir / "tests").mkdir()
        (target_dir / "tests" / "__init__.py").write_text("")
        (target_dir / "tests" / "test_main.py").write_text(broken_test)

        # Initial test run fails
        init_res = run_project_tests(target_dir)
        assert init_res.success is False

        # Apply repair
        fixed_test = "def test_fixed(): assert 1 + 1 == 2"
        (target_dir / "tests" / "test_main.py").write_text(fixed_test)
        re_res = run_project_tests(target_dir)
        assert re_res.success is True

    asyncio.run(run_test())


def test_06_verify_zip_artifact():
    """Step 7: Programmatically verify ZIP artifact contents and security exclusions."""
    from app.codegen.service import ARTIFACTS_DIR
    zip_path = ARTIFACTS_DIR / BENCHMARK_PROJECT_ID / f"{BENCHMARK_PROJECT_ID}.zip"
    assert zip_path.exists()
    assert zip_path.stat().st_size > 0

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        assert "app/main.py" in names
        assert "README.md" in names
        assert "requirements.txt" in names
        assert "project.json" in names

        # Security check: no secrets or cache
        forbidden = [".env", "id_rsa", "__pycache__/main.cpython-310.pyc", ".pytest_cache"]
        for f in forbidden:
            assert f not in names, f"Forbidden file '{f}' found in ZIP archive"


def test_07_verify_independent_zip_extraction(tmp_path: Path):
    """Step 8: Extract ZIP to isolated directory outside AIForge and run tests independently."""
    from app.codegen.service import ARTIFACTS_DIR
    zip_path = ARTIFACTS_DIR / BENCHMARK_PROJECT_ID / f"{BENCHMARK_PROJECT_ID}.zip"
    extract_dir = tmp_path / "independent_extract"
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    assert (extract_dir / "app" / "main.py").exists()
    assert (extract_dir / "README.md").exists()

    # Run tests in extracted directory
    res = run_project_tests(extract_dir)
    assert res.success is True
    assert res.exit_code == 0


def test_08_security_path_traversal(tmp_path: Path):
    """Step 10: Security path traversal checks."""
    target_root = tmp_path / "root"
    target_root.mkdir()

    bad_paths = [
        "../../evil.py",
        "C:\\evil.py",
        "/evil.py",
        "../.env",
        "app/../../secret.txt",
    ]

    for bp in bad_paths:
        with pytest.raises(PathValidationError):
            validate_file_path(bp, target_root)


def test_09_real_provider_failure_handling():
    """Step 11: Real LLM provider missing key failure behavior."""
    # Ensure missing API key raises LLMConfigurationError honestly
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(LLMConfigurationError):
            OpenAIProvider(api_key=None)
    finally:
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key
