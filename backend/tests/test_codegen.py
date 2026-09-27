"""
Unit & Integration Tests for AIForge Codegen Engine (Phase 9)
"""
import asyncio
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.codegen import (
    CodegenError,
    PathValidationError,
    FileSpec,
    ManifestChunk,
    CodegenRequest,
    CodegenService,
)
from app.codegen.validator import validate_file_path, sanitize_relative_path
from app.codegen.file_writer import write_file_spec, build_file_tree, read_file_content
from app.codegen.tester import run_project_tests
from app.codegen.packager import package_project_zip
from app.codegen.generator import CodegenEngine
from app.codegen.repair import RepairEngine
from app.main import app

client = TestClient(app)


def test_path_validator_security(tmp_path: Path):
    target_root = tmp_path / "project"
    target_root.mkdir()

    # Valid relative paths
    safe1 = validate_file_path("app/main.py", target_root)
    assert safe1 == (target_root / "app" / "main.py").resolve()

    safe2 = validate_file_path("README.md", target_root)
    assert safe2 == (target_root / "README.md").resolve()

    # Insecure path traversal attempts
    with pytest.raises(PathValidationError):
        validate_file_path("../outside.py", target_root)

    with pytest.raises(PathValidationError):
        validate_file_path("app/../../outside.py", target_root)

    with pytest.raises(PathValidationError):
        validate_file_path("", target_root)


def test_file_writer_and_tree(tmp_path: Path):
    target_root = tmp_path / "project"
    target_root.mkdir()

    spec1 = FileSpec(path="app/main.py", content="print('hello')", description="main file")
    spec2 = FileSpec(path="tests/test_main.py", content="def test_pass(): assert True", description="test file")

    write_file_spec(target_root, spec1)
    write_file_spec(target_root, spec2)

    assert (target_root / "app" / "main.py").exists()
    assert (target_root / "tests" / "test_main.py").exists()

    content = read_file_content(target_root, "app/main.py")
    assert content == "print('hello')"

    tree = build_file_tree(target_root)
    assert tree.is_dir is True
    children_names = [c.name for c in tree.children]
    assert "app" in children_names
    assert "tests" in children_names


def test_tester_execution(tmp_path: Path):
    target_root = tmp_path / "project"
    target_root.mkdir()

    # Create a project with passing pytest test
    write_file_spec(
        target_root,
        FileSpec(path="tests/test_demo.py", content="def test_success(): assert 1 + 1 == 2"),
    )

    res = run_project_tests(target_root)
    assert res.success is True
    assert res.exit_code == 0


def test_packager_zip(tmp_path: Path):
    target_root = tmp_path / "project"
    target_root.mkdir()

    write_file_spec(target_root, FileSpec(path="app/main.py", content="code"))
    (target_root / ".env").write_text("SECRET_KEY=12345", encoding="utf-8")
    (target_root / "__pycache__").mkdir(exist_ok=True)
    (target_root / "__pycache__" / "main.cpython-310.pyc").write_text("bin", encoding="utf-8")

    zip_dest = tmp_path / "out" / "project.zip"
    packaged = package_project_zip(target_root, zip_dest)

    assert packaged.exists()
    assert packaged.stat().st_size > 0

    import zipfile
    with zipfile.ZipFile(packaged, "r") as z:
        names = z.namelist()
        assert "app/main.py" in names
        assert ".env" not in names
        assert "__pycache__/main.cpython-310.pyc" not in names


def test_codegen_engine_and_repair(tmp_path: Path):
    async def run_test():
        engine = CodegenEngine(provider=None)
        manifest = await engine.generate_manifest(requirement_text="Build sample REST API")
        assert len(manifest.files) >= 2
        paths = [f.path for f in manifest.files]
        assert "app/main.py" in paths
        assert "tests/test_main.py" in paths

        # Test repair engine on passing project
        target_root = tmp_path / "project_repair"
        target_root.mkdir()
        write_file_spec(target_root, FileSpec(path="tests/test_ok.py", content="def test_pass(): pass"))

        repair_engine = RepairEngine(provider=None)
        res = await repair_engine.repair_project(target_root)
        assert res.success is True
        assert res.retry_count == 0

    asyncio.run(run_test())


def test_codegen_service_flow(tmp_path: Path):
    async def run_test():
        service = CodegenService(provider=None)
        req = CodegenRequest(
            project_id="test_proj_001",
            requirement_text="Create a text analysis service",
            project_name="text-analyzer",
        )
        result = await service.generate_project(req)
        assert result.project_id == "test_proj_001"
        assert result.status in ("completed", "repaired"), f"Service generation failed: {result.test_result.stdout}\n{result.test_result.stderr}"
        assert len(result.generated_files) > 0
        assert result.test_result is not None
        assert result.test_result.success is True
        assert result.zip_path is not None

    asyncio.run(run_test())


def test_codegen_api_endpoints():
    # 1. Generate code via API
    resp = client.post(
        "/api/codegen/generate",
        json={
            "project_id": "api_test_proj",
            "requirement_text": "Build API server with text processing",
            "project_name": "api-demo",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["project_id"] == "api_test_proj"
    assert data["status"] in ("completed", "repaired"), f"Generation failed: {data.get('error')}"

    # 2. Get status
    resp_status = client.get("/api/codegen/api_test_proj")
    assert resp_status.status_code == 200
    assert resp_status.json()["project_id"] == "api_test_proj"

    # 3. Get files tree
    resp_files = client.get("/api/codegen/api_test_proj/files")
    assert resp_files.status_code == 200
    assert resp_files.json()["name"] == "api_test_proj"

    # 4. Get file content
    resp_content = client.get("/api/codegen/api_test_proj/file?path=app/main.py")
    assert resp_content.status_code == 200
    assert "FastAPI" in resp_content.json()["content"]

    # 5. Run test endpoint
    resp_test = client.post("/api/codegen/api_test_proj/test")
    assert resp_test.status_code == 200
    assert resp_test.json()["success"] is True

    # 6. Download ZIP
    resp_dl = client.get("/api/codegen/api_test_proj/download")
    assert resp_dl.status_code == 200
    assert resp_dl.headers["content-type"] == "application/zip"
