"""
AIForge Builder Test Suite — Task 4

Tests the Builder module (builder.py) and the /api/build endpoint.
Covers:
 - project generation
 - expected file structure
 - pipeline.json correctness
 - project.json correctness
 - executable component handling (text_classifier)
 - catalog-only component handling
 - generated FastAPI server behaviour
 - generated pipeline abstraction
 - execution_status logic
 - /api/build HTTP endpoint
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.builder import (
    GENERATED_DIR,
    BuildResult,
    _execution_status,
    build_project,
)
from app.main import app
from app.planner import analyze_spec

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(name: str, spec: str) -> BuildResult:
    return build_project(name, spec)


CLASSIFIER_SPEC = "Classify legal documents into categories and expose a REST API."
# This spec contains text_classifier (executable) plus catalog components
# (opencv, ocr) — so execution_status must be "partial", not "blocked".
CATALOG_SPEC = "Extract text from images, classify the documents, and expose an API."
FULL_SPEC = (
    "Analyze photos of legal documents, extract text, classify them, "
    "summarize the content, and expose a REST API."
)


# ===========================================================================
# execution_status helper
# ===========================================================================

class TestExecutionStatus:
    def test_ready_for_classifier_plus_api(self):
        plan = analyze_spec(CLASSIFIER_SPEC)
        assert _execution_status(plan) == "ready"

    def test_partial_when_catalog_components_present(self):
        plan = analyze_spec(CATALOG_SPEC)
        assert _execution_status(plan) == "partial"

    def test_partial_for_full_pipeline(self):
        plan = analyze_spec(FULL_SPEC)
        assert _execution_status(plan) == "partial"

    def test_blocked_for_empty_components(self):
        fake_plan = {"selected_components": [], "fastapi": []}
        assert _execution_status(fake_plan) == "blocked"


# ===========================================================================
# BuildResult
# ===========================================================================

class TestBuildResult:
    def test_to_dict_has_all_required_keys(self):
        result = _build("test-result", CLASSIFIER_SPEC)
        d = result.to_dict()
        for key in (
            "project_id", "project_name", "project_path",
            "execution_status", "execution_ready",
            "pipeline", "pipeline_steps",
            "selected_components", "implemented_components",
            "catalog_components", "warnings", "generated_files",
        ):
            assert key in d, f"missing key: {key}"

    def test_project_id_starts_with_slugified_name(self):
        result = _build("my-awesome-project", CLASSIFIER_SPEC)
        assert result.project_id.startswith("my-awesome-project-")

    def test_project_dir_exists(self):
        result = _build("dir-check", CLASSIFIER_SPEC)
        assert result.project_dir.exists()

    def test_execution_status_is_string(self):
        result = _build("status-type", CLASSIFIER_SPEC)
        assert isinstance(result.execution_status, str)
        assert result.execution_status in ("ready", "partial", "blocked")


# ===========================================================================
# File structure
# ===========================================================================

class TestGeneratedFileStructure:
    @pytest.fixture(scope="class")
    def result(self):
        return _build("file-structure-test", CLASSIFIER_SPEC)

    def test_project_directory_created(self, result):
        assert result.project_dir.exists()
        assert result.project_dir.is_dir()

    def test_agent_subdirectory_exists(self, result):
        assert (result.project_dir / "agent").is_dir()

    def test_models_subdirectory_exists(self, result):
        assert (result.project_dir / "models").is_dir()

    def test_tests_subdirectory_exists(self, result):
        assert (result.project_dir / "tests").is_dir()

    def test_readme_exists(self, result):
        assert (result.project_dir / "README.md").exists()

    def test_requirements_txt_exists(self, result):
        assert (result.project_dir / "requirements.txt").exists()

    def test_dockerfile_exists(self, result):
        assert (result.project_dir / "Dockerfile").exists()

    def test_pipeline_json_exists(self, result):
        assert (result.project_dir / "pipeline.json").exists()

    def test_project_json_exists(self, result):
        assert (result.project_dir / "project.json").exists()

    def test_agent_pipeline_py_exists(self, result):
        assert (result.project_dir / "agent" / "pipeline.py").exists()

    def test_agent_server_py_exists(self, result):
        assert (result.project_dir / "agent" / "server.py").exists()

    def test_tests_test_pipeline_py_exists(self, result):
        assert (result.project_dir / "tests" / "test_pipeline.py").exists()

    def test_generated_files_list_matches_disk(self, result):
        for rel in result.generated_files:
            assert (result.project_dir / rel).exists(), f"listed but missing: {rel}"


# ===========================================================================
# pipeline.json
# ===========================================================================

class TestPipelineJson:
    @pytest.fixture(scope="class")
    def data(self):
        result = _build("pipeline-json-test", CLASSIFIER_SPEC)
        return json.loads((result.project_dir / "pipeline.json").read_text(encoding="utf-8"))

    def test_has_project_id(self, data):
        assert "project_id" in data

    def test_has_specification(self, data):
        assert "specification" in data
        assert len(data["specification"]) > 5

    def test_has_execution_status(self, data):
        assert "execution_status" in data
        assert data["execution_status"] in ("ready", "partial", "blocked")

    def test_has_execution_ready(self, data):
        assert "execution_ready" in data
        assert isinstance(data["execution_ready"], bool)

    def test_has_selected_components(self, data):
        assert "selected_components" in data
        assert isinstance(data["selected_components"], list)
        assert len(data["selected_components"]) > 0

    def test_has_pipeline_steps(self, data):
        assert "pipeline_steps" in data
        assert isinstance(data["pipeline_steps"], list)
        assert len(data["pipeline_steps"]) > 0

    def test_has_implemented_components(self, data):
        assert "implemented_components" in data

    def test_has_catalog_components(self, data):
        assert "catalog_components" in data

    def test_has_warnings(self, data):
        assert "warnings" in data

    def test_classifier_spec_execution_ready_true(self, data):
        assert data["execution_ready"] is True

    def test_classifier_spec_execution_status_ready(self, data):
        assert data["execution_status"] == "ready"


# ===========================================================================
# project.json
# ===========================================================================

class TestProjectJson:
    @pytest.fixture(scope="class")
    def data(self):
        result = _build("project-json-test", CLASSIFIER_SPEC)
        return json.loads((result.project_dir / "project.json").read_text(encoding="utf-8"))

    def test_has_project_id(self, data):
        assert "project_id" in data

    def test_has_project_name(self, data):
        assert "project_name" in data

    def test_has_specification(self, data):
        assert "specification" in data

    def test_has_status(self, data):
        assert data["status"] == "generated"

    def test_has_execution_status(self, data):
        assert "execution_status" in data

    def test_has_created_at(self, data):
        assert "created_at" in data
        assert isinstance(data["created_at"], float)

    def test_has_components(self, data):
        assert "components" in data
        assert isinstance(data["components"], list)

    def test_has_executable_components(self, data):
        assert "executable_components" in data

    def test_has_generated_files(self, data):
        assert "generated_files" in data
        assert isinstance(data["generated_files"], list)
        assert len(data["generated_files"]) > 0


# ===========================================================================
# Executable component handling (text_classifier)
# ===========================================================================

class TestExecutableComponents:
    @pytest.fixture(scope="class")
    def result(self):
        return _build("exec-test", CLASSIFIER_SPEC)

    def test_text_classifier_in_selected(self, result):
        assert "text_classifier" in result.plan["selected_components"]

    def test_text_classifier_in_implemented(self, result):
        assert "text_classifier" in result.plan["implemented_components"]

    def test_execution_status_ready(self, result):
        assert result.execution_status == "ready"

    def test_server_references_model_joblib(self, result):
        server = (result.project_dir / "agent" / "server.py").read_text(encoding="utf-8")
        assert "model.joblib" in server

    def test_server_has_real_predict_endpoint(self, result):
        server = (result.project_dir / "agent" / "server.py").read_text(encoding="utf-8")
        assert "model.predict(" in server
        assert "model.predict_proba(" in server

    def test_server_handles_missing_model(self, result):
        server = (result.project_dir / "agent" / "server.py").read_text(encoding="utf-8")
        assert "503" in server or "not trained" in server.lower()

    def test_pipeline_has_text_classifier_component(self, result):
        pipeline_src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "TextClassifierComponent" in pipeline_src

    def test_pipeline_loads_model_joblib(self, result):
        pipeline_src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "model.joblib" in pipeline_src

    def test_requirements_includes_scikit_learn(self, result):
        reqs = (result.project_dir / "requirements.txt").read_text(encoding="utf-8")
        assert "scikit-learn" in reqs

    def test_requirements_includes_joblib(self, result):
        reqs = (result.project_dir / "requirements.txt").read_text(encoding="utf-8")
        assert "joblib" in reqs

    def test_no_warnings_for_ready_pipeline(self, result):
        assert result.plan.get("warnings", []) == []


# ===========================================================================
# Catalog-only component handling
# ===========================================================================

# A spec with zero executable components — only catalog ones selected.
# "Summarize documents using an LLM" → llm + summarization (both catalog)
# No classification → no text_classifier → execution_status = "blocked"
# "Extract text from images" → opencv + ocr (both catalog, no classifier)
# Use the latter since it avoids LLM ambiguity.
_PURE_CATALOG_SPEC = "Extract text from scanned document images."


class TestCatalogOnlyComponents:
    @pytest.fixture(scope="class")
    def result(self):
        return _build("catalog-test", _PURE_CATALOG_SPEC)

    def test_execution_ready_false(self, result):
        assert result.plan.get("execution_ready") is False

    def test_warnings_present(self, result):
        assert len(result.plan.get("warnings", [])) > 0

    def test_server_does_not_pretend_to_work(self, result):
        server = (result.project_dir / "agent" / "server.py").read_text(encoding="utf-8")
        assert "not_implemented" in server

    def test_server_honest_about_catalog_components(self, result):
        server = (result.project_dir / "agent" / "server.py").read_text(encoding="utf-8")
        assert "catalog" in server.lower() or "not yet executable" in server.lower()

    def test_pipeline_contains_not_implemented_raises(self, result):
        pipeline_src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "NotImplementedError" in pipeline_src

    def test_pipeline_catches_not_implemented(self, result):
        """Catalog pipeline must catch NotImplementedError and record warnings."""
        pipeline_src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "pipeline_warnings" in pipeline_src

    def test_catalog_components_in_catalog_list(self, result):
        # ocr is still non-executable (Tesseract absent) so catalog_components is non-empty
        catalog = result.plan.get("catalog_components", [])
        assert len(catalog) > 0, (
            f"Expected at least ocr in catalog_components, got: {catalog}"
        )


# ===========================================================================
# Full pipeline (mixed executable + catalog)
# ===========================================================================

class TestFullPipeline:
    @pytest.fixture(scope="class")
    def result(self):
        return _build("full-pipeline-test", FULL_SPEC)

    def test_opencv_in_selected(self, result):
        assert "opencv" in result.plan["selected_components"]

    def test_ocr_in_selected(self, result):
        assert "ocr" in result.plan["selected_components"]

    def test_text_classifier_in_selected(self, result):
        assert "text_classifier" in result.plan["selected_components"]

    def test_llm_in_selected(self, result):
        assert "llm" in result.plan["selected_components"]

    def test_fastapi_in_selected(self, result):
        assert "fastapi" in result.plan["selected_components"]

    def test_execution_status_partial(self, result):
        assert result.execution_status == "partial"

    def test_all_files_generated(self, result):
        required = [
            "README.md", "requirements.txt", "Dockerfile",
            "pipeline.json", "project.json",
            "agent/pipeline.py", "agent/server.py",
            "tests/test_pipeline.py",
        ]
        for f in required:
            assert (result.project_dir / f).exists(), f"missing: {f}"

    def test_readme_mentions_catalog_status(self, result):
        readme = (result.project_dir / "README.md").read_text(encoding="utf-8")
        assert "Catalog" in readme or "not yet executable" in readme.lower()

    def test_pipeline_json_has_warnings(self, result):
        data = json.loads((result.project_dir / "pipeline.json").read_text(encoding="utf-8"))
        assert len(data.get("warnings", [])) > 0


# ===========================================================================
# generated server.py — static content checks
# ===========================================================================

class TestGeneratedServer:
    def test_classifier_server_has_health_endpoint(self):
        result = _build("server-health-test", CLASSIFIER_SPEC)
        server = (result.project_dir / "agent" / "server.py").read_text(encoding="utf-8")
        assert "@app.get(\"/health\")" in server or 'def health' in server

    def test_classifier_server_has_predict_endpoint(self):
        result = _build("server-predict-test", CLASSIFIER_SPEC)
        server = (result.project_dir / "agent" / "server.py").read_text(encoding="utf-8")
        assert "@app.post(\"/predict\")" in server or 'def predict' in server

    def test_catalog_server_has_health_endpoint(self):
        result = _build("catalog-server-health", CATALOG_SPEC)
        server = (result.project_dir / "agent" / "server.py").read_text(encoding="utf-8")
        assert "def health" in server

    def test_catalog_server_has_predict_stub(self):
        result = _build("catalog-server-predict", _PURE_CATALOG_SPEC)
        server = (result.project_dir / "agent" / "server.py").read_text(encoding="utf-8")
        assert "def predict" in server
        assert "not_implemented" in server


# ===========================================================================
# generated pipeline.py — structural checks
# ===========================================================================

class TestGeneratedPipeline:
    def test_pipeline_has_component_base_class(self):
        result = _build("pipeline-base-test", CLASSIFIER_SPEC)
        src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "class Component" in src

    def test_pipeline_has_pipeline_class(self):
        result = _build("pipeline-class-test", CLASSIFIER_SPEC)
        src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "class Pipeline" in src

    def test_pipeline_has_run_method(self):
        result = _build("pipeline-run-test", CLASSIFIER_SPEC)
        src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "def run(" in src

    def test_pipeline_has_run_pipeline_function(self):
        result = _build("pipeline-fn-test", CLASSIFIER_SPEC)
        src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "def run_pipeline(" in src

    def test_pipeline_has_steps_list(self):
        result = _build("pipeline-steps-test", CLASSIFIER_SPEC)
        src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "self.steps" in src

    def test_catalog_pipeline_has_not_implemented_error(self):
        result = _build("pipeline-ni-test", CATALOG_SPEC)
        src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "NotImplementedError" in src

    def test_catalog_pipeline_catches_errors_as_warnings(self):
        result = _build("pipeline-catch-test", CATALOG_SPEC)
        src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "pipeline_warnings" in src


# ===========================================================================
# README.md
# ===========================================================================

class TestReadme:
    def test_readme_contains_specification(self):
        result = _build("readme-spec-test", CLASSIFIER_SPEC)
        readme = (result.project_dir / "README.md").read_text(encoding="utf-8")
        assert CLASSIFIER_SPEC in readme

    def test_readme_contains_project_name(self):
        result = _build("readme-name-test", CLASSIFIER_SPEC)
        readme = (result.project_dir / "README.md").read_text(encoding="utf-8")
        assert "readme-name-test" in readme

    def test_readme_has_execution_status(self):
        result = _build("readme-status-test", CLASSIFIER_SPEC)
        readme = (result.project_dir / "README.md").read_text(encoding="utf-8")
        assert "Execution status" in readme

    def test_readme_lists_components(self):
        result = _build("readme-comps-test", CLASSIFIER_SPEC)
        readme = (result.project_dir / "README.md").read_text(encoding="utf-8")
        assert "Task Classifier" in readme or "text_classifier" in readme


# ===========================================================================
# requirements.txt
# ===========================================================================

class TestRequirements:
    def test_always_includes_fastapi(self):
        result = _build("req-fastapi-test", CLASSIFIER_SPEC)
        reqs = (result.project_dir / "requirements.txt").read_text(encoding="utf-8")
        assert "fastapi" in reqs

    def test_always_includes_uvicorn(self):
        result = _build("req-uvicorn-test", CLASSIFIER_SPEC)
        reqs = (result.project_dir / "requirements.txt").read_text(encoding="utf-8")
        assert "uvicorn" in reqs

    def test_classifier_includes_sklearn(self):
        result = _build("req-sklearn-test", CLASSIFIER_SPEC)
        reqs = (result.project_dir / "requirements.txt").read_text(encoding="utf-8")
        assert "scikit-learn" in reqs

    def test_catalog_notes_non_executable_packages(self):
        result = _build("req-catalog-test", CATALOG_SPEC)
        reqs = (result.project_dir / "requirements.txt").read_text(encoding="utf-8")
        # Catalog components should be commented out, not silently omitted
        assert "#" in reqs


# ===========================================================================
# Dockerfile
# ===========================================================================

class TestDockerfile:
    def test_dockerfile_uses_python_base(self):
        result = _build("docker-test", CLASSIFIER_SPEC)
        docker = (result.project_dir / "Dockerfile").read_text(encoding="utf-8")
        assert "python:" in docker

    def test_dockerfile_has_cmd(self):
        result = _build("docker-cmd-test", CLASSIFIER_SPEC)
        docker = (result.project_dir / "Dockerfile").read_text(encoding="utf-8")
        assert "CMD" in docker or "ENTRYPOINT" in docker

    def test_classifier_dockerfile_includes_sklearn(self):
        result = _build("docker-sklearn-test", CLASSIFIER_SPEC)
        docker = (result.project_dir / "Dockerfile").read_text(encoding="utf-8")
        assert "scikit-learn" in docker


# ===========================================================================
# /api/build HTTP endpoint
# ===========================================================================

class TestApiBuildEndpoint:
    def test_returns_200(self):
        r = client.post("/api/build", json={"project_name": "test-build", "spec": CLASSIFIER_SPEC})
        assert r.status_code == 200

    def test_response_has_project_id(self):
        r = client.post("/api/build", json={"project_name": "test-pid", "spec": CLASSIFIER_SPEC})
        body = r.json()
        assert "project_id" in body
        assert body["project_id"].startswith("test-pid-")

    def test_response_has_execution_status(self):
        r = client.post("/api/build", json={"project_name": "test-exec", "spec": CLASSIFIER_SPEC})
        body = r.json()
        assert "execution_status" in body
        assert body["execution_status"] in ("ready", "partial", "blocked")

    def test_response_has_execution_ready(self):
        r = client.post("/api/build", json={"project_name": "test-rdy", "spec": CLASSIFIER_SPEC})
        body = r.json()
        assert "execution_ready" in body
        assert isinstance(body["execution_ready"], bool)

    def test_classifier_spec_returns_ready(self):
        r = client.post("/api/build", json={"project_name": "ready-check", "spec": CLASSIFIER_SPEC})
        assert r.json()["execution_status"] == "ready"
        assert r.json()["execution_ready"] is True

    def test_catalog_spec_returns_partial(self):
        r = client.post("/api/build", json={"project_name": "partial-check", "spec": CATALOG_SPEC})
        assert r.json()["execution_status"] == "partial"
        assert r.json()["execution_ready"] is False

    def test_response_has_pipeline(self):
        r = client.post("/api/build", json={"project_name": "pipeline-check", "spec": CLASSIFIER_SPEC})
        body = r.json()
        assert "pipeline" in body
        assert isinstance(body["pipeline"], list)
        assert len(body["pipeline"]) > 0

    def test_response_has_pipeline_steps(self):
        r = client.post("/api/build", json={"project_name": "steps-check", "spec": CLASSIFIER_SPEC})
        body = r.json()
        assert "pipeline_steps" in body
        assert len(body["pipeline_steps"]) > 0

    def test_response_has_generated_files(self):
        r = client.post("/api/build", json={"project_name": "files-check", "spec": CLASSIFIER_SPEC})
        body = r.json()
        assert "generated_files" in body
        assert len(body["generated_files"]) >= 7

    def test_response_has_project_path(self):
        r = client.post("/api/build", json={"project_name": "path-check", "spec": CLASSIFIER_SPEC})
        body = r.json()
        assert "project_path" in body

    def test_response_has_warnings_for_catalog_spec(self):
        r = client.post("/api/build", json={"project_name": "warn-check", "spec": CATALOG_SPEC})
        body = r.json()
        assert "warnings" in body
        assert len(body["warnings"]) > 0

    def test_full_spec_returns_partial_with_all_components(self):
        r = client.post("/api/build", json={"project_name": "full-check", "spec": FULL_SPEC})
        body = r.json()
        assert body["execution_status"] == "partial"
        sel = body["selected_components"]
        assert "opencv" in sel
        assert "ocr" in sel
        assert "text_classifier" in sel
        assert "fastapi" in sel

    def test_missing_spec_returns_422(self):
        r = client.post("/api/build", json={"project_name": "bad"})
        assert r.status_code == 422

    def test_short_spec_returns_422(self):
        r = client.post("/api/build", json={"project_name": "bad", "spec": "hi"})
        assert r.status_code == 422

    def test_default_project_name_accepted(self):
        r = client.post("/api/build", json={"spec": CLASSIFIER_SPEC})
        assert r.status_code == 200

    def test_project_directory_created_on_disk(self):
        r = client.post("/api/build", json={"project_name": "disk-check", "spec": CLASSIFIER_SPEC})
        body = r.json()
        project_dir = Path(body["project_path"])
        assert project_dir.exists()
        assert (project_dir / "project.json").exists()
