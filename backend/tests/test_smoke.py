"""
AIForge smoke + integration test suite.

Each test is named after what it proves so failures are self-explanatory.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, analyze_spec

client = TestClient(app)

DEMO_CSV = Path(__file__).resolve().parents[2] / "data" / "legal_demo.csv"


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 2. Catalog
# ---------------------------------------------------------------------------

def test_catalog_lists_all_components():
    r = client.get("/api/catalog")
    assert r.status_code == 200
    keys = {c["key"] for c in r.json()["components"]}
    # The catalog has been expanded beyond the original 7; assert the original
    # 7 are all present (subset check) rather than requiring an exact count.
    assert {"opencv", "ocr", "text_classifier", "llm", "embeddings", "vector_db", "fastapi"}.issubset(keys)


def test_catalog_components_have_status_fields():
    """Every component must expose executable and tested booleans."""
    r = client.get("/api/catalog")
    for comp in r.json()["components"]:
        assert "executable" in comp, f"{comp['key']} missing 'executable'"
        assert "tested" in comp, f"{comp['key']} missing 'tested'"
        assert isinstance(comp["executable"], bool)
        assert isinstance(comp["tested"], bool)


def test_only_text_classifier_and_fastapi_are_executable():
    r = client.get("/api/catalog")
    exec_keys = {c["key"] for c in r.json()["components"] if c["executable"]}
    # OpenCV is now executable (pure Python, no system binary required).
    # OCR stays False until Tesseract binary is installed.
    assert "text_classifier" in exec_keys
    assert "fastapi" in exec_keys
    assert "opencv" in exec_keys
    assert "ocr" not in exec_keys  # Tesseract binary absent in test env


# ---------------------------------------------------------------------------
# 3. Planner — analyze_spec correctness
# ---------------------------------------------------------------------------

def test_analyze_selects_document_components():
    """Full pipeline spec: should include vision, OCR, classifier, LLM and API."""
    result = client.post(
        "/api/analyze",
        json={
            "specification": (
                "Build an AI system that accepts images of legal documents, "
                "extracts text, classifies the document type, summarizes the "
                "content, and exposes a REST API."
            )
        },
    )
    assert result.status_code == 200
    body = result.json()
    assert "opencv" in body["selected_components"]
    assert "ocr" in body["selected_components"]
    assert "text_classifier" in body["selected_components"]
    assert "llm" in body["selected_components"]
    assert "fastapi" in body["selected_components"]


def test_analyze_classifier_not_suppressed_by_llm():
    """
    Classification intent must survive even when generation/summarization terms
    are also present.  This was the original planner bug.
    """
    body = analyze_spec("Classify legal documents and summarize them.")
    assert "text_classifier" in body["selected_components"], (
        "text_classifier must be selected when both classification and "
        "generation terms are present"
    )
    assert "llm" in body["selected_components"]


def test_analyze_pure_classification_spec():
    body = analyze_spec("Classify customer feedback into categories and expose a REST API.")
    assert "text_classifier" in body["selected_components"]
    assert "fastapi" in body["selected_components"]


def test_analyze_fallback_adds_classifier_for_generic_spec():
    """When no capability keyword matches, the fallback must add text_classifier."""
    body = analyze_spec("Build an AI system and expose an API.")
    assert "text_classifier" in body["selected_components"]
    assert "fastapi" in body["selected_components"]


def test_analyze_fastapi_always_present():
    for spec in [
        "Classify images.",
        "Summarize documents.",
        "Search a knowledge base.",
        "Do something useful.",
    ]:
        body = analyze_spec(spec)
        assert "fastapi" in body["selected_components"], f"fastapi missing for: {spec!r}"


def test_analyze_returns_implemented_and_catalog_split():
    body = analyze_spec(
        "Classify legal documents and summarize them using an API."
    )
    assert "text_classifier" in body["implemented_components"]
    assert "fastapi" in body["implemented_components"]
    assert "llm" in body["catalog_components"]


def test_analyze_deduplicates_components():
    """Repeated terms should not produce duplicate component entries."""
    body = analyze_spec(
        "Classify classify classify and expose a REST API REST API."
    )
    assert body["selected_components"].count("text_classifier") == 1
    assert body["selected_components"].count("fastapi") == 1


# ---------------------------------------------------------------------------
# 4. Project generation
# ---------------------------------------------------------------------------

def _generate_project(spec: str, name: str, plan: dict) -> dict:
    r = client.post(
        "/api/generate",
        json={"specification": spec, "project_name": name, "plan": plan},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_generate_returns_project_id():
    plan = analyze_spec("Classify legal documents and expose a REST API.")
    out = _generate_project(plan["specification"], "gen-test", plan)
    assert "project_id" in out
    assert out["project_id"].startswith("gen-test-")


def test_generated_project_files_exist():
    """All expected scaffold files must be written to disk."""
    plan = analyze_spec("Classify legal documents and expose a REST API.")
    out = _generate_project(plan["specification"], "file-check", plan)
    from app.main import PROJECTS_DIR

    project_dir = PROJECTS_DIR / out["project_id"]
    assert project_dir.exists(), "project directory not created"
    assert (project_dir / "project.json").exists()
    assert (project_dir / "README.md").exists()
    assert (project_dir / "config" / "pipeline.json").exists()
    assert (project_dir / "src" / "pipeline.py").exists()
    assert (project_dir / "src" / "server.py").exists()
    assert (project_dir / "tests" / "test_pipeline.py").exists()
    assert (project_dir / "Dockerfile").exists()


def test_generated_pipeline_json_is_valid():
    plan = analyze_spec("Classify legal documents.")
    out = _generate_project(plan["specification"], "pipeline-json-test", plan)
    from app.main import PROJECTS_DIR

    pipeline_path = PROJECTS_DIR / out["project_id"] / "config" / "pipeline.json"
    data = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert "selected_components" in data
    assert "pipeline" in data


def test_generated_project_metadata_fields():
    plan = analyze_spec("Classify legal documents.")
    out = _generate_project(plan["specification"], "meta-test", plan)
    from app.main import PROJECTS_DIR

    meta = json.loads(
        (PROJECTS_DIR / out["project_id"] / "project.json").read_text(encoding="utf-8")
    )
    assert meta["project_id"] == out["project_id"]
    assert "specification" in meta
    assert "components" in meta
    assert "created_at" in meta
    assert meta["status"] == "generated"


def test_generated_server_references_model_for_classifier_projects():
    """
    When text_classifier is in the plan, the generated server.py must reference
    model.joblib — it is no longer allowed to call the dummy pipeline.run().
    """
    plan = analyze_spec("Classify legal documents and expose a REST API.")
    out = _generate_project(plan["specification"], "server-check", plan)
    from app.main import PROJECTS_DIR

    server_src = (PROJECTS_DIR / out["project_id"] / "src" / "server.py").read_text(
        encoding="utf-8"
    )
    assert "model.joblib" in server_src, (
        "Generated server.py must reference model.joblib for classifier projects"
    )
    assert "pipeline.run" not in server_src, (
        "Generated server.py must not call the unimplemented pipeline.run()"
    )


def test_generated_server_is_honest_for_catalog_only_projects():
    """
    When only catalog components are selected (no text_classifier), server.py
    must not pretend to serve predictions.
    """
    # Force a plan that contains only catalog components (no classification terms)
    plan = analyze_spec("Summarize documents using an LLM and expose an API.")
    # Manually remove text_classifier if fallback added it
    plan["components"] = [c for c in plan["components"] if c["key"] != "text_classifier"]
    plan["selected_components"] = [k for k in plan["selected_components"] if k != "text_classifier"]

    out = _generate_project(plan["specification"], "catalog-server-check", plan)
    from app.main import PROJECTS_DIR

    server_src = (PROJECTS_DIR / out["project_id"] / "src" / "server.py").read_text(
        encoding="utf-8"
    )
    assert "not_implemented" in server_src or "not yet executable" in server_src, (
        "Generated server.py for catalog-only projects must declare it is not executable"
    )


# ---------------------------------------------------------------------------
# 5. Project list endpoint
# ---------------------------------------------------------------------------

def test_projects_list_returns_list():
    r = client.get("/api/projects")
    assert r.status_code == 200
    body = r.json()
    assert "projects" in body
    assert "total" in body
    assert isinstance(body["projects"], list)
    assert body["total"] == len(body["projects"])


def test_projects_list_includes_generated_project():
    plan = analyze_spec("Classify legal documents.")
    out = _generate_project(plan["specification"], "list-test", plan)
    project_id = out["project_id"]

    r = client.get("/api/projects")
    ids = [p["project_id"] for p in r.json()["projects"]]
    assert project_id in ids


# ---------------------------------------------------------------------------
# 6. Demo data route
# ---------------------------------------------------------------------------

def test_demo_csv_is_served():
    r = client.get("/demo/legal_demo.csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")
    # Verify it is real CSV content
    first_line = r.text.split("\n")[0]
    assert "text" in first_line.lower() or "label" in first_line.lower()


def test_demo_unknown_file_returns_404():
    r = client.get("/demo/malicious.py")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 7. Training + prediction (full end-to-end)
# ---------------------------------------------------------------------------

def test_real_training_and_prediction():
    plan = client.post(
        "/api/analyze",
        json={"specification": "Classify Indian legal documents and expose a REST API."},
    ).json()
    generated = client.post(
        "/api/generate",
        json={
            "specification": plan["specification"],
            "project_name": "smoke-test",
            "plan": plan,
        },
    ).json()
    project_id = generated["project_id"]

    with DEMO_CSV.open("rb") as handle:
        response = client.post(
            f"/api/train?project_id={project_id}",
            files={"dataset": ("legal_demo.csv", handle, "text/csv")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["accuracy"] > 0.80
    assert body["metrics"]["f1"] > 0.80

    prediction = client.post(
        "/api/predict",
        json={
            "project_id": project_id,
            "text": "The petitioner respectfully submits this petition before the competent court.",
        },
    )
    assert prediction.status_code == 200
    assert prediction.json()["prediction"] == "Petition"
    # 7-class problem: random baseline is ~0.14; anything > 0.25 indicates
    # the model is well above chance.
    assert prediction.json()["confidence"] > 0.25
    assert len(prediction.json()["top_predictions"]) == 3


def test_trained_project_metadata_updated():
    """After training, project.json status must be 'trained'."""
    plan = analyze_spec("Classify legal documents.")
    out = _generate_project(plan["specification"], "status-update-test", plan)
    project_id = out["project_id"]

    with DEMO_CSV.open("rb") as handle:
        client.post(
            f"/api/train?project_id={project_id}",
            files={"dataset": ("legal_demo.csv", handle, "text/csv")},
        )

    r = client.get(f"/api/projects/{project_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["trained"] is True
    assert "metrics" in body
    assert body["status"] == "trained"


def test_predict_without_trained_model_returns_404():
    plan = analyze_spec("Classify legal documents.")
    out = _generate_project(plan["specification"], "no-model-test", plan)
    r = client.post(
        "/api/predict",
        json={"project_id": out["project_id"], "text": "Some text"},
    )
    assert r.status_code == 404


def test_train_invalid_csv_missing_columns():
    plan = analyze_spec("Classify legal documents.")
    out = _generate_project(plan["specification"], "bad-csv-test", plan)
    bad_csv = b"col_a,col_b\nfoo,bar\nbaz,qux\n" * 3
    r = client.post(
        f"/api/train?project_id={out['project_id']}",
        files={"dataset": ("bad.csv", bad_csv, "text/csv")},
    )
    assert r.status_code == 400
    assert "text" in r.json()["detail"].lower() or "label" in r.json()["detail"].lower()


def test_train_too_few_rows_returns_400():
    plan = analyze_spec("Classify legal documents.")
    out = _generate_project(plan["specification"], "tiny-csv-test", plan)
    tiny_csv = b"text,label\nhello,A\nworld,B\n"
    r = client.post(
        f"/api/train?project_id={out['project_id']}",
        files={"dataset": ("tiny.csv", tiny_csv, "text/csv")},
    )
    assert r.status_code == 400
