"""
AIForge Planner Test Suite — Task 3

Tests the intelligent pipeline planner across all 7 mandatory specification
scenarios plus edge cases, dependency resolution, execution readiness, and
the expanded component registry.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.planner import (
    REGISTRY,
    analyze_spec,
    _resolve_dependencies,
    _sort_pipeline,
)

client = TestClient(app)


# ===========================================================================
# Helper
# ===========================================================================

def _sel(spec: str) -> list[str]:
    """Return selected_components for a spec."""
    return analyze_spec(spec)["selected_components"]


def _plan(spec: str) -> dict:
    return analyze_spec(spec)


# ===========================================================================
# Registry integrity
# ===========================================================================

class TestRegistry:
    def test_all_required_keys_present(self):
        required_ids = {
            "opencv", "ocr", "text_classifier", "llm", "embeddings",
            "vector_db", "fastapi", "summarization", "image_classification",
            "object_detection", "video_processing", "pdf_parser", "rag",
            "batch_inference",
        }
        assert required_ids.issubset(REGISTRY.keys())

    def test_every_component_has_required_fields(self):
        required_fields = {
            "name", "category", "description", "icon",
            "input_type", "output_type", "dependencies",
            "status", "executable", "tested",
        }
        for cid, comp in REGISTRY.items():
            for field in required_fields:
                assert field in comp, f"{cid} missing field '{field}'"

    def test_executable_components_are_only_text_classifier_and_fastapi(self):
        exec_ids = {cid for cid, c in REGISTRY.items() if c["executable"]}
        # OpenCV is now executable (pure Python, no system binary needed).
        # OCR is executable only when Tesseract binary is installed (False in CI).
        assert "text_classifier" in exec_ids
        assert "fastapi" in exec_ids
        assert "opencv" in exec_ids
        assert "ocr" not in exec_ids  # requires Tesseract binary

    def test_all_dependencies_reference_valid_component_ids(self):
        for cid, comp in REGISTRY.items():
            for dep in comp.get("dependencies", []):
                assert dep in REGISTRY, (
                    f"{cid} declares dependency '{dep}' which is not in REGISTRY"
                )

    def test_status_values_are_valid(self):
        valid = {"implemented", "catalog"}
        for cid, comp in REGISTRY.items():
            assert comp["status"] in valid, (
                f"{cid} has invalid status '{comp['status']}'"
            )


# ===========================================================================
# Dependency resolution
# ===========================================================================

class TestDependencyResolution:
    def test_ocr_requires_opencv(self):
        result = _resolve_dependencies(["ocr"])
        assert "opencv" in result
        assert result.index("opencv") < result.index("ocr")

    def test_vector_db_requires_embeddings(self):
        result = _resolve_dependencies(["vector_db"])
        assert "embeddings" in result
        assert result.index("embeddings") < result.index("vector_db")

    def test_rag_requires_embeddings_vector_db_llm(self):
        result = _resolve_dependencies(["rag"])
        assert "embeddings" in result
        assert "vector_db" in result
        assert "llm" in result
        assert result.index("embeddings") < result.index("rag")
        assert result.index("vector_db") < result.index("rag")

    def test_image_classification_requires_opencv(self):
        result = _resolve_dependencies(["image_classification"])
        assert "opencv" in result
        assert result.index("opencv") < result.index("image_classification")

    def test_summarization_requires_llm(self):
        result = _resolve_dependencies(["summarization"])
        assert "llm" in result
        assert result.index("llm") < result.index("summarization")

    def test_no_duplicates_in_result(self):
        result = _resolve_dependencies(["rag", "vector_db", "embeddings"])
        assert len(result) == len(set(result))

    def test_text_classifier_has_no_dependencies(self):
        result = _resolve_dependencies(["text_classifier"])
        assert result == ["text_classifier"]


# ===========================================================================
# Mandatory scenario 1: "Classify legal documents."
# ===========================================================================

class TestSpec1_ClassifyLegalDocuments:
    SPEC = "Classify legal documents."

    def test_text_classifier_selected(self):
        assert "text_classifier" in _sel(self.SPEC)

    def test_fastapi_always_present(self):
        assert "fastapi" in _sel(self.SPEC)

    def test_no_ocr_no_opencv_for_text_only(self):
        sel = _sel(self.SPEC)
        assert "ocr" not in sel
        assert "opencv" not in sel

    def test_execution_ready_true(self):
        plan = _plan(self.SPEC)
        assert plan["execution_ready"] is True

    def test_no_warnings(self):
        plan = _plan(self.SPEC)
        assert plan["warnings"] == []

    def test_text_classifier_in_implemented_components(self):
        plan = _plan(self.SPEC)
        assert "text_classifier" in plan["implemented_components"]


# ===========================================================================
# Mandatory scenario 2: "Classify legal documents and summarize them."
# ===========================================================================

class TestSpec2_ClassifyAndSummarize:
    SPEC = "Classify legal documents and summarize them."

    def test_text_classifier_selected(self):
        assert "text_classifier" in _sel(self.SPEC)

    def test_llm_selected(self):
        assert "llm" in _sel(self.SPEC)

    def test_summarization_selected(self):
        # summarization rule adds both llm and summarization
        assert "summarization" in _sel(self.SPEC)

    def test_execution_ready_false(self):
        # llm/summarization are not executable
        plan = _plan(self.SPEC)
        assert plan["execution_ready"] is False

    def test_warnings_mention_non_executable(self):
        plan = _plan(self.SPEC)
        assert len(plan["warnings"]) > 0
        combined = " ".join(plan["warnings"]).lower()
        assert "not yet executable" in combined

    def test_text_classifier_in_implemented(self):
        plan = _plan(self.SPEC)
        assert "text_classifier" in plan["implemented_components"]

    def test_llm_in_catalog(self):
        plan = _plan(self.SPEC)
        assert "llm" in plan["catalog_components"]


# ===========================================================================
# Mandatory scenario 3: "Extract text from images."
# ===========================================================================

class TestSpec3_ExtractTextFromImages:
    SPEC = "Extract text from images."

    def test_opencv_selected(self):
        assert "opencv" in _sel(self.SPEC)

    def test_ocr_selected(self):
        assert "ocr" in _sel(self.SPEC)

    def test_opencv_before_ocr(self):
        sel = _sel(self.SPEC)
        assert sel.index("opencv") < sel.index("ocr")

    def test_execution_ready_false(self):
        plan = _plan(self.SPEC)
        assert plan["execution_ready"] is False

    def test_opencv_is_now_implemented(self):
        # OpenCV is now executable — it should NOT be in catalog_components
        plan = _plan(self.SPEC)
        assert "opencv" not in plan["catalog_components"]
        assert "opencv" in plan["implemented_components"]

    def test_ocr_in_catalog_without_tesseract(self):
        # OCR requires Tesseract binary — absent in test env → still catalog
        plan = _plan(self.SPEC)
        assert "ocr" in plan["catalog_components"]


# ===========================================================================
# Mandatory scenario 4: "Extract text from images and classify the document."
# ===========================================================================

class TestSpec4_ExtractAndClassify:
    SPEC = "Extract text from images and classify the document."

    def test_opencv_selected(self):
        assert "opencv" in _sel(self.SPEC)

    def test_ocr_selected(self):
        assert "ocr" in _sel(self.SPEC)

    def test_text_classifier_selected(self):
        assert "text_classifier" in _sel(self.SPEC)

    def test_pipeline_order_opencv_ocr_classifier(self):
        sel = _sel(self.SPEC)
        assert sel.index("opencv") < sel.index("ocr")
        assert sel.index("ocr") < sel.index("text_classifier")

    def test_execution_ready_false(self):
        # opencv + ocr are not executable
        plan = _plan(self.SPEC)
        assert plan["execution_ready"] is False


# ===========================================================================
# Mandatory scenario 5: "Build a REST API for a document classifier."
# ===========================================================================

class TestSpec5_RestApiForClassifier:
    SPEC = "Build a REST API for a document classifier."

    def test_text_classifier_selected(self):
        assert "text_classifier" in _sel(self.SPEC)

    def test_fastapi_selected(self):
        assert "fastapi" in _sel(self.SPEC)

    def test_execution_ready_true(self):
        plan = _plan(self.SPEC)
        assert plan["execution_ready"] is True

    def test_classifier_before_fastapi(self):
        sel = _sel(self.SPEC)
        assert sel.index("text_classifier") < sel.index("fastapi")


# ===========================================================================
# Mandatory scenario 6: "Build a semantic document search system."
# ===========================================================================

class TestSpec6_SemanticSearch:
    SPEC = "Build a semantic document search system."

    def test_embeddings_selected(self):
        assert "embeddings" in _sel(self.SPEC)

    def test_vector_db_selected(self):
        assert "vector_db" in _sel(self.SPEC)

    def test_embeddings_before_vector_db(self):
        sel = _sel(self.SPEC)
        assert sel.index("embeddings") < sel.index("vector_db")

    def test_execution_ready_false(self):
        plan = _plan(self.SPEC)
        assert plan["execution_ready"] is False

    def test_warnings_non_empty(self):
        assert len(_plan(self.SPEC)["warnings"]) > 0


# ===========================================================================
# Mandatory scenario 7: Full pipeline spec
# ===========================================================================

class TestSpec7_FullPipeline:
    SPEC = (
        "Build an AI that analyzes photos of legal documents, "
        "extracts text, classifies them, summarizes them "
        "and exposes a REST API."
    )

    def test_opencv_selected(self):
        assert "opencv" in _sel(self.SPEC)

    def test_ocr_selected(self):
        assert "ocr" in _sel(self.SPEC)

    def test_text_classifier_selected(self):
        assert "text_classifier" in _sel(self.SPEC)

    def test_llm_selected(self):
        assert "llm" in _sel(self.SPEC)

    def test_fastapi_selected(self):
        assert "fastapi" in _sel(self.SPEC)

    def test_pipeline_order_complete(self):
        sel = _sel(self.SPEC)
        assert sel.index("opencv") < sel.index("ocr")
        assert sel.index("ocr") < sel.index("text_classifier")
        assert sel.index("text_classifier") < sel.index("fastapi")

    def test_execution_ready_false(self):
        plan = _plan(self.SPEC)
        assert plan["execution_ready"] is False

    def test_warnings_identify_non_executable_components(self):
        plan = _plan(self.SPEC)
        non_exec = plan["task"]["non_executable_components"]
        # opencv is now executable — it must NOT be in non_exec
        assert "opencv" not in non_exec
        # ocr still requires Tesseract binary → still non-executable in test env
        assert "ocr" in non_exec

    def test_text_classifier_in_implemented_components(self):
        plan = _plan(self.SPEC)
        assert "text_classifier" in plan["implemented_components"]

    def test_catalog_components_listed(self):
        plan = _plan(self.SPEC)
        # opencv is now implemented/executable
        assert "opencv" not in plan["catalog_components"]
        # ocr still requires Tesseract → catalog in test env
        assert "ocr" in plan["catalog_components"]

    def test_via_http_endpoint(self):
        r = client.post("/api/analyze", json={"specification": self.SPEC})
        assert r.status_code == 200
        body = r.json()
        assert "opencv" in body["selected_components"]
        assert "ocr" in body["selected_components"]
        assert "text_classifier" in body["selected_components"]
        assert "llm" in body["selected_components"]
        assert "fastapi" in body["selected_components"]
        assert body["execution_ready"] is False


# ===========================================================================
# Structured output fields
# ===========================================================================

class TestStructuredOutput:
    def test_response_has_input_block(self):
        plan = _plan("Classify legal documents.")
        assert "input" in plan
        assert "primary" in plan["input"]
        assert "detected_types" in plan["input"]

    def test_response_has_task_block(self):
        plan = _plan("Classify legal documents.")
        assert "task" in plan
        assert "execution_ready" in plan["task"]
        assert "non_executable_components" in plan["task"]
        assert "warnings" in plan["task"]

    def test_response_has_pipeline_steps(self):
        plan = _plan("Classify legal documents.")
        assert "pipeline_steps" in plan
        for step in plan["pipeline_steps"]:
            assert "order" in step
            assert "id" in step
            assert "name" in step
            assert "status" in step
            assert "executable" in step
            assert "reason" in step
            assert "dependencies" in step

    def test_pipeline_steps_are_ordered(self):
        plan = _plan("Extract text from images and classify documents.")
        orders = [s["order"] for s in plan["pipeline_steps"]]
        assert orders == list(range(1, len(orders) + 1))

    def test_reasons_present_for_selected_components(self):
        plan = _plan("Classify legal documents and summarize them.")
        reason_components = {r["component"] for r in plan["reasons"]}
        # At least classifier and LLM should have reasons
        assert any("Classifier" in c or "classifier" in c for c in reason_components)

    def test_execution_ready_top_level_matches_task_block(self):
        plan = _plan("Extract text from images and classify documents.")
        assert plan["execution_ready"] == plan["task"]["execution_ready"]

    def test_warnings_top_level_matches_task_block(self):
        plan = _plan("Extract text from images.")
        assert plan["warnings"] == plan["task"]["warnings"]

    def test_input_primary_is_image_for_photo_specs(self):
        plan = _plan("Analyze photos of legal documents.")
        assert plan["input"]["primary"] == "image"

    def test_input_primary_is_text_for_text_specs(self):
        plan = _plan("Classify customer feedback.")
        assert plan["input"]["primary"] == "text"


# ===========================================================================
# Auto-injection of OpenCV for image inputs
# ===========================================================================

class TestVisionAutoInjection:
    def test_opencv_injected_when_image_mentioned_without_explicit_vision(self):
        # "photo of a document" should trigger opencv auto-injection
        sel = _sel("Analyze photos of legal documents and classify them.")
        assert "opencv" in sel

    def test_opencv_injected_before_ocr(self):
        sel = _sel("Extract text from images of contracts.")
        assert "opencv" in sel
        assert sel.index("opencv") < sel.index("ocr")

    def test_opencv_not_duplicated(self):
        sel = _sel("Use OpenCV and OCR to extract text from images.")
        assert sel.count("opencv") == 1

    def test_no_opencv_for_plain_text_spec(self):
        sel = _sel("Classify customer reviews into positive, negative, neutral.")
        assert "opencv" not in sel


# ===========================================================================
# Execution readiness
# ===========================================================================

class TestExecutionReadiness:
    def test_text_only_classifier_is_ready(self):
        assert _plan("Classify legal documents.")["execution_ready"] is True

    def test_classifier_plus_api_is_ready(self):
        assert _plan(
            "Build a REST API for a document classifier."
        )["execution_ready"] is True

    def test_ocr_pipeline_is_not_ready(self):
        assert _plan("Extract text from images.")["execution_ready"] is False

    def test_llm_pipeline_is_not_ready(self):
        assert _plan("Summarize documents.")["execution_ready"] is False

    def test_embeddings_pipeline_is_not_ready(self):
        assert _plan("Search a knowledge base semantically.")["execution_ready"] is False

    def test_full_pipeline_is_not_ready(self):
        assert _plan(
            "Analyze photos, extract text, classify and summarize."
        )["execution_ready"] is False

    def test_non_executable_components_listed_in_task(self):
        plan = _plan("Extract text from images and classify them.")
        non_exec = plan["task"]["non_executable_components"]
        # opencv is now executable — must NOT be in non_exec
        assert "opencv" not in non_exec
        # ocr requires Tesseract → non-executable in test env
        assert "ocr" in non_exec
        assert "text_classifier" not in non_exec  # text_classifier IS executable


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_fallback_for_fully_generic_spec(self):
        sel = _sel("Build an AI system and expose an API.")
        assert "text_classifier" in sel
        assert "fastapi" in sel

    def test_fastapi_always_included(self):
        for spec in [
            "Classify images.",
            "Summarize documents.",
            "Search a knowledge base.",
            "Do something useful.",
            "Train a model.",
        ]:
            assert "fastapi" in _sel(spec), f"fastapi missing for: {spec!r}"

    def test_deduplication_of_repeated_terms(self):
        sel = _sel("Classify classify classify documents and expose API API API.")
        assert sel.count("text_classifier") == 1
        assert sel.count("fastapi") == 1

    def test_output_has_id_field(self):
        plan = _plan("Classify legal documents.")
        assert "id" in plan
        import uuid
        uuid.UUID(plan["id"])  # must be a valid UUID

    def test_output_has_specification_field(self):
        spec = "Classify legal documents."
        plan = _plan(spec)
        assert plan["specification"] == spec

    def test_complexity_increases_with_components(self):
        simple = _plan("Classify legal documents.")
        complex_ = _plan(
            "Analyze photos, extract text, classify, summarize, search and expose API."
        )
        complexity_rank = {"Low": 0, "Medium": 1, "High": 2}
        assert complexity_rank[complex_["estimated_complexity"]] >= complexity_rank[simple["estimated_complexity"]]

    def test_rag_spec_selects_full_retrieval_stack(self):
        sel = _sel("Build a retrieval augmented generation system for legal documents.")
        assert "embeddings" in sel
        assert "vector_db" in sel

    def test_object_detection_spec(self):
        sel = _sel("Detect objects in images and expose an API.")
        assert "object_detection" in sel
        assert "opencv" in sel

    def test_video_processing_spec(self):
        sel = _sel("Process video streams and analyze footage.")
        assert "video_processing" in sel

    def test_pdf_parsing_spec(self):
        sel = _sel("Parse PDF documents and extract text.")
        assert "pdf_parser" in sel

    def test_batch_inference_spec(self):
        sel = _sel("Run batch inference on a large dataset of documents.")
        assert "batch_inference" in sel

    def test_components_all_in_registry(self):
        """Every ID returned in selected_components must exist in REGISTRY."""
        for spec in [
            "Classify legal documents.",
            "Extract text from images and classify.",
            "Build a semantic search system.",
            "Summarize and classify documents and expose an API.",
        ]:
            plan = _plan(spec)
            for cid in plan["selected_components"]:
                assert cid in REGISTRY, f"Unknown component id '{cid}' returned for spec: {spec!r}"


# ===========================================================================
# HTTP endpoint shape
# ===========================================================================

class TestApiEndpointShape:
    def test_analyze_returns_execution_ready(self):
        r = client.post("/api/analyze", json={"specification": "Classify legal documents."})
        assert r.status_code == 200
        body = r.json()
        assert "execution_ready" in body
        assert isinstance(body["execution_ready"], bool)

    def test_analyze_returns_warnings(self):
        r = client.post("/api/analyze", json={"specification": "Extract text from images."})
        body = r.json()
        assert "warnings" in body
        assert isinstance(body["warnings"], list)

    def test_analyze_returns_pipeline_steps(self):
        r = client.post("/api/analyze", json={"specification": "Classify legal documents."})
        body = r.json()
        assert "pipeline_steps" in body
        assert len(body["pipeline_steps"]) > 0

    def test_analyze_returns_input_block(self):
        r = client.post("/api/analyze", json={"specification": "Classify legal documents."})
        body = r.json()
        assert "input" in body
        assert "primary" in body["input"]

    def test_analyze_returns_task_block(self):
        r = client.post("/api/analyze", json={"specification": "Classify legal documents."})
        body = r.json()
        assert "task" in body
        assert "execution_ready" in body["task"]

    def test_catalog_returns_all_registry_components(self):
        r = client.get("/api/catalog")
        assert r.status_code == 200
        keys = {c["key"] for c in r.json()["components"]}
        assert keys == set(REGISTRY.keys())

    def test_catalog_components_have_new_metadata_fields(self):
        r = client.get("/api/catalog")
        for comp in r.json()["components"]:
            assert "input_type" in comp, f"{comp['key']} missing input_type"
            assert "output_type" in comp, f"{comp['key']} missing output_type"
            assert "dependencies" in comp, f"{comp['key']} missing dependencies"
            assert "description" in comp, f"{comp['key']} missing description"
