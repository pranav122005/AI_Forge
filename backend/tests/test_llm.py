"""
AIForge LLM Provider Tests — Task 6

Tests the LLM provider abstraction, deterministic fallback, Pydantic models,
/api/understand, and /api/build-intelligent endpoints.

All LLM tests use unittest.mock — no internet/API access required.
"""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app.llm import (
    DeterministicProvider,
    LLMProvider,
    RequirementSpec,
    UnderstandingResult,
    WatsonxProvider,
    get_provider,
    understand_requirement,
)
from app.llm.provider import (
    LLMParseError,
    LLMProviderError,
    _deterministic_parse,
    _parse_llm_json,
    _spec_to_planner_string,
)
from app.main import app

client = TestClient(app)


# ===========================================================================
# RequirementSpec model
# ===========================================================================

class TestRequirementSpec:
    def test_minimal_valid_spec(self):
        spec = RequirementSpec(goal="Classify legal documents.")
        assert spec.goal == "Classify legal documents."
        assert spec.input_type == "text"
        assert spec.output_type == "prediction"

    def test_all_fields_set(self):
        spec = RequirementSpec(
            goal="Extract text from images and classify them.",
            input_type="image",
            output_type="prediction",
            tasks=["preprocess_image", "ocr", "classify"],
            requires_vision=True,
            requires_ocr=True,
            requires_classification=True,
            requires_api=True,
            domain="legal",
            language="en",
        )
        assert spec.requires_vision is True
        assert spec.requires_ocr is True
        assert spec.requires_classification is True
        assert "ocr" in spec.tasks

    def test_defaults_are_sensible(self):
        spec = RequirementSpec(goal="Build an AI.")
        assert spec.requires_vision is False
        assert spec.requires_ocr is False
        assert spec.requires_classification is False
        assert spec.requires_llm is False
        assert spec.requires_api is True      # default True
        assert spec.requires_training is True  # default True

    def test_goal_min_length_enforced(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            RequirementSpec(goal="Hi")

    def test_serialisation_roundtrip(self):
        spec = RequirementSpec(
            goal="Classify legal documents.",
            requires_classification=True,
        )
        d = spec.model_dump()
        spec2 = RequirementSpec.model_validate(d)
        assert spec2.goal == spec.goal
        assert spec2.requires_classification is True


# ===========================================================================
# UnderstandingResult model
# ===========================================================================

class TestUnderstandingResult:
    def test_valid_llm_result(self):
        spec = RequirementSpec(goal="Classify documents.")
        result = UnderstandingResult(
            source="llm",
            requirement=spec,
            planner_spec="Classify documents. The system must expose a REST API.",
            confidence=0.9,
        )
        assert result.source == "llm"
        assert result.confidence == 0.9
        assert result.raw_llm_response is None

    def test_valid_deterministic_result(self):
        spec = RequirementSpec(goal="Classify documents.")
        result = UnderstandingResult(
            source="deterministic",
            requirement=spec,
            planner_spec="Classify documents.",
            confidence=1.0,
        )
        assert result.source == "deterministic"
        assert result.confidence == 1.0

    def test_confidence_bounds(self):
        spec = RequirementSpec(goal="Classify documents.")
        with pytest.raises(Exception):
            UnderstandingResult(
                source="llm",
                requirement=spec,
                planner_spec="x",
                confidence=1.5,  # > 1.0 invalid
            )

    def test_model_dump_excludes_raw_response(self):
        spec = RequirementSpec(goal="Classify documents.")
        result = UnderstandingResult(
            source="llm",
            requirement=spec,
            planner_spec="Classify documents.",
            confidence=0.8,
            raw_llm_response="some raw output",
        )
        d = result.model_dump(exclude={"raw_llm_response"})
        assert "raw_llm_response" not in d


# ===========================================================================
# LLMProvider base class
# ===========================================================================

class TestLLMProviderBase:
    def test_generate_raises_not_implemented(self):
        provider = LLMProvider()
        with pytest.raises(NotImplementedError):
            provider.generate("test prompt")

    def test_name_attribute_exists(self):
        provider = LLMProvider()
        assert hasattr(provider, "name")


# ===========================================================================
# DeterministicProvider
# ===========================================================================

class TestDeterministicProvider:
    def setup_method(self):
        self.provider = DeterministicProvider()

    def test_name_is_deterministic(self):
        assert self.provider.name == "deterministic"

    def test_generate_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.provider.generate("some prompt")

    def test_understand_returns_result(self):
        result = self.provider.understand("Classify legal documents.")
        assert isinstance(result, UnderstandingResult)

    def test_understand_source_is_deterministic(self):
        result = self.provider.understand("Classify legal documents.")
        assert result.source == "deterministic"

    def test_understand_confidence_is_1(self):
        result = self.provider.understand("Classify legal documents.")
        assert result.confidence == 1.0

    def test_understand_has_planner_spec(self):
        result = self.provider.understand("Classify legal documents.")
        assert len(result.planner_spec) > 5

    def test_understand_warns_about_no_llm(self):
        result = self.provider.understand("Classify legal documents.")
        assert len(result.warnings) > 0
        assert any("deterministic" in w.lower() or "not configured" in w.lower()
                   for w in result.warnings)

    def test_classify_spec_sets_classification(self):
        result = self.provider.understand("Classify legal documents.")
        assert result.requirement.requires_classification is True

    def test_image_spec_sets_vision(self):
        result = self.provider.understand("Analyze photos of documents.")
        assert result.requirement.requires_vision is True

    def test_summarize_spec_sets_llm(self):
        result = self.provider.understand("Summarize legal documents.")
        assert result.requirement.requires_llm is True

    def test_search_spec_sets_retrieval(self):
        result = self.provider.understand("Search a knowledge base semantically.")
        assert result.requirement.requires_retrieval is True

    def test_ocr_spec_sets_ocr(self):
        result = self.provider.understand("Extract text from scanned documents.")
        assert result.requirement.requires_ocr is True

    def test_api_spec_sets_api(self):
        result = self.provider.understand("Classify documents and expose a REST API.")
        assert result.requirement.requires_api is True

    def test_legal_domain_detected(self):
        result = self.provider.understand("Classify Indian legal documents.")
        assert result.requirement.domain == "legal"

    def test_general_domain_fallback(self):
        result = self.provider.understand("Classify some items.")
        assert result.requirement.domain == "general"

    def test_input_type_image_for_photo_spec(self):
        result = self.provider.understand("Analyze photos of documents.")
        assert result.requirement.input_type == "image"

    def test_input_type_text_default(self):
        result = self.provider.understand("Classify text documents.")
        assert result.requirement.input_type == "text"

    def test_tasks_list_non_empty(self):
        result = self.provider.understand("Classify documents.")
        assert len(result.requirement.tasks) > 0

    def test_image_pipeline_tasks_ordered(self):
        result = self.provider.understand("Extract text from images and classify them.")
        tasks = result.requirement.tasks
        if "preprocess_image" in tasks and "ocr" in tasks:
            assert tasks.index("preprocess_image") < tasks.index("ocr")


# ===========================================================================
# WatsonxProvider
# ===========================================================================

class TestWatsonxProvider:
    def test_raises_without_credentials(self):
        env = {"WATSONX_API_KEY": "", "WATSONX_PROJECT_ID": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.dict(os.environ, {"WATSONX_API_KEY": "", "WATSONX_PROJECT_ID": ""}):
                with pytest.raises(LLMProviderError):
                    WatsonxProvider(api_key="", project_id="")

    def test_is_configured_false_without_env(self):
        env_backup = {
            "WATSONX_API_KEY": os.environ.pop("WATSONX_API_KEY", ""),
            "WATSONX_PROJECT_ID": os.environ.pop("WATSONX_PROJECT_ID", ""),
        }
        try:
            assert WatsonxProvider.is_configured() is False
        finally:
            for k, v in env_backup.items():
                if v:
                    os.environ[k] = v

    def test_is_configured_true_with_env(self):
        with mock.patch.dict(os.environ, {
            "WATSONX_API_KEY": "test-key",
            "WATSONX_PROJECT_ID": "test-project",
        }):
            assert WatsonxProvider.is_configured() is True

    def test_name_is_watsonx(self):
        with mock.patch.dict(os.environ, {
            "WATSONX_API_KEY": "test-key",
            "WATSONX_PROJECT_ID": "test-project",
        }):
            provider = WatsonxProvider()
            assert provider.name == "watsonx"

    def test_generate_raises_when_sdk_unavailable(self):
        with mock.patch.dict(os.environ, {
            "WATSONX_API_KEY": "test-key",
            "WATSONX_PROJECT_ID": "test-project",
        }):
            provider = WatsonxProvider()
            with mock.patch("builtins.__import__", side_effect=ImportError("ibm_watsonx_ai")):
                with pytest.raises(LLMProviderError):
                    provider.generate("test")

    @mock.patch.dict(os.environ, {
        "WATSONX_API_KEY": "test-key",
        "WATSONX_PROJECT_ID": "test-project",
    })
    def test_understand_with_valid_mocked_response(self):
        """Full understand() flow with mocked generate()."""
        valid_json = json.dumps({
            "goal": "Classify legal documents and expose a REST API.",
            "input_type": "text",
            "output_type": "prediction",
            "tasks": ["classify", "serve_api"],
            "requires_vision": False,
            "requires_ocr": False,
            "requires_classification": True,
            "requires_llm": False,
            "requires_retrieval": False,
            "requires_training": True,
            "requires_api": True,
            "constraints": [],
            "domain": "legal",
            "language": "en",
            "notes": "",
        })
        provider = WatsonxProvider()
        with mock.patch.object(provider, "generate", return_value=valid_json):
            result = provider.understand("Classify legal documents.")
        assert result.source == "llm"
        assert result.requirement.requires_classification is True
        assert result.requirement.domain == "legal"
        assert result.confidence == 0.85


# ===========================================================================
# get_provider factory
# ===========================================================================

class TestGetProvider:
    def test_returns_deterministic_without_credentials(self):
        env_backup = {k: os.environ.pop(k, "") for k in [
            "WATSONX_API_KEY", "WATSONX_PROJECT_ID"
        ]}
        try:
            provider = get_provider()
            assert isinstance(provider, DeterministicProvider)
        finally:
            for k, v in env_backup.items():
                if v:
                    os.environ[k] = v

    def test_returns_watsonx_with_credentials(self):
        with mock.patch.dict(os.environ, {
            "WATSONX_API_KEY": "test-key",
            "WATSONX_PROJECT_ID": "test-project",
        }):
            provider = get_provider()
            assert isinstance(provider, WatsonxProvider)

    def test_falls_back_to_deterministic_on_init_failure(self):
        with mock.patch.dict(os.environ, {
            "WATSONX_API_KEY": "test-key",
            "WATSONX_PROJECT_ID": "test-project",
        }):
            with mock.patch(
                "app.llm.provider.WatsonxProvider.__init__",
                side_effect=LLMProviderError("init failed"),
            ):
                provider = get_provider()
        assert isinstance(provider, DeterministicProvider)


# ===========================================================================
# understand_requirement top-level function
# ===========================================================================

class TestUnderstandRequirement:
    def test_returns_understanding_result(self):
        result = understand_requirement("Classify legal documents.")
        assert isinstance(result, UnderstandingResult)

    def test_deterministic_fallback_always_works(self):
        # Ensure no LLM credentials are present
        env_backup = {k: os.environ.pop(k, "") for k in [
            "WATSONX_API_KEY", "WATSONX_PROJECT_ID"
        ]}
        try:
            result = understand_requirement("Classify legal documents.")
            assert result.source == "deterministic"
        finally:
            for k, v in env_backup.items():
                if v:
                    os.environ[k] = v

    def test_falls_back_when_llm_fails(self):
        with mock.patch.dict(os.environ, {
            "WATSONX_API_KEY": "bad-key",
            "WATSONX_PROJECT_ID": "bad-project",
        }):
            with mock.patch(
                "app.llm.provider.WatsonxProvider.understand",
                side_effect=LLMProviderError("network error"),
            ):
                result = understand_requirement("Classify legal documents.")
        assert result.source == "deterministic"
        # Warning about LLM failure must be present
        assert any("unavailable" in w or "failed" in w.lower() for w in result.warnings)

    def test_uses_llm_when_available(self):
        valid_json = json.dumps({
            "goal": "Classify documents.", "input_type": "text",
            "output_type": "prediction", "tasks": ["classify"],
            "requires_classification": True, "requires_api": True,
            "domain": "general", "language": "en",
        })
        with mock.patch.dict(os.environ, {
            "WATSONX_API_KEY": "test-key",
            "WATSONX_PROJECT_ID": "test-project",
        }):
            with mock.patch.object(WatsonxProvider, "generate", return_value=valid_json):
                result = understand_requirement("Classify documents.")
        assert result.source == "llm"

    def test_planner_spec_non_empty(self):
        result = understand_requirement("Classify legal documents.")
        assert len(result.planner_spec) > 5

    def test_result_has_requirement(self):
        result = understand_requirement("Classify legal documents.")
        assert isinstance(result.requirement, RequirementSpec)


# ===========================================================================
# JSON parsing helpers
# ===========================================================================

class TestParseLlmJson:
    def test_parses_valid_json(self):
        valid = json.dumps({
            "goal": "Classify documents.",
            "input_type": "text",
            "output_type": "prediction",
            "tasks": ["classify"],
            "requires_classification": True,
        })
        spec, warnings = _parse_llm_json(valid, "Classify documents.")
        assert spec.requires_classification is True
        assert warnings == []

    def test_strips_markdown_fences(self):
        fenced = "```json\n{\"goal\": \"Test goal for classify\", \"input_type\": \"text\"}\n```"
        spec, _ = _parse_llm_json(fenced, "original")
        assert spec.goal == "Test goal for classify"

    def test_extracts_json_from_mixed_text(self):
        mixed = 'Sure! Here is the JSON:\n{"goal": "Classify stuff here", "input_type": "text"}'
        spec, warnings = _parse_llm_json(mixed, "original")
        assert "Classify" in spec.goal

    def test_raises_on_no_json_object(self):
        with pytest.raises(LLMParseError):
            _parse_llm_json("This is not JSON at all.", "original")

    def test_raises_on_malformed_json(self):
        with pytest.raises(LLMParseError):
            _parse_llm_json('{"goal": "Test", "unclosed": [}', "original")

    def test_partial_parse_on_validation_error(self):
        """A goal-less JSON should produce a partial spec with a warning."""
        incomplete = json.dumps({"input_type": "text"})
        spec, warnings = _parse_llm_json(incomplete, "Original requirement text here")
        # Should not raise — should produce a partial spec
        assert isinstance(spec, RequirementSpec)
        assert len(warnings) > 0

    def test_extra_fields_are_ignored(self):
        """Pydantic should not choke on extra fields in the LLM response."""
        extra = json.dumps({
            "goal": "Classify documents here.",
            "input_type": "text",
            "UNKNOWN_FIELD": "some value",
        })
        spec, _ = _parse_llm_json(extra, "Classify documents.")
        assert spec.goal == "Classify documents here."


# ===========================================================================
# _deterministic_parse
# ===========================================================================

class TestDeterministicParse:
    def test_classify_sets_requires_classification(self):
        spec = _deterministic_parse("Classify legal documents.")
        assert spec.requires_classification is True

    def test_image_sets_requires_vision(self):
        spec = _deterministic_parse("Analyze photos of documents.")
        assert spec.requires_vision is True

    def test_summarize_sets_requires_llm(self):
        spec = _deterministic_parse("Summarize legal documents.")
        assert spec.requires_llm is True

    def test_semantic_search_sets_retrieval(self):
        spec = _deterministic_parse("Build a semantic search system.")
        assert spec.requires_retrieval is True

    def test_rest_api_sets_requires_api(self):
        spec = _deterministic_parse("Expose a REST API for the classifier.")
        assert spec.requires_api is True

    def test_pdf_sets_input_type(self):
        spec = _deterministic_parse("Parse PDF documents.")
        assert spec.input_type == "pdf"

    def test_legal_domain_detected(self):
        spec = _deterministic_parse("Classify Indian legal court documents.")
        assert spec.domain == "legal"

    def test_ocr_keyword_sets_requires_ocr(self):
        spec = _deterministic_parse("Use OCR to extract text from images.")
        assert spec.requires_ocr is True

    def test_full_pipeline_spec(self):
        spec = _deterministic_parse(
            "Analyze photos of legal documents, extract text using OCR, "
            "classify them, and expose a REST API."
        )
        assert spec.requires_vision is True
        assert spec.requires_ocr is True
        assert spec.requires_classification is True
        assert spec.requires_api is True
        assert spec.domain == "legal"


# ===========================================================================
# _spec_to_planner_string
# ===========================================================================

class TestSpecToPlannerString:
    def test_includes_goal(self):
        spec = RequirementSpec(goal="Classify legal documents.")
        s = _spec_to_planner_string(spec)
        assert "Classify legal documents." in s

    def test_adds_vision_text_when_required(self):
        spec = RequirementSpec(goal="Process images.", requires_vision=True)
        s = _spec_to_planner_string(spec)
        assert "image" in s.lower()

    def test_adds_ocr_text_when_required(self):
        spec = RequirementSpec(goal="Extract text.", requires_ocr=True)
        s = _spec_to_planner_string(spec)
        assert "ocr" in s.lower() or "extract text" in s.lower()

    def test_adds_api_text_when_required(self):
        spec = RequirementSpec(goal="Classify documents.", requires_api=True)
        s = _spec_to_planner_string(spec)
        assert "api" in s.lower() or "rest" in s.lower()

    def test_no_duplication_when_nothing_required(self):
        spec = RequirementSpec(goal="Do something useful.")
        s = _spec_to_planner_string(spec)
        # Should just be the goal
        assert s == "Do something useful."


# ===========================================================================
# Planner integration — ensure understand() + analyze_spec() chain works
# ===========================================================================

class TestPlannerIntegration:
    def test_deterministic_understanding_feeds_planner(self):
        from app.planner import analyze_spec
        result = understand_requirement("Classify legal documents and expose a REST API.")
        plan = analyze_spec(result.planner_spec)
        assert "text_classifier" in plan["selected_components"]
        assert "fastapi" in plan["selected_components"]

    def test_image_understanding_feeds_planner(self):
        from app.planner import analyze_spec
        result = understand_requirement(
            "Extract text from images and classify documents."
        )
        plan = analyze_spec(result.planner_spec)
        assert "opencv" in plan["selected_components"]
        assert "ocr" in plan["selected_components"]
        assert "text_classifier" in plan["selected_components"]

    def test_llm_json_parsing_feeds_planner(self):
        """Simulate LLM output → parse → planner_spec → Planner."""
        from app.planner import analyze_spec
        llm_output = json.dumps({
            "goal": "Classify legal documents and expose a REST API.",
            "input_type": "text",
            "output_type": "prediction",
            "tasks": ["classify", "serve_api"],
            "requires_classification": True,
            "requires_api": True,
            "domain": "legal",
            "language": "en",
        })
        spec, _ = _parse_llm_json(llm_output, "Classify legal documents.")
        planner_spec = _spec_to_planner_string(spec)
        plan = analyze_spec(planner_spec)
        assert "text_classifier" in plan["selected_components"]


# ===========================================================================
# /api/understand endpoint
# ===========================================================================

class TestUnderstandEndpoint:
    def test_returns_200(self):
        r = client.post("/api/understand", json={"spec": "Classify legal documents."})
        assert r.status_code == 200

    def test_response_has_source(self):
        r = client.post("/api/understand", json={"spec": "Classify legal documents."})
        assert "source" in r.json()
        assert r.json()["source"] in ("llm", "deterministic")

    def test_response_has_requirement(self):
        r = client.post("/api/understand", json={"spec": "Classify legal documents."})
        body = r.json()
        assert "requirement" in body
        assert "goal" in body["requirement"]

    def test_response_has_confidence(self):
        r = client.post("/api/understand", json={"spec": "Classify legal documents."})
        body = r.json()
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0

    def test_response_has_warnings(self):
        r = client.post("/api/understand", json={"spec": "Classify legal documents."})
        body = r.json()
        assert "warnings" in body
        assert isinstance(body["warnings"], list)

    def test_response_has_planner_spec(self):
        r = client.post("/api/understand", json={"spec": "Classify legal documents."})
        body = r.json()
        assert "planner_spec" in body
        assert len(body["planner_spec"]) > 5

    def test_response_no_raw_llm_response(self):
        """raw_llm_response must be excluded from the API response."""
        r = client.post("/api/understand", json={"spec": "Classify legal documents."})
        body = r.json()
        assert "raw_llm_response" not in body

    def test_short_spec_returns_422(self):
        r = client.post("/api/understand", json={"spec": "AI"})
        assert r.status_code == 422

    def test_missing_spec_returns_422(self):
        r = client.post("/api/understand", json={})
        assert r.status_code == 422

    def test_classify_spec_sets_classification(self):
        r = client.post("/api/understand", json={"spec": "Classify legal documents."})
        req = r.json()["requirement"]
        assert req["requires_classification"] is True

    def test_image_spec_sets_vision(self):
        r = client.post("/api/understand", json={
            "spec": "Analyze photos of legal documents."
        })
        req = r.json()["requirement"]
        assert req["requires_vision"] is True

    def test_api_spec_sets_api(self):
        r = client.post("/api/understand", json={
            "spec": "Classify documents and expose a REST API endpoint."
        })
        req = r.json()["requirement"]
        assert req["requires_api"] is True

    def test_deterministic_source_when_no_credentials(self):
        env_backup = {k: os.environ.pop(k, "") for k in [
            "WATSONX_API_KEY", "WATSONX_PROJECT_ID"
        ]}
        try:
            r = client.post("/api/understand", json={"spec": "Classify legal documents."})
            assert r.json()["source"] == "deterministic"
        finally:
            for k, v in env_backup.items():
                if v:
                    os.environ[k] = v

    def test_llm_source_with_mocked_credentials_and_response(self):
        valid_json = json.dumps({
            "goal": "Classify legal documents.", "input_type": "text",
            "output_type": "prediction", "tasks": ["classify"],
            "requires_classification": True, "requires_api": True,
            "domain": "legal", "language": "en",
        })
        with mock.patch.dict(os.environ, {
            "WATSONX_API_KEY": "test-key",
            "WATSONX_PROJECT_ID": "test-project",
        }):
            with mock.patch.object(WatsonxProvider, "generate", return_value=valid_json):
                r = client.post("/api/understand", json={"spec": "Classify legal documents."})
        assert r.json()["source"] == "llm"


# ===========================================================================
# /api/build-intelligent endpoint
# ===========================================================================

class TestBuildIntelligentEndpoint:
    def test_returns_200(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "test-intelligent",
            "spec": "Classify legal documents and expose a REST API.",
        })
        assert r.status_code == 200

    def test_response_has_project_id(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "test-pid-intel",
            "spec": "Classify legal documents and expose a REST API.",
        })
        assert "project_id" in r.json()

    def test_response_has_understanding(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "test-understanding",
            "spec": "Classify legal documents and expose a REST API.",
        })
        body = r.json()
        assert "understanding" in body
        assert "source" in body["understanding"]
        assert "requirement" in body["understanding"]

    def test_response_has_execution_status(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "test-status-intel",
            "spec": "Classify legal documents and expose a REST API.",
        })
        body = r.json()
        assert "execution_status" in body
        assert body["execution_status"] in ("ready", "partial", "blocked")

    def test_response_has_generated_files(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "test-files-intel",
            "spec": "Classify legal documents and expose a REST API.",
        })
        body = r.json()
        assert "generated_files" in body
        assert len(body["generated_files"]) >= 7

    def test_response_has_selected_components(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "test-comps-intel",
            "spec": "Classify legal documents and expose a REST API.",
        })
        body = r.json()
        assert "selected_components" in body
        assert len(body["selected_components"]) > 0

    def test_classifier_spec_is_ready(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "test-ready-intel",
            "spec": "Classify legal documents and expose a REST API.",
        })
        assert r.json()["execution_status"] == "ready"

    def test_short_spec_returns_422(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "bad", "spec": "Hi"
        })
        assert r.status_code == 422

    def test_understanding_source_deterministic_without_credentials(self):
        env_backup = {k: os.environ.pop(k, "") for k in [
            "WATSONX_API_KEY", "WATSONX_PROJECT_ID"
        ]}
        try:
            r = client.post("/api/build-intelligent", json={
                "project_name": "test-det",
                "spec": "Classify legal documents and expose a REST API.",
            })
            assert r.json()["understanding"]["source"] == "deterministic"
        finally:
            for k, v in env_backup.items():
                if v:
                    os.environ[k] = v

    def test_full_image_pipeline_intelligent(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "test-image-intel",
            "spec": (
                "Extract text from photos of legal documents, "
                "classify them, and expose a REST API."
            ),
        })
        assert r.status_code == 200
        body = r.json()
        sel = body["selected_components"]
        assert "opencv" in sel
        assert "ocr" in sel
        assert "text_classifier" in sel
        assert "fastapi" in sel

    def test_project_files_created_on_disk(self):
        r = client.post("/api/build-intelligent", json={
            "project_name": "disk-intel",
            "spec": "Classify legal documents and expose a REST API.",
        })
        from pathlib import Path
        project_path = Path(r.json()["project_path"])
        assert project_path.exists()
        assert (project_path / "project.json").exists()
