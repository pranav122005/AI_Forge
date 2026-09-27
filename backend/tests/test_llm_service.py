"""
Tests for Phase 6.1 — LLM Provider, Service, and FastAPI Parsing Endpoint
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.llm.models import RequirementSpec
from app.llm.providers.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMParseError,
    LLMProviderError,
)
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.service import parse_requirement
from app.main import app

client = TestClient(app)


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, return_spec: RequirementSpec | None = None, raise_exc: Exception | None = None):
        self.return_spec = return_spec
        self.raise_exc = raise_exc
        self.last_prompt = None

    async def generate_structured(self, prompt, response_model, system_prompt=None):
        self.last_prompt = prompt
        if self.raise_exc:
            raise self.raise_exc
        if self.return_spec:
            return self.return_spec
        return RequirementSpec(goal=prompt[:100] if len(prompt) >= 5 else "Default valid goal")


def test_valid_structured_response():
    expected_spec = RequirementSpec(
        goal="Classify legal documents into categories.",
        requires_classification=True,
        domain="legal",
    )
    provider = MockLLMProvider(return_spec=expected_spec)

    result = asyncio.run(parse_requirement("Classify legal documents", provider=provider))

    assert isinstance(result, RequirementSpec)
    assert result.goal == "Classify legal documents into categories."
    assert result.requires_classification is True
    assert result.domain == "legal"


def test_malformed_llm_response():
    provider = MockLLMProvider(raise_exc=LLMParseError("LLM response is not valid JSON"))

    with pytest.raises(LLMParseError) as exc_info:
        asyncio.run(parse_requirement("Do something", provider=provider))

    assert "not valid JSON" in str(exc_info.value)


def test_missing_required_fields():
    with pytest.raises(ValidationError):
        # min_length=5 for goal is violated
        RequirementSpec(goal="hi")


def test_api_requirement_detection():
    spec = RequirementSpec(
        goal="Build document classifier with REST API",
        requires_api=True,
        tasks=["classify", "serve_api"],
    )
    provider = MockLLMProvider(return_spec=spec)

    result = asyncio.run(parse_requirement("Build classifier with REST API", provider=provider))

    assert result.requires_api is True
    assert "serve_api" in result.tasks


def test_ocr_requirement_detection():
    spec = RequirementSpec(
        goal="Extract text from scanned PDF images using OCR",
        requires_ocr=True,
        requires_vision=True,
        tasks=["preprocess_image", "ocr"],
    )
    provider = MockLLMProvider(return_spec=spec)

    result = asyncio.run(parse_requirement("Extract text from scanned PDFs", provider=provider))

    assert result.requires_ocr is True
    assert result.requires_vision is True
    assert "ocr" in result.tasks


def test_text_classification_requirement():
    spec = RequirementSpec(
        goal="Categorize support tickets",
        requires_classification=True,
        tasks=["classify"],
    )
    provider = MockLLMProvider(return_spec=spec)

    result = asyncio.run(parse_requirement("Categorize support tickets", provider=provider))

    assert result.requires_classification is True
    assert "classify" in result.tasks


def test_provider_configuration_errors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMConfigurationError) as exc_info:
        OpenAIProvider(api_key=None)

    assert "OPENAI_API_KEY environment variable is not set" in str(exc_info.value)


def test_endpoint_success():
    expected_spec = RequirementSpec(
        goal="Build a legal document classifier with OCR and a REST API",
        requires_ocr=True,
        requires_classification=True,
        requires_api=True,
        domain="legal",
    )

    with patch("app.llm.service.parse_requirement", new_callable=AsyncMock) as mock_parse:
        mock_parse.return_value = expected_spec

        response = client.post(
            "/api/llm/parse",
            json={"requirement": "Build a legal document classifier with OCR and a REST API"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["goal"] == "Build a legal document classifier with OCR and a REST API"
        assert data["requires_ocr"] is True
        assert data["requires_classification"] is True
        assert data["requires_api"] is True
        assert "OPENAI_API_KEY" not in str(data)


def test_endpoint_validation_failure():
    # Prompt shorter than 5 chars
    response = client.post(
        "/api/llm/parse",
        json={"requirement": "bad"},
    )
    assert response.status_code == 422


def test_no_api_key_leakage():
    fake_key = "sk-proj-secret1234567890abcdef"
    provider = OpenAIProvider(api_key=fake_key, model="gpt-4o-mini")

    mock_client = AsyncMock()
    # Simulate an error message from OpenAI that contains the API key
    mock_error_type = type("OpenAIError", (Exception,), {})
    mock_error_inst = mock_error_type(f"Invalid API Key: {fake_key} is denied")

    mock_client.beta.chat.completions.parse.side_effect = mock_error_inst

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        with pytest.raises(LLMProviderError) as exc_info:
            asyncio.run(
                provider.generate_structured("test requirement", RequirementSpec)
            )

        err_str = str(exc_info.value)
        assert fake_key not in err_str
        assert "[REDACTED_API_KEY]" in err_str
