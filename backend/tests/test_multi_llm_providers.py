"""
Unit and Integration Tests for AIForge Multi-LLM Provider System
================================================================
Covers:
- Provider registry & metadata
- BaseLLMProvider contract
- Gemini, OpenAI, and Anthropic provider implementations
- LLMProviderFactory dynamic instantiation & credential management
- REST API endpoints for provider listing, configuration, activation, and connectivity testing
- Secret redaction for Anthropic and other API keys
- Backward compatibility and non-breaking default behaviors
"""
import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.llm.registry import ProviderRegistry, registry
from app.llm.factory import LLMProviderFactory
from app.llm.providers.base import BaseLLMProvider, LLMConfigurationError
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.providers.anthropic_provider import AnthropicProvider
from app.security import redact_secrets


@pytest.fixture
def client():
    return TestClient(app)


def test_base_provider_contract():
    class DummyProvider(BaseLLMProvider):
        provider_id = "dummy"
        display_name = "Dummy Provider"
        default_model = "dummy-v1"
        supported_models = ["dummy-v1", "dummy-v2"]

        async def generate_structured(self, prompt, schema_class, **kwargs):
            return schema_class()

    dp = DummyProvider()
    assert dp.provider_id == "dummy"
    assert dp.display_name == "Dummy Provider"
    assert dp.default_model == "dummy-v1"
    assert "dummy-v2" in dp.supported_models


def test_secret_redaction_anthropic_key():
    anthropic_key = "sk-ant-api03-abcdef1234567890abcdef1234567890-test1234"
    log_line = f"Failed to authenticate with token {anthropic_key} on server"
    redacted = redact_secrets(log_line)
    assert anthropic_key not in redacted
    assert "[REDACTED_ANTHROPIC_KEY]" in redacted or "[REDACTED_KEY]" in redacted


def test_registry_list_providers():
    test_reg = ProviderRegistry()
    providers = test_reg.list_providers()
    assert len(providers) >= 3
    ids = [p.id for p in providers]
    assert "gemini" in ids
    assert "openai" in ids
    assert "anthropic" in ids

    gemini_info = next(p for p in providers if p.id == "gemini")
    assert gemini_info.name == "Google Gemini"
    assert "gemini-3.8-flash" in gemini_info.models

    anthropic_info = next(p for p in providers if p.id == "anthropic")
    assert anthropic_info.name == "Anthropic Claude"
    assert "claude-3-5-sonnet-20241022" in anthropic_info.models


def test_registry_active_provider_switching():
    test_reg = ProviderRegistry()
    test_reg.set_active_provider("anthropic", "claude-3-5-sonnet-20241022")
    assert test_reg.get_active_provider_id() == "anthropic"
    assert test_reg.get_configured_model("anthropic") == "claude-3-5-sonnet-20241022"

    test_reg.set_active_provider("openai", "gpt-4o")
    assert test_reg.get_active_provider_id() == "openai"
    assert test_reg.get_configured_model("openai") == "gpt-4o"

    with pytest.raises(ValueError):
        test_reg.set_active_provider("unsupported_provider_xyz")


def test_registry_session_config():
    test_reg = ProviderRegistry()
    test_reg.set_provider_config("openai", api_key="sk-test-session-key", model="gpt-4o-mini")
    assert test_reg.is_provider_configured("openai") is True
    assert test_reg.get_configured_model("openai") == "gpt-4o-mini"


def test_factory_get_provider_fallback():
    prov = LLMProviderFactory.get_provider("fallback")
    assert prov is None


def test_anthropic_provider_mocked_generation():
    provider = AnthropicProvider(api_key="sk-ant-test-key", model="claude-3-5-sonnet-20241022")
    assert provider.provider_id == "anthropic"
    assert provider.display_name == "Anthropic Claude"

    from pydantic import BaseModel
    class TestResponseModel(BaseModel):
        answer: str
        score: int

    mock_resp_json = {
        "content": [
            {"type": "text", "text": '{"answer": "Claude generated this code", "score": 100}'}
        ]
    }

    async def _run():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_resp_json
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = await provider.generate_structured("Test prompt", TestResponseModel)
            assert isinstance(result, TestResponseModel)
            assert result.answer == "Claude generated this code"
            assert result.score == 100

    asyncio.run(_run())


def test_anthropic_provider_connection_test():
    provider = AnthropicProvider(api_key="sk-ant-test-key", model="claude-3-5-sonnet-20241022")

    mock_resp_json = {
        "content": [
            {"type": "text", "text": '{"status": "CONNECTED", "model": "claude-3-5-sonnet-20241022"}'}
        ]
    }

    async def _run():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_resp_json
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            test_res = await provider.test_connection()
            assert test_res["status"] in ("SUCCESS", "CONNECTED")
            assert test_res["provider"] == "anthropic"

    asyncio.run(_run())


def test_api_list_providers(client):
    res = client.get("/api/llm/providers")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    ids = [p["id"] for p in data]
    assert "gemini" in ids
    assert "openai" in ids
    assert "anthropic" in ids
    for p in data:
        assert "api_key" not in p  # Keys must never be exposed!


def test_api_active_provider_endpoints(client):
    res = client.get("/api/llm/active")
    assert res.status_code == 200
    data = res.json()
    assert "provider" in data
    assert "model" in data

    # Set active provider
    post_res = client.post("/api/llm/active", json={"provider": "gemini", "model": "gemini-3.8-flash"})
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "ok"
    assert post_res.json()["provider"] == "gemini"

    # Set invalid provider
    err_res = client.post("/api/llm/active", json={"provider": "nonexistent_provider"})
    assert err_res.status_code == 400


def test_api_configure_provider(client):
    res = client.post("/api/llm/providers/openai/configure", json={"api_key": "sk-test-key", "model": "gpt-4o"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o"
    assert data["configured"] is True


def test_api_test_provider_unconfigured(client):
    res = client.post("/api/llm/providers/anthropic/test", json={})
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "anthropic"


def test_api_health_and_capabilities_include_providers(client):
    h_res = client.get("/api/health")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert "providers" in h_data
    assert "active_provider" in h_data

    c_res = client.get("/api/capabilities")
    assert c_res.status_code == 200
    c_data = c_res.json()
    assert "providers" in c_data
    assert "active_provider" in c_data
