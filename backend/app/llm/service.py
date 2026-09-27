"""
AIForge LLM Service & Multi-Provider Integration
==================================================

High-level LLM requirement parsing service and multi-provider factory.
Supports Google Gemini, OpenAI, Anthropic Claude, and deterministic fallback provider selection.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    _backend_env = Path(__file__).resolve().parents[1] / ".env"
    _root_env = Path(__file__).resolve().parents[2] / ".env"
    if _backend_env.exists():
        load_dotenv(_backend_env)
    if _root_env.exists():
        load_dotenv(_root_env)
    load_dotenv()
except Exception:
    pass

from app.llm.factory import LLMProviderFactory
from app.llm.models import RequirementSpec
from app.llm.providers.anthropic_provider import AnthropicProvider
from app.llm.providers.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMParseError,
    LLMProviderError,
)
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.registry import ProviderInfo, registry

DEFAULT_SYSTEM_PROMPT = """\
You are AIForge Requirement Analyzer.
Your job is to analyze natural language user requirements for AI engineering systems
and extract a structured specification object matching the RequirementSpec schema.

Capabilities to detect:
- requires_vision: True if image/video processing (OpenCV) is requested.
- requires_ocr: True if text extraction from images/scans is requested.
- requires_classification: True if document/text classification is requested.
- requires_llm: True if text generation/summarization/QA using an LLM is requested.
- requires_retrieval: True if semantic search/vector retrieval/RAG is requested.
- requires_api: True if a REST API serving layer is requested.

Return ONLY structured JSON conforming to RequirementSpec.
"""


def get_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[BaseLLMProvider]:
    """
    Instantiate configured LLM provider based on requested name, session storage, or environment variables.
    """
    if provider_name and provider_name.lower().strip() == "fallback":
        return None
    return registry.get_provider(provider_id=provider_name, api_key=api_key, model=model)


def get_default_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Get active LLM provider, raising LLMConfigurationError if unconfigured or fallback active.
    """
    provider = get_provider(provider_name=provider_name, api_key=api_key, model=model)
    if provider is None:
        active = provider_name or registry.get_active_provider_id()
        raise LLMConfigurationError(
            f"No active LLM provider configured for '{active}'. Please configure GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
        )
    return provider


def get_provider_info() -> Dict[str, Any]:
    """
    Get safe, redacted provider metadata for health and capabilities endpoints.
    """
    active_id = registry.get_active_provider_id()
    provider = registry.get_provider(active_id)
    if provider is None:
        return {
            "provider": active_id or "unconfigured",
            "available": False,
            "model": "none",
        }

    return {
        "provider": getattr(provider, "provider_id", active_id),
        "available": True,
        "model": getattr(provider, "model", "unknown"),
        "name": getattr(provider, "display_name", active_id),
    }


async def parse_requirement(
    user_text: str,
    provider: Optional[BaseLLMProvider] = None,
) -> RequirementSpec:
    """
    Parse natural-language user requirement text into a validated RequirementSpec.
    """
    if not user_text or not user_text.strip():
        raise ValueError("Requirement text cannot be empty.")

    if provider is None:
        provider = get_default_provider()

    result = await provider.generate_structured(
        prompt=user_text.strip(),
        response_model=RequirementSpec,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    if not isinstance(result, RequirementSpec):
        raise LLMParseError("LLM output did not validate as RequirementSpec.")

    return result
