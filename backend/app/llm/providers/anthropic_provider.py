"""
AIForge Anthropic Claude LLM Provider
======================================

Anthropic Claude LLM provider implementation using Anthropic REST API (/v1/messages) via HTTPX.
Features structured schema parsing, exponential retries, secret redaction, and timeout enforcement.
Configured via environment variables:
- ANTHROPIC_API_KEY (required)
- ANTHROPIC_MODEL (optional, default: 'claude-3-5-sonnet-20241022')
- ANTHROPIC_BASE_URL (optional, default: 'https://api.anthropic.com/v1/messages')
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.llm.providers.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMParseError,
    LLMProviderError,
)
from app.security import redact_secrets

T = TypeVar("T", bound=BaseModel)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
SUPPORTED_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    "claude-sonnet",
]


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM provider implementation.
    """

    provider_id: str = "anthropic"
    display_name: str = "Anthropic Claude"
    default_model: str = DEFAULT_ANTHROPIC_MODEL
    supported_models: List[str] = SUPPORTED_ANTHROPIC_MODELS

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 45.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY environment variable is not set"
            )

        self.model = model if model is not None else os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.base_url = (
            base_url
            if base_url is not None
            else os.getenv("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL)
        )
        self.max_retries = max_retries
        self.timeout = timeout

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """
        Generate a structured Pydantic model instance from prompt using Anthropic's Messages API.
        """
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        system_instruction = (
            f"{system_prompt}\n\n"
            f"You MUST respond ONLY with a raw, valid JSON object adhering to this JSON schema:\n{schema_json}\n"
            f"Do not include any conversational preamble or markdown code block formatting."
            if system_prompt
            else f"You MUST respond ONLY with a raw, valid JSON object adhering to this JSON schema:\n{schema_json}\n"
                 f"Do not include any conversational preamble or markdown code block formatting."
        )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system_instruction,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.base_url, headers=headers, json=payload)
                    
                    if resp.status_code == 401 or resp.status_code == 403:
                        raise LLMConfigurationError(f"Anthropic authentication failed (HTTP {resp.status_code}): Invalid API key")

                    if resp.status_code == 429:
                        raise LLMProviderError("Anthropic rate limit exceeded (HTTP 429)")

                    if resp.status_code >= 500:
                        raise LLMProviderError(f"Anthropic server error (HTTP {resp.status_code})")

                    if not resp.is_success:
                        err_text = redact_secrets(resp.text, extra_secrets=[self.api_key])
                        raise LLMProviderError(f"Anthropic API returned HTTP {resp.status_code}: {err_text}")

                    data = resp.json()
                    content_blocks = data.get("content", [])
                    raw_text = ""
                    for block in content_blocks:
                        if block.get("type") == "text":
                            raw_text += block.get("text", "")

                    if not raw_text.strip():
                        raise LLMParseError("Anthropic returned an empty response")

                    cleaned = self._extract_json_text(raw_text)
                    parsed_dict = json.loads(cleaned)
                    return response_model.model_validate(parsed_dict)

            except (httpx.TimeoutException, httpx.RequestError, Exception) as exc:
                last_exception = exc
                err_str = redact_secrets(str(exc), extra_secrets=[self.api_key])
                is_transient = isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)) or "429" in err_str or "500" in err_str or "503" in err_str

                if is_transient and attempt < self.max_retries:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                break

        final_err = redact_secrets(str(last_exception), extra_secrets=[self.api_key]) if last_exception else "Anthropic call failed"
        if isinstance(last_exception, (json.JSONDecodeError, ValidationError)):
            raise LLMParseError(f"Failed to parse Claude response into {response_model.__name__}: {final_err}") from last_exception
        raise LLMProviderError(f"Anthropic API call failed after {self.max_retries} attempt(s): {final_err}") from last_exception

    async def test_connection(self) -> Dict[str, Any]:
        """
        Perform a live lightweight test with Anthropic Claude.
        """
        class PingResponse(BaseModel):
            status: str
            model: str

        try:
            res = await self.generate_structured(
                prompt="Respond with status 'CONNECTED' and your model name in JSON format.",
                response_model=PingResponse,
                system_prompt="You are an AIForge provider test probe. Return JSON conforming to schema.",
            )
            return {
                "provider": self.provider_id,
                "model": self.model,
                "status": "CONNECTED",
                "message": "Provider connection successful",
            }
        except Exception as exc:
            err_msg = redact_secrets(str(exc), extra_secrets=[self.api_key])
            return {
                "provider": self.provider_id,
                "model": self.model,
                "status": "FAILED",
                "message": f"Provider request failed: {err_msg}",
            }

    def _extract_json_text(self, text: str) -> str:
        """Strip markdown codeblock wrappers if present e.g. ```json ... ```."""
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean = "\n".join(lines).strip()
        return clean
