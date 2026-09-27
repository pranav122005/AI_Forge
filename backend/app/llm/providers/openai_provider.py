"""
AIForge OpenAI LLM Provider
===========================

OpenAI-compatible LLM provider implementation using the official OpenAI Python SDK.
Features retries, timeout enforcement, secret redaction, and schema parsing.
Configured via environment variables:
- OPENAI_API_KEY (required)
- OPENAI_MODEL (optional, default: 'gpt-4o-mini')
- OPENAI_BASE_URL (optional)
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.providers.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMParseError,
    LLMProviderError,
)
from app.security import redact_secrets

T = TypeVar("T", bound=BaseModel)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
SUPPORTED_OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5",
    "gpt-5-mini",
    "gpt-3.5-turbo",
]


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation using OpenAI SDK with structured JSON parsing.
    """

    provider_id: str = "openai"
    display_name: str = "OpenAI"
    default_model: str = DEFAULT_OPENAI_MODEL
    supported_models: List[str] = SUPPORTED_OPENAI_MODELS

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 45.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY environment variable is not set"
            )

        self.model = model if model is not None else os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.base_url = base_url if base_url is not None else os.getenv("OPENAI_BASE_URL")
        self.max_retries = max_retries
        self.timeout = timeout

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """
        Generate a structured Pydantic model instance from prompt using OpenAI's API.
        """
        try:
            from openai import AsyncOpenAI, OpenAIError
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'openai' package is required to use OpenAIProvider. "
                "Install it with `pip install openai`."
            ) from exc

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Attempt structured outputs parsing via OpenAI SDK
                try:
                    response = await asyncio.wait_for(
                        client.beta.chat.completions.parse(
                            model=self.model,
                            messages=messages,
                            response_format=response_model,
                        ),
                        timeout=self.timeout,
                    )
                    choice = response.choices[0]
                    if getattr(choice.message, "refusal", None):
                        raise LLMParseError(f"Model refused request: {choice.message.refusal}")

                    if choice.message.parsed is not None:
                        return choice.message.parsed

                    if choice.message.content:
                        cleaned = self._extract_json_text(choice.message.content)
                        data = json.loads(cleaned)
                        return response_model.model_validate(data)
                except (AttributeError, TypeError):
                    # Fallback to chat completion with json_object
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_format={"type": "json_object"},
                        ),
                        timeout=self.timeout,
                    )
                    content = response.choices[0].message.content
                    if not content:
                        raise LLMParseError("LLM returned empty response")

                    cleaned = self._extract_json_text(content)
                    data = json.loads(cleaned)
                    return response_model.model_validate(data)

            except (asyncio.TimeoutError, OpenAIError, Exception) as exc:
                last_exception = exc
                err_str = redact_secrets(str(exc), extra_secrets=[self.api_key])
                is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
                is_transient = is_rate_limit or isinstance(exc, asyncio.TimeoutError) or "500" in err_str or "503" in err_str

                if is_transient and attempt < self.max_retries:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                break

        final_err = redact_secrets(str(last_exception), extra_secrets=[self.api_key]) if last_exception else "LLM returned empty response"
        if isinstance(last_exception, (json.JSONDecodeError, ValidationError)):
            raise LLMParseError(f"Failed to parse LLM response into {response_model.__name__}: {final_err}") from last_exception
        raise LLMProviderError(f"OpenAI API call failed after {self.max_retries} attempt(s): {final_err}") from last_exception

    async def test_connection(self) -> Dict[str, Any]:
        """
        Perform a live lightweight test with OpenAI.
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
