"""
AIForge Abstract LLM Provider Interface
========================================

Defines the contract for LLM providers. All concrete implementations
(Google Gemini, OpenAI, Anthropic Claude, local models, etc.) must implement BaseLLMProvider.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider call or generation fails."""


class LLMConfigurationError(LLMProviderError):
    """Raised when an LLM provider is missing configuration (e.g., API keys)."""


class LLMParseError(LLMProviderError, ValueError):
    """Raised when LLM output cannot be parsed or validated into the expected Pydantic model."""


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    
    Provider implementations must be replaceable and credentials must come
    from environment variables or runtime configuration.
    """

    provider_id: str = "base"
    display_name: str = "Base Provider"
    default_model: str = "default"
    supported_models: List[str] = []

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """
        Generate a structured Pydantic object from a prompt.

        Parameters
        ----------
        prompt : str
            The user prompt or requirement text.
        response_model : type[T]
            The Pydantic BaseModel class expected as output.
        system_prompt : str | None, optional
            Optional system prompt instructions.

        Returns
        -------
        T
            An instance of `response_model`.
        """
        pass

    async def test_connection(self) -> Dict[str, Any]:
        """
        Verify provider connectivity and credentials.
        Returns a dictionary with status: 'CONNECTED' or 'FAILED' and sanitized message.
        """
        class PingResponse(BaseModel):
            status: str
            service: str

        try:
            res = await self.generate_structured(
                prompt="Respond with status 'ok' and service 'AIForge' in JSON format.",
                response_model=PingResponse,
                system_prompt="You are a health probe. Output JSON.",
            )
            return {
                "provider": getattr(self, "provider_id", "unknown"),
                "model": getattr(self, "model", "default"),
                "status": "CONNECTED",
                "message": "Provider connection successful",
            }
        except Exception as exc:
            from app.security import redact_secrets
            err_msg = redact_secrets(str(exc), extra_secrets=[getattr(self, "api_key", None)])
            return {
                "provider": getattr(self, "provider_id", "unknown"),
                "model": getattr(self, "model", "default"),
                "status": "FAILED",
                "message": f"Provider request failed: {err_msg}",
            }
