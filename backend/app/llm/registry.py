"""
AIForge LLM Provider Registry
==============================

Central registry for managing multi-LLM providers (Google Gemini, OpenAI, Anthropic Claude),
session credentials, active provider selection, and provider metadata.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from app.llm.providers.anthropic_provider import (
    DEFAULT_ANTHROPIC_MODEL,
    SUPPORTED_ANTHROPIC_MODELS,
    AnthropicProvider,
)
from app.llm.providers.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMProviderError,
)
from app.llm.providers.gemini_provider import (
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GEMINI_MODELS,
    GeminiProvider,
)
from app.llm.providers.openai_provider import (
    DEFAULT_OPENAI_MODEL,
    SUPPORTED_OPENAI_MODELS,
    OpenAIProvider,
)


class ProviderInfo(BaseModel):
    id: str = Field(..., description="Provider unique identifier")
    name: str = Field(..., description="Provider human-readable display name")
    configured: bool = Field(..., description="Whether credentials are configured")
    models: List[str] = Field(default_factory=list, description="Supported models list")
    default_model: str = Field(..., description="Default model for this provider")
    active: bool = Field(default=False, description="Whether this is the currently active provider")


class ProviderRegistry:
    """
    Registry and runtime configuration manager for LLM providers.
    """

    def __init__(self) -> None:
        self._provider_classes: Dict[str, Type[BaseLLMProvider]] = {
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
        }
        self._provider_metadata: Dict[str, Dict[str, Any]] = {
            "gemini": {
                "name": "Google Gemini",
                "default_model": DEFAULT_GEMINI_MODEL,
                "models": SUPPORTED_GEMINI_MODELS,
                "env_key": "GEMINI_API_KEY",
                "env_model": "GEMINI_MODEL",
            },
            "openai": {
                "name": "OpenAI",
                "default_model": DEFAULT_OPENAI_MODEL,
                "models": SUPPORTED_OPENAI_MODELS,
                "env_key": "OPENAI_API_KEY",
                "env_model": "OPENAI_MODEL",
            },
            "anthropic": {
                "name": "Anthropic Claude",
                "default_model": DEFAULT_ANTHROPIC_MODEL,
                "models": SUPPORTED_ANTHROPIC_MODELS,
                "env_key": "ANTHROPIC_API_KEY",
                "env_model": "ANTHROPIC_MODEL",
            },
        }
        # In-memory runtime/session overrides (e.g. from UI)
        self._session_configs: Dict[str, Dict[str, str]] = {}
        self._active_provider: Optional[str] = None
        self._active_model: Optional[str] = None

    def list_providers(self) -> List[ProviderInfo]:
        """
        List all registered providers with their configuration status (never exposing keys!).
        """
        result = []
        for p_id, meta in self._provider_metadata.items():
            configured = self.is_provider_configured(p_id)
            active = (p_id == self.get_active_provider_id())
            current_model = self.get_configured_model(p_id)
            models = list(meta["models"])
            if current_model and current_model not in models:
                models.insert(0, current_model)

            result.append(
                ProviderInfo(
                    id=p_id,
                    name=meta["name"],
                    configured=configured,
                    models=models,
                    default_model=current_model or meta["default_model"],
                    active=active,
                )
            )
        return result

    def is_provider_configured(self, provider_id: str) -> bool:
        """
        Check if API key is present in session storage or environment variables.
        """
        p_id = provider_id.lower().strip()
        if p_id in self._session_configs and self._session_configs[p_id].get("api_key"):
            return True
        meta = self._provider_metadata.get(p_id)
        if meta and os.getenv(meta["env_key"]):
            return True
        return False

    def get_configured_model(self, provider_id: str) -> str:
        """
        Get currently active model for provider from session or env.
        """
        p_id = provider_id.lower().strip()
        if p_id == self.get_active_provider_id() and self._active_model:
            return self._active_model
        if p_id in self._session_configs and self._session_configs[p_id].get("model"):
            return self._session_configs[p_id]["model"]
        meta = self._provider_metadata.get(p_id)
        if meta:
            return os.getenv(meta["env_model"], meta["default_model"])
        return "default"

    def get_active_provider_id(self) -> str:
        """
        Get the currently active provider ID.
        """
        if self._active_provider:
            return self._active_provider

        env_pref = (os.getenv("LLM_PROVIDER") or os.getenv("DEFAULT_LLM_PROVIDER") or "").lower().strip()
        if env_pref and env_pref in self._provider_classes:
            return env_pref
        if env_pref == "fallback":
            return "fallback"

        if self.is_provider_configured("gemini"):
            return "gemini"
        if self.is_provider_configured("openai"):
            return "openai"
        if self.is_provider_configured("anthropic"):
            return "anthropic"

        return "gemini"

    def set_active_provider(self, provider_id: str, model: Optional[str] = None) -> None:
        """
        Set active provider and optional model.
        """
        p_id = provider_id.lower().strip()
        if p_id not in self._provider_classes and p_id != "fallback":
            raise ValueError(f"Unsupported LLM provider '{provider_id}'. Choose from: {list(self._provider_classes.keys())}")
        self._active_provider = p_id
        if model:
            self._active_model = model

    def set_provider_config(
        self,
        provider_id: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Save runtime credentials / model for a provider in memory session storage.
        """
        p_id = provider_id.lower().strip()
        if p_id not in self._provider_classes:
            raise ValueError(f"Unsupported LLM provider '{provider_id}'.")
        
        if p_id not in self._session_configs:
            self._session_configs[p_id] = {}
        
        if api_key and api_key.strip():
            self._session_configs[p_id]["api_key"] = api_key.strip()
        if model and model.strip():
            self._session_configs[p_id]["model"] = model.strip()

    def get_provider(
        self,
        provider_id: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[BaseLLMProvider]:
        """
        Instantiate and return the requested (or active) provider instance.
        """
        p_id = (provider_id or self.get_active_provider_id()).lower().strip()
        if p_id == "fallback":
            return None

        cls = self._provider_classes.get(p_id)
        if not cls:
            return None

        # Resolve API Key
        key = api_key
        if not key and p_id in self._session_configs:
            key = self._session_configs[p_id].get("api_key")
        if not key:
            meta = self._provider_metadata.get(p_id)
            if meta:
                key = os.getenv(meta["env_key"])

        # Resolve Model
        m = model or self.get_configured_model(p_id)

        try:
            return cls(api_key=key, model=m)
        except LLMConfigurationError:
            return None

    async def test_provider(
        self,
        provider_id: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a test connection against the specified provider.
        """
        p_id = provider_id.lower().strip()
        cls = self._provider_classes.get(p_id)
        if not cls:
            return {
                "provider": provider_id,
                "status": "FAILED",
                "message": f"Unsupported provider '{provider_id}'",
            }

        # Resolve Key
        key = api_key
        if not key and p_id in self._session_configs:
            key = self._session_configs[p_id].get("api_key")
        if not key:
            meta = self._provider_metadata.get(p_id)
            if meta:
                key = os.getenv(meta["env_key"])

        if not key:
            return {
                "provider": p_id,
                "status": "NOT_CONFIGURED",
                "message": f"No API key configured for {self._provider_metadata[p_id]['name']}. Please configure an API key.",
            }

        m = model or self.get_configured_model(p_id)

        try:
            instance = cls(api_key=key, model=m)
            return await instance.test_connection()
        except Exception as exc:
            from app.security import redact_secrets
            err_msg = redact_secrets(str(exc), extra_secrets=[key])
            return {
                "provider": p_id,
                "model": m,
                "status": "FAILED",
                "message": f"Connection test failed: {err_msg}",
            }


# Singleton registry instance
registry = ProviderRegistry()
