"""
AIForge LLM Provider Factory
============================

Factory functions for obtaining and configuring LLM providers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.llm.providers.base import BaseLLMProvider, LLMConfigurationError
from app.llm.registry import ProviderInfo, registry


class LLMProviderFactory:
    """
    Factory class providing clean access to ProviderRegistry.
    """

    @staticmethod
    def get_provider(
        provider_id: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[BaseLLMProvider]:
        """Get an instantiated provider."""
        return registry.get_provider(provider_id=provider_id, api_key=api_key, model=model)

    @staticmethod
    def get_required_provider(
        provider_id: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> BaseLLMProvider:
        """Get an instantiated provider or raise LLMConfigurationError."""
        p = registry.get_provider(provider_id=provider_id, api_key=api_key, model=model)
        if p is None:
            active = provider_id or registry.get_active_provider_id()
            raise LLMConfigurationError(
                f"LLM Provider '{active}' is not configured or missing credentials."
            )
        return p

    @staticmethod
    def list_providers() -> List[ProviderInfo]:
        """List all available providers with configuration status."""
        return registry.list_providers()

    @staticmethod
    def set_active(provider_id: str, model: Optional[str] = None) -> None:
        """Set the globally active provider and optional model."""
        registry.set_active_provider(provider_id=provider_id, model=model)

    @staticmethod
    def configure_provider(
        provider_id: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Store session credentials for a provider."""
        registry.set_provider_config(provider_id=provider_id, api_key=api_key, model=model)

    @staticmethod
    async def test_connection(
        provider_id: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test connection to a provider."""
        return await registry.test_provider(provider_id=provider_id, api_key=api_key, model=model)
