"""
AIForge LLM Provider package.

Exports the public interface used by the rest of AIForge:

    from app.llm import understand_requirement, get_provider

The LLM layer converts free-form natural-language requirements into a
structured RequirementSpec that the deterministic Planner can consume.

Architecture:
    User requirement (str)
        │
        ▼
    LLMProvider.understand()        ← optional, requires credentials
        │
        ▼
    RequirementSpec (Pydantic)      ← structured understanding
        │
        ▼
    AIForge Planner                 ← deterministic component selection
        │
        ▼
    AIForge Builder                 ← project generation

If the LLM is unavailable, understand_requirement() falls back to the
deterministic provider so the rest of AIForge continues to work.
"""
from app.llm.provider import (
    DeterministicProvider,
    LLMProvider,
    UnderstandingResult,
    WatsonxProvider,
    get_provider,
    understand_requirement,
)
from app.llm.models import RequirementSpec

__all__ = [
    "LLMProvider",
    "DeterministicProvider",
    "WatsonxProvider",
    "UnderstandingResult",
    "RequirementSpec",
    "get_provider",
    "understand_requirement",
]
