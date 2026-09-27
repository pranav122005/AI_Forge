"""
AIForge Agent Runtime Exceptions
================================
"""
from __future__ import annotations


class AgentExecutionError(RuntimeError):
    """Base exception for agent execution runtime errors."""


class ComponentNotFoundError(AgentExecutionError, KeyError):
    """Raised when a requested component key is not in the component registry."""


class ComponentNotExecutableError(AgentExecutionError, NotImplementedError):
    """Raised when attempting to execute a catalog-only or un-executable component."""


class PipelineValidationError(AgentExecutionError, ValueError):
    """Raised when an agent pipeline definition is malformed, invalid, or un-allowlisted."""
