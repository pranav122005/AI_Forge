"""
AIForge Orchestration Exceptions
================================
"""
from __future__ import annotations


class OrchestratorError(RuntimeError):
    """Base exception for AIForge orchestrator errors."""


class RequirementAnalysisError(OrchestratorError):
    """Raised when natural-language requirement analysis fails."""


class PlanningError(OrchestratorError):
    """Raised when architecture planning fails."""


class BuildError(OrchestratorError):
    """Raised when project scaffold build fails."""
