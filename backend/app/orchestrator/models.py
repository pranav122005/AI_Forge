"""
AIForge Orchestrator Models & Lifecycle States
==============================================
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProjectLifecycle(str, Enum):
    """
    Project lifecycle states.
    """

    PLANNED = "PLANNED"
    BUILT = "BUILT"
    READY_FOR_TRAINING = "READY_FOR_TRAINING"
    TRAINED = "TRAINED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OrchestrationRequest(BaseModel):
    """
    Input request to the POST /api/generate endpoint.
    """

    requirement: str = Field(
        min_length=5,
        description="Natural-language software or AI system requirement",
    )
    project_name: str | None = Field(
        default=None,
        description="Optional project name slug",
    )


class OrchestrationResult(BaseModel):
    """
    Structured output returned by AIForgeOrchestrator.
    """

    project_id: str
    project_name: str
    requirement: dict[str, Any]
    plan: dict[str, Any]
    execution_status: str
    lifecycle_status: ProjectLifecycle
    selected_components: list[str]
    implemented_components: list[str]
    catalog_components: list[str]
    generated_files: list[str]
    training: dict[str, Any] | None = None
