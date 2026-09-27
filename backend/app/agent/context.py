"""
AIForge Agent Execution Context
===============================

Maintains the state, inputs, intermediate artifacts, and execution log during
an agent pipeline run.
"""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """
    State container passed through pipeline components during execution.
    """

    project_id: str = Field(description="Unique project identifier")
    input_data: Any = Field(default=None, description="Initial input data provided to the agent")
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Intermediate and final outputs produced by pipeline components",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata and environment info",
    )
    execution_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological log of component execution steps",
    )

    def log_step(
        self,
        component_name: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record an execution step in the context log."""
        step_record = {
            "component": component_name,
            "status": status,
            "timestamp": time.time(),
            "details": details or {},
        }
        self.execution_log.append(step_record)

    def set_artifact(self, key: str, value: Any) -> None:
        """Store an artifact produced by a component."""
        self.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored artifact."""
        return self.artifacts.get(key, default)
