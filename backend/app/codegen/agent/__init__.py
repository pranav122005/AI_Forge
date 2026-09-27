"""
AIForge Agent Package
=====================

Provides autonomous Agentic Development Loop, project understanding, structured planning, multi-step execution, testing, repair, verification, and state tracking.
"""
from __future__ import annotations

from app.codegen.agent.context import (
    build_project_summary,
    read_selected_file_contents,
    select_relevant_files,
)
from app.codegen.agent.engine import AgentExecutionEngine, get_agent_run_state
from app.codegen.agent.models import (
    AgentPlan,
    AgentRunRequest,
    AgentRunState,
    FileSelection,
    PlanStep,
    VerificationResult,
)
from app.codegen.agent.planner import generate_agent_plan, generate_fallback_plan
from app.codegen.agent.verifier import verify_project

__all__ = [
    "AgentExecutionEngine",
    "get_agent_run_state",
    "build_project_summary",
    "select_relevant_files",
    "read_selected_file_contents",
    "generate_agent_plan",
    "generate_fallback_plan",
    "verify_project",
    "AgentPlan",
    "PlanStep",
    "AgentRunRequest",
    "AgentRunState",
    "FileSelection",
    "VerificationResult",
]
