"""
AIForge Agentic Development Loop Models
=========================================

Pydantic data models for agent context, planning, step execution, execution state, verification, and API payload schemas.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """Represents a single step within an AI agent execution plan."""

    id: str = Field(..., description="Unique step identifier, e.g. 'step_1'")
    description: str = Field(..., description="Description of the action to perform")
    action: str = Field(..., description="Action type: 'inspect', 'create', 'modify', 'delete', 'test', 'verify'")
    files: List[str] = Field(default_factory=list, description="List of target relative file paths")
    rationale: Optional[str] = Field(default="", description="Reasoning behind this step")
    status: str = Field(default="PENDING", description="Step status: 'PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'SKIPPED'")


class AgentPlan(BaseModel):
    """Structured execution plan produced by the Gemini agent or fallback planner."""

    summary: str = Field(..., description="High-level plan summary")
    rationale: Optional[str] = Field(default="", description="Overall architectural strategy")
    steps: List[PlanStep] = Field(default_factory=list, description="Ordered list of execution steps")
    estimated_files: List[str] = Field(default_factory=list, description="Estimated files to be affected")


class FileSelection(BaseModel):
    """Structured file relevance output."""

    relevant_files: List[str] = Field(default_factory=list, description="Paths of relevant project files")
    reason: Optional[str] = Field(default="", description="Justification for selecting these files")


class VerificationResult(BaseModel):
    """Result of the multi-check project verification stage."""

    success: bool = Field(..., description="True if all verification checks passed")
    passed_checks: List[str] = Field(default_factory=list, description="List of checks that passed")
    failed_checks: List[str] = Field(default_factory=list, description="List of checks that failed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed diagnostic information per check")


class AgentRunRequest(BaseModel):
    """Request payload to trigger the Agentic Development Loop."""

    instruction: str = Field(..., min_length=3, description="User modification/feature instruction")
    user_feedback: Optional[str] = Field(default=None, description="Optional extra user guidance")
    provider: Optional[str] = Field(default=None, description="Optional LLM provider override")
    model: Optional[str] = Field(default=None, description="Optional model override")


class AgentRunState(BaseModel):
    """Persistent execution state tracking the lifecycle of an agent run."""

    run_id: str = Field(..., description="Unique identifier for this agent run")
    project_id: str = Field(..., description="Target project ID")
    instruction: str = Field(..., description="User instruction being executed")
    status: str = Field(
        default="IDLE",
        description="Status: 'IDLE', 'ANALYZING', 'PLANNING', 'EXECUTING', 'TESTING', 'REPAIRING', 'VERIFYING', 'PACKAGING', 'COMPLETED', 'FAILED', 'ROLLED_BACK'",
    )
    current_step_index: int = Field(default=0, description="0-indexed current step position")
    total_steps: int = Field(default=0, description="Total planned steps")
    plan: Optional[AgentPlan] = Field(default=None, description="Current execution plan")
    files_created: List[str] = Field(default_factory=list, description="Files created during execution")
    files_modified: List[str] = Field(default_factory=list, description="Files modified during execution")
    files_deleted: List[str] = Field(default_factory=list, description="Files deleted during execution")
    test_result: Optional[Dict[str, Any]] = Field(default=None, description="Latest test execution outcome")
    repair_attempts: int = Field(default=0, description="Count of repair attempts made")
    verification_result: Optional[VerificationResult] = Field(default=None, description="Final verification result")
    zip_available: bool = Field(default=False, description="True if packaged ZIP artifact is ready")
    zip_path: Optional[str] = Field(default=None, description="Download API URL for packaged ZIP")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Chronological audit log")
    error: Optional[str] = Field(default=None, description="Error message if run failed")
    created_at: float = Field(default_factory=time.time, description="Start timestamp")
    updated_at: float = Field(default_factory=time.time, description="Last updated timestamp")


VALID_STATE_TRANSITIONS: Dict[str, Set[str]] = {
    "IDLE": {"QUEUED", "ANALYZING", "FAILED", "ROLLED_BACK"},
    "QUEUED": {"ANALYZING", "FAILED", "ROLLED_BACK"},
    "ANALYZING": {"PLANNING", "FAILED", "ROLLED_BACK"},
    "PLANNING": {"EXECUTING", "FAILED", "ROLLED_BACK"},
    "EXECUTING": {"TESTING", "FAILED", "ROLLED_BACK"},
    "TESTING": {"REPAIRING", "VERIFYING", "FAILED", "ROLLED_BACK"},
    "REPAIRING": {"TESTING", "FAILED", "ROLLED_BACK"},
    "VERIFYING": {"PACKAGING", "FAILED", "ROLLED_BACK"},
    "PACKAGING": {"COMPLETED", "FAILED", "ROLLED_BACK"},
    "COMPLETED": set(),
    "FAILED": set(),
    "ROLLED_BACK": set(),
}


def validate_state_transition(current_state: str, new_state: str) -> bool:
    """
    Validate that an agent state transition is permitted by the lifecycle state machine graph.
    """
    if current_state == new_state:
        return True
    allowed = VALID_STATE_TRANSITIONS.get(current_state, set())
    return new_state in allowed
