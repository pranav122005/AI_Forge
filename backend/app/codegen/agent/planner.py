"""
AIForge Agent Structured Planning Engine
========================================

Generates structured multi-step execution plans (`AgentPlan`) via Gemini provider or deterministic fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.codegen.agent.models import AgentPlan, PlanStep
from app.llm.providers.base import BaseLLMProvider

PLANNER_SYSTEM_PROMPT = """\
You are AIForge Lead Agentic Architect.
Your role is to analyze software modification instructions, inspect existing project structure and selected file contents, and generate a structured multi-step execution plan (`AgentPlan`).

Rules:
1. Output MUST strictly adhere to the `AgentPlan` Pydantic model.
2. Each step in `steps` MUST specify:
   - `id`: unique step ID (e.g., 'step_1', 'step_2')
   - `description`: clear action statement (e.g. 'Add JWT dependencies to requirements.txt')
   - `action`: one of 'inspect', 'create', 'modify', 'delete', 'test', 'verify'
   - `files`: list of target relative file paths (e.g., ['requirements.txt'], ['app/auth.py'])
   - `rationale`: architectural justification for this step
3. Step sequence MUST logically order dependencies:
   Dependency/Utility additions -> Service/Feature implementation -> API route integration -> Automated unit tests -> Final verification.
"""


async def generate_agent_plan(
    target_root: Path,
    instruction: str,
    summary: Dict[str, Any],
    selected_files: List[str],
    file_contents: str,
    provider: Optional[BaseLLMProvider] = None,
) -> AgentPlan:
    """
    Generate structured AgentPlan using LLM or rule-based fallback.
    """
    if provider is not None:
        prompt = (
            f"User Requirement / Instruction:\n{instruction}\n\n"
            f"Project Metadata Summary:\n{summary}\n\n"
            f"Selected Relevant File Contents:\n{file_contents}\n\n"
            "Generate a structured, step-by-step execution plan to implement this instruction safely and completely."
        )
        try:
            plan = await provider.generate_structured(
                prompt=prompt,
                response_model=AgentPlan,
                system_prompt=PLANNER_SYSTEM_PROMPT,
            )
            if plan and plan.steps:
                return plan
        except Exception:
            pass

    # Deterministic Fallback Planner for common instructions / offline operation
    return generate_fallback_plan(instruction, summary, selected_files)


def generate_fallback_plan(
    instruction: str,
    summary: Dict[str, Any],
    selected_files: List[str],
) -> AgentPlan:
    """
    Deterministic fallback planner creating structured plans when LLM is unavailable.
    """
    instr_lower = instruction.lower()
    entrypoint = summary.get("entrypoints", ["app/main.py"])[0] if summary.get("entrypoints") else "app/main.py"

    # Case 1: JWT Authentication
    if "jwt" in instr_lower or "auth" in instr_lower:
        return AgentPlan(
            summary="Add JWT Authentication Module and Route Security",
            rationale="Modular JWT auth service with pyjwt token creation and decode tests",
            estimated_files=["app/auth.py", "app/main.py", "tests/test_auth.py", "requirements.txt"],
            steps=[
                PlanStep(
                    id="step_1",
                    description="Add PyJWT dependency",
                    action="modify",
                    files=["requirements.txt"],
                    rationale="Required library for JWT encoding and decoding",
                ),
                PlanStep(
                    id="step_2",
                    description="Create authentication service module",
                    action="create",
                    files=["app/auth.py"],
                    rationale="Encapsulate token creation and validation logic",
                ),
                PlanStep(
                    id="step_3",
                    description="Integrate auth routes in entrypoint",
                    action="modify",
                    files=[entrypoint],
                    rationale="Expose login and token verification endpoints",
                ),
                PlanStep(
                    id="step_4",
                    description="Add automated test suite for authentication",
                    action="create",
                    files=["tests/test_auth.py"],
                    rationale="Verify JWT token issuance and invalid token handling",
                ),
                PlanStep(
                    id="step_5",
                    description="Execute project test suite and verify build",
                    action="test",
                    files=[],
                    rationale="Ensure full test suite compliance",
                ),
            ],
        )

    # Case 2: Docker Support
    if "docker" in instr_lower:
        return AgentPlan(
            summary="Add Docker Containerization Support",
            rationale="Create production Dockerfile and .dockerignore for containerized execution",
            estimated_files=["Dockerfile", ".dockerignore"],
            steps=[
                PlanStep(
                    id="step_1",
                    description="Create Dockerfile container configuration",
                    action="create",
                    files=["Dockerfile"],
                    rationale="Defines Python runtime container environment",
                ),
                PlanStep(
                    id="step_2",
                    description="Create .dockerignore build filter",
                    action="create",
                    files=[".dockerignore"],
                    rationale="Prevents copying cache and temporary artifacts into image",
                ),
                PlanStep(
                    id="step_3",
                    description="Verify container project files",
                    action="verify",
                    files=["Dockerfile"],
                    rationale="Ensure container config is syntactically valid",
                ),
            ],
        )

    # Case 3: Health Check
    if "health" in instr_lower:
        return AgentPlan(
            summary="Add /health Monitoring Endpoint",
            rationale="Expose status health endpoint and unit tests",
            estimated_files=[entrypoint, "tests/test_main.py"],
            steps=[
                PlanStep(
                    id="step_1",
                    description="Add /health route to main FastAPI app",
                    action="modify",
                    files=[entrypoint],
                    rationale="Provides status check for monitoring probes",
                ),
                PlanStep(
                    id="step_2",
                    description="Add automated test for /health endpoint",
                    action="modify",
                    files=["tests/test_main.py"],
                    rationale="Assert 200 OK HTTP response from /health",
                ),
                PlanStep(
                    id="step_3",
                    description="Execute test suite",
                    action="test",
                    files=[],
                    rationale="Validate route execution",
                ),
            ],
        )

    # Generic Default Plan
    return AgentPlan(
        summary=f"Execute feature modification: {instruction}",
        rationale="Apply structured changes, run unit tests, and verify project integrity",
        estimated_files=[entrypoint],
        steps=[
            PlanStep(
                id="step_1",
                description=f"Modify {entrypoint} for requirement",
                action="modify",
                files=[entrypoint],
                rationale="Apply code changes",
            ),
            PlanStep(
                id="step_2",
                description="Run automated pytest test suite",
                action="test",
                files=[],
                rationale="Confirm test pass",
            ),
            PlanStep(
                id="step_3",
                description="Perform final project verification",
                action="verify",
                files=[],
                rationale="Ensure no syntax errors or secret leaks",
            ),
        ],
    )
