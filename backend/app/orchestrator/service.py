"""
AIForge Single Orchestration Service
===================================

Coordinates LLM requirement understanding, intelligent planner component selection,
scaffold building, training configuration generation, and project lifecycle management.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.builder import build_project
from app.llm.models import RequirementSpec
from app.llm.provider import _deterministic_parse, _spec_to_planner_string
from app.llm.providers.base import LLMConfigurationError, LLMProviderError
from app.llm.service import parse_requirement
from app.orchestrator.errors import (
    BuildError,
    PlanningError,
    RequirementAnalysisError,
)
from app.orchestrator.models import ProjectLifecycle

BASE_DIR = Path(__file__).resolve().parents[2]
GENERATED_PROJECTS_DIR = BASE_DIR / "generated_projects"


class AIForgeOrchestrator:
    """
    Master end-to-end AIForge pipeline orchestrator.
    """

    async def generate_project(
        self,
        requirement: str,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the end-to-end AIForge pipeline:
        Natural language -> LLM Spec -> Planner -> Builder -> Project Scaffold.
        """
        if not requirement or len(requirement.strip()) < 5:
            raise RequirementAnalysisError("Requirement text must be at least 5 characters.")

        req_text = requirement.strip()

        # Step 1: LLM Requirement Understanding
        try:
            spec = await parse_requirement(req_text)
        except (LLMConfigurationError, LLMProviderError):
            # Fall back deterministically if LLM credentials/API are unconfigured
            spec = _deterministic_parse(req_text)
        except Exception as exc:
            raise RequirementAnalysisError(f"Failed to analyze requirement: {exc}") from exc

        # Step 2: Convert Spec to Planner Input
        planner_spec = _spec_to_planner_string(spec)

        # Step 3: Component Selection & Scaffolding Build via Builder
        p_name = project_name or "aiforge-project"
        try:
            build_res = build_project(p_name, planner_spec)
        except Exception as exc:
            raise BuildError(f"Failed to build project scaffold: {exc}") from exc

        plan = build_res.plan
        project_id = build_res.project_id
        project_dir = build_res.project_dir

        # Step 4: Training Configuration
        training_config = None
        if spec.requires_training or "text_classifier" in plan.get("selected_components", []):
            training_config = {
                "enabled": True,
                "task": "text_classification",
                "input_column": "text",
                "target_column": "label",
                "model": "tfidf_logistic_regression",
            }

        # Step 5: Determine Project Lifecycle State
        exec_status = plan.get("execution_status", "partial")
        if spec.requires_training:
            lifecycle_status = ProjectLifecycle.READY_FOR_TRAINING
        elif exec_status == "ready":
            lifecycle_status = ProjectLifecycle.READY
        else:
            lifecycle_status = ProjectLifecycle.BUILT

        # Step 6: Persist Lifecycle & Metadata to project.json
        meta_path = project_dir / "project.json"
        metadata = {
            "project_id": project_id,
            "project_name": build_res.project_name,
            "specification": req_text,
            "status": lifecycle_status.value,
            "lifecycle_status": lifecycle_status.value,
            "execution_status": exec_status,
            "execution_ready": plan.get("execution_ready", False),
            "requirement": spec.model_dump(),
            "plan": plan,
            "components": [c["name"] for c in plan.get("components", [])],
            "selected_components": plan.get("selected_components", []),
            "implemented_components": plan.get("implemented_components", []),
            "catalog_components": plan.get("catalog_components", []),
            "generated_files": build_res.generated_files,
            "training": training_config,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "project_id": project_id,
            "project_name": build_res.project_name,
            "requirement": spec.model_dump(),
            "plan": plan,
            "execution_status": exec_status,
            "lifecycle_status": lifecycle_status.value,
            "selected_components": plan.get("selected_components", []),
            "implemented_components": plan.get("implemented_components", []),
            "catalog_components": plan.get("catalog_components", []),
            "generated_files": build_res.generated_files,
            "training": training_config,
        }


# Global orchestrator instance
default_orchestrator = AIForgeOrchestrator()
