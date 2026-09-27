"""
AIForge Agent Pipeline Execution Engine
=======================================

Validates and executes a sequence of pipeline steps in order.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent.context import AgentContext
from app.agent.errors import ComponentNotExecutableError, PipelineValidationError
from app.agent.registry import ComponentRegistry, default_registry


class PipelineStep(BaseModel):
    """Represents a single component step in a pipeline."""

    component_id: str = Field(description="Component key matching registry")
    name: str = Field(default="", description="Human-readable component name")
    executable: bool = Field(default=True, description="Whether step is executable")


class AgentPipeline:
    """
    Sequence of pipeline steps to be executed sequentially by AgentRuntime.
    """

    def __init__(self, steps: list[PipelineStep | dict[str, Any]] | None = None) -> None:
        self.steps: list[PipelineStep] = []
        if steps:
            for s in steps:
                if isinstance(s, PipelineStep):
                    self.steps.append(s)
                elif isinstance(s, dict):
                    comp_id = s.get("component_id") or s.get("key") or s.get("id") or s.get("name")
                    if not comp_id:
                        raise PipelineValidationError(f"Invalid pipeline step format: {s}")
                    self.steps.append(
                        PipelineStep(
                            component_id=str(comp_id),
                            name=str(s.get("name", comp_id)),
                            executable=bool(s.get("executable", True)),
                        )
                    )

    def validate(self, registry: ComponentRegistry | None = None) -> None:
        """
        Safety check: ensure all steps match allowlisted components in registry.
        """
        reg = registry or default_registry
        for step in self.steps:
            comp_id = step.component_id
            # Security safety check: reject arbitrary injection strings or dangerous symbols
            if not comp_id or not comp_id.replace("_", "").isalnum():
                raise PipelineValidationError(f"Unsafe component ID in pipeline: {comp_id}")

            try:
                comp = reg.get(comp_id)
            except Exception as exc:
                raise PipelineValidationError(
                    f"Pipeline validation failed: unknown component '{comp_id}'"
                ) from exc

            if not getattr(comp, "executable", False):
                # Catalog component step validation note
                pass

    def execute(
        self,
        input_data: Any,
        context: AgentContext,
        registry: ComponentRegistry | None = None,
    ) -> dict[str, Any]:
        """
        Execute all pipeline steps sequentially.
        """
        reg = registry or default_registry
        self.validate(reg)

        current_data = input_data
        step_log: list[dict[str, Any]] = []

        for step in self.steps:
            comp_id = step.component_id
            comp = reg.get(comp_id)

            if not getattr(comp, "executable", False):
                context.log_step(comp_id, "failed", {"error": "Component is catalog-only"})
                raise ComponentNotExecutableError(
                    f"Pipeline step '{comp_id}' is catalog-only and not executable."
                )

            try:
                comp.validate_input(current_data, context)
                output_data = comp.execute(current_data, context)
                comp.validate_output(output_data, context)

                context.log_step(comp_id, "completed")
                step_log.append({"component": comp_id, "status": "completed"})

                # Pass output as input to next step if applicable
                if output_data is not None:
                    current_data = output_data

            except Exception as exc:
                context.log_step(comp_id, "failed", {"error": str(exc)})
                step_log.append({"component": comp_id, "status": "failed", "error": str(exc)})
                raise

        # Determine final result object
        final_result = current_data
        if "prediction" in context.artifacts:
            final_result = context.artifacts["prediction"]
        elif isinstance(current_data, dict):
            final_result = current_data
        elif isinstance(current_data, str):
            final_result = {"text": current_data}

        return {
            "project_id": context.project_id,
            "status": "completed",
            "result": final_result,
            "steps": step_log,
        }
