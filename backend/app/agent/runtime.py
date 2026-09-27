"""
AIForge Agent Runtime Engine
============================

Loads generated project configurations, executes pipeline steps, tracks step execution,
and handles execution errors safely.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agent.context import AgentContext
from app.agent.errors import AgentExecutionError, ComponentNotExecutableError, ComponentNotFoundError
from app.agent.pipeline import AgentPipeline, PipelineStep
from app.agent.registry import ComponentRegistry, default_registry

BASE_DIR = Path(__file__).resolve().parents[2]
GENERATED_PROJECTS_DIR = BASE_DIR / "generated_projects"
PROJECTS_DIR = BASE_DIR / "generated"


def get_project_dir(project_id: str) -> Path:
    p1 = GENERATED_PROJECTS_DIR / project_id
    if p1.exists():
        return p1
    p2 = PROJECTS_DIR / project_id
    if p2.exists():
        return p2
    return p1


class AgentRuntime:
    """
    Agent Execution Runtime Engine.
    """

    def __init__(self, registry: ComponentRegistry | None = None) -> None:
        self.registry = registry or default_registry

    def load_project_pipeline(self, project_id: str) -> AgentPipeline:
        """
        Load pipeline definition for a project from disk.
        """
        project_dir = get_project_dir(project_id)

        if not project_dir.exists():
            raise AgentExecutionError(f"Project '{project_id}' not found.")

        pipeline_json_path = project_dir / "config" / "pipeline.json"
        steps: list[PipelineStep] = []

        if pipeline_json_path.exists():
            try:
                data = json.loads(pipeline_json_path.read_text(encoding="utf-8"))
                # Handle planner analyze_spec output format
                comps = data.get("components") or data.get("pipeline") or []
                for c in comps:
                    if isinstance(c, dict):
                        key = c.get("key") or c.get("id") or c.get("name")
                        if key and key != "fastapi":
                            steps.append(
                                PipelineStep(
                                    component_id=str(key),
                                    name=str(c.get("name", key)),
                                    executable=bool(c.get("executable", True)),
                                )
                            )
                    elif isinstance(c, str) and c != "fastapi":
                        steps.append(PipelineStep(component_id=c, name=c))
            except Exception as exc:
                raise AgentExecutionError(f"Could not load pipeline config: {exc}") from exc

        # Fallback to project metadata if pipeline.json step list was empty
        if not steps:
            meta_path = project_dir / "project.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    exec_comps = meta.get("executable_components", ["text_classifier"])
                    for key in exec_comps:
                        if key != "fastapi":
                            steps.append(PipelineStep(component_id=str(key), name=str(key)))
                except Exception:
                    pass

        # Default fallback for classification pipelines
        if not steps:
            steps = [PipelineStep(component_id="text_classifier", name="Task Classifier")]

        return AgentPipeline(steps)

    def run_project(self, project_id: str, input_data: Any) -> dict[str, Any]:
        """
        Execute an agent pipeline end-to-end for a given project and input data.
        """
        context = AgentContext(project_id=project_id, input_data=input_data)
        
        # If raw bytes or image data were provided in dict input
        if isinstance(input_data, dict):
            if "image_bytes" in input_data:
                context.set_artifact("image_bytes", input_data["image_bytes"])
            if "text" in input_data:
                context.set_artifact("input_text", input_data["text"])
        elif isinstance(input_data, bytes):
            context.set_artifact("image_bytes", input_data)
        elif isinstance(input_data, str):
            context.set_artifact("input_text", input_data)

        failed_component = None
        executed_steps: list[dict[str, Any]] = []

        try:
            pipeline = self.load_project_pipeline(project_id)
            exec_result = pipeline.execute(input_data, context, self.registry)
            return exec_result

        except (ComponentNotExecutableError, ComponentNotFoundError, AgentExecutionError) as exc:
            # Find the last logged step to identify failed component
            if context.execution_log:
                failed_component = context.execution_log[-1].get("component")
            return {
                "project_id": project_id,
                "status": "failed",
                "error": str(exc),
                "failed_component": failed_component,
                "steps": [
                    {"component": log["component"], "status": log["status"]}
                    for log in context.execution_log
                ],
            }
        except Exception as exc:
            if context.execution_log:
                failed_component = context.execution_log[-1].get("component")
            return {
                "project_id": project_id,
                "status": "failed",
                "error": str(exc),
                "failed_component": failed_component,
                "steps": [
                    {"component": log["component"], "status": log["status"]}
                    for log in context.execution_log
                ],
            }


# Global runtime instance
default_runtime = AgentRuntime()
