"""
Tests for AIForge Agent Runtime, Component Registry, Pipeline Execution & Security
"""
from __future__ import annotations

import pytest
from app.agent.context import AgentContext
from app.agent.errors import (
    AgentExecutionError,
    ComponentNotExecutableError,
    ComponentNotFoundError,
    PipelineValidationError,
)
from app.agent.pipeline import AgentPipeline, PipelineStep
from app.agent.registry import (
    BaseComponent,
    CatalogComponent,
    ComponentRegistry,
    FastAPIComponent,
    LLMComponent,
    OCRComponent,
    OpenCVComponent,
    TextClassifierComponent,
)
from app.agent.runtime import AgentRuntime


def test_agent_context_logging():
    ctx = AgentContext(project_id="test-proj", input_data="hello")
    assert ctx.project_id == "test-proj"
    assert ctx.input_data == "hello"

    ctx.set_artifact("key", "val")
    assert ctx.get_artifact("key") == "val"

    ctx.log_step("comp1", "completed", {"detail": "ok"})
    assert len(ctx.execution_log) == 1
    assert ctx.execution_log[0]["component"] == "comp1"
    assert ctx.execution_log[0]["status"] == "completed"


def test_component_registry_get():
    reg = ComponentRegistry()

    opencv_comp = reg.get("opencv")
    assert isinstance(opencv_comp, OpenCVComponent)
    assert opencv_comp.executable is True

    fastapi_comp = reg.get("fastapi")
    assert isinstance(fastapi_comp, FastAPIComponent)

    catalog_comp = reg.get("vector_db")
    assert isinstance(catalog_comp, CatalogComponent)
    assert catalog_comp.executable is False

    with pytest.raises(ComponentNotFoundError):
        reg.get("non_existent_component_xyz")


def test_catalog_component_execution_raises():
    catalog_comp = CatalogComponent("rag", "RAG Pipeline")
    ctx = AgentContext(project_id="test-proj")

    with pytest.raises(ComponentNotExecutableError) as exc_info:
        catalog_comp.execute("input", ctx)

    assert "catalog-only and not executable" in str(exc_info.value)


def test_pipeline_validation_security_check():
    # Dangerous or illegal component IDs must be rejected by pipeline validation
    unsafe_pipeline = AgentPipeline([
        PipelineStep(component_id="__import__('os').system('dir')"),
    ])

    with pytest.raises(PipelineValidationError) as exc_info:
        unsafe_pipeline.validate()

    assert "Unsafe component ID" in str(exc_info.value)


def test_pipeline_execution_success():
    reg = ComponentRegistry()

    class CustomStep(BaseComponent):
        name = "custom"
        executable = True

        def execute(self, input_data: str, context: AgentContext) -> str:
            res = input_data.upper()
            context.set_artifact("custom_out", res)
            return res

    reg.register("custom", CustomStep())

    pipeline = AgentPipeline([
        PipelineStep(component_id="custom", name="Custom"),
    ])

    ctx = AgentContext(project_id="test-proj")
    result = pipeline.execute("hello world", ctx, reg)

    assert result["status"] == "completed"
    assert ctx.get_artifact("custom_out") == "HELLO WORLD"
    assert result["steps"][0]["status"] == "completed"


def test_agent_runtime_missing_project():
    runtime = AgentRuntime()
    res = runtime.run_project("non_existent_project_id_9999", "sample text")

    assert res["status"] == "failed"
    assert "not found" in res["error"].lower()
