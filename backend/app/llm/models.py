"""
AIForge LLM Data Models
=======================

Pydantic models that represent the structured output of the LLM understanding
layer.  These are the contract between the LLM provider and the AIForge Planner.

The LLM is responsible for *understanding* the user's intent.
The Planner is responsible for *engineering decisions* (component selection).
The Builder is responsible for *project generation*.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RequirementSpec(BaseModel):
    """
    Structured representation of a user's AI system requirement.

    Produced by the LLM provider (or the deterministic fallback) and
    consumed by the AIForge Planner.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "goal": "Classify Indian legal documents into categories and expose a REST API.",
                "input_type": "text",
                "output_type": "prediction",
                "tasks": ["classify"],
                "requires_classification": True,
                "requires_api": True,
                "domain": "legal",
                "language": "en",
            }
        }
    )


    # What the user wants to achieve
    goal: str = Field(
        description="A one-sentence summary of the user's goal.",
        min_length=5,
    )

    # Input characteristics
    input_type: str = Field(
        default="text",
        description=(
            "Primary input type: 'text', 'image', 'video', 'document', "
            "'pdf', 'csv', 'audio', or 'unknown'."
        ),
    )
    input_description: str = Field(
        default="",
        description="Free-form description of the input data.",
    )

    # Output characteristics
    output_type: str = Field(
        default="prediction",
        description=(
            "Primary output type: 'prediction', 'summary', 'search_results', "
            "'generated_text', 'structured_data', or 'unknown'."
        ),
    )

    # AI tasks required (ordered list)
    tasks: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of AI tasks required, e.g. "
            "['preprocess_image', 'ocr', 'classify', 'summarize']."
        ),
    )

    # Capability flags — used to guide the planner
    requires_vision: bool = Field(
        default=False,
        description="True if image/video preprocessing (OpenCV) is required.",
    )
    requires_ocr: bool = Field(
        default=False,
        description="True if OCR text extraction from images is required.",
    )
    requires_classification: bool = Field(
        default=False,
        description="True if text/document classification is required.",
    )
    requires_llm: bool = Field(
        default=False,
        description="True if an LLM (generation, summarization, QA) is required.",
    )
    requires_retrieval: bool = Field(
        default=False,
        description="True if semantic search / RAG / vector retrieval is required.",
    )
    requires_training: bool = Field(
        default=True,
        description="True if ML model training from labeled data is required.",
    )
    requires_api: bool = Field(
        default=True,
        description="True if a REST API serving layer is required.",
    )

    # Constraints and context
    constraints: list[str] = Field(
        default_factory=list,
        description="Known constraints, e.g. ['local-only', 'no-GPU', 'low-latency'].",
    )
    domain: str = Field(
        default="general",
        description="Application domain, e.g. 'legal', 'medical', 'e-commerce'.",
    )
    language: str = Field(
        default="en",
        description="Primary language of the input data.",
    )

    # Free-form context the planner may use
    notes: str = Field(
        default="",
        description="Any additional notes or context from the understanding step.",
    )



class UnderstandingResult(BaseModel):
    """
    The full response returned by the /api/understand endpoint.

    Wraps a RequirementSpec with metadata about how the understanding
    was produced (LLM vs deterministic fallback) and any warnings.
    """

    source: Literal["llm", "deterministic"] = Field(
        description=(
            "'llm' when a language model produced the understanding; "
            "'deterministic' when the rule-based fallback was used."
        ),
    )

    requirement: RequirementSpec = Field(
        description="The structured requirement specification.",
    )

    # Natural-language interpretation sent to the planner
    planner_spec: str = Field(
        description=(
            "A clean natural-language specification string derived from the "
            "RequirementSpec, ready to be passed to analyze_spec()."
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the understanding. "
            "1.0 for deterministic; 0.0–1.0 for LLM."
        ),
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about ambiguity, missing info, or degraded operation.",
    )

    raw_llm_response: str | None = Field(
        default=None,
        description="Raw LLM output before parsing (for debugging). Never logged in production.",
    )

    model_used: str | None = Field(
        default=None,
        description="LLM model identifier used, if applicable.",
    )
